# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Detect performance-core counts and resolve worker-pool sizes across macOS, Linux, and Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from typing import Optional


# ==================================================
# CPU TOPOLOGY / WORKER-COUNT POLICY
# ==================================================
#
# Reusable across pipelines: any stage that wants to fan work out over a process
# pool can ask for a sensible worker count without re-deriving the policy.
#
# The "performance core" count is what you usually want as the parallel width:
# on heterogeneous CPUs (Apple Silicon, Intel P/E hybrids) the efficiency cores
# are much slower at compute-bound work, so spinning up workers for them buys
# little and can hurt. Likewise hyper-threading/SMT siblings share execution
# resources, so counting logical CPUs over-commits compute-bound work.
#
# Detection is best-effort and zero-dependency (no psutil). Per platform:
#   - macOS:   `sysctl hw.perflevel0.physicalcpu` -> P-core count (P+E hybrids)
#   - Linux:   parse /sys topology for unique physical cores, then cap by the
#              scheduler affinity mask so cgroup/container CPU limits are honored
#   - Windows: PowerShell CIM query for physical NumberOfCores (summed over
#              sockets); `wmic` is intentionally avoided as it is deprecated and
#              already absent on recent Windows builds
# Every branch falls back to os.cpu_count() (logical) if its probe fails, so the
# function always returns a usable >= 1.


def _affinity_cap() -> Optional[int]:
    """Logical CPUs this process is actually allowed to run on, or None.

    On Linux this reflects cgroup/cpuset limits (Docker --cpuset, k8s, taskset),
    so it is the real ceiling regardless of how many cores the box physically
    has. Not a physical-core count -- used only to cap other estimates.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                return n
        except OSError:
            pass
    return None


def _macos_pcores() -> Optional[int]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        n = int(out.stdout.strip())
        return n if n > 0 else None
    except Exception:
        return None


def _linux_physical_cores() -> Optional[int]:
    """Count unique (package, core) pairs from sysfs -> physical cores, no SMT."""
    base = "/sys/devices/system/cpu"
    cores: set[tuple[str, str]] = set()
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for entry in entries:
        topo = f"{base}/{entry}/topology"
        try:
            with open(f"{topo}/physical_package_id") as f:
                pkg = f.read().strip()
            with open(f"{topo}/core_id") as f:
                core = f.read().strip()
        except OSError:
            continue
        cores.add((pkg, core))
    return len(cores) or None


def _windows_physical_cores() -> Optional[int]:
    """Physical cores via GetLogicalProcessorInformationEx (no SMT siblings).

    Native Win32 call through ctypes: ~sub-millisecond and no subprocess, unlike
    a PowerShell/wmic query. Counts RelationProcessorCore records, each of which
    is exactly one physical core. Handles >64 logical CPUs (processor groups),
    which the older non-Ex API and registry enumeration do not.
    """
    try:
        import ctypes
        from ctypes import wintypes

        RELATION_PROCESSOR_CORE = 0
        kernel32 = ctypes.windll.kernel32

        # First call sizes the buffer; it is expected to "fail" with
        # ERROR_INSUFFICIENT_BUFFER (122) and fill `length`.
        length = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE, None, ctypes.byref(length)
        )
        if length.value == 0:
            return None

        buf = (ctypes.c_byte * length.value)()
        if not kernel32.GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE, buf, ctypes.byref(length)
        ):
            return None

        # Walk the variable-length records: each begins with
        # DWORD Relationship, DWORD Size. We filtered to cores, so every
        # record is one physical core.
        count = 0
        offset = 0
        end = length.value
        while offset + 8 <= end:
            size = wintypes.DWORD.from_buffer(buf, offset + 4).value
            if size == 0:
                break
            count += 1
            offset += size
        return count or None
    except Exception:
        return None


@lru_cache(maxsize=1)
def performance_core_count() -> int:
    """Best-effort count of performance cores. Falls back to logical CPU count.

    Always returns >= 1. See module header for per-platform detection details.
    """
    estimate: Optional[int] = None

    if sys.platform == "darwin":
        estimate = _macos_pcores()
    elif sys.platform == "win32":
        estimate = _windows_physical_cores()
    elif sys.platform.startswith("linux"):
        estimate = _linux_physical_cores()

    if estimate is None:
        estimate = os.cpu_count() or 1

    # Never claim more cores than the scheduler will actually give us
    # (cgroup/cpuset/taskset limits in containers).
    cap = _affinity_cap()
    if cap is not None:
        estimate = min(estimate, cap)

    return max(1, estimate)


def resolve_worker_count(
    requested: Optional[int],
    *,
    n_items: int,
    reserve: int = 0,
) -> int:
    """
    Resolve a process-pool width from an optional caller request.

    requested:
      - None -> performance_core_count() (the tuned default)
      - int  -> taken as-is, then clamped
    n_items : never spawn more workers than there is work
    reserve : leave this many cores free (e.g. for other concurrent pipeline stages)

    Always returns >= 1.
    """
    if n_items <= 0:
        return 1

    if requested is None:
        requested = performance_core_count()
        if reserve:
            requested = max(1, requested - reserve)

    return max(1, min(requested, n_items))
