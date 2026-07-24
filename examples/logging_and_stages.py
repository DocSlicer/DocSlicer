"""
logging_and_stages.py — Two ways to observe a parse in progress: step logs and on_stage.

docslicer exposes progress at two granularities:

  * Logging  — each pipeline step (box extraction, cell building, ...) logs a
    structured INFO record via timed_step(), with step_name + duration_sec.
    Fine-grained, per-orchestrator, good for perf debugging.

  * on_stage — a coarse callback fired at a handful of named checkpoints
    (extract_elements, process_layouts, extract_tables, detect_hierarchy,
    build_chunks, ...) shared across every format. Good for a progress bar
    or "which phase are we in" UI, without caring about per-step internals.

Pass --json to switch the step logs from plain text to the JSON shape a
log-aggregation backend (Datadog, CloudWatch, ELK, ...) would ingest, and
read back the structured fields (event, step_name, duration_sec) instead of
just the rendered message.

Either way, a StepDurationCollector taps every record's `duration_sec` field
to print a per-step timing table and a total at the end.

The work runs under `if __name__ == "__main__":` — docslicer parses CPU-bound
steps (PDF word extraction, OCR, ...) across a process pool, and on spawn-based
platforms (macOS, Windows) each worker re-imports this file. The guard keeps that
re-import from re-running the parse in every worker.

Usage:
    python examples/logging_and_stages.py
    python examples/logging_and_stages.py path/to/your/document.pdf
    python examples/logging_and_stages.py path/to/your/document.pdf --json
"""

import json
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docslicer

# ── Reading the structured log_data ─────────────────────────────────────────
# timed_step() attaches event/step_name/duration_sec (and any extra_meta) via
# `extra=`, which land as plain attributes on the LogRecord — the default
# formatter just doesn't render them. Two ways to get at that data:

_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class StructuredJsonFormatter(logging.Formatter):
    """Renders a record as the JSON payload a monitoring backend would ingest."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"logger": record.name, "level": record.levelname, "message": record.getMessage()}
        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_ATTRS}
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class StepDurationCollector(logging.Handler):
    """Reads record.duration_sec directly (no string parsing) to total step time."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.steps: list[tuple[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        duration = getattr(record, "duration_sec", None)
        step_name = getattr(record, "step_name", None)
        if duration is not None and step_name is not None:
            self.steps.append((step_name, duration))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    AS_JSON = "--json" in sys.argv[1:]
    SOURCE = args[0] if args else Path(__file__).parent / "sample_docs" / "financial_report.pdf"

    # ── Logging setup ────────────────────────────────────────────────────────
    # Root at WARNING so third-party libs (pikepdf, httpx, ...) stay quiet; only
    # the docslicer logger is bumped to INFO so per-step timing lines show up.

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    docslicer_logger = logging.getLogger("docslicer")
    docslicer_logger.setLevel(logging.INFO)

    if AS_JSON:
        json_handler = logging.StreamHandler()
        json_handler.setFormatter(StructuredJsonFormatter())
        docslicer_logger.addHandler(json_handler)
        docslicer_logger.propagate = False  # avoid double-printing via root's plain handler

    collector = StepDurationCollector()
    docslicer_logger.addHandler(collector)

    # ── on_stage callback ─────────────────────────────────────────────────────
    # Fired once per coarse stage; use it to drive a progress bar or print phase
    # transitions with their own timing, independent of the step-level logs above.

    _stage_t0 = perf_counter()

    def on_stage(stage: str) -> None:
        nonlocal _stage_t0
        now = perf_counter()
        print(f"[stage] {stage:<20} (+{now - _stage_t0:.3f}s)")
        _stage_t0 = now

    # ── Parse ──────────────────────────────────────────────────────────────────

    print(f"Parsing: {SOURCE}\n")
    t0 = perf_counter()
    result = docslicer.parse_document(SOURCE, on_stage=on_stage)
    elapsed = perf_counter() - t0

    print(f"\nDone in {elapsed:.2f}s — {len(result.chunks)} chunks  {len(result.blocks)} blocks  {len(result.tables)} tables")

    # ── Totals, read back from the collected records ──────────────────────────

    step_total = sum(duration for _, duration in collector.steps)
    print(f"\n{'Step':<30} Duration")
    print("-" * 42)
    for step_name, duration in collector.steps:
        print(f"{step_name:<30} {duration:>7.3f}s")
    print("-" * 42)
    print(f"{'sum of steps':<30} {step_total:>7.3f}s")
    print(f"{'wall clock':<30} {elapsed:>7.3f}s")
    print(f"{'unaccounted (I/O, browser launch, glue code)':<30} {elapsed - step_total:>7.3f}s")


if __name__ == "__main__":
    main()
