# SPDX-FileCopyrightText: 2026 Market Framer Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Invariants the hierarchy navigation API must hold, on every document.

These are not unit tests of individual methods — they are properties of the
whole structure. The distinction matters because every navigation method can
pass its own unit test while the structure as a whole is unusable: if chunks
reference a heading_id that never became a node, then ``chunks_under`` returns
correct results for every heading you ask about and still cannot reach most of
the document.

The load-bearing one is **reachability**: content that no heading can reach is
invisible to any agent navigating by outline, and invisible without erroring —
``get_section`` returns a short, plausible-looking section and the agent answers
from it confidently. Silent partial content is worse than a crash.

Run::

    pytest tests/test_hierarchy_invariants.py -v
    pytest tests/test_hierarchy_invariants.py -v -k reachable   # the important one
"""

from __future__ import annotations

import pytest

# Documents to check. Kept as fixture names so the session-scoped parses in
# conftest.py are shared with the rest of the suite rather than re-run.
DOC_FIXTURES = [
    "pdf_result",
    "academic_result",
    "html_result",
    "docx_result",
    "pptx_result",
]


@pytest.fixture(params=DOC_FIXTURES)
def result(request):
    """Each supported format, one at a time."""
    return request.getfixturevalue(request.param)


# ── Structural integrity ──────────────────────────────────────────────────────


def test_heading_ids_are_unique(result):
    """Two nodes sharing an id makes every id-based lookup ambiguous."""
    ids = [n.heading_id for n in result.hierarchy.flatten()]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate heading_ids: {sorted(duplicates)}"


def test_every_node_is_resolvable_by_id(result):
    """flatten() and the id lookup used by chunks_under must agree."""
    for node in result.hierarchy.flatten():
        assert result._resolve_heading(node.heading_id) is not None, (
            f"heading_id {node.heading_id} is in flatten() but not resolvable"
        )


def test_chunk_ids_reference_real_chunks(result):
    """No node may point at a chunk id that isn't in result.chunks."""
    real = {c.id for c in result.chunks}
    for node in result.hierarchy.flatten():
        phantom = set(node.chunk_ids) - real
        assert not phantom, f"heading {node.heading_id} references missing chunks: {phantom}"


def test_no_chunk_claimed_by_two_headings(result):
    """A chunk belonging to two sections would be double-counted on retrieval."""
    seen: dict[str, int] = {}
    for node in result.hierarchy.flatten():
        for cid in node.chunk_ids:
            assert cid not in seen, (
                f"chunk {cid} claimed by headings {seen[cid]} and {node.heading_id}"
            )
            seen[cid] = node.heading_id


# ── Reachability: the invariant that actually matters ─────────────────────────


def test_all_body_chunks_are_reachable_from_some_heading(result):
    """Every body chunk must be reachable by navigating down from a root.

    This is the promise the agent navigation story depends on. If it fails, an
    agent reading the outline and drilling into sections sees only a fraction of
    the document — and gets no error telling it so.

    Scoped to section == "body" deliberately: headers, footers, TOC entries and
    coverpage furniture are legitimately outside the heading tree.
    """
    reachable: set[str] = set()
    for root in result.hierarchy.roots:
        reachable.update(c.id for c in result.chunks_under(root, recursive=True))

    body = [c for c in result.chunks if c.section == "body"]
    if not body:
        pytest.skip("no body chunks in this document")

    orphans = [c for c in body if c.id not in reachable]
    orphan_tokens = sum(c.token_count for c in orphans)
    total_tokens = sum(c.token_count for c in body)
    pct = 100 * orphan_tokens / total_tokens if total_tokens else 0

    assert not orphans, (
        f"{len(orphans)}/{len(body)} body chunks unreachable from any root "
        f"({orphan_tokens:,}/{total_tokens:,} tokens, {pct:.1f}% of the document).\n"
        f"First few: {[(c.id[:8], c.heading, c.text[:40]) for c in orphans[:3]]}"
    )


def test_reachable_token_share_is_near_total(result):
    """Softer form of the above, as a coverage ratio.

    Useful as a regression tripwire once the strict test passes: a drop from
    100% to 80% is a hierarchy regression even if a few furniture chunks are
    legitimately excluded.
    """
    reachable: set[str] = set()
    for root in result.hierarchy.roots:
        reachable.update(c.id for c in result.chunks_under(root, recursive=True))

    body_tokens = sum(c.token_count for c in result.chunks if c.section == "body")
    if not body_tokens:
        pytest.skip("no body chunks")
    got = sum(
        c.token_count for c in result.chunks if c.section == "body" and c.id in reachable
    )
    share = got / body_tokens
    assert share >= 0.95, f"only {share:.1%} of body tokens reachable via the hierarchy"


def test_content_is_not_concentrated_on_one_heading(result):
    """No single heading may hold most of the document's text directly.

    The second failure mode, and the one reachability misses. A tree where every
    chunk is attached to the root title node is 100% "reachable" and still
    useless: the outline shows 30 promising subsections, every one of them
    returns nothing, and only the root has content. Reachability and
    navigability are different properties and both need asserting.
    """
    body_by_id = {c.id: c.token_count for c in result.chunks if c.section == "body"}
    total = sum(body_by_id.values())
    nodes = result.hierarchy.flatten()
    if total < 500 or len(nodes) < 5:
        pytest.skip("document too small or too flat to be meaningful")

    own = {
        n.heading_id: sum(body_by_id.get(i, 0) for i in n.chunk_ids) for n in nodes
    }
    worst = max(own, key=own.get)
    share = own[worst] / total
    assert share < 0.5, (
        f"heading {worst} holds {share:.0%} of body tokens directly, across "
        f"{len(nodes)} headings — content is not distributed over the sections"
    )


def test_most_leaf_sections_have_content(result):
    """A leaf heading with no text is a dead end in the agent's outline.

    Some genuinely empty leaves are fine (a heading immediately followed by a
    subheading, a figure-only section). A majority of them being empty means
    chunk-to-heading attribution is not working.
    """
    body_ids = {c.id for c in result.chunks if c.section == "body"}
    leaves = [n for n in result.hierarchy.flatten() if not n.children]
    if len(leaves) < 4:
        pytest.skip("too few leaf headings to be meaningful")

    empty = [n for n in leaves if not (set(n.chunk_ids) & body_ids)]
    assert len(empty) <= len(leaves) * 0.4, (
        f"{len(empty)}/{len(leaves)} leaf sections have no body text attached. "
        f"Examples: {[(n.heading_id, n.text[:35]) for n in empty[:4]]}"
    )


# ── Subtree semantics ─────────────────────────────────────────────────────────


def test_recursive_is_a_superset_of_non_recursive(result):
    """recursive=True must include everything recursive=False returns."""
    for node in result.hierarchy.flatten():
        shallow = {c.id for c in result.chunks_under(node, recursive=False)}
        deep = {c.id for c in result.chunks_under(node, recursive=True)}
        assert shallow <= deep, (
            f"heading {node.heading_id}: recursive=False returned "
            f"{shallow - deep} not present in recursive=True"
        )


def test_child_chunks_are_contained_in_parent(result):
    """A parent's recursive set must contain each child's recursive set.

    Catches broken parent/child wiring — the failure mode where drilling into a
    subsection returns content that was not in the parent you drilled from.
    """
    for node in result.hierarchy.flatten():
        parent_set = {c.id for c in result.chunks_under(node, recursive=True)}
        for child in node.children:
            child_set = {c.id for c in result.chunks_under(child, recursive=True)}
            assert child_set <= parent_set, (
                f"child {child.heading_id} has chunks outside parent {node.heading_id}: "
                f"{child_set - parent_set}"
            )


def test_subtree_chunks_are_in_document_order(result):
    """chunks_under must preserve reading order.

    An agent concatenates these into a prompt; out-of-order text reads as a
    non-sequitur and quietly degrades answers.
    """
    for node in result.hierarchy.flatten():
        indexes = [c.chunk_index for c in result.chunks_under(node, recursive=True)]
        assert indexes == sorted(indexes), (
            f"heading {node.heading_id} returned chunks out of order: {indexes[:10]}"
        )


def test_subtree_chunks_are_contiguous(result):
    """A section's chunks should form an unbroken run of chunk_index values.

    A gap means content physically inside the section was attributed elsewhere.
    Marked xfail-friendly: documents where a section is interrupted by a page
    header or floating table may legitimately produce a gap, so treat a failure
    here as a prompt to look rather than a hard defect.
    """
    for node in result.hierarchy.flatten():
        indexes = sorted(c.chunk_index for c in result.chunks_under(node, recursive=True))
        if len(indexes) < 2:
            continue
        gaps = [
            (a, b) for a, b in zip(indexes, indexes[1:]) if b - a > 1
        ]
        assert not gaps, f"heading {node.heading_id} has index gaps {gaps[:5]}"


# ── Lookup round-trips ────────────────────────────────────────────────────────


def test_find_heading_round_trips_every_node(result):
    """Searching a node's own text must return that node.

    The agent's only route from a rendered outline back to a node is by text, so
    a node that cannot find itself is unreachable in practice.
    """
    for node in result.hierarchy.flatten():
        if not node.text.strip():
            continue
        found = {n.heading_id for n in result.find_heading(node.text)}
        assert node.heading_id in found, (
            f"find_heading({node.text[:40]!r}) did not return heading {node.heading_id}"
        )


def test_outline_is_not_larger_than_the_document(result):
    """An outline that costs more than the content defeats its own purpose."""
    outline_tokens = len(result.hierarchy.to_outline()) // 4
    body_tokens = sum(c.token_count for c in result.chunks if c.section == "body")
    if body_tokens < 500:
        pytest.skip("document too small for this to be meaningful")
    assert outline_tokens < body_tokens * 0.5, (
        f"outline is {outline_tokens:,} tokens against {body_tokens:,} of body text"
    )


def test_level_matches_tree_depth(result):
    """hierarchy.level(n) must return nodes actually at depth n."""
    for depth in (1, 2, 3):
        for node in result.hierarchy.level(depth):
            assert len(node.path) == depth - 1, (
                f"level({depth}) returned heading {node.heading_id} with "
                f"ancestor path {node.path}"
            )
