"""max_chunk_chars is a hard ceiling — an embedder rejects an over-length input.

The regression this guards: a heading group whose content blocks all fit under
max, but where one block fits only *just* under it. Every chunk carries the
heading text, so heading_chars + that block exceeded the ceiling, no partition
scored finite, and the DP fell back to emitting the whole heading group as a
single chunk — many times over max instead of a few chars over.
"""

import pandas as pd
import pytest

from docslicer.shared.step_08_chunk_builder import build_chunks

MAX = 3200


def _table_text(n_chars: int, seed: int, ncols: int = 3) -> str:
    """Markdown-ish table rows of exactly n_chars, split-able on newlines."""
    rows, out, i = [], 0, 0
    while out < n_chars:
        rows.append("| " + " | ".join(f"cell{seed}-{i}-{c} data data" for c in range(ncols)) + " |")
        out += len(rows[-1]) + 1
        i += 1
    return "\n".join(rows)[:n_chars]


def _blocks(heading_chars: int, content_lens: list[int]) -> pd.DataFrame:
    rows = [{"page_number": 1, "block_id": 1, "text": "H" * heading_chars,
             "block_type": "heading", "heading_id": 6}]
    for i, length in enumerate(content_lens, start=2):
        rows.append({"page_number": 1, "block_id": i, "text": _table_text(length, i),
                     "block_type": "table", "heading_id": pd.NA})
    df = pd.DataFrame(rows)
    df["embed_char_count"] = df["text"].str.len()
    return df


@pytest.mark.parametrize(
    "heading_chars, content_lens",
    [
        # The real fixture: several tables under one heading, block 1 sits in the
        # dead zone between max - heading_chars and max.
        (59, [3175, 3467, 2882, 3219, 3018, 3525, 3574]),
        # Minimal case: a single block one char inside max, heading pushes it over.
        (59, [MAX - 1]),
        # Nothing in the dead zone — must stay correct too.
        (59, [1200, 1400, 1100, 1600]),
        # Long heading squeezes the content budget hard.
        (2000, [3100, 2900, 3150]),
    ],
)
def test_no_chunk_exceeds_max_chunk_chars(heading_chars, content_lens):
    chunks = build_chunks(_blocks(heading_chars, content_lens), max_chunk_chars=MAX)

    assert not chunks.empty
    over = chunks[chunks["text"].str.len() > MAX]
    assert over.empty, (
        f"{len(over)} chunk(s) over the ceiling; largest={int(chunks['text'].str.len().max())} "
        f"(max_chunk_chars={MAX})"
    )
    # embed_char_count must agree with the text it describes
    assert (chunks["embed_char_count"] == chunks["text"].str.len()).all()


def test_oversize_group_is_split_not_collapsed():
    """The fixture group must yield many chunks, not one giant one."""
    chunks = build_chunks(_blocks(59, [3175, 3467, 2882, 3219, 3018, 3525, 3574]), max_chunk_chars=MAX)
    assert len(chunks) >= 8, f"expected the group to be partitioned, got {len(chunks)} chunk(s)"


def test_no_content_is_lost_when_splitting():
    """Splitting must preserve every non-whitespace character of the input."""
    blocks = _blocks(59, [3175, 3467, 2882])
    chunks = build_chunks(blocks, max_chunk_chars=MAX)

    src = " ".join(blocks["text"]).split()
    got = " ".join(chunks["text"]).split()
    assert set(src).issubset(set(got))
