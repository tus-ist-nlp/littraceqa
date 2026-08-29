"""The chunk_types filter, and keeping index paths apart by index_name.

Without these two, a model cannot be used at the granularity it was designed for:
* chunk_types: SPECTER2 is a whole-paper model trained on title+abstract, so
  embedding body chunks individually is off-design. This is what narrows it to the
  abstract alone.
* index_name: two variants of the same indexer (all chunks / abstract only) sitting
  side by side **overwrite each other** unless their index directories differ.
"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.chunk_filter import ALL_CHUNK_TYPES, filter_chunk_types


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"p0#c{i}", paper_id="p0", text=f"t{i}", chunk_type=t, metadata={})
        for i, t in enumerate(
            ["title_abstract", "text_span", "text_span", "table", "figure"]
        )
    ]


def test_none_keeps_every_chunk():
    assert len(filter_chunk_types(_chunks(), None)) == 5
    assert len(filter_chunk_types(_chunks(), [])) == 5


def test_filters_to_the_requested_types():
    kept = filter_chunk_types(_chunks(), ["title_abstract"])
    assert [c.chunk_type for c in kept] == ["title_abstract"]

    body = filter_chunk_types(_chunks(), ["text_span", "table", "figure"])
    assert [c.chunk_type for c in body] == ["text_span", "text_span", "table", "figure"]


def test_typo_in_chunk_types_is_rejected_loudly():
    """Failing here beats quietly building an index of zero chunks."""
    with pytest.raises(ValueError, match="unknown chunk_types"):
        filter_chunk_types(_chunks(), ["titel_abstract"])


def test_all_chunk_types_matches_what_the_chunkers_emit():
    assert "title_abstract" in ALL_CHUNK_TYPES
    assert "text_span" in ALL_CHUNK_TYPES
    assert "table" in ALL_CHUNK_TYPES
