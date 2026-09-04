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

from littraceqa.search.contracts import ALL_CHUNK_TYPES, Chunk, filter_chunk_types


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


# ---- facts more than one module has to agree on -----------------------------


def test_index_directory_layout_is_defined_once():
    """The index filenames are a contract between modules that never import each other.

    `indexes.py` and `faiss_qwen3.py` write these; `expander.py` opens them
    directly. **A rename on one side alone fails silently** — the reader just does
    not find the file — so the names live in one place and this pins their values.
    """
    from littraceqa.search import contracts

    assert contracts.INDEX_FILENAME == "index.faiss"
    assert contracts.CHUNKS_FILENAME == "chunks.jsonl"
    assert contracts.PAPERS_FILENAME == "papers.jsonl"


def test_candidate_papers_limit_is_what_the_recall_curve_stops_at():
    """scripts/evaluate.py measures no deeper than a prediction records.

    The curve used to end at a hard-coded 50 with a comment saying it matched
    CANDIDATE_PAPERS_LIMIT. It now imports it, so raising the limit moves the curve
    with it.
    """
    import importlib.util
    from pathlib import Path

    from littraceqa.search.contracts import CANDIDATE_PAPERS_LIMIT

    assert CANDIDATE_PAPERS_LIMIT == 50

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_ev", root / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    assert CANDIDATE_PAPERS_LIMIT in evaluate.CANDIDATE_RECALL_KS
    assert max(evaluate.CANDIDATE_RECALL_KS) == CANDIDATE_PAPERS_LIMIT + 20
