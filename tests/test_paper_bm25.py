"""Tests for deterministic paper-level BM25 aggregation."""

from __future__ import annotations

import pytest

from litqa.contracts import Chunk
from litqa.index.paper_bm25 import PaperBM25Index, aggregate_papers


def _chunk(
    chunk_id: str,
    paper_id: str,
    text: str,
    chunk_type: str = "text_span",
    **metadata,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        chunk_type=chunk_type,
        metadata={"title": f"Title {paper_id}", **metadata},
    )


def test_aggregate_papers_removes_repeated_prefix_and_references():
    chunks = [
        _chunk("p1#c0", "p1", "[ACL 2025] Title p1\nAbstract text", "title_abstract"),
        _chunk("p1#c1", "p1", "[ACL 2025] Title p1\n1 Method", text_level=2),
        _chunk("p1#c2", "p1", "[ACL 2025] Title p1\nUseful method details"),
        _chunk("p1#c3", "p1", "[ACL 2025] Title p1\nReferences", text_level=2),
        _chunk("p1#c4", "p1", "[ACL 2025] Title p1\nUnrelated citation title"),
    ]

    papers = list(aggregate_papers(chunks))

    assert len(papers) == 1
    assert papers[0].chunk_id == "p1#paper"
    assert papers[0].text.count("Title p1") == 1
    assert "1 Method\nUseful method details" in papers[0].text
    assert "Unrelated citation title" not in papers[0].text
    assert papers[0].metadata["source_chunk_count"] == 5


def test_aggregate_papers_rejects_noncontiguous_papers():
    chunks = [
        _chunk("p1#c1", "p1", "prefix\none"),
        _chunk("p2#c1", "p2", "prefix\ntwo"),
        _chunk("p1#c2", "p1", "prefix\nthree"),
    ]

    with pytest.raises(ValueError, match="contiguous"):
        list(aggregate_papers(chunks))


def test_paper_bm25_build_load_and_search(tmp_path):
    chunks = [
        _chunk("p1#c1", "p1", "prefix\nlayer parallel speculative decoding"),
        _chunk("p2#c1", "p2", "prefix\nvisual object detection"),
    ]
    index_dir = tmp_path / "paper-bm25"
    index = PaperBM25Index(str(index_dir), result_text_chars=20)
    index.build(chunks)

    loaded = PaperBM25Index(str(index_dir), result_text_chars=20)
    loaded.load()
    results = loaded.search("speculative decoding", top_k=2)

    assert results[0].paper_id == "p1"
    assert results[0].chunk_type == "paper"
    assert results[0].source == "paper_bm25"
    assert len(results[0].text) <= 20
