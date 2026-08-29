"""Paper-to-paper expansion, checked against a small real faiss index.

The behaviour that has to hold:

- an anchor never comes back among its own neighbours
- the existing candidate ranking is untouched (the expander returns ids, nothing more)
- an anchor absent from the index is skipped silently, never failing the query
- several anchors interleave by rank
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from littraceqa.di_pipeline.expander import Specter2PaperExpander


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    # p0 and p1 are close, p2/p3 next, p4 far away.
    vectors = np.array(
        [
            [1.00, 0.00],  # p0
            [0.99, 0.14],  # p1 (p0's nearest neighbour)
            [0.90, 0.44],  # p2
            [0.80, 0.60],  # p3
            [0.00, 1.00],  # p4 (far)
        ],
        dtype="float32",
    )
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    with open(tmp_path / "chunks.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(vectors)):
            f.write(json.dumps({"chunk_id": f"p{i}#c0000", "paper_id": f"p{i}"}) + "\n")
    return tmp_path


def test_rank_returns_neighbors_in_order(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=3)
    # An anchor (p0) is not among its own neighbours. **Everything passed in is an
    # anchor**, so "the existing candidates are not dropped" holds structurally: the
    # expander never sees the candidate list at all.
    assert expander.rank(["p0"]) == ["p1", "p2", "p3"]


def test_rank_unknown_anchor_is_silent(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=3)
    assert expander.rank(["unknown_paper"]) == []
    assert expander.rank([]) == []


def test_rank_multi_anchor_interleaves_by_rank(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=3)
    # p0's neighbours: p1,p2,p3; p4's: p3,p2,p1
    # Interleaved: rank0 -> p1 (p0's), p3 (p4's); rank1 -> p2 ...; repeats dropped.
    assert expander.rank(["p0", "p4"]) == ["p1", "p3", "p2"]


def test_rank_caps_at_neighbors(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=2)
    assert len(expander.rank(["p0", "p4"])) <= 2


def _bib_corpus(tmp_path: Path) -> Path:
    """A small corpus for bibliographic coupling: p0 and p1 share two references,
    p2 shares one, p3 shares none."""
    rows = [
        ("p0", "[ACL 2025] Paper Zero\nabstract zero", "title_abstract"),
        ("p0", "refs: arXiv:2401.00001 arXiv:2401.00002 arXiv:2401.00003", "text_span"),
        ("p1", "[ACL 2025] Paper One\nabstract one", "title_abstract"),
        ("p1", "refs: arXiv:2401.00001 arXiv:2401.00002", "text_span"),
        ("p2", "[ACL 2025] Paper Two\nabstract two", "title_abstract"),
        ("p2", "refs: arXiv:2401.00001 arXiv:2409.99999", "text_span"),
        ("p3", "[ACL 2025] Paper Three\nabstract three", "title_abstract"),
        ("p3", "refs: ar X iv : 2405.55555", "text_span"),
    ]
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, (pid, text, ct) in enumerate(rows):
            f.write(json.dumps({"chunk_id": f"{pid}#c{i}", "paper_id": pid,
                                "text": text, "chunk_type": ct}) + "\n")
    return path


def test_bib_coupling_ranks_by_shared_references(tmp_path: Path) -> None:
    from littraceqa.di_pipeline.expander import BibCouplingExpander

    ex = BibCouplingExpander(
        chunks=str(_bib_corpus(tmp_path)), cache_path=str(tmp_path / "refs.pkl"),
        neighbors=5, min_shared=2,
    )
    # Only p1, which shares two with p0; p2 shares one and min_shared drops it.
    assert ex.rank(["p0"]) == ["p1"]
    # The representative text is the title_abstract chunk
    # The second time comes from the cache, with no walk over the corpus
    assert (tmp_path / "refs.pkl").exists()
    again = BibCouplingExpander(
        chunks="/nonexistent", cache_path=str(tmp_path / "refs.pkl"),
        neighbors=5, min_shared=2,
    )
    assert again.rank(["p0"]) == ["p1"]


def test_bib_coupling_min_shared_one_includes_weak_links(tmp_path: Path) -> None:
    from littraceqa.di_pipeline.expander import BibCouplingExpander

    ex = BibCouplingExpander(
        chunks=str(_bib_corpus(tmp_path)), cache_path=str(tmp_path / "refs1.pkl"),
        neighbors=5, min_shared=1,
    )
    added = ex.rank(["p0"])
    assert added[0] == "p1"          # ordered by Jaccard
    assert set(added) == {"p1", "p2"}  # p3 shares nothing, so it is absent


def test_fused_expander_rrf_merges_sources(index_dir: Path, tmp_path: Path) -> None:
    from littraceqa.di_pipeline.expander import (
        BibCouplingExpander, FusedPaperExpander, Specter2PaperExpander,
    )

    class _Fixed:
        def __init__(self, ids): self.ids = ids
        def rank(self, ranked): return list(self.ids)

    # A: [x, y], B: [y, z] -> y is high in both, so RRF puts it first
    fused = FusedPaperExpander(
        sources=[_Fixed(["x", "y"]), _Fixed(["y", "z"])], neighbors=5, rrf_k=1
    )
    assert fused.rank(["anchor"])[0] == "y"
    assert set(fused.rank(["anchor"])) == {"x", "y", "z"}
    # The text comes from whichever source had it first


def _bm25_paper_index(tmp_path: Path) -> Path:
    """A minimal bm25s_paper index: p1 shares vocabulary with p0, p2 does not."""
    import bm25s

    papers = [
        ("p0", "[ICLR 2025] Truncated Consistency Models\nconsistency distillation diffusion"),
        ("p1", "[ICLR 2025] Simplifying Consistency Models\nconsistency distillation sampler"),
        ("p2", "[ACL 2025] Machine Translation Survey\nalignment corpora multilingual"),
    ]
    index_dir = tmp_path / "bm25s_paper"
    index_dir.mkdir()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([text for _, text in papers], stopwords="en"))
    retriever.save(str(index_dir))
    with (index_dir / "papers.jsonl").open("w", encoding="utf-8") as f:
        for paper_id, text in papers:
            f.write(json.dumps({"chunk_id": f"{paper_id}#paper", "paper_id": paper_id,
                                "text": text, "chunk_type": "paper", "metadata": {}}) + "\n")
    return index_dir


def test_bm25_mlt_ranks_lexically_close_papers(tmp_path: Path) -> None:
    from littraceqa.di_pipeline.expander import BM25MLTExpander

    index_dir = _bm25_paper_index(tmp_path)
    ex = BM25MLTExpander(str(index_dir), cache_path=str(tmp_path / "mlt.pkl"), neighbors=5)
    # The anchor is not returned, and p1, which shares vocabulary, comes before p2
    added = ex.rank(["p0"])
    assert added[0] == "p1"
    assert "p0" not in added
    # The anchor's query text is the head of its papers.jsonl line (title+abstract)
    # The second time comes from the cache, without walking papers.jsonl
    assert (tmp_path / "mlt.pkl").exists()
    again = BM25MLTExpander(str(index_dir), cache_path=str(tmp_path / "mlt.pkl"), neighbors=5)
    assert again.rank(["p0"])[0] == "p1"


def test_fused_rank_keeps_existing_candidates() -> None:
    """Existing candidates are not dropped; a paper in both sources comes first."""
    from littraceqa.di_pipeline.expander import FusedPaperExpander

    class _Fixed:
        def __init__(self, ids): self.ids = ids
        def rank(self, ranked): return list(self.ids)

    fused = FusedPaperExpander(
        sources=[_Fixed(["x", "y"]), _Fixed(["x", "z"])], neighbors=5, rrf_k=1
    )
    assert fused.rank(["x"])[0] == "x"
