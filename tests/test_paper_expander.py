"""論文→論文展開（Specter2PaperExpander）のテスト。

本物の faiss 索引を小さく作って検証する。押さえるべき仕様:

- anchor 自身と既存候補は返さない（追記分だけを返す）
- 既存候補の順位に触らない（expand は追加IDのリストを返すだけ）
- 索引に居ない anchor は黙って飛ばす（クエリ全体を壊さない）
- 複数 anchor はランク順の交互配置
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from littraceqa.di_pipeline.retrieve.paper_expander import Specter2PaperExpander


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    # p0 と p1 が近く、p2/p3 がその次、p4 は遠い、という配置。
    vectors = np.array(
        [
            [1.00, 0.00],  # p0
            [0.99, 0.14],  # p1 (p0 の最近傍)
            [0.90, 0.44],  # p2
            [0.80, 0.60],  # p3
            [0.00, 1.00],  # p4 (遠い)
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
    # anchor 自身（p0）は近傍に入らない。**渡されたリストは全部 anchor**なので、
    # 「既存候補を落とさない」は expander が候補列を見ないことで構造的に保たれる。
    assert expander.rank(["p0"]) == ["p1", "p2", "p3"]


def test_rank_unknown_anchor_is_silent(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=3)
    assert expander.rank(["unknown_paper"]) == []
    assert expander.rank([]) == []


def test_rank_multi_anchor_interleaves_by_rank(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=3)
    # anchor p0 の近傍: p1,p2,p3 / anchor p4 の近傍: p3,p2,p1
    # 交互配置: rank0 -> p1(p0側), p3(p4側), rank1 -> p2 ... 重複は除去。
    assert expander.rank(["p0", "p4"]) == ["p1", "p3", "p2"]


def test_rank_caps_at_neighbors(index_dir: Path) -> None:
    expander = Specter2PaperExpander(str(index_dir), neighbors=2)
    assert len(expander.rank(["p0", "p4"])) <= 2


def _bib_corpus(tmp_path: Path) -> Path:
    """書誌結合用の小さなコーパス。p0 と p1 は文献を2本共有、p2 は1本だけ、p3 は無関係。"""
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
    from littraceqa.di_pipeline.retrieve.paper_expander import BibCouplingExpander

    ex = BibCouplingExpander(
        chunks=str(_bib_corpus(tmp_path)), cache_path=str(tmp_path / "refs.pkl"),
        neighbors=5, min_shared=2,
    )
    # p0 と2本共有する p1 のみ。p2 は1本しか共有しないので min_shared で落ちる。
    assert ex.rank(["p0"]) == ["p1"]
    # 代表テキストは title_abstract
    # 2回目はキャッシュから読む（走査しない）
    assert (tmp_path / "refs.pkl").exists()
    again = BibCouplingExpander(
        chunks="/nonexistent", cache_path=str(tmp_path / "refs.pkl"),
        neighbors=5, min_shared=2,
    )
    assert again.rank(["p0"]) == ["p1"]


def test_bib_coupling_min_shared_one_includes_weak_links(tmp_path: Path) -> None:
    from littraceqa.di_pipeline.retrieve.paper_expander import BibCouplingExpander

    ex = BibCouplingExpander(
        chunks=str(_bib_corpus(tmp_path)), cache_path=str(tmp_path / "refs1.pkl"),
        neighbors=5, min_shared=1,
    )
    added = ex.rank(["p0"])
    assert added[0] == "p1"          # Jaccard が高い順
    assert set(added) == {"p1", "p2"}  # p3 は共有ゼロなので入らない


def test_fused_expander_rrf_merges_sources(index_dir: Path, tmp_path: Path) -> None:
    from littraceqa.di_pipeline.retrieve.paper_expander import (
        BibCouplingExpander, FusedPaperExpander, Specter2PaperExpander,
    )

    class _Fixed:
        def __init__(self, ids): self.ids = ids
        def rank(self, ranked): return list(self.ids)

    # A: [x, y] / B: [y, z] -> y が両方で上位なので RRF で先頭に来る
    fused = FusedPaperExpander(
        sources=[_Fixed(["x", "y"]), _Fixed(["y", "z"])], neighbors=5, rrf_k=1
    )
    assert fused.rank(["anchor"])[0] == "y"
    assert set(fused.rank(["anchor"])) == {"x", "y", "z"}
    # テキストは最初に持っていたソースから引く


def _bm25_paper_index(tmp_path: Path) -> Path:
    """bm25s_paper 索引の最小版。p1 は p0 と語彙が重なり、p2 は無関係。"""
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
    from littraceqa.di_pipeline.retrieve.paper_expander import BM25MLTExpander

    index_dir = _bm25_paper_index(tmp_path)
    ex = BM25MLTExpander(str(index_dir), cache_path=str(tmp_path / "mlt.pkl"), neighbors=5)
    # anchor 自身は返さず、語彙が重なる p1 が p2 より先
    added = ex.rank(["p0"])
    assert added[0] == "p1"
    assert "p0" not in added
    # anchor のクエリ文は papers.jsonl の先頭（title+abstract）から取る
    # 2回目はキャッシュから読む（papers.jsonl を走査しない）
    assert (tmp_path / "mlt.pkl").exists()
    again = BM25MLTExpander(str(index_dir), cache_path=str(tmp_path / "mlt.pkl"), neighbors=5)
    assert again.rank(["p0"])[0] == "p1"


def test_fused_rank_keeps_existing_candidates() -> None:
    """既存候補は落とさない。両ソースに乗っている論文が先頭に来る。"""
    from littraceqa.di_pipeline.retrieve.paper_expander import FusedPaperExpander

    class _Fixed:
        def __init__(self, ids): self.ids = ids
        def rank(self, ranked): return list(self.ids)

    fused = FusedPaperExpander(
        sources=[_Fixed(["x", "y"]), _Fixed(["x", "z"])], neighbors=5, rrf_k=1
    )
    assert fused.rank(["x"])[0] == "x"
