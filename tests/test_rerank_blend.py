"""`rerank_blend`（reranker の順位を融合前の順位と RRF で混ぜる）のテスト。

一番大事なのは2つ:

* **書かなければ現行と完全に同一**（reranker が順位を置き換える）
* **融合した順位が `score` に載っている**こと。下流はどこも score で並べ直す
  （agent/reading.py の貯め込み・_candidate_papers・to_gold_papers）ので、
  返り値の並び順にしか順位が無いと100%捨てられる。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever
from littraceqa.di_pipeline.retrieve.rrf import RRFFuser


def _result(name: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{name}#c0",
        paper_id=name,
        score=score,
        text=f"body {name}",
        chunk_type="text_span",
        metadata={},
    )


class _StubIndexer:
    """固定のランキングを返す索引。"""

    def __init__(self, order: list[str]):
        self.order = order

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [_result(name, 1.0 - i / 100) for i, name in enumerate(self.order)][:top_k]


class _ReverseReranker:
    """融合結果を丸ごとひっくり返す reranker（置換と融合の差が見えるように）。"""

    def rerank(self, query, candidates, top_k):
        # 本物と同じく score を yes 確率で上書きしてから返す。
        ordered = list(reversed(candidates))
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                score=1.0 - i / 100,
                text=r.text,
                chunk_type=r.chunk_type,
                metadata=r.metadata,
            )
            for i, r in enumerate(ordered)
        ][:top_k]


def _retriever(**kwargs) -> HybridRetriever:
    return HybridRetriever(
        indexers=[_StubIndexer(["a", "b", "c", "d", "e"])],
        fuser=RRFFuser(k=60),
        reranker=_ReverseReranker(),
        per_index_k=10,
        pool_k=5,
        **kwargs,
    )


def test_without_blend_the_reranker_replaces_the_ranking():
    """既定（rerank_blend なし）は従来どおり reranker の順位そのもの。"""
    got = _retriever().retrieve("q", top_k=5)
    assert [r.paper_id for r in got] == ["e", "d", "c", "b", "a"]


def test_blend_keeps_the_original_ranking_partly_alive():
    """重みを元順位側に寄せると、reranker に丸ごと反転されない。"""
    got = _retriever(
        rerank_blend={"original_weight": 1.0, "rerank_weight": 0.0, "rrf_k": 60}
    ).retrieve("q", top_k=5)
    assert [r.paper_id for r in got] == ["a", "b", "c", "d", "e"]


def test_blend_score_matches_the_returned_order():
    """**融合順位が score に載っている。** 下流が score で並べ直しても崩れない。"""
    got = _retriever(
        rerank_blend={"original_weight": 0.6, "rerank_weight": 0.4, "rrf_k": 60}
    ).retrieve("q", top_k=5)
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)
    # score で並べ直しても同じ順序になる（これが崩れると保護も融合も効かない）
    assert [r.paper_id for r in sorted(got, key=lambda r: -r.score)] == [
        r.paper_id for r in got
    ]


def test_protect_top_keeps_the_original_head_set():
    """融合前の上位集合は reranker に押し出させない。"""
    got = _retriever(
        rerank_blend={
            "original_weight": 0.0,  # reranker だけを見る設定でも……
            "rerank_weight": 1.0,
            "rrf_k": 60,
            "protect_top": 2,
        }
    ).retrieve("q", top_k=5)
    # ……融合前の上位2件（a, b）は先頭に残る（並び順は融合結果に従う）
    assert set(r.paper_id for r in got[:2]) == {"a", "b"}
    # 保護も score に載っている（並べ直しで元に戻らない）
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)


def test_no_reranker_is_untouched():
    """reranker が無ければ rerank_blend を書いても何も起きない。"""
    retriever = HybridRetriever(
        indexers=[_StubIndexer(["a", "b", "c"])],
        fuser=RRFFuser(k=60),
        reranker=None,
        per_index_k=10,
        rerank_blend={"original_weight": 0.6, "rerank_weight": 0.4},
    )
    assert [r.paper_id for r in retriever.retrieve("q", top_k=3)] == ["a", "b", "c"]
