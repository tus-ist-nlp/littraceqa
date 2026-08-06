"""Tests for paper-level two-lane reranking and fail-soft fusion."""

from __future__ import annotations

from dataclasses import replace

import pytest

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.retrieve.seed_expansion.two_lane import (
    PaperTwoLaneReranker,
)


def _result(
    paper_id: str,
    *,
    chunk_id: str | None = None,
    text: str | None = None,
    title: str | None = None,
    source: str = "test",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"{paper_id}#candidate",
        paper_id=paper_id,
        score=1.0,
        text=text or f"candidate text for {paper_id}",
        chunk_type="text_span",
        metadata={"title": title or f"Title {paper_id}"},
        source=source,
    )


def _document(paper_id: str, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{paper_id}#paper",
        paper_id=paper_id,
        text=text or f"paper-level document for {paper_id}",
        chunk_type="paper",
        metadata={"title": f"Document {paper_id}"},
    )


class _PaperIndex:
    name = "paper_bm25"

    def __init__(self, documents: dict[str, Chunk | None]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def get_document(self, paper_id: str) -> Chunk | None:
        self.calls.append(paper_id)
        return self.documents.get(paper_id)


class _SharedScoreReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.score_calls: list[tuple[str, list[RetrievalResult]]] = []
        self.rerank_scored_calls: list[
            tuple[list[RetrievalResult], list[float], int]
        ] = []
        self.rerank_calls: list[object] = []

    def score_candidates(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[float]:
        self.score_calls.append((query, list(candidates)))
        return [self.scores[candidate.paper_id] for candidate in candidates]

    def rerank_scored(
        self,
        candidates: list[RetrievalResult],
        scores: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.rerank_scored_calls.append((list(candidates), list(scores), top_k))
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].paper_id),
        )
        return [
            replace(
                candidate,
                score=score,
                metadata={
                    **candidate.metadata,
                    "title": "reranker must not overwrite this",
                    "qwen3_score": score,
                },
            )
            for candidate, score in ranked[:top_k]
        ]

    def rerank(self, query, candidates, top_k):
        self.rerank_calls.append((query, candidates, top_k))
        raise AssertionError("shared scoring should avoid ordinary rerank calls")


class _IndependentReranker:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, list[RetrievalResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((query, list(candidates), top_k))
        if self.error is not None:
            raise self.error
        return [
            replace(candidate, score=float(rank), metadata={
                **candidate.metadata,
                "test_reranked": True,
            })
            for rank, candidate in enumerate(reversed(candidates), start=1)
        ][:top_k]


def _two_lane(reranker, **overrides) -> PaperTwoLaneReranker:
    params = {
        "reranker": reranker,
        "document_chars": 12,
        "rrf_k": 60,
        "base_weight": 1.0,
        "expansion_weight": 1.15,
    }
    params.update(overrides)
    return PaperTwoLaneReranker(**params)


def test_shared_scores_union_once_and_fuses_reranked_lanes() -> None:
    base_b = _result("b", chunk_id="b#base", title="Base B", source="base")
    expansion_b = _result(
        "b",
        chunk_id="b#expansion",
        title="Expansion B",
        source="expansion",
    )
    base = [_result("a"), base_b, _result("a", chunk_id="a#duplicate")]
    expansion = [expansion_b, _result("c", source="expansion")]
    documents = {
        paper_id: _document(paper_id, f"document-{paper_id}-long-tail")
        for paper_id in ("a", "b", "c")
    }
    index = _PaperIndex(documents)
    reranker = _SharedScoreReranker({"a": 0.1, "b": 0.9, "c": 0.8})

    fused = _two_lane(reranker).fuse(
        "question",
        base,
        expansion,
        [index],
        max_candidates=3,
    )

    assert [result.paper_id for result in fused] == ["b", "c", "a"]
    assert len(reranker.score_calls) == 1
    assert reranker.rerank_calls == []
    assert len(reranker.rerank_scored_calls) == 2
    assert [candidate.paper_id for candidate in reranker.score_calls[0][1]] == [
        "b",
        "c",
        "a",
    ]
    assert all(
        len(candidate.text) <= 12
        for candidate in reranker.score_calls[0][1]
    )
    assert index.calls == ["b", "c", "a"]

    by_id = {result.paper_id: result for result in fused}
    assert by_id["b"].chunk_id == "b#base"
    assert by_id["b"].metadata["title"] == "Base B"
    assert by_id["b"].metadata["qwen3_score"] == pytest.approx(0.9)
    assert by_id["b"].metadata["two_lane_base_rank"] == 1
    assert by_id["b"].metadata["two_lane_expansion_rank"] == 1
    assert by_id["b"].metadata["two_lane_sources"] == ["base", "expansion"]
    assert by_id["c"].metadata["two_lane_base_rank"] is None
    assert by_id["c"].metadata["two_lane_expansion_rank"] == 2
    assert fused[0].metadata["two_lane_rerank_status"] == "applied_shared"


def test_reranker_without_shared_api_reranks_each_lane() -> None:
    base = [_result("a"), _result("b")]
    expansion = [_result("b"), _result("c")]
    index = _PaperIndex(
        {paper_id: _document(paper_id) for paper_id in ("a", "b", "c")}
    )
    reranker = _IndependentReranker()

    fused = _two_lane(reranker).fuse(
        "question",
        base,
        expansion,
        [index],
        max_candidates=3,
    )

    assert len(reranker.calls) == 2
    assert [result.paper_id for result in fused] == ["b", "c", "a"]
    assert fused[0].metadata["two_lane_rerank_status"] == "applied_independent"
    assert all(result.metadata["test_reranked"] for result in fused)


@pytest.mark.parametrize(
    ("index", "reranker", "error_type"),
    [
        (_PaperIndex({"a": _document("a"), "b": None}), _IndependentReranker(), "ValueError"),
        (
            _PaperIndex({"a": _document("a"), "b": _document("b")}),
            _IndependentReranker(error=RuntimeError("model failure")),
            "RuntimeError",
        ),
    ],
)
def test_invalid_document_or_reranker_error_falls_back_to_unscored_rrf(
    index: _PaperIndex,
    reranker: _IndependentReranker,
    error_type: str,
) -> None:
    base = [_result("a"), _result("b")]
    expansion = [_result("b")]

    fused = _two_lane(reranker).fuse(
        "question",
        base,
        expansion,
        [index],
        max_candidates=2,
    )

    assert [result.paper_id for result in fused] == ["b", "a"]
    assert fused[0].metadata["two_lane_rerank_status"] == "fallback"
    assert fused[0].metadata["two_lane_rerank_error_type"] == error_type
    assert fused[0].metadata["two_lane_base_rank"] == 2
    assert fused[0].metadata["two_lane_expansion_rank"] == 1
    assert fused[0].metadata["two_lane_rrf_score"] == pytest.approx(
        1.0 / 62 + 1.15 / 61
    )


def test_candidate_limit_is_applied_after_unscored_lane_fusion() -> None:
    reranker = _IndependentReranker(error=RuntimeError("stop before scoring"))
    index = _PaperIndex(
        {paper_id: _document(paper_id) for paper_id in ("a", "b", "c", "d")}
    )

    fused = _two_lane(reranker).fuse(
        "question",
        [_result("a"), _result("b")],
        [_result("c"), _result("d")],
        [index],
        max_candidates=2,
    )

    assert [result.paper_id for result in fused] == ["c", "d"]
    assert len(fused) == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_max_candidates(value) -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        _two_lane(_IndependentReranker()).fuse(
            "question",
            [],
            [],
            [],
            max_candidates=value,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_document_chars(value) -> None:
    with pytest.raises(ValueError, match="document_chars"):
        _two_lane(_IndependentReranker(), document_chars=value)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rrf_k", -1, ValueError),
        ("rrf_k", float("inf"), ValueError),
        ("base_weight", -0.1, ValueError),
        ("expansion_weight", "1.0", TypeError),
    ],
)
def test_rejects_invalid_numeric_settings(field, value, error) -> None:
    with pytest.raises(error, match=field):
        _two_lane(_IndependentReranker(), **{field: value})


def test_requires_at_least_one_positive_weight() -> None:
    with pytest.raises(ValueError, match="lane weight"):
        _two_lane(
            _IndependentReranker(),
            base_weight=0,
            expansion_weight=0,
        )
