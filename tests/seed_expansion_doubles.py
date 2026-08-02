"""Test doubles shared by the seed expansion test modules.

Each fake records the calls it receives so a test can assert not only the
final ranking but also which stage asked for what, and in which order.
"""

from __future__ import annotations

from dataclasses import replace

from littraceqa.di_pipeline.contracts import (
    Chunk,
    RetrievalResult,
    SearchHints,
)


def _result(
    paper_id: str,
    *,
    score: float = 1.0,
    title: str = "Seed title",
    text: str = "Representative text",
    chunk_id: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"{paper_id}#c0000",
        paper_id=paper_id,
        score=score,
        text=text,
        chunk_type="text_span",
        metadata={"title": title},
        source="test",
    )


class _FakeRetriever:
    def __init__(
        self,
        responses: list[list[RetrievalResult]],
        *,
        reranker=None,
        indexers: list | None = None,
        fail_on_calls: set[int] | None = None,
    ) -> None:
        self.responses = responses
        self.reranker = reranker
        self.indexers = indexers if indexers is not None else [object()]
        self.fail_on_calls = fail_on_calls or set()
        self.calls: list[tuple[str, int, SearchHints | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k, hints))
        if len(self.calls) in self.fail_on_calls:
            raise RuntimeError("search failed")
        response_index = min(len(self.calls) - 1, len(self.responses) - 1)
        return list(self.responses[response_index])


class _FakePaperIndex:
    name = "paper_bm25"

    def __init__(
        self,
        documents: dict[str, Chunk],
        *,
        owners: tuple[dict, ...] = (),
        neighbors: dict[str, tuple[dict, ...]] | None = None,
        search_results: dict[str, tuple[RetrievalResult, ...]] | None = None,
    ) -> None:
        self.documents = documents
        self.owners = owners
        self.neighbors = neighbors or {}
        self.search_results = search_results or {}
        self.calls: list[str] = []
        self.owner_calls: list[tuple[tuple[str, ...], int]] = []
        self.neighbor_calls: list[tuple[str, int]] = []
        self.search_calls: list[tuple[str, int]] = []

    def get_document(self, paper_id: str) -> Chunk | None:
        self.calls.append(paper_id)
        return self.documents.get(paper_id)

    def find_method_owners(self, methods, limit=10):
        normalized = (methods,) if isinstance(methods, str) else tuple(methods)
        self.owner_calls.append((normalized, limit))
        return self.owners[:limit]

    def get_method_neighbors(self, paper_id: str, limit=10):
        self.neighbor_calls.append((paper_id, limit))
        return self.neighbors.get(paper_id, ())[:limit]

    def search(self, query: str, top_k: int):
        self.search_calls.append((query, top_k))
        return list(self.search_results.get(query, ()))[:top_k]


class _FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((query, list(candidates), top_k))
        return list(reversed(candidates))[:top_k]


class _RecordingFinalReranker:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        invalid_result: str | None = None,
    ) -> None:
        self.error = error
        self.invalid_result = invalid_result
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

        reranked = []
        for rank, candidate in enumerate(reversed(candidates), start=1):
            metadata = dict(candidate.metadata)
            metadata["title"] = "Mutated proxy title"
            metadata["test_final_rerank_rank"] = rank
            reranked.append(
                replace(
                    candidate,
                    score=float(len(candidates) - rank + 1),
                    metadata=metadata,
                )
            )
        if self.invalid_result == "missing":
            return reranked[:-1]
        if self.invalid_result == "duplicate":
            return [*reranked[:-1], reranked[0]]
        if self.invalid_result == "non_finite":
            return [
                replace(reranked[0], score=float("nan")),
                *reranked[1:],
            ]
        return reranked


class _PushPaperLastReranker:
    def __init__(self, paper_id: str) -> None:
        self.paper_id = paper_id
        self.calls: list[tuple[str, list[RetrievalResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((query, list(candidates), top_k))
        ordered = [
            candidate
            for candidate in candidates
            if candidate.paper_id != self.paper_id
        ]
        ordered.extend(
            candidate
            for candidate in candidates
            if candidate.paper_id == self.paper_id
        )
        return [
            replace(candidate, score=float(len(ordered) - index))
            for index, candidate in enumerate(ordered)
        ]


class _FakePaperEmbeddingStore:
    def __init__(
        self,
        results_by_paper: dict[str, list[RetrievalResult]],
        *,
        failing_papers: set[str] | None = None,
    ) -> None:
        self.results_by_paper = results_by_paper
        self.failing_papers = failing_papers or set()
        self.calls: list[tuple[str, int]] = []

    def search_by_paper_id(
        self,
        paper_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((paper_id, top_k))
        if paper_id in self.failing_papers:
            raise RuntimeError("dense neighbor lookup failed")
        return list(self.results_by_paper.get(paper_id, ()))[:top_k]
