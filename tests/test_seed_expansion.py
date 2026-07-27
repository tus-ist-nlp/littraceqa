from __future__ import annotations

from dataclasses import replace

import pytest

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers
from littraceqa.di_pipeline.retrieve.seed_expansion import SeedExpansionRetriever


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
    ) -> None:
        self.responses = responses
        self.reranker = reranker
        self.indexers = indexers if indexers is not None else [object()]
        self.calls: list[tuple[str, int, SearchHints | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k, hints))
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


def test_expands_from_top_result_and_forwards_hints():
    representative_text = f"{'x' * 300}\n\n{'y' * 300}"
    inner = _FakeRetriever(
        [
            [_result("p1", title="A paper", text=representative_text)],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(inner)
    hints = SearchHints(venues=("ACL",), years=(2024,))

    retriever.retrieve("original question", 10, hints=hints)

    assert len(inner.calls) == 2
    assert inner.calls[0] == ("original question", 50, hints)
    assert inner.calls[1] == (
        f"original question A paper {'x' * 300} {'y' * 211}",
        50,
        hints,
    )


def test_prefers_paper_context_and_does_not_duplicate_leading_title():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = (
        "A paper\nUseful abstract and introduction"
    )
    seed.metadata["paper_rank_expansion_source"] = "paper_bm25"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("original question", 10)

    assert len(inner.calls) == 2
    assert inner.calls[1][0] == (
        "original question A paper Useful abstract and introduction"
    )
    assert "Query-matched table" not in inner.calls[1][0]
    assert all(
        "seed_expansion_local_rank" not in result.metadata
        for result in results
    )


def test_opt_in_local_expansion_uses_legacy_seed_query_and_forwards_hints():
    seed = _result(
        "p1",
        title="A paper",
        text="Query-matched table details",
    )
    seed.metadata["paper_rank_expansion_text"] = "A paper\nUseful abstract"
    inner = _FakeRetriever(
        [
            [seed],
            [_result("p2")],
            [_result("p3")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        seed_text_chars=19,
        local_expansion_weight=0.5,
    )
    hints = SearchHints(venues=("ACL",), years=(2024,))

    retriever.retrieve("original question", 10, hints=hints)

    assert len(inner.calls) == 3
    assert inner.calls[1] == (
        "original question A paper Useful abst",
        50,
        hints,
    )
    assert inner.calls[2] == (
        "original question A paper Query-matched table",
        50,
        hints,
    )


def test_opt_in_literal_attribute_hints_are_forwarded_to_every_lane():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever(
        [
            [seed],
            [_result("p2")],
            [_result("p3")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
        literal_attribute_hints=True,
    )

    retriever.retrieve(
        "Which CVPR 2025 papers cite UniAD "
        "(Planning-oriented Autonomous Driving, CVPR2023)?",
        10,
    )

    expected = SearchHints(venues=("CVPR",), years=(2025,))
    assert len(inner.calls) == 3
    assert all(hints == expected for _, _, hints in inner.calls)


def test_opt_in_target_method_hints_are_deferred_until_final_ranking():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(
        inner,
        literal_attribute_hints=True,
        literal_method_hints=True,
    )

    retriever.retrieve("In the TCM paper, what is the batch size?", 10)

    expected = SearchHints()
    assert len(inner.calls) == 2
    assert all(hints == expected for _, _, hints in inner.calls)


@pytest.mark.parametrize(
    "provided",
    [
        SearchHints(),
        SearchHints(venues=("ACL",), years=(2024,)),
        SearchHints(
            venues=("ACL",),
            years=(2024,),
            methods=("TCM",),
        ),
    ],
)
def test_caller_hints_take_precedence_over_literal_extraction(provided):
    inner = _FakeRetriever(
        [
            [_result("p1")],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        literal_attribute_hints=True,
    )

    retriever.retrieve("Which CVPR 2025 papers are relevant?", 10, hints=provided)

    assert len(inner.calls) == 2
    expected = SearchHints(
        venues=provided.venues,
        years=provided.years,
    )
    assert all(hints == expected for _, _, hints in inner.calls)


def test_default_does_not_extract_literal_attribute_hints():
    inner = _FakeRetriever(
        [
            [_result("p1")],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("Which CVPR 2025 papers are relevant?", 10)

    assert len(inner.calls) == 2
    assert all(hints is None for _, _, hints in inner.calls)


def test_local_expansion_still_runs_when_paper_context_search_is_empty():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever(
        [
            [seed],
            [],
            [_result("local")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
    )

    results = retriever.retrieve("question", 10)

    assert len(inner.calls) == 3
    assert [result.paper_id for result in results] == ["p1", "local"]
    assert results[0].metadata["seed_expansion_expanded_rank"] is None
    assert results[1].metadata["seed_expansion_local_rank"] == 1


def test_local_expansion_is_not_duplicated_without_paper_context():
    initial = [_result("p1", title="A paper", text="Local evidence")]
    inner = _FakeRetriever(
        [
            initial,
            [_result("p2")],
            [_result("unexpected")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
    )

    results = retriever.retrieve("question", 10)

    assert len(inner.calls) == 2
    assert inner.calls[1][0] == "question A paper Local evidence"
    assert all(
        "seed_expansion_local_rank" not in result.metadata
        for result in results
    )


def test_paper_context_keeps_title_when_context_does_not_begin_with_it():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = "Useful abstract"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("original question", 10)

    assert inner.calls[1][0] == "original question A paper Useful abstract"


def test_empty_paper_context_falls_back_to_legacy_seed_text():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = "  "
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("original question", 10)

    assert inner.calls[1][0] == "original question A paper Query-matched table"


def test_fuses_unique_papers_with_equal_weight_rrf():
    initial = [_result("p1"), _result("p2"), _result("p3")]
    expanded = [_result("p2"), _result("p3"), _result("p4")]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p2", "p3", "p1", "p4"]
    assert results[0].chunk_id == "p2#c0000"
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 63 + 1 / 62)
    assert results[2].score == pytest.approx(1 / 61)
    assert results[3].score == pytest.approx(1 / 63)
    assert results[0].source == "seed_expansion_rrf"
    assert results[0].metadata["seed_expansion_original_rank"] == 2
    assert results[0].metadata["seed_expansion_expanded_rank"] == 1


def test_fuses_opt_in_local_expansion_with_configured_rrf_weight():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    initial = [seed, _result("p2")]
    expanded = [_result("p2"), _result("p3")]
    local = [_result("p3"), _result("p1")]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded, local]),
        rrf_k=0,
        local_expansion_weight=0.5,
    )

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p2", "p1", "p3"]
    assert results[0].score == pytest.approx(1.5)
    assert results[1].score == pytest.approx(1.25)
    assert results[2].score == pytest.approx(1.0)
    assert results[0].metadata["seed_expansion_local_rank"] is None
    assert results[1].metadata["seed_expansion_local_rank"] == 2
    assert results[2].metadata["seed_expansion_local_rank"] == 1


def test_opt_in_paper_neighborhood_recovers_a_related_candidate():
    seed = _result(
        "seed",
        title="AnchorNet: A Reliable Anchor Method",
    )
    related = _result(
        "related",
        title="Related Method for Scientific Document Retrieval",
    )
    distractors = [
        _result(f"distractor-{index}", title=f"Distractor paper number {index}")
        for index in range(1, 12)
    ]
    candidates = [seed, *distractors, related]
    paper_index = _FakePaperIndex(
        {
            candidate.paper_id: Chunk(
                chunk_id=f"{candidate.paper_id}#paper",
                paper_id=candidate.paper_id,
                text=(
                    "This paper discusses Related Method for Scientific "
                    "Document Retrieval."
                    if candidate.paper_id == "seed"
                    else "Independent full-paper text."
                ),
                chunk_type="paper",
                metadata={"title": candidate.metadata["title"]},
            )
            for candidate in candidates
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("What does AnchorNet report?", 10)

    assert "related" in [result.paper_id for result in results]
    assert "related" not in [result.paper_id for result in candidates[:10]]
    assert paper_index.calls


def test_paper_neighborhood_is_disabled_by_default():
    candidates = [_result("seed"), _result("other")]
    paper_index = _FakePaperIndex({})
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        )
    )

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["seed", "other"]
    assert paper_index.calls == []


def test_method_owner_in_normal_pool_adds_related_candidate():
    original = [
        _result(
            "owner",
            title="Truncated Consistency Models",
            text="TCM full paper",
        ),
        *(
            _result(f"p{index}", title=f"Paper {index}")
            for index in range(1, 12)
        ),
    ]
    owner = Chunk(
        chunk_id="owner#paper",
        paper_id="owner",
        text="TCM full paper",
        chunk_type="paper",
        metadata={"title": "Truncated Consistency Models"},
    )
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {"owner": owner, "related": related},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [original, original],
            indexers=[paper_index],
        ),
        literal_attribute_hints=True,
        literal_method_hints=True,
        method_owner_weight=1.0,
        method_relation_weight=0.5,
        method_relation_seed_k=1,
    )

    results = retriever.retrieve(
        "In the TCM paper, what is the batch size?",
        10,
    )

    paper_ids = [result.paper_id for result in results]
    assert "owner" in paper_ids
    assert "related" in paper_ids
    assert paper_ids.index("owner") < paper_ids.index("related")
    owner_result = next(result for result in results if result.paper_id == "owner")
    related_result = next(
        result for result in results if result.paper_id == "related"
    )
    assert owner_result.metadata["method_owner_aliases"] == ["TCM"]
    assert related_result.metadata["method_relation_aliases"] == ["TCM"]
    assert related_result.metadata["method_relation_via_papers"] == ["owner"]
    assert paper_index.owner_calls == [(("TCM",), 10)]
    assert ("owner", 10) in paper_index.neighbor_calls


def test_method_relation_requires_owner_within_normal_candidate_window():
    initial = [_result("p1"), _result("p2")]
    expanded = [_result("p3"), _result("owner")]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {"related": related},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [initial, expanded],
            indexers=[paper_index],
        ),
        candidate_k=2,
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "question",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    assert "owner" in [result.paper_id for result in results]
    assert "related" not in [result.paper_id for result in results]
    assert paper_index.owner_calls == [(("TCM",), 10)]
    assert paper_index.neighbor_calls == []
    assert paper_index.calls == []


def test_method_topic_lane_reranks_only_normal_candidates_and_protects_top_eight():
    original = [
        *(_result(f"p{index}") for index in range(1, 9)),
        _result("owner", title="Dobi-SVD: Owner paper"),
        _result("related", title="Related compression paper"),
        _result("tail"),
    ]
    owner_document = Chunk(
        chunk_id="owner#paper",
        paper_id="owner",
        text="Dobi-SVD topic text that must be truncated",
        chunk_type="paper",
        metadata={"title": "Dobi-SVD: Owner paper"},
    )
    paper_index = _FakePaperIndex(
        {"owner": owner_document},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["Dobi-SVD"],
                "strength": 2,
            },
        ),
    )
    topic_results = [
        _result("owner"),
        _result("outside"),
        _result("p1"),
        _result("related"),
    ]
    inner = _FakeRetriever(
        [original, original, topic_results],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        method_topic_weight=1.0,
        method_topic_seed_chars=20,
        method_topic_seed_k=1,
        method_topic_max_results=50,
    )

    results = retriever.retrieve(
        "Question about Dobi-SVD",
        10,
        hints=SearchHints(methods=("Dobi-SVD",)),
    )

    paper_ids = [result.paper_id for result in results]
    assert {f"p{index}" for index in range(1, 9)} <= set(paper_ids)
    assert "related" in paper_ids
    assert "outside" not in paper_ids
    assert paper_ids.index("related") < paper_ids.index("owner")
    related = next(result for result in results if result.paper_id == "related")
    assert related.metadata["method_topic_rank"] == 1
    assert related.metadata["method_topic_via_papers"] == ["owner"]
    assert "method_topic_via_papers" not in next(
        result for result in results if result.paper_id == "owner"
    ).metadata
    assert inner.calls[2][0] == owner_document.text[:20].strip()
    assert inner.calls[2][1] == 50
    assert inner.calls[2][2] == SearchHints()


def test_method_topic_lane_requires_owner_in_normal_candidate_window():
    original = [_result("p1"), _result("p2"), _result("owner")]
    owner_document = Chunk(
        chunk_id="owner#paper",
        paper_id="owner",
        text="Owner topic text",
        chunk_type="paper",
        metadata={"title": "TCM owner"},
    )
    paper_index = _FakePaperIndex(
        {"owner": owner_document},
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    inner = _FakeRetriever(
        [original, original],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        candidate_k=2,
        method_topic_weight=0.15,
    )

    results = retriever.retrieve(
        "Question about TCM",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    assert [result.paper_id for result in results] == ["p1", "p2"]
    assert len(inner.calls) == 2
    assert paper_index.calls == []


def test_method_topic_lane_is_disabled_by_default():
    candidates = [_result("owner"), _result("related")]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic text",
                chunk_type="paper",
                metadata={"title": "TCM owner"},
            )
        },
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    inner = _FakeRetriever(
        [candidates, candidates],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve(
        "Question about TCM",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    assert len(inner.calls) == 2
    assert paper_index.owner_calls == []
    assert paper_index.calls == []


def test_method_topic_failure_returns_the_unmodified_ranking():
    candidates = [_result("owner"), _result("related")]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic text",
                chunk_type="paper",
                metadata={"title": "TCM owner"},
            )
        },
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )

    class _FailingTopicRetriever(_FakeRetriever):
        def retrieve(self, query, top_k, *, hints=None):
            if len(self.calls) == 2:
                self.calls.append((query, top_k, hints))
                raise RuntimeError("topic search failed")
            return super().retrieve(query, top_k, hints=hints)

    inner = _FailingTopicRetriever(
        [candidates, candidates],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        method_topic_weight=0.15,
    )

    results = retriever.retrieve(
        "Question about TCM",
        10,
        hints=SearchHints(methods=("TCM",)),
    )
    control_inner = _FakeRetriever(
        [candidates, candidates],
        indexers=[paper_index],
    )
    expected = SeedExpansionRetriever(control_inner).retrieve(
        "Question about TCM",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    assert results == expected
    assert len(inner.calls) == 3


def test_method_topic_failure_preserves_exact_owner_reranking():
    candidates = [_result("other"), _result("owner")]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic text",
                chunk_type="paper",
                metadata={"title": "TCM owner"},
            )
        },
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )

    class _FailingTopicRetriever(_FakeRetriever):
        def retrieve(self, query, top_k, *, hints=None):
            if len(self.calls) == 2:
                self.calls.append((query, top_k, hints))
                raise RuntimeError("topic search failed")
            return super().retrieve(query, top_k, hints=hints)

    inner = _FailingTopicRetriever(
        [candidates, candidates],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        method_owner_weight=1.0,
        method_topic_weight=0.35,
    )

    results = retriever.retrieve(
        "Question about TCM",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    assert [result.paper_id for result in results] == ["owner", "other"]
    assert results[0].metadata["method_owner_rank"] == 1
    assert all(
        "method_topic_rank" not in result.metadata for result in results
    )


def test_method_relations_limit_new_papers_and_protect_baseline_top_eight():
    original = [
        *(_result(f"p{index}") for index in range(1, 9)),
        _result("owner", title="Truncated Consistency Models"),
        _result("tail"),
    ]
    related_documents = {
        f"related-{index}": Chunk(
            chunk_id=f"related-{index}#paper",
            paper_id=f"related-{index}",
            text=f"Related paper {index}",
            chunk_type="paper",
            metadata={"title": f"Related paper {index}"},
        )
        for index in range(1, 4)
    }
    paper_index = _FakePaperIndex(
        related_documents,
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": tuple(
                {
                    "paper_id": f"related-{index}",
                    "aliases": ["TCM"],
                    "strength": 4 - index,
                }
                for index in range(1, 4)
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [original, original],
            indexers=[paper_index],
        ),
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "question",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    paper_ids = [result.paper_id for result in results]
    assert {f"p{index}" for index in range(1, 9)} <= set(paper_ids)
    assert paper_ids.index("owner") < paper_ids.index("related-1")
    assert "related-2" not in paper_ids
    assert "related-3" not in paper_ids
    assert paper_index.calls == ["related-1", "related-2"]


def test_explicit_title_guard_cannot_evict_method_protected_top_eight():
    original = [
        *(
            _result(f"p{index}", title=f"Paper {index}")
            for index in range(1, 9)
        ),
        _result("owner", title="Truncated Consistency Models"),
        _result("named", title="D-FINE: Distribution Refinement"),
    ]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {"related": related},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [original, original],
            indexers=[paper_index],
        ),
        method_owner_weight=1.0,
        method_relation_weight=0.5,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve(
        "Compare TCM with the D-FINE paper.",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    paper_ids = [result.paper_id for result in results]
    assert {f"p{index}" for index in range(1, 9)} <= set(paper_ids)
    assert "named" in paper_ids
    assert "related" not in paper_ids


def test_final_reranker_cannot_evict_method_protected_top_eight():
    original = [
        *(
            _result(f"p{index}", title=f"Paper {index}")
            for index in range(1, 9)
        ),
        _result("owner", title="Truncated Consistency Models"),
        _result("tail", title="Tail paper"),
    ]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {"related": related},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [original, original],
            indexers=[paper_index],
        ),
        reranker=_FakeReranker(),
        max_results=8,
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "Question about TCM",
        8,
        hints=SearchHints(methods=("TCM",)),
    )

    assert {result.paper_id for result in results} == {
        f"p{index}" for index in range(1, 9)
    }


def test_method_relations_run_after_paper_neighborhood():
    owner_result = _result(
        "owner",
        title="TCM: Truncated Consistency Models",
    )
    baseline = [owner_result, _result("other", title="Other paper")]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Independent owner paper.",
                chunk_type="paper",
                metadata=owner_result.metadata,
            ),
            "other": Chunk(
                chunk_id="other#paper",
                paper_id="other",
                text="Independent other paper.",
                chunk_type="paper",
                metadata=baseline[1].metadata,
            ),
            "related": related,
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [baseline, baseline],
            indexers=[paper_index],
        ),
        paper_neighborhood_weight=0.2,
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "question",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    by_id = {result.paper_id: result for result in results}
    assert "paper_neighborhood_baseline_rank" in by_id["owner"].metadata
    assert by_id["owner"].metadata["method_owner_aliases"] == ["TCM"]
    assert (
        "paper_neighborhood_baseline_rank"
        not in by_id["related"].metadata
    )


def test_method_relation_expansion_is_disabled_by_default():
    candidates = [_result("seed"), _result("other")]
    paper_index = _FakePaperIndex(
        {},
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        literal_attribute_hints=True,
        literal_method_hints=True,
    )

    results = retriever.retrieve("In the TCM paper, what is reported?", 10)

    assert [result.paper_id for result in results] == ["seed", "other"]
    assert paper_index.owner_calls == []
    assert paper_index.neighbor_calls == []


def test_method_relations_require_an_explicit_target_method_hint():
    candidates = [_result("seed"), _result("other")]
    paper_index = _FakePaperIndex(
        {},
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "Which papers cite a consistency-model baseline?",
        10,
    )

    assert [result.paper_id for result in results] == ["seed", "other"]
    assert paper_index.owner_calls == []
    assert paper_index.neighbor_calls == []


def test_enabled_paper_neighborhood_without_provider_keeps_ranking():
    candidates = [_result("seed"), _result("other")]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["seed", "other"]


def test_duplicate_chunks_do_not_consume_paper_ranks():
    initial = [
        _result("p1", chunk_id="p1#c0000"),
        _result("p1", chunk_id="p1#c0001"),
        _result("p2"),
    ]
    expanded = [_result("p2"), _result("p3")]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p2", "p1", "p3"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)


def test_ties_are_deterministic_and_prefer_first_seen_run():
    initial = [_result("original")]
    expanded = [_result("expanded")]

    first = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded])
    ).retrieve("question", 10)
    second = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded])
    ).retrieve("question", 10)

    assert [result.paper_id for result in first] == ["original", "expanded"]
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


@pytest.mark.parametrize(
    ("title", "text"),
    [
        ("", "Representative text"),
        ("Seed title", ""),
    ],
)
def test_missing_seed_content_falls_back_without_second_retrieval(title, text):
    initial = [
        _result("p1", title=title, text=text, chunk_id="p1#c0000"),
        _result("p1", title="Duplicate", chunk_id="p1#c0001"),
        _result("p2"),
    ]
    inner = _FakeRetriever([initial])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p1", "p2"]
    assert len(inner.calls) == 1
    assert results[0] is initial[0]


def test_final_reranker_also_applies_when_seed_content_is_missing():
    initial = [
        _result("p1", title="", text="Representative text"),
        _result("p2"),
        _result("p3"),
    ]
    inner = _FakeRetriever([initial])
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        inner,
        reranker=reranker,
        rerank_pool_k=2,
    )

    results = retriever.retrieve("question", 2)

    assert len(inner.calls) == 1
    assert len(reranker.calls) == 1
    assert [candidate.paper_id for candidate in reranker.calls[0][1]] == [
        "p1",
        "p2",
    ]
    assert [result.paper_id for result in results] == ["p2", "p1"]


def test_empty_initial_results_return_without_second_retrieval():
    inner = _FakeRetriever([[]])
    retriever = SeedExpansionRetriever(inner)

    assert retriever.retrieve("question", 10) == []
    assert len(inner.calls) == 1


def test_empty_expansion_falls_back_to_initial_ranking():
    initial = [_result("p1"), _result("p2")]
    inner = _FakeRetriever([initial, []])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("question", 10)

    assert results == initial
    assert len(inner.calls) == 2


def test_final_reranker_also_applies_when_expanded_search_is_empty():
    initial = [_result("p1"), _result("p2"), _result("p3")]
    inner = _FakeRetriever([initial, []])
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        inner,
        reranker=reranker,
        rerank_pool_k=2,
    )

    results = retriever.retrieve("question", 2)

    assert len(inner.calls) == 2
    assert len(reranker.calls) == 1
    assert [candidate.paper_id for candidate in reranker.calls[0][1]] == [
        "p1",
        "p2",
    ]
    assert [result.paper_id for result in results] == ["p2", "p1"]


def test_default_max_results_caps_output_at_ten():
    initial = [_result(f"p{i:02d}") for i in range(15)]
    expanded = [_result(f"p{i:02d}") for i in range(15)]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    results = retriever.retrieve("question", 50)

    assert len(results) == 10


@pytest.mark.parametrize("extended_k", [20, 50])
def test_stable_prefix_keeps_top_ten_when_appending_a_deeper_tail(
    extended_k: int,
):
    original = [
        *(
            _result(f"p{index}", title=f"Paper {index}")
            for index in range(1, 9)
        ),
        _result("owner", title="Truncated Consistency Models"),
        _result("tail-10", title="Tail paper 10"),
        *(
            _result(f"tail-{index}", title=f"Tail paper {index}")
            for index in range(11, 16)
        ),
        _result("named", title="D-FINE: Distribution Refinement"),
        *(
            _result(f"tail-{index}", title=f"Tail paper {index}")
            for index in range(17, 61)
        ),
    ]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {"related": related},
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )

    def _retrieve(
        max_results: int,
        stable_prefix_k: int | None = None,
    ) -> list[RetrievalResult]:
        retriever = SeedExpansionRetriever(
            _FakeRetriever(
                [original, original],
                indexers=[paper_index],
            ),
            max_results=max_results,
            stable_prefix_k=stable_prefix_k,
            protect_explicit_title_matches=True,
            method_owner_weight=1.0,
            method_relation_weight=0.5,
        )
        return retriever.retrieve(
            "Compare TCM with the D-FINE paper.",
            max_results,
            hints=SearchHints(methods=("TCM",)),
        )

    top_ten = _retrieve(10)
    extended = _retrieve(extended_k, stable_prefix_k=10)

    assert extended[:10] == top_ten
    assert len(extended) == extended_k
    assert len({result.paper_id for result in extended}) == extended_k
    assert any(result.paper_id.startswith("tail-") for result in extended[10:])
    assert all(
        left.score > right.score
        for left, right in zip(extended[9:], extended[10:])
    )


def test_method_dense_tail_preserves_top_ten_and_adds_valid_neighbors():
    candidates = [
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(
            _result(f"p{index}", title=f"Paper {index}")
            for index in range(2, 21)
        ),
    ]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": f"Title for {paper_id}"},
        )
        for paper_id in ("owner", "p15", "dense-new")
    }
    paper_index = _FakePaperIndex(
        documents,
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "owner": [
                _result("owner", score=1.0),
                _result("dense-new", score=0.9),
                _result("p15", score=0.8),
                _result("outside-corpus", score=0.7),
            ]
        }
    )

    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve(
        "Question about TCM",
        20,
        hints=SearchHints(methods=("TCM",)),
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=3.0,
        method_dense_tail_max_new_papers=1,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "Question about TCM",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert results[:10] == baseline[:10]
    assert [
        result.to_dict() for result in results[:10]
    ] == [result.to_dict() for result in baseline[:10]]
    assert "dense-new" in [result.paper_id for result in results[10:]]
    assert "outside-corpus" not in [result.paper_id for result in results]
    assert dense_store.calls == [("owner", 20)]
    dense_result = next(
        result for result in results if result.paper_id == "dense-new"
    )
    assert dense_result.source == "method_dense_tail_rrf"
    assert dense_result.metadata["method_dense_tail_baseline_rank"] is None
    assert dense_result.metadata["method_dense_tail_best_neighbor_rank"] == 2
    assert dense_result.metadata["method_dense_tail_best_similarity"] == 0.9
    assert dense_result.metadata["method_dense_tail_via_papers"] == ["owner"]
    assert dense_result.metadata["method_dense_tail_is_new"] is True


def test_method_dense_tail_limits_new_papers_and_is_deterministic():
    candidates = [
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(_result(f"p{index}") for index in range(2, 21)),
    ]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in ("owner", "new-a", "new-b", "new-c")
    }
    paper_index = _FakePaperIndex(
        documents,
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "owner": [
                _result("new-b", score=0.8),
                _result("new-a", score=0.8),
                _result("new-c", score=0.7),
            ]
        }
    )

    def retrieve() -> list[RetrievalResult]:
        retriever = SeedExpansionRetriever(
            _FakeRetriever(
                [candidates, candidates],
                indexers=[paper_index],
            ),
            max_results=20,
            stable_prefix_k=10,
            paper_embedding_index_dir="/tmp/paper-embeddings",
            method_dense_tail_weight=4.0,
            method_dense_tail_max_new_papers=2,
        )
        retriever._paper_embedding_store = dense_store
        return retriever.retrieve(
            "Question about TCM",
            20,
            hints=SearchHints(methods=("TCM",)),
        )

    first = retrieve()
    second = retrieve()

    first_ids = [result.paper_id for result in first]
    assert "new-a" in first_ids
    assert "new-b" in first_ids
    assert "new-c" not in first_ids
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


def test_method_dense_tail_owner_outside_prefix_keeps_exact_baseline():
    candidates = [
        *(_result(f"p{index}") for index in range(1, 11)),
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(_result(f"p{index}") for index in range(12, 21)),
    ]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner paper",
                chunk_type="paper",
                metadata={"title": "TCM owner"},
            )
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {"owner": [_result("dense-new", score=0.9)]}
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20, hints=SearchHints(methods=("TCM",)))
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=1.0,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "question",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert results == baseline
    assert dense_store.calls == []


def test_method_dense_tail_provider_failure_keeps_exact_baseline():
    candidates = [
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(_result(f"p{index}") for index in range(2, 21)),
    ]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner paper",
                chunk_type="paper",
                metadata={"title": "TCM owner"},
            )
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20, hints=SearchHints(methods=("TCM",)))
    dense_store = _FakePaperEmbeddingStore(
        {},
        failing_papers={"owner"},
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=1.0,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "question",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert results == baseline
    assert dense_store.calls == [("owner", 20)]


def test_method_dense_tail_continues_after_one_seed_fails():
    candidates = [
        _result("owner-a", title="TCM-A: First owner"),
        _result("owner-b", title="TCM-B: Second owner"),
        *(_result(f"p{index}") for index in range(3, 21)),
    ]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in ("owner-a", "owner-b", "dense-new")
    }
    paper_index = _FakePaperIndex(
        documents,
        owners=(
            {
                "paper_id": "owner-a",
                "aliases": ["TCM-A"],
                "strength": 2,
            },
            {
                "paper_id": "owner-b",
                "aliases": ["TCM-B"],
                "strength": 2,
            },
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {"owner-b": [_result("dense-new", score=0.9)]},
        failing_papers={"owner-a"},
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=4.0,
        method_dense_tail_seed_k=2,
        method_dense_tail_max_new_papers=1,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "Question about TCM-A and TCM-B",
        20,
        hints=SearchHints(methods=("TCM-A", "TCM-B")),
    )

    assert "dense-new" in [result.paper_id for result in results[10:]]
    assert dense_store.calls == [("owner-a", 20), ("owner-b", 20)]
    dense_result = next(
        result for result in results if result.paper_id == "dense-new"
    )
    assert dense_result.metadata["method_dense_tail_via_papers"] == [
        "owner-b"
    ]


def test_method_dense_tail_is_disabled_by_default():
    candidates = [_result("owner"), _result("other")]
    paper_index = _FakePaperIndex(
        {},
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=1,
        paper_embedding_index_dir="/tmp/paper-embeddings",
    )
    dense_store = _FakePaperEmbeddingStore(
        {"owner": [_result("dense-new", score=0.9)]}
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "question",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert [result.paper_id for result in results] == ["owner", "other"]
    assert dense_store.calls == []


def test_paper_dense_tail_preserves_top_ten_and_adds_rank_one_neighbors():
    candidates = [
        _result("rank-one", title="Leading paper"),
        *(_result(f"p{index}") for index in range(2, 21)),
    ]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in ("rank-one", "dense-new")
    }
    paper_index = _FakePaperIndex(documents)
    dense_store = _FakePaperEmbeddingStore(
        {
            "rank-one": [
                _result("dense-new", score=0.9),
                _result("outside-corpus", score=0.8),
            ]
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_tail_weight=1.0,
        paper_dense_tail_max_results=7,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert results[:10] == baseline[:10]
    assert [result.to_dict() for result in results[:10]] == [
        result.to_dict() for result in baseline[:10]
    ]
    assert "dense-new" in [result.paper_id for result in results[10:]]
    assert "outside-corpus" not in [result.paper_id for result in results]
    assert dense_store.calls == [("rank-one", 7)]
    dense_result = next(
        result for result in results if result.paper_id == "dense-new"
    )
    assert dense_result.source == "paper_dense_tail_rrf"
    assert dense_result.metadata["paper_dense_tail_via_papers"] == [
        "rank-one"
    ]
    assert dense_result.metadata["paper_dense_tail_best_neighbor_rank"] == 1
    assert dense_result.metadata["paper_dense_tail_is_new"] is True
    assert "method_dense_tail_via_papers" not in dense_result.metadata


def test_paper_and_method_dense_tail_deduplicate_the_same_seed_and_score():
    candidates = [
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(_result(f"p{index}") for index in range(2, 21)),
    ]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in ("owner", "dense-new")
    }
    paper_index = _FakePaperIndex(
        documents,
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {"owner": [_result("dense-new", score=0.9)]}
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        rrf_k=60,
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=1.0,
        method_dense_tail_max_results=20,
        method_dense_tail_max_new_papers=1,
        paper_dense_tail_weight=3.0,
        paper_dense_tail_max_results=7,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "Question about TCM",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert dense_store.calls == [("owner", 20)]
    dense_result = next(
        result for result in results if result.paper_id == "dense-new"
    )
    expected_score = 3.0 / 61.0
    assert dense_result.source == "paper_method_dense_tail_rrf"
    assert dense_result.metadata["method_dense_tail_via_papers"] == ["owner"]
    assert dense_result.metadata["paper_dense_tail_via_papers"] == ["owner"]
    assert dense_result.metadata["method_dense_tail_rrf_score"] == pytest.approx(
        expected_score
    )
    assert dense_result.metadata["paper_dense_tail_rrf_score"] == pytest.approx(
        expected_score
    )


def test_paper_and_method_dense_tail_share_the_new_paper_budget():
    candidates = [
        _result("prefix-seed", title="Leading paper"),
        _result("owner", title="TCM: Truncated Consistency Models"),
        *(_result(f"p{index}") for index in range(3, 21)),
    ]
    new_paper_ids = {"generic-a", "generic-b", "method-a", "method-b"}
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Full paper text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in {"prefix-seed", "owner", *new_paper_ids}
    }
    paper_index = _FakePaperIndex(
        documents,
        owners=(
            {"paper_id": "owner", "aliases": ["TCM"], "strength": 2},
        ),
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "owner": [
                _result("method-a", score=0.9),
                _result("method-b", score=0.8),
            ],
            "prefix-seed": [
                _result("generic-a", score=0.9),
                _result("generic-b", score=0.8),
            ],
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        method_dense_tail_weight=1.0,
        method_dense_tail_max_results=7,
        method_dense_tail_max_new_papers=2,
        paper_dense_tail_weight=4.0,
        paper_dense_tail_max_results=7,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve(
        "Question about TCM",
        20,
        hints=SearchHints(methods=("TCM",)),
    )

    assert dense_store.calls == [("owner", 7), ("prefix-seed", 7)]
    added_ids = {
        result.paper_id for result in results if result.paper_id in new_paper_ids
    }
    assert added_ids == {"generic-a", "generic-b"}
    assert len(added_ids) == 2


def test_paper_dense_tail_failure_keeps_exact_baseline():
    candidates = [
        _result("rank-one"),
        *(_result(f"p{index}") for index in range(2, 21)),
    ]
    paper_index = _FakePaperIndex(
        {
            "rank-one": Chunk(
                chunk_id="rank-one#paper",
                paper_id="rank-one",
                text="Full paper text",
                chunk_type="paper",
                metadata={"title": "Leading paper"},
            )
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_tail_weight=1.0,
    )
    dense_store = _FakePaperEmbeddingStore(
        {},
        failing_papers={"rank-one"},
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
    assert dense_store.calls == [("rank-one", 7)]


def test_paper_dense_consensus_is_disabled_by_default():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    paper_index = _FakePaperIndex({})
    dense_store = _FakePaperEmbeddingStore(
        {"p1": [_result("consensus-new", score=0.9)]}
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
    )
    retriever._paper_embedding_store = dense_store

    retriever.retrieve("question", 20)

    assert dense_store.calls == []
    assert paper_index.calls == []


def test_paper_dense_reciprocal_replaces_only_the_final_slot():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    reciprocal_document = Chunk(
        chunk_id="reciprocal-new#paper",
        paper_id="reciprocal-new",
        text="Reciprocal paper text",
        chunk_type="paper",
        metadata={"title": "Reciprocal paper"},
    )
    paper_index = _FakePaperIndex(
        {"reciprocal-new": reciprocal_document}
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "p7": [
                _result("p15", score=0.99),
                _result("reciprocal-new", score=0.9),
            ],
            "reciprocal-new": [
                _result(f"p{index}", score=1.0 - index / 100)
                for index in range(1, 7)
            ],
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_reciprocal_seed_k=8,
        paper_dense_reciprocal_forward_k=20,
        paper_dense_reciprocal_reverse_k=10,
        paper_dense_reciprocal_min_support=6,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results[:-1]] == [
        result.to_dict() for result in baseline[:-1]
    ]
    assert results[-1].paper_id == "reciprocal-new"
    assert results[-1].score == baseline[-1].score
    assert results[-1].source == "paper_dense_reciprocal_exploration"
    assert dense_store.calls == [
        *((f"p{index}", 20) for index in range(1, 9)),
        ("reciprocal-new", 10),
    ]
    assert paper_index.calls == ["reciprocal-new"]
    assert paper_index.owner_calls == []
    assert paper_index.neighbor_calls == []
    assert results[-1].metadata == {
        "title": "Reciprocal paper",
        "paper_dense_reciprocal_seed_count": 8,
        "paper_dense_reciprocal_discovered_candidates": 1,
        "paper_dense_reciprocal_examined_candidates": 1,
        "paper_dense_reciprocal_support": 6,
        "paper_dense_reciprocal_forward_support": 1,
        "paper_dense_reciprocal_best_forward_rank": 2,
        "paper_dense_reciprocal_best_reverse_rank": 1,
        "paper_dense_reciprocal_best_similarity": 0.9,
        "paper_dense_reciprocal_forward_rrf_score": pytest.approx(1 / 62),
        "paper_dense_reciprocal_reverse_rrf_score": pytest.approx(
            sum(1 / (60 + rank) for rank in range(1, 7))
        ),
        "paper_dense_reciprocal_forward_via_papers": ["p7"],
        "paper_dense_reciprocal_reverse_via_papers": [
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
        ],
        "paper_dense_reciprocal_replaced_paper_id": "p20",
        "paper_dense_reciprocal_is_new": True,
    }
    assert all(
        left.score >= right.score
        for left, right in zip(results, results[1:])
    )


def test_paper_dense_reciprocal_requires_distinct_reverse_seed_support():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    paper_index = _FakePaperIndex({})
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [_result("candidate", score=0.9)],
            "candidate": [
                _result("p1", score=0.99),
                _result("p1", score=0.98),
                _result("p2", score=0.97),
            ],
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_reciprocal_seed_k=3,
        paper_dense_reciprocal_min_support=3,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
    assert paper_index.calls == []


def test_paper_dense_reciprocal_breaks_complete_ties_by_paper_id():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Text for {paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for paper_id in ("candidate-a", "candidate-z")
    }
    paper_index = _FakePaperIndex(documents)
    reverse_neighbors = [
        _result(f"p{index}", score=1.0 - index / 100)
        for index in range(1, 4)
    ]
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [_result("candidate-z", score=0.9)],
            "p2": [_result("candidate-a", score=0.9)],
            "candidate-a": reverse_neighbors,
            "candidate-z": reverse_neighbors,
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_reciprocal_seed_k=3,
        paper_dense_reciprocal_min_support=3,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert results[-1].paper_id == "candidate-a"


def test_paper_dense_reciprocal_caps_reverse_candidate_searches():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Text for {paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for paper_id in ("candidate-a", "candidate-b")
    }
    paper_index = _FakePaperIndex(documents)
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [
                _result("candidate-a", score=0.9),
                _result("candidate-b", score=0.8),
            ],
            "candidate-a": [
                _result("p1", score=0.99),
                _result("p2", score=0.98),
            ],
            "candidate-b": [
                _result("p1", score=0.99),
                _result("p2", score=0.98),
            ],
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_reciprocal_seed_k=2,
        paper_dense_reciprocal_min_support=2,
        paper_dense_reciprocal_max_candidates=1,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert results[-1].paper_id == "candidate-a"
    assert dense_store.calls == [
        ("p1", 20),
        ("p2", 20),
        ("candidate-a", 10),
    ]
    assert results[-1].metadata[
        "paper_dense_reciprocal_discovered_candidates"
    ] == 2
    assert results[-1].metadata[
        "paper_dense_reciprocal_examined_candidates"
    ] == 1


def test_paper_dense_reciprocal_failure_falls_through_to_existing_ranking():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    paper_index = _FakePaperIndex({})
    dense_store = _FakePaperEmbeddingStore(
        {"p1": [_result("candidate", score=0.9)]},
        failing_papers={"candidate"},
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_reciprocal_seed_k=3,
        paper_dense_reciprocal_min_support=2,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
    assert paper_index.calls == []


def test_method_bridge_replaces_only_final_slot_with_dual_support():
    candidates = [
        _result("owner"),
        *[_result(f"p{index}") for index in range(2, 21)],
    ]
    linked_document = Chunk(
        chunk_id="linked#paper",
        paper_id="linked",
        text="Linked paper text",
        chunk_type="paper",
        metadata={"title": "Linked paper"},
    )
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic document",
                chunk_type="paper",
                metadata={"title": "Owner paper"},
            ),
            "linked": linked_document,
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["OWNER"],
                "strength": 2,
            },
        ),
        neighbors={
            "p11": (
                {
                    "paper_id": "linked",
                    "aliases": ["LINK"],
                    "strength": 1,
                },
            ),
        },
    )
    inner = _FakeRetriever(
        [
            candidates,
            candidates,
            [_result("p2"), _result("linked")],
        ],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        max_results=20,
        stable_prefix_k=10,
        literal_attribute_hints=True,
        literal_method_hints=True,
        method_topic_seed_chars=100,
        method_topic_max_results=10,
        method_bridge_topic_max_rank=2,
    )

    results = retriever.retrieve(
        "In the OWNER paper, what is reported?",
        20,
    )

    assert [result.paper_id for result in results[:-1]] == [
        result.paper_id for result in candidates[:-1]
    ]
    assert results[-1].paper_id == "linked"
    assert results[-1].source == "method_bridge_exploration"
    assert results[-1].metadata == {
        "title": "Linked paper",
        "method_bridge_topic_rank": 2,
        "method_bridge_owner_papers": ["owner"],
        "method_bridge_via_papers": ["p11"],
        "method_bridge_aliases": ["LINK"],
        "method_bridge_strength": 1,
        "method_bridge_replaced_paper_id": "p20",
        "method_bridge_is_new": True,
    }


def test_method_bridge_requires_owner_topic_support():
    candidates = [
        _result("owner"),
        *[_result(f"p{index}") for index in range(2, 21)],
    ]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic document",
                chunk_type="paper",
                metadata={},
            ),
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["OWNER"],
                "strength": 2,
            },
        ),
        neighbors={
            "p11": (
                {
                    "paper_id": "unsupported",
                    "aliases": ["METHOD"],
                    "strength": 1,
                },
            ),
        },
    )
    inner = _FakeRetriever(
        [
            candidates,
            candidates,
            [_result("different")],
        ],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        max_results=20,
        stable_prefix_k=10,
        literal_attribute_hints=True,
        literal_method_hints=True,
        method_topic_max_results=10,
        method_bridge_topic_max_rank=5,
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("In the OWNER paper, what is reported?", 20)

    results = retriever.retrieve(
        "In the OWNER paper, what is reported?",
        20,
    )

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]


def test_paper_dense_consensus_replaces_only_the_final_slot():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    consensus_document = Chunk(
        chunk_id="consensus-new#paper",
        paper_id="consensus-new",
        text="Consensus paper text",
        chunk_type="paper",
        metadata={"title": "Consensus paper"},
    )
    paper_index = _FakePaperIndex(
        {"consensus-new": consensus_document}
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [
                _result("p15", score=0.99),
                _result("consensus-new", score=0.9),
                _result("consensus-new", score=0.1),
            ],
            "p2": [_result("consensus-new", score=0.93)],
            "p3": [_result("unsupported", score=0.95)],
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_consensus_seed_k=3,
        paper_dense_consensus_max_results=7,
        paper_dense_consensus_min_support=2,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results[:-1]] == [
        result.to_dict() for result in baseline[:-1]
    ]
    assert results[-1].paper_id == "consensus-new"
    assert results[-1].score == baseline[-1].score
    assert results[-1].source == "paper_dense_consensus_exploration"
    assert dense_store.calls == [("p1", 7), ("p2", 7), ("p3", 7)]
    assert paper_index.calls == ["consensus-new"]
    assert results[-1].metadata == {
        "title": "Consensus paper",
        "paper_dense_consensus_support": 2,
        "paper_dense_consensus_via_papers": ["p1", "p2"],
        "paper_dense_consensus_best_neighbor_rank": 1,
        "paper_dense_consensus_best_similarity": 0.93,
        "paper_dense_consensus_rrf_score": pytest.approx(1 / 62 + 1 / 61),
        "paper_dense_consensus_replaced_paper_id": "p20",
        "paper_dense_consensus_is_new": True,
    }
    assert all(
        left.score >= right.score
        for left, right in zip(results, results[1:])
    )


def test_paper_dense_consensus_requires_support_from_distinct_seeds():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    document = Chunk(
        chunk_id="candidate#paper",
        paper_id="candidate",
        text="Candidate text",
        chunk_type="paper",
        metadata={},
    )
    paper_index = _FakePaperIndex({"candidate": document})
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [
                _result("candidate", score=0.9),
                _result("candidate", score=0.8),
            ],
            "p2": [_result("other", score=0.9)],
        }
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_consensus_seed_k=2,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
    assert dense_store.calls == [("p1", 7), ("p2", 7)]
    assert paper_index.calls == []


def test_paper_dense_consensus_breaks_complete_ties_by_paper_id():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Text for {paper_id}",
            chunk_type="paper",
            metadata={"title": paper_id},
        )
        for paper_id in ("candidate-a", "candidate-z")
    }
    paper_index = _FakePaperIndex(documents)
    results_by_paper = {
        "p1": [
            _result("candidate-z", score=0.9),
            _result("candidate-a", score=0.9),
        ],
        "p2": [
            _result("candidate-a", score=0.9),
            _result("candidate-z", score=0.9),
        ],
    }

    def retrieve() -> list[RetrievalResult]:
        retriever = SeedExpansionRetriever(
            _FakeRetriever(
                [candidates, candidates],
                indexers=[paper_index],
            ),
            max_results=20,
            stable_prefix_k=10,
            paper_embedding_index_dir="/tmp/paper-embeddings",
            paper_dense_consensus_seed_k=2,
        )
        retriever._paper_embedding_store = _FakePaperEmbeddingStore(
            results_by_paper
        )
        return retriever.retrieve("question", 20)

    first = retrieve()
    second = retrieve()

    assert first[-1].paper_id == "candidate-a"
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


def test_paper_dense_consensus_skips_invalid_corpus_documents():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    paper_index = _FakePaperIndex(
        {
            "invalid": Chunk(
                chunk_id="invalid#paper",
                paper_id="invalid",
                text=" ",
                chunk_type="paper",
                metadata={},
            ),
            "valid": Chunk(
                chunk_id="valid#paper",
                paper_id="valid",
                text="Valid paper text",
                chunk_type="paper",
                metadata={},
            ),
        }
    )
    dense_store = _FakePaperEmbeddingStore(
        {
            "p1": [
                _result("invalid", score=0.99),
                _result("valid", score=0.8),
            ],
            "p2": [
                _result("invalid", score=0.99),
                _result("valid", score=0.8),
            ],
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_consensus_seed_k=2,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert results[-1].paper_id == "valid"
    assert paper_index.calls == ["invalid", "valid"]


def test_paper_dense_consensus_failure_keeps_exact_final_ranking():
    candidates = [_result(f"p{index}") for index in range(1, 21)]
    paper_index = _FakePaperIndex({})
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("question", 20)
    dense_store = _FakePaperEmbeddingStore(
        {"p1": [_result("candidate", score=0.9)]},
        failing_papers={"p2"},
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=20,
        stable_prefix_k=10,
        paper_embedding_index_dir="/tmp/paper-embeddings",
        paper_dense_consensus_seed_k=2,
    )
    retriever._paper_embedding_store = dense_store

    results = retriever.retrieve("question", 20)

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
    assert dense_store.calls == [("p1", 7), ("p2", 7)]
    assert paper_index.calls == []


def test_final_candidate_rerank_runs_once_after_dense_tail_and_consensus():
    candidates = [
        _result(
            f"p{index}",
            text=f"Original result text for p{index}",
        )
        for index in range(1, 21)
    ]
    paper_ids = {
        *(candidate.paper_id for candidate in candidates),
        "tail-new",
        "consensus-new",
    }
    documents = {
        paper_id: Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=f"Proxy document text for {paper_id}",
            chunk_type="paper",
            metadata={"title": f"Title for {paper_id}"},
        )
        for paper_id in paper_ids
    }
    dense_results = {
        "p1": [
            _result("tail-new", score=0.95),
            _result("consensus-new", score=0.9),
        ],
        "p2": [_result("consensus-new", score=0.92)],
        "p3": [_result("consensus-new", score=0.91)],
    }

    def retrieve(reranker=None):
        paper_index = _FakePaperIndex(documents)
        dense_store = _FakePaperEmbeddingStore(dense_results)
        retriever = SeedExpansionRetriever(
            _FakeRetriever(
                [candidates, candidates],
                indexers=[paper_index],
            ),
            reranker=reranker,
            max_results=20,
            stable_prefix_k=10,
            paper_embedding_index_dir="/tmp/paper-embeddings",
            paper_dense_tail_weight=4.0,
            paper_dense_tail_seed_k=1,
            paper_dense_tail_max_results=1,
            paper_dense_consensus_seed_k=3,
            paper_dense_consensus_max_results=7,
            paper_dense_consensus_min_support=2,
            rerank_final_candidates=reranker is not None,
            final_rerank_document_chars=12,
        )
        retriever._paper_embedding_store = dense_store
        return retriever.retrieve("question", 20)

    baseline = retrieve()
    final_reranker = _RecordingFinalReranker()
    results = retrieve(final_reranker)

    baseline_ids = [result.paper_id for result in baseline]
    assert {"tail-new", "consensus-new"} <= set(baseline_ids)
    assert len(final_reranker.calls) == 1
    rerank_query, proxies, rerank_top_k = final_reranker.calls[0]
    assert rerank_query == "question"
    assert rerank_top_k == 20
    assert [proxy.paper_id for proxy in proxies] == baseline_ids
    assert all(
        proxy.text == documents[proxy.paper_id].text[:12]
        for proxy in proxies
    )

    assert [result.paper_id for result in results] == list(
        reversed(baseline_ids)
    )
    assert {result.paper_id for result in results} == set(baseline_ids)
    baseline_by_id = {result.paper_id: result for result in baseline}
    for result in results:
        original = baseline_by_id[result.paper_id]
        assert result.text == original.text
        assert result.chunk_id == original.chunk_id
        assert result.chunk_type == original.chunk_type
        assert result.source == original.source
        assert result.metadata["title"] == original.metadata["title"]
        assert "test_final_rerank_rank" in result.metadata
    assert results[0].metadata["final_rerank_status"] == "applied"
    assert (
        results[0].metadata["final_rerank_candidate_set_preserved"] is True
    )
    assert results[0].metadata["pre_rerank_candidate_papers"] == baseline_ids
    assert "final_rerank_error_type" not in results[0].metadata


def test_final_candidate_rerank_exception_falls_back_to_original_ranking():
    candidates = [
        _result(f"p{index}", text=f"Original text {index}")
        for index in range(1, 4)
    ]
    documents = {
        candidate.paper_id: Chunk(
            chunk_id=f"{candidate.paper_id}#paper",
            paper_id=candidate.paper_id,
            text=f"Proxy document for {candidate.paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for candidate in candidates
    }
    paper_index = _FakePaperIndex(documents)
    baseline = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        max_results=3,
    ).retrieve("question", 3)
    final_reranker = _RecordingFinalReranker(
        error=RuntimeError("inference failed")
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=final_reranker,
        max_results=3,
        rerank_final_candidates=True,
    )

    results = retriever.retrieve("question", 3)

    assert [result.paper_id for result in results] == [
        result.paper_id for result in baseline
    ]
    assert [result.to_dict() for result in results[1:]] == [
        result.to_dict() for result in baseline[1:]
    ]
    fallback_metadata = dict(results[0].metadata)
    assert fallback_metadata.pop("final_rerank_status") == "fallback"
    assert (
        fallback_metadata.pop("final_rerank_candidate_set_preserved") is True
    )
    assert fallback_metadata.pop("final_rerank_error_type") == "RuntimeError"
    assert fallback_metadata == baseline[0].metadata
    assert results[0].score == baseline[0].score
    assert results[0].text == baseline[0].text
    assert results[0].chunk_id == baseline[0].chunk_id
    assert results[0].source == baseline[0].source
    assert len(final_reranker.calls) == 1


@pytest.mark.parametrize(
    "invalid_result",
    ["missing", "duplicate", "non_finite"],
)
def test_final_candidate_rerank_invalid_output_falls_back(invalid_result):
    candidates = [_result(f"p{index}") for index in range(1, 4)]
    paper_index = _FakePaperIndex(
        {
            candidate.paper_id: Chunk(
                chunk_id=f"{candidate.paper_id}#paper",
                paper_id=candidate.paper_id,
                text=f"Proxy document for {candidate.paper_id}",
                chunk_type="paper",
                metadata={},
            )
            for candidate in candidates
        }
    )
    final_reranker = _RecordingFinalReranker(
        invalid_result=invalid_result
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=final_reranker,
        max_results=3,
        rerank_final_candidates=True,
    )

    results = retriever.retrieve("question", 3)

    assert [result.paper_id for result in results] == ["p1", "p2", "p3"]
    assert results[0].metadata["final_rerank_status"] == "fallback"
    assert (
        results[0].metadata["final_rerank_candidate_set_preserved"] is True
    )
    assert results[0].metadata["final_rerank_error_type"] == "ValueError"


def test_final_candidate_rerank_missing_proxy_document_falls_back():
    candidates = [_result("p1"), _result("p2")]
    paper_index = _FakePaperIndex(
        {
            "p1": Chunk(
                chunk_id="p1#paper",
                paper_id="p1",
                text="Proxy document for p1",
                chunk_type="paper",
                metadata={},
            )
        }
    )
    final_reranker = _RecordingFinalReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=final_reranker,
        max_results=2,
        rerank_final_candidates=True,
    )

    results = retriever.retrieve("question", 2)

    assert [result.paper_id for result in results] == ["p1", "p2"]
    assert results[0].metadata["final_rerank_status"] == "fallback"
    assert (
        results[0].metadata["final_rerank_candidate_set_preserved"] is True
    )
    assert results[0].metadata["final_rerank_error_type"] == "ValueError"
    assert final_reranker.calls == []
    assert paper_index.calls == ["p1", "p2"]


def test_requested_top_k_can_reduce_max_results():
    initial = [_result(f"p{i}") for i in range(5)]
    expanded = [_result(f"p{i}") for i in range(5)]
    retriever = SeedExpansionRetriever(_FakeRetriever([initial, expanded]))

    assert len(retriever.retrieve("question", 3)) == 3


def test_final_reranker_runs_once_after_fusion_with_original_query():
    initial = [_result("p1"), _result("p2"), _result("p3")]
    expanded = [_result("p2"), _result("p3"), _result("p4")]
    inner = _FakeRetriever([initial, expanded])
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        inner,
        reranker=reranker,
        rerank_pool_k=3,
    )

    results = retriever.retrieve("original question", 2)

    assert len(inner.calls) == 2
    assert len(reranker.calls) == 1
    rerank_query, candidates, output_k = reranker.calls[0]
    assert rerank_query == "original question"
    assert [candidate.paper_id for candidate in candidates] == ["p2", "p3", "p1"]
    assert output_k == 2
    assert [result.paper_id for result in results] == ["p1", "p3"]
    assert results[0].metadata["pre_rerank_candidate_papers"] == [
        "p2",
        "p3",
        "p1",
    ]


def test_final_reranker_pool_defaults_to_fifty():
    initial = [_result(f"initial-{index:02d}") for index in range(40)]
    expanded = [_result(f"expanded-{index:02d}") for index in range(40)]
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever([initial, expanded]),
        reranker=reranker,
    )

    retriever.retrieve("question", 10)

    assert retriever.rerank_pool_k == 50
    assert len(reranker.calls[0][1]) == 50


@pytest.mark.parametrize(
    ("alias", "question"),
    [
        ("EasySpec", "Which reference is cited in the EasySpec paper?"),
        ("SecEmb", "What is reported in paper ＳｅｃＥｍｂ?"),
        ("PMA", "What accuracy does PMA achieve?"),
        ("D-FINE", "Which setting is used by D-FINE?"),
        ("Model2", "How does Model2 perform?"),
    ],
)
def test_explicit_title_guard_restores_identifier_like_aliases(alias, question):
    candidates = [
        _result("p1", title="Unrelated paper"),
        _result("p2", title="Another unrelated paper"),
        _result("named", title=f"{alias}: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve(question, 2)

    assert [result.paper_id for result in results] == ["p1", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == alias
    assert results[-1].metadata["pre_title_guard_rank"] == 3


def test_explicit_title_guard_keeps_selected_ranking_unchanged():
    candidates = [
        _result("p1", title="Unrelated paper"),
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p3", title="Another unrelated paper"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in the SecEmb paper?", 2)

    assert [result.paper_id for result in results] == ["p1", "named"]
    assert "explicit_title_guard_alias" not in results[1].metadata
    assert "pre_title_guard_rank" not in results[1].metadata


def test_explicit_title_guard_restores_candidate_dropped_by_reranker():
    candidates = [
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p2", title="Second paper"),
        _result("p3", title="Third paper"),
        _result("p4", title="Fourth paper"),
    ]
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        reranker=reranker,
        rerank_pool_k=4,
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 2)

    assert [result.paper_id for result in results] == ["p4", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == "SecEmb"
    assert results[-1].metadata["pre_title_guard_rank"] == 1
    assert results[0].metadata["pre_rerank_candidate_papers"] == [
        "named",
        "p2",
        "p3",
        "p4",
    ]


def test_neighborhood_runs_before_final_reranker_and_title_guard():
    candidates = [
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p2", title="Second paper"),
        _result("p3", title="Third paper"),
        _result("p4", title="Fourth paper"),
    ]
    paper_index = _FakePaperIndex(
        {
            candidate.paper_id: Chunk(
                chunk_id=f"{candidate.paper_id}#paper",
                paper_id=candidate.paper_id,
                text="Independent full-paper text.",
                chunk_type="paper",
                metadata={"title": candidate.metadata["title"]},
            )
            for candidate in candidates
        }
    )
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=reranker,
        rerank_pool_k=4,
        max_results=2,
        protect_explicit_title_matches=True,
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 2)

    assert all(
        candidate.source == "paper_neighborhood_rrf"
        for candidate in reranker.calls[0][1]
    )
    assert [result.paper_id for result in results] == ["p4", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == "SecEmb"


def test_explicit_title_guard_ignores_generic_alias():
    candidates = [
        _result("p1", title="First paper"),
        _result("p2", title="Second paper"),
        _result("generic", title="RAG: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in the RAG paper?", 2)

    assert [result.paper_id for result in results] == ["p1", "p2"]


def test_explicit_title_guard_ignores_ambiguous_alias():
    candidates = [
        _result("p1", title="First paper"),
        _result("ambiguous-1", title="SecEmb: First interpretation"),
        _result("ambiguous-2", title="SecEmb: Second interpretation"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=1,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 1)

    assert [result.paper_id for result in results] == ["p1"]


def test_explicit_title_guard_requires_standalone_alias_occurrence():
    candidates = [
        _result("p1", title="First paper"),
        _result("named", title="SecEmb: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=1,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("How does SecEmbedding perform?", 1)

    assert [result.paper_id for result in results] == ["p1"]


def test_rejects_wrapping_retriever_with_reranker():
    with pytest.raises(ValueError, match="reranker twice"):
        SeedExpansionRetriever(_FakeRetriever([[]], reranker=object()))


def test_proxies_indexers_and_exposes_final_reranker():
    inner = _FakeRetriever([[]])
    final_reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(inner, reranker=final_reranker)

    assert retriever.indexers is inner.indexers
    assert retriever.reranker is final_reranker


def test_final_output_order_is_preserved_by_score_based_paper_aggregation():
    results = [
        _result("first", score=0.2),
        _result("second", score=0.9),
    ]

    aligned = SeedExpansionRetriever._align_scores_with_output_order(results)

    assert [result.paper_id for result in aligned] == ["first", "second"]
    assert [result.score for result in aligned] == [0.9, 0.2]
    assert to_gold_papers(aligned) == ["first", "second"]
    assert aligned[0].metadata["pre_output_order_score"] == 0.2
    assert aligned[1].metadata["pre_output_order_score"] == 0.9
    assert [result.metadata["output_order_rank"] for result in aligned] == [1, 2]


def test_aligned_scores_leave_already_ranked_results_unchanged():
    results = [
        _result("first", score=0.9),
        _result("second", score=0.2),
    ]

    aligned = SeedExpansionRetriever._align_scores_with_output_order(results)

    assert aligned is results
    assert aligned[0] is results[0]
    assert aligned[1] is results[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_k": 0}, "candidate_k"),
        ({"seed_text_chars": 0}, "seed_text_chars"),
        ({"rrf_k": -1}, "rrf_k"),
        ({"max_results": 0}, "max_results"),
        ({"stable_prefix_k": 0}, "stable_prefix_k"),
        ({"rerank_pool_k": 0}, "rerank_pool_k"),
        (
            {"final_rerank_document_chars": 0},
            "final_rerank_document_chars",
        ),
        (
            {"rerank_final_candidates": True},
            "rerank_final_candidates",
        ),
        ({"max_protected_titles": 0}, "max_protected_titles"),
        ({"max_protected_titles": 5}, "max_protected_titles"),
        ({"method_topic_seed_chars": 0}, "method_topic_seed_chars"),
        ({"method_topic_seed_k": 0}, "method_topic_seed_k"),
        ({"method_topic_max_results": 0}, "method_topic_max_results"),
        (
            {"method_bridge_topic_max_rank": -1},
            "method_bridge_topic_max_rank",
        ),
        (
            {
                "method_topic_max_results": 5,
                "method_bridge_topic_max_rank": 6,
            },
            "method_bridge_topic_max_rank",
        ),
        ({"method_relation_seed_k": 0}, "method_relation_seed_k"),
        ({"method_relation_max_results": 0}, "method_relation_max_results"),
        ({"method_dense_tail_seed_k": 0}, "method_dense_tail_seed_k"),
        ({"paper_dense_tail_seed_k": 0}, "paper_dense_tail_seed_k"),
        (
            {"paper_dense_consensus_seed_k": -1},
            "paper_dense_consensus_seed_k",
        ),
        (
            {"paper_dense_reciprocal_seed_k": -1},
            "paper_dense_reciprocal_seed_k",
        ),
        (
            {"method_dense_tail_max_results": 0},
            "method_dense_tail_max_results",
        ),
        (
            {"paper_dense_tail_max_results": 0},
            "paper_dense_tail_max_results",
        ),
        (
            {"paper_dense_consensus_max_results": 0},
            "paper_dense_consensus_max_results",
        ),
        (
            {"paper_dense_consensus_min_support": 0},
            "paper_dense_consensus_min_support",
        ),
        (
            {"paper_dense_reciprocal_forward_k": 0},
            "paper_dense_reciprocal_forward_k",
        ),
        (
            {"paper_dense_reciprocal_reverse_k": 0},
            "paper_dense_reciprocal_reverse_k",
        ),
        (
            {"paper_dense_reciprocal_min_support": 0},
            "paper_dense_reciprocal_min_support",
        ),
        (
            {"paper_dense_reciprocal_max_candidates": 0},
            "paper_dense_reciprocal_max_candidates",
        ),
        (
            {"paper_dense_reciprocal_max_candidates": 129},
            "paper_dense_reciprocal_max_candidates",
        ),
        (
            {
                "paper_dense_consensus_seed_k": 1,
                "paper_dense_consensus_min_support": 2,
            },
            "paper_dense_consensus_min_support",
        ),
        (
            {
                "paper_dense_reciprocal_seed_k": 5,
                "paper_dense_reciprocal_min_support": 6,
            },
            "paper_dense_reciprocal_min_support",
        ),
        (
            {
                "paper_dense_reciprocal_seed_k": 8,
                "paper_dense_reciprocal_reverse_k": 5,
                "paper_dense_reciprocal_min_support": 6,
            },
            "paper_dense_reciprocal_min_support",
        ),
        (
            {
                "max_results": 5,
                "paper_dense_reciprocal_seed_k": 8,
                "paper_dense_reciprocal_min_support": 6,
            },
            "paper_dense_reciprocal_min_support",
        ),
        (
            {"method_relation_max_new_papers": -1},
            "method_relation_max_new_papers",
        ),
        (
            {"method_relation_protected_top_k": -1},
            "method_relation_protected_top_k",
        ),
        (
            {"method_dense_tail_max_new_papers": -1},
            "method_dense_tail_max_new_papers",
        ),
        (
            {"method_dense_tail_max_new_papers": 11},
            "method_dense_tail_max_new_papers",
        ),
        (
            {"paper_embedding_index_dir": ""},
            "paper_embedding_index_dir",
        ),
    ],
)
def test_validates_constructor_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SeedExpansionRetriever(_FakeRetriever([[]]), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"protect_explicit_title_matches": 1}, "protect_explicit_title_matches"),
        ({"literal_attribute_hints": 1}, "literal_attribute_hints"),
        ({"literal_method_hints": 1}, "literal_method_hints"),
        ({"rerank_final_candidates": 1}, "rerank_final_candidates"),
        (
            {"final_rerank_document_chars": True},
            "final_rerank_document_chars",
        ),
        (
            {"final_rerank_document_chars": 2.5},
            "final_rerank_document_chars",
        ),
        ({"stable_prefix_k": True}, "stable_prefix_k"),
        ({"stable_prefix_k": 2.5}, "stable_prefix_k"),
        ({"max_protected_titles": True}, "max_protected_titles"),
        ({"max_protected_titles": "4"}, "max_protected_titles"),
        ({"method_topic_seed_chars": True}, "method_topic_seed_chars"),
        ({"method_topic_seed_k": 1.5}, "method_topic_seed_k"),
        ({"method_topic_max_results": "50"}, "method_topic_max_results"),
        (
            {"method_bridge_topic_max_rank": True},
            "method_bridge_topic_max_rank",
        ),
        (
            {"method_bridge_topic_max_rank": 2.5},
            "method_bridge_topic_max_rank",
        ),
        ({"method_relation_seed_k": True}, "method_relation_seed_k"),
        ({"method_relation_max_results": 2.5}, "method_relation_max_results"),
        ({"method_dense_tail_seed_k": True}, "method_dense_tail_seed_k"),
        ({"paper_dense_tail_seed_k": True}, "paper_dense_tail_seed_k"),
        (
            {"paper_dense_consensus_seed_k": True},
            "paper_dense_consensus_seed_k",
        ),
        (
            {"paper_dense_reciprocal_seed_k": True},
            "paper_dense_reciprocal_seed_k",
        ),
        (
            {"method_dense_tail_max_results": 2.5},
            "method_dense_tail_max_results",
        ),
        (
            {"paper_dense_tail_max_results": 2.5},
            "paper_dense_tail_max_results",
        ),
        (
            {"paper_dense_consensus_max_results": 2.5},
            "paper_dense_consensus_max_results",
        ),
        (
            {"paper_dense_consensus_min_support": True},
            "paper_dense_consensus_min_support",
        ),
        (
            {"paper_dense_reciprocal_forward_k": 2.5},
            "paper_dense_reciprocal_forward_k",
        ),
        (
            {"paper_dense_reciprocal_reverse_k": True},
            "paper_dense_reciprocal_reverse_k",
        ),
        (
            {"paper_dense_reciprocal_min_support": "6"},
            "paper_dense_reciprocal_min_support",
        ),
        (
            {"paper_dense_reciprocal_max_candidates": True},
            "paper_dense_reciprocal_max_candidates",
        ),
        (
            {"method_relation_max_new_papers": True},
            "method_relation_max_new_papers",
        ),
        (
            {"method_relation_protected_top_k": 8.0},
            "method_relation_protected_top_k",
        ),
        (
            {"method_dense_tail_max_new_papers": True},
            "method_dense_tail_max_new_papers",
        ),
        (
            {"paper_embedding_index_dir": 123},
            "paper_embedding_index_dir",
        ),
    ],
)
def test_validates_explicit_title_guard_parameter_types(kwargs, message):
    with pytest.raises(TypeError, match=message):
        SeedExpansionRetriever(_FakeRetriever([[]]), **kwargs)


@pytest.mark.parametrize("weight", [True, "0.5", None])
def test_rejects_non_numeric_local_expansion_weight(weight):
    with pytest.raises(TypeError, match="local_expansion_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            local_expansion_weight=weight,
        )


@pytest.mark.parametrize("weight", [True, "0.2", None])
def test_rejects_non_numeric_paper_neighborhood_weight(weight):
    with pytest.raises(TypeError, match="paper_neighborhood_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_weight=weight,
        )


@pytest.mark.parametrize("weight", [True, "0.05", None])
def test_rejects_non_numeric_paper_neighborhood_two_hop_weight(weight):
    with pytest.raises(TypeError, match="paper_neighborhood_two_hop_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_two_hop_weight=weight,
        )


@pytest.mark.parametrize(
    "name",
    [
        "method_owner_weight",
        "method_relation_weight",
        "method_topic_weight",
        "method_dense_tail_weight",
        "paper_dense_tail_weight",
    ],
)
@pytest.mark.parametrize("weight", [True, "0.5", None])
def test_rejects_non_numeric_method_relation_weights(name, weight):
    with pytest.raises(TypeError, match=name):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            **{name: weight},
        )


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_rejects_non_finite_or_negative_local_expansion_weight(weight):
    with pytest.raises(ValueError, match="local_expansion_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            local_expansion_weight=weight,
        )


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_rejects_non_finite_or_negative_paper_neighborhood_weight(weight):
    with pytest.raises(ValueError, match="paper_neighborhood_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_weight=weight,
        )


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_rejects_invalid_paper_neighborhood_two_hop_weight(weight):
    with pytest.raises(ValueError, match="paper_neighborhood_two_hop_weight"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_two_hop_weight=weight,
        )


@pytest.mark.parametrize(
    "name",
    [
        "method_owner_weight",
        "method_relation_weight",
        "method_topic_weight",
        "method_dense_tail_weight",
        "paper_dense_tail_weight",
    ],
)
@pytest.mark.parametrize(
    "weight",
    [-0.1, float("inf"), float("-inf"), float("nan")],
)
def test_rejects_invalid_method_relation_weights(name, weight):
    with pytest.raises(ValueError, match=name):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            **{name: weight},
        )


@pytest.mark.parametrize("degree", [True, 4.0, "4", None])
def test_rejects_non_integer_paper_neighborhood_max_hub_degree(degree):
    with pytest.raises(TypeError, match="paper_neighborhood_max_hub_degree"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_max_hub_degree=degree,
        )


@pytest.mark.parametrize("degree", [0, -1])
def test_rejects_non_positive_paper_neighborhood_max_hub_degree(degree):
    with pytest.raises(ValueError, match="paper_neighborhood_max_hub_degree"):
        SeedExpansionRetriever(
            _FakeRetriever([[]]),
            paper_neighborhood_max_hub_degree=degree,
        )


def test_nonpositive_top_k_returns_without_retrieval():
    inner = _FakeRetriever([[]])
    retriever = SeedExpansionRetriever(inner)

    assert retriever.retrieve("question", 0) == []
    assert inner.calls == []
