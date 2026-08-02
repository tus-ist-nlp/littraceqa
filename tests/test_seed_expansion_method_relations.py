"""Method ownership, method-to-method edges and the protected prefix."""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import (
    Chunk,
    SearchHints,
)
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakePaperIndex,
    _FakeReranker,
    _FakeRetriever,
    _result,
)


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
