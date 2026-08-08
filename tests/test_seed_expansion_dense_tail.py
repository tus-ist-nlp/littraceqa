"""Dense tail fusion behind the stable prefix, for both seed lanes."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import (
    Chunk,
    RetrievalResult,
    SearchHints,
)
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakePaperEmbeddingStore,
    _FakePaperIndex,
    _FakeRetriever,
    _result,
)


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
    paper_index = _FakePaperIndex({})

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


