"""Reciprocal dense exploration of the final slot."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakePaperEmbeddingStore,
    _FakePaperIndex,
    _FakeRetriever,
    _result,
)


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
