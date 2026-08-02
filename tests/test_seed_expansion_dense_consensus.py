"""Consensus dense exploration of the final slot."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import (
    Chunk,
    RetrievalResult,
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
