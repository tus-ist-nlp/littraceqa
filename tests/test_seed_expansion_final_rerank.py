"""Final candidate reranking, its fixed pool and its fail-soft fallbacks."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Chunk, SearchHints
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.final_rerank import (
    FinalCandidateReranker,
)
from seed_expansion_doubles import (
    _FakePaperEmbeddingStore,
    _FakePaperIndex,
    _FakeReranker,
    _FakeRetriever,
    _RecordingFinalReranker,
    _result,
)


def test_final_candidate_rerank_runs_once_after_the_dense_tail():
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
        "tail-second",
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
            _result("tail-second", score=0.9),
        ],
    }

    def retrieve(reranker=None):
        paper_index = _FakePaperIndex(
            documents,
            owners=({"paper_id": "p1", "alias": "MethodX"},),
        )
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
            method_dense_tail_weight=4.0,
            method_dense_tail_seed_k=1,
            method_dense_tail_max_results=2,
            method_dense_tail_max_new_papers=2,
            rerank_final_candidates=reranker is not None,
            rerank_pool_k=20,
            final_rerank_document_chars=12,
        )
        retriever._paper_embedding_store = dense_store
        return retriever.retrieve(
            "question",
            20,
            hints=SearchHints(methods=("MethodX",)),
        )

    baseline = retrieve()
    final_reranker = _RecordingFinalReranker()
    results = retrieve(final_reranker)

    baseline_ids = [result.paper_id for result in baseline]
    assert {"tail-new", "tail-second"} <= set(baseline_ids)
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


def test_final_rerank_wider_pool_protects_requested_output_set():
    candidates = [_result(f"p{index}") for index in range(1, 5)]
    documents = {
        candidate.paper_id: Chunk(
            chunk_id=f"{candidate.paper_id}#paper",
            paper_id=candidate.paper_id,
            text=f"Paper document for {candidate.paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for candidate in candidates
    }
    final_reranker = _RecordingFinalReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[_FakePaperIndex(documents)],
        ),
        reranker=final_reranker,
        max_results=2,
        rerank_pool_k=4,
        rerank_final_candidates=True,
        final_rerank_protected_top_k=2,
    )

    results = retriever.retrieve("question", 2)

    assert len(final_reranker.calls) == 1
    assert len(final_reranker.calls[0][1]) == 4
    assert [result.paper_id for result in results] == ["p2", "p1"]


def test_final_candidate_rerank_can_protect_original_top_k_set():
    candidates = [_result(f"p{index}") for index in range(1, 5)]
    candidates[0].metadata["final_rerank_pre_protection_rank"] = 999
    documents = {
        candidate.paper_id: Chunk(
            chunk_id=f"{candidate.paper_id}#paper",
            paper_id=candidate.paper_id,
            text=f"Paper document for {candidate.paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for candidate in candidates
    }
    reranker = _RecordingFinalReranker()
    stage = FinalCandidateReranker(
        reranker=reranker,
        document_chars=100,
        protected_top_k=2,
    )

    results = stage.rerank(
        "question",
        candidates,
        [_FakePaperIndex(documents)],
    )

    assert [result.paper_id for result in results] == [
        "p2",
        "p1",
        "p4",
        "p3",
    ]
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )
    assert results[0].metadata["final_rerank_protected_top_k"] == 2
    assert results[0].metadata["final_rerank_prefix_protected"] is True
    assert results[2].metadata["final_rerank_prefix_protected"] is False
    by_id = {result.paper_id: result for result in results}
    assert by_id["p1"].metadata["final_rerank_pre_protection_rank"] == 4
    assert by_id["p1"].metadata["final_rerank_pre_protection_score"] == 1.0
    assert by_id["p2"].metadata["final_rerank_pre_protection_score"] == 2.0
    assert by_id["p4"].metadata["final_rerank_pre_protection_score"] == 4.0


def test_final_candidate_rerank_uses_bounded_paper_head():
    candidate = _result("p1", text="Query-matched passage")
    document = Chunk(
        chunk_id="p1#paper",
        paper_id="p1",
        text="Full paper document text",
        chunk_type="paper",
        metadata={},
    )
    reranker = _RecordingFinalReranker()
    stage = FinalCandidateReranker(
        reranker=reranker,
        document_chars=12,
    )

    stage.rerank(
        "question",
        [candidate],
        [_FakePaperIndex({"p1": document})],
    )

    assert reranker.calls[0][1][0].text == document.text[:12]


def test_final_candidate_rerank_exception_falls_back_to_original_ranking():
    candidates = [
        _result(f"p{index}", text=f"Original text {index}")
        for index in range(1, 5)
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
        max_results=2,
    ).retrieve("question", 2)
    final_reranker = _RecordingFinalReranker(
        error=RuntimeError("inference failed")
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=final_reranker,
        max_results=2,
        rerank_pool_k=4,
        rerank_final_candidates=True,
    )

    results = retriever.retrieve("question", 2)

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
    assert len(final_reranker.calls[0][1]) == 4


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
