"""Constructor validation and the interface the wrapper exposes."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakeReranker,
    _FakeRetriever,
)


def test_rejects_wrapping_retriever_with_reranker():
    with pytest.raises(ValueError, match="reranker twice"):
        SeedExpansionRetriever(_FakeRetriever([[]], reranker=object()))


def test_proxies_indexers_and_exposes_final_reranker():
    inner = _FakeRetriever([[]])
    final_reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(inner, reranker=final_reranker)

    assert retriever.indexers is inner.indexers
    assert retriever.reranker is final_reranker
    assert retriever.structured_filter is False
    assert retriever.exact_method_search is False
    assert retriever.method_dense_tail_weight == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_k": 0}, "candidate_k"),
        ({"seed_text_chars": 0}, "seed_text_chars"),
        ({"rrf_k": -1}, "rrf_k"),
        ({"max_results": 0}, "max_results"),
        ({"stable_prefix_k": 0}, "stable_prefix_k"),
        ({"open_set_seed_k": 0}, "open_set_seed_k"),
        ({"open_set_seed_k": 9}, "open_set_seed_k"),
        ({"open_set_min_support": 1}, "open_set_min_support"),
        ({"open_set_max_seed_rank": 0}, "open_set_max_seed_rank"),
        (
            {"candidate_k": 5, "open_set_max_seed_rank": 6},
            "open_set_max_seed_rank",
        ),
        ({"open_set_slot_k": 0}, "open_set_slot_k"),
        (
            {
                "max_results": 10,
                "open_set_seed_k": 5,
                "open_set_slot_k": 20,
            },
            "open_set_slot_k",
        ),
        (
            {"open_set_seed_k": 2, "open_set_min_support": 2},
            "open_set_min_support",
        ),
        ({"rerank_pool_k": 0}, "rerank_pool_k"),
        (
            {"final_rerank_document_chars": 0},
            "final_rerank_document_chars",
        ),
        (
            {"final_rerank_protected_top_k": -1},
            "final_rerank_protected_top_k",
        ),
        (
            {"rerank_final_candidates": True},
            "rerank_final_candidates",
        ),
        ({"max_protected_titles": 0}, "max_protected_titles"),
        ({"max_protected_titles": 5}, "max_protected_titles"),
        ({"method_dense_tail_seed_k": 0}, "method_dense_tail_seed_k"),
        (
            {"method_dense_tail_max_results": 0},
            "method_dense_tail_max_results",
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
        )    ],
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
        ({"open_set_seed_k": True}, "open_set_seed_k"),
        ({"open_set_min_support": 2.5}, "open_set_min_support"),
        ({"open_set_max_seed_rank": "2"}, "open_set_max_seed_rank"),
        ({"open_set_slot_k": None}, "open_set_slot_k"),
        ({"rerank_final_candidates": 1}, "rerank_final_candidates"),
        ({"two_lane_rerank": 1}, "two_lane_rerank"),
        (
            {"final_rerank_document_chars": True},
            "final_rerank_document_chars",
        ),
        (
            {"final_rerank_document_chars": 2.5},
            "final_rerank_document_chars",
        ),
        (
            {"final_rerank_protected_top_k": True},
            "final_rerank_protected_top_k",
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
