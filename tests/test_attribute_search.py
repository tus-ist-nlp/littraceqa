"""Tests for deterministic soft reranking with structured search hints."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.attributes import (
    extract_literal_search_hints,
    extract_target_method_hints,
    rerank_by_attributes,
)


def _candidate(index: int, **metadata) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"paper-{index}#c0000",
        paper_id=f"paper-{index}",
        score=1.0 - index / 10,
        text=f"candidate {index}",
        chunk_type="text_span",
        metadata=metadata,
        source="rrf",
    )


def test_search_hints_accept_singular_fields_and_ignore_invalid_years():
    hints = SearchHints.from_dict(
        {"venue": "ACL 2024", "year": "2024", "method": "RAG"}
    )

    assert hints == SearchHints(venues=("ACL 2024",), years=(2024,), methods=("RAG",))
    assert hints.to_dict() == {
        "venues": ["ACL 2024"],
        "years": [2024],
        "methods": ["RAG"],
    }
    assert SearchHints.from_dict({"years": ["unknown"]}).is_empty


def test_empty_hints_preserve_exact_order_scores_and_objects():
    candidates = [_candidate(0, venue="ACL"), _candidate(1, venue="EMNLP")]

    results = rerank_by_attributes(candidates, SearchHints())

    assert results == candidates
    assert all(actual is original for actual, original in zip(results, candidates))


def test_matching_venue_and_year_are_softly_promoted_without_filtering():
    candidates = [
        _candidate(0, venue="EMNLP", year=2023),
        _candidate(1),
        _candidate(
            2,
            venue="62nd Annual Meeting of the Association for Computational Linguistics 2024",
            year="2024",
        ),
        _candidate(3, venue="ICLR", year=2024),
    ]

    results = rerank_by_attributes(
        candidates,
        SearchHints(venues=("ACL",), years=(2024,)),
        attribute_weight=1.0,
    )

    assert [result.paper_id for result in results] == [
        "paper-2",
        "paper-1",
        "paper-0",
        "paper-3",
    ]
    assert {result.paper_id for result in results} == {
        candidate.paper_id for candidate in candidates
    }
    matched = results[0]
    assert matched.metadata["attribute_matches"] == ["venue", "year"]
    assert matched.metadata["pre_attribute_rank"] == 3
    assert candidates[2].metadata == {
        "venue": "62nd Annual Meeting of the Association for Computational Linguistics 2024",
        "year": "2024",
    }


def test_method_names_support_lists_and_separator_normalized_exact_matching():
    candidates = [
        _candidate(0, method_names=["contrastive learning"]),
        _candidate(1, methods=["retrieval-augmented generation"]),
    ]

    results = rerank_by_attributes(
        candidates,
        SearchHints(methods=("retrieval augmented generation",)),
        attribute_weight=2.0,
    )

    assert [result.paper_id for result in results] == ["paper-1", "paper-0"]
    assert results[0].metadata["attribute_matches"] == ["method"]


@pytest.mark.parametrize(
    ("requested", "nonmatching"),
    [
        ("sCT", "SCT"),
        ("SCT", "sCT"),
        ("sCM", "SCM"),
        ("SVD", "Dobi-SVD"),
        (
            "retrieval augmented generation",
            "retrieval augmented generation model",
        ),
    ],
)
def test_method_matching_rejects_case_collisions_and_partial_aliases(
    requested,
    nonmatching,
):
    candidates = [
        _candidate(0, method_names=[nonmatching]),
        _candidate(1, method_names=[requested]),
    ]

    results = rerank_by_attributes(
        candidates,
        SearchHints(methods=(requested,)),
        attribute_weight=2.0,
    )

    by_id = {result.paper_id: result for result in results}
    assert results[0].paper_id == "paper-1"
    assert by_id["paper-0"].metadata["attribute_signal"] == 0.0
    assert by_id["paper-1"].metadata["attribute_matches"] == ["method"]


def test_method_hint_can_match_an_explicit_paper_title():
    candidates = [
        _candidate(0, title="A Generic Retrieval Baseline"),
        _candidate(1, title="D-FINE: Fine-grained Distribution Refinement"),
    ]

    results = rerank_by_attributes(
        candidates,
        SearchHints(methods=("D-FINE",)),
        attribute_weight=2.0,
    )

    assert [result.paper_id for result in results] == ["paper-1", "paper-0"]


def test_method_hint_is_positive_only_for_related_papers():
    candidates = [
        _candidate(0, method_names=["ECT"]),
        _candidate(1, method_names=["sCT"]),
        _candidate(2),
    ]

    results = rerank_by_attributes(
        candidates,
        SearchHints(methods=("sCT",)),
        attribute_weight=1.0,
    )

    by_id = {result.paper_id: result for result in results}
    assert results[0].paper_id == "paper-1"
    assert by_id["paper-0"].metadata["attribute_signal"] == 0.0
    assert by_id["paper-2"].metadata["attribute_signal"] == 0.0


def test_unknown_hint_falls_back_to_unchanged_normal_ranking():
    candidates = [_candidate(0, venue="ACL"), _candidate(1), _candidate(2, venue="ICLR")]

    results = rerank_by_attributes(candidates, SearchHints(venues=("SIGIR",)))

    assert results == candidates
    assert all(actual is original for actual, original in zip(results, candidates))


def test_equal_combined_scores_use_original_order_deterministically():
    candidates = [_candidate(0, venue="ACL"), _candidate(1, venue="ACL")]

    first = rerank_by_attributes(candidates, SearchHints(venues=("ACL",)))
    second = rerank_by_attributes(candidates, SearchHints(venues=("ACL",)))

    assert [result.paper_id for result in first] == ["paper-0", "paper-1"]
    assert [result.paper_id for result in second] == ["paper-0", "paper-1"]


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_invalid_attribute_weight_is_rejected(weight: float):
    with pytest.raises(ValueError, match="attribute_weight"):
        rerank_by_attributes([], SearchHints(), attribute_weight=weight)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Which CVPR 2025 papers cite UniAD "
            "(Planning-oriented Autonomous Driving, CVPR2023)?",
            SearchHints(venues=("CVPR",), years=(2025,)),
        ),
        (
            "By how much does a 2025 NeurIPS method improve the result?",
            SearchHints(venues=("NeurIPS",), years=(2025,)),
        ),
        (
            "Who is cited in a CVPR-2025 paper on human motion?",
            SearchHints(venues=("CVPR",), years=(2025,)),
        ),
        (
            "Compare ACL/2024 and EMNLP 2025 papers.",
            SearchHints(
                venues=("ACL", "EMNLP"),
                years=(2024, 2025),
            ),
        ),
    ],
)
def test_extracts_literal_venue_year_pairs(question, expected):
    assert extract_literal_search_hints(question) == expected


def test_paired_constraint_ignores_other_unpaired_venue_mentions():
    question = "Compare an ACL baseline with the target CVPR 2025 papers."

    assert extract_literal_search_hints(question) == SearchHints(
        venues=("CVPR",),
        years=(2025,),
    )


@pytest.mark.parametrize(
    "question",
    [
        "What COCO val2017 mAP does the model achieve?",
        "Compare train2017 and test2017 accuracy.",
        "Which method cites UniAD (Planning-oriented Autonomous Driving, CVPR2023)?",
        "Does a paper mention (CVPR 2025) as related work?",
        "What happened at CVPR2023?",
        "How does the model perform on AIME 2024?",
    ],
)
def test_does_not_infer_broad_or_parenthetical_year_constraints(question):
    assert extract_literal_search_hints(question).is_empty


def test_extracts_a_standalone_known_venue_without_a_year():
    assert extract_literal_search_hints(
        "Which EMNLP papers evaluate retrieval?"
    ) == SearchHints(venues=("EMNLP",))


def test_extracts_year_only_from_conservative_candidate_scope():
    assert extract_literal_search_hints(
        "Among the 2025 retrieval methods, which use reinforcement learning?"
    ) == SearchHints(years=(2025,))


def test_handles_nested_and_full_width_parentheses_conservatively():
    question = "Which ACL 2024 papers cite UniAD（CVPR 2023 (oral)）?"

    assert extract_literal_search_hints(question) == SearchHints(
        venues=("ACL",),
        years=(2024,),
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("In the TCM paper, what is the batch size?", ("TCM",)),
        (
            "What optimizer and learning rate does sCT use for training?",
            ("sCT",),
        ),
        (
            "What are the scores for TCM, sCT, ECM-XL (100k), and IMM?",
            ("TCM", "sCT", "ECM-XL", "IMM"),
        ),
        (
            "What accuracy does VTI achieve, and what does MoD achieve?",
            ("VTI", "MoD"),
        ),
        (
            "What mAP does Mr. DETR achieve?",
            ("Mr. DETR",),
        ),
        (
            "How many trainable parameters does LLM-Pruner require?",
            ("LLM-Pruner",),
        ),
        (
            "What mAP does DEIM-D-FINE-X achieve, and what mAP does "
            "Mr. DETR with Swin-L achieve?",
            ("DEIM-D-FINE-X", "Mr. DETR"),
        ),
    ],
)
def test_extracts_only_target_method_identifiers(question, expected):
    assert extract_target_method_hints(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Which papers cite UniAD (Planning-oriented Autonomous Driving)?",
        "Compare the proposed model with BERT.",
        "What score is reported for training on CIFAR-10?",
        "Which CVPR 2025 papers mention D-FINE?",
    ],
)
def test_target_method_extraction_rejects_baselines_and_generic_terms(question):
    assert extract_target_method_hints(question) == ()


def test_literal_method_hints_are_opt_in():
    question = "In the TCM paper, what is the batch size?"

    assert extract_literal_search_hints(question).methods == ()
    assert extract_literal_search_hints(
        question,
        include_methods=True,
    ).methods == ("TCM",)


@pytest.mark.parametrize("query", ["", "   ", None, 2025])
def test_empty_or_non_string_queries_produce_empty_literal_hints(query):
    assert extract_literal_search_hints(query).is_empty
