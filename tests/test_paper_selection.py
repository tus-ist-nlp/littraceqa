"""Paper-set selection: the stated count, the cut, and the official arithmetic."""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.evaluation.paper_selection import (
    load_gold_paper_sets,
    prf,
    score_selection,
)
from littraceqa.di_pipeline.select.cardinality import (
    expected_paper_count,
    is_open_ended_enumeration,
)
from littraceqa.di_pipeline.select.selector import CardinalityPaperSelector


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("For the two ICCV 2025 papers, compare their inference speed.", 2),
        ("For these two detection papers, report the speedups.", 2),
        ("Two NAACL 2025 generalization studies each report a metric.", 2),
        ("Across the three ICML 2025 works, what batch size is used?", 3),
        (
            "What are the 2-step FID scores on CIFAR-10 for TCM, sCT, "
            "ECM-XL (100k iterations), and IMM?",
            4,
        ),
        (
            "What are the 1-step FID scores for TCM, ECM-XL "
            "(with 102.4M training budget), iCT-deep, and SiD?",
            4,
        ),
        (
            "Consider the encoder-free VLM paper that introduces EVEv2.0 and "
            "the event-understanding paper that introduces SymbolicDet.",
            2,
        ),
    ],
)
def test_reads_the_count_a_question_states(question, expected):
    assert expected_paper_count(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "How many subfigures are there in Figure 4 of the DynaPipe paper?",
        "How many parentheses does equation 6 of the autoencoder paper have?",
        "In the TCM paper, does the denoising FID at t=0.2 exceed 3.0?",
        "What optimizer does sCT use for training on CIFAR-10?",
        "In Table 3 of the IMM paper, what 2-step FID is reported?",
    ],
)
def test_a_number_belonging_to_a_figure_is_not_a_paper_count(question):
    # "Figure 4 of the ... paper" names one paper, not four.
    assert expected_paper_count(question) == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which CVPR 2025 papers cite UniAD as a baseline?", True),
        (
            "Among ICML 2025 papers that propose an objective, what does "
            "each method optimize?",
            True,
        ),
        ("In the TCM paper, what batch size is used?", False),
        ("For the two ICCV 2025 papers, compare their speed.", False),
    ],
)
def test_detects_open_ended_enumeration(question, expected):
    assert is_open_ended_enumeration(question) is expected


def test_default_is_one_paper_and_bounded_by_max():
    selector = CardinalityPaperSelector()
    selection = selector.select("What batch size does TCM use?", ["a", "b", "c"])

    assert selection.paper_ids == ("a",)
    assert selection.expected_count == 1
    assert selection.reason == "default"


def test_stated_count_cuts_the_ranking_there():
    selector = CardinalityPaperSelector()
    selection = selector.select(
        "For the two ICCV 2025 papers, compare their speed.",
        ["a", "b", "c", "d"],
    )

    assert selection.paper_ids == ("a", "b")
    assert selection.reason == "stated_in_question"


def test_open_set_count_applies_only_to_open_ended_enumerations():
    selector = CardinalityPaperSelector(open_set_count=5)
    candidates = ["a", "b", "c", "d", "e", "f"]

    enumeration = selector.select("Which CVPR 2025 papers cite UniAD?", candidates)
    assert enumeration.paper_ids == ("a", "b", "c", "d", "e")
    assert enumeration.reason == "open_set_enumeration"

    plain = selector.select("What batch size does TCM use?", candidates)
    assert plain.paper_ids == ("a",)


def test_selection_deduplicates_and_ignores_invalid_ids():
    selector = CardinalityPaperSelector()
    selection = selector.select(
        "For the three ICML 2025 papers, compare their speed.",
        ["a", "a", "", None, "b", "c", "d"],
    )

    assert selection.paper_ids == ("a", "b", "c")


def test_max_papers_bounds_a_larger_stated_count():
    selector = CardinalityPaperSelector(max_papers=2)
    selection = selector.select(
        "Across the six ICML 2025 papers, what batch size is used?",
        ["a", "b", "c", "d", "e", "f"],
    )

    assert selection.paper_ids == ("a", "b")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_count": 0},
        {"open_set_count": 0},
        {"max_papers": 0},
        {"default_count": True},
        {"default_count": 5, "max_papers": 2},
        {"open_set_count": 5, "max_papers": 2},
    ],
)
def test_rejects_invalid_selector_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        CardinalityPaperSelector(**kwargs)


def test_prf_matches_the_official_edge_cases():
    assert prf(set(), set()) == (1.0, 1.0, 1.0)
    assert prf({"a"}, set()) == (0.0, 0.0, 0.0)
    assert prf(set(), {"a"}) == (0.0, 1.0, 0.0)
    precision, recall, f1 = prf({"a", "b"}, {"a", "c"})
    assert (precision, recall) == (0.5, 0.5)
    assert f1 == pytest.approx(0.5)


def test_submitting_twenty_papers_for_one_gold_paper_scores_badly():
    # The reason a selector exists: rank quality does not survive a wide
    # submission. Twenty papers containing the single gold paper score 0.095.
    gold = {"q": {"a"}}
    selected = {"q": ["a"] + [f"p{index}" for index in range(19)]}

    metrics = score_selection(gold, selected)

    assert metrics.paper_recall_macro == pytest.approx(1.0)
    assert metrics.paper_f1_macro == pytest.approx(2 / 21)


def test_score_selection_treats_a_missing_query_as_an_empty_submission():
    metrics = score_selection({"a": {"p1"}, "b": {"p2"}}, {"a": ["p1"]})

    assert metrics.query_count == 2
    assert metrics.paper_f1_macro == pytest.approx(0.5)


def test_score_selection_rejects_empty_gold():
    with pytest.raises(ValueError, match="gold_by_query"):
        score_selection({}, {})


def test_load_gold_paper_sets_reads_both_id_shapes(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "gold_papers": [{"paper_id": "a"}]})
        + "\n"
        + json.dumps({"query_id": "q2", "gold_papers": ["b", "c"]})
        + "\n",
        encoding="utf-8",
    )

    assert load_gold_paper_sets(path) == {"q1": {"a"}, "q2": {"b", "c"}}


def test_load_gold_paper_sets_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no queries"):
        load_gold_paper_sets(path)
