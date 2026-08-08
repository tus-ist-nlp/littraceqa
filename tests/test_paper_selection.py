"""Paper-set selection: the stated count, the cut, and the official arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from littraceqa.di_pipeline.evaluation.submission_scoring import (
    load_gold_paper_sets,
    prf,
    score_selection,
)
from littraceqa.di_pipeline.select.cardinality import (
    expected_paper_count,
    is_open_ended_enumeration,
)
from littraceqa.di_pipeline.select import build_paper_selector
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


def test_stated_count_margin_widens_only_a_stated_count():
    selector = CardinalityPaperSelector(default_count=2, stated_count_margin=1)
    candidates = ["a", "b", "c", "d", "e"]

    stated = selector.select(
        "For the two ICCV 2025 papers, compare their speed.", candidates
    )
    assert stated.paper_ids == ("a", "b", "c")

    unstated = selector.select("What batch size does TCM use?", candidates)
    assert unstated.paper_ids == ("a", "b")


def test_require_evidence_drops_unsupported_candidates():
    selector = CardinalityPaperSelector(
        default_count=2, require_evidence=True, max_papers=10
    )

    selection = selector.select(
        "What batch size does TCM use?",
        ["a", "b", "c"],
        evidence_paper_ids={"b", "c"},
    )

    assert selection.paper_ids == ("b", "c")
    assert selection.dropped_without_evidence == ("a",)
    assert selection.reason.endswith("+evidence")


def test_require_evidence_is_inert_without_evidence_information():
    selector = CardinalityPaperSelector(default_count=2, require_evidence=True)

    selection = selector.select("What batch size does TCM use?", ["a", "b", "c"])

    assert selection.paper_ids == ("a", "b")
    assert selection.dropped_without_evidence == ()


def test_evidence_filter_never_empties_the_submission():
    # An empty submission scores zero on both precision and recall, so the
    # raw ranking is kept when nothing could be verified.
    selector = CardinalityPaperSelector(require_evidence=True)

    selection = selector.select(
        "What batch size does TCM use?", ["a", "b"], evidence_paper_ids=set()
    )

    assert selection.paper_ids == ("a",)


@pytest.mark.parametrize("margin", [-1, True, 1.5])
def test_rejects_invalid_stated_count_margin(margin):
    with pytest.raises((TypeError, ValueError)):
        CardinalityPaperSelector(stated_count_margin=margin)


def test_rejects_non_boolean_require_evidence():
    with pytest.raises(TypeError, match="require_evidence"):
        CardinalityPaperSelector(require_evidence="yes")


# --- the shipped select_style configurations -------------------------------


SELECT_STYLES = sorted(Path("configs/select_style").glob("*.yaml"))


def test_every_shipped_select_style_builds():
    assert [path.stem for path in SELECT_STYLES] == [
        "f1_balanced",
        "high_precision",
        "high_recall",
    ]
    for path in SELECT_STYLES:
        assert build_paper_selector(yaml.safe_load(path.read_text())) is not None


@pytest.mark.parametrize(
    ("style", "open_set", "default", "stated_two"),
    [
        ("high_precision", 1, 1, 2),
        ("f1_balanced", 3, 1, 2),
        ("high_recall", 5, 2, 3),
    ],
)
def test_shipped_styles_bracket_the_tradeoff(style, open_set, default, stated_two):
    selector = build_paper_selector(
        yaml.safe_load(Path(f"configs/select_style/{style}.yaml").read_text())
    )

    assert selector.expected_count("Which CVPR 2025 papers cite UniAD?")[0] == open_set
    assert selector.expected_count("What batch size does TCM use?")[0] == default
    assert (
        selector.expected_count("For the two ICCV 2025 papers, compare speed.")[0]
        == stated_two
    )


def test_build_paper_selector_passes_through_none():
    assert build_paper_selector(None) is None


@pytest.mark.parametrize("spec", [{"name": "nope"}, {}, {"name": None}])
def test_build_paper_selector_rejects_unknown_names(spec):
    with pytest.raises(ValueError, match="unknown paper selector"):
        build_paper_selector(spec)


def test_build_paper_selector_rejects_a_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        build_paper_selector("cardinality")
