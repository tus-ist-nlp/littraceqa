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
from littraceqa.di_pipeline.select.owner_aware import (
    MethodOwnerIndex,
    OwnerAwarePaperSelector,
    explicitly_names_paper,
)
from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
)


def _method_owner_index(tmp_path, owners=None):
    path = tmp_path / "method_alias_graph.json"
    path.write_text(
        json.dumps({"schema_version": 3, "owners": owners or {}}),
        encoding="utf-8",
    )
    return path


def test_with_papers_preserves_evidence_diagnostics():
    selection = PaperSelection(
        ("old",),
        expected_count=1,
        reason="default",
        dropped_without_evidence=("dropped",),
    )

    refined = selection.with_papers(("new-1", "new-2"), "coverage")

    assert refined == PaperSelection(
        ("new-1", "new-2"),
        expected_count=2,
        reason="default+coverage",
        dropped_without_evidence=("dropped",),
    )


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
        (
            "In the real-world experiments of Fast-in-Slow, for which task "
            "do pi0 and FiS-VLA achieve the same performance?"
        ),
        (
            "What is the $AP^{kit}_{3D}$ performance of DetAny3D on Omni3D "
            "with ground-truth prompts and Cube R-CNN 2D detections?"
        ),
    ],
)
def test_single_paper_or_system_wording_defaults_to_one(question):
    # These questions identify one paper or system, not one paper per setting.
    assert expected_paper_count(question) == 1


@pytest.mark.parametrize(
    "question",
    [
        "What is the performance of TCM, sCT, and IMM?",
        (
            "In the experiments of TCM, sCT, and IMM, what score does each "
            "method report?"
        ),
    ],
)
def test_single_subject_wording_does_not_hide_multiple_subjects(question):
    assert expected_paper_count(question) == 3


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


def test_method_owner_matching_prefers_exact_spelling_and_requires_candidates():
    index = MethodOwnerIndex(
        {
            "MoST": "paper-most",
            "MosT": "paper-other",
            "AD-GS": "paper-adgs",
        }
    )

    assert index.owner_matches(
        "Compare MoST with ad-gs.",
        {"paper-most", "paper-other", "paper-adgs"},
    ) == (("MoST", "paper-most"), ("ad-gs", "paper-adgs"))
    assert index.owner_matches("Compare MOST.", {"paper-most", "paper-other"}) == ()
    assert index.owner_matches("Compare AD-GS.", {"paper-most"}) == ()


def test_owner_aware_selector_replaces_top_one_only_for_an_explicit_paper(tmp_path):
    selector = OwnerAwarePaperSelector(
        method_owner_index_path=_method_owner_index(
            tmp_path, {"EasySpec": "owner"}
        )
    )

    explicit = selector.select(
        "What is reported in the EasySpec paper?", ["rank-one", "owner"]
    )
    implicit = selector.select(
        "How does EasySpec improve accuracy?", ["rank-one", "owner"]
    )

    assert explicit.paper_ids == ("owner",)
    assert implicit.paper_ids == ("rank-one",)


def test_owner_index_can_be_built_after_the_selector(tmp_path):
    path = tmp_path / "method_alias_graph.json"
    selector = OwnerAwarePaperSelector(method_owner_index_path=path)
    path.write_text(
        json.dumps({"schema_version": 3, "owners": {"EasySpec": "owner"}}),
        encoding="utf-8",
    )

    selection = selector.select(
        "What is reported in the EasySpec paper?", ["wrong", "owner"]
    )

    assert selection.paper_ids == ("owner",)


def test_owner_aware_selector_fills_multi_paper_slots_and_protects_top_one(
    tmp_path,
):
    selector = OwnerAwarePaperSelector(
        method_owner_index_path=_method_owner_index(
            tmp_path,
            {"EasySpec": "easy", "AgentNet": "agent"},
        )
    )

    selection = selector.select(
        "For the EasySpec paper and the AgentNet paper, compare accuracy.",
        ["easy", "unrelated", "agent"],
    )

    assert selection.paper_ids == ("easy", "agent")
    assert selection.expected_count == 2


def test_distinct_method_owners_raise_the_minimum_paper_count(tmp_path):
    selector = OwnerAwarePaperSelector(
        method_owner_index_path=_method_owner_index(
            tmp_path,
            {"DiTFastAttnV2": "first", "DLFR-Gen": "second"},
        )
    )

    selection = selector.select(
        "These efficiency papers report DiTFastAttnV2 results alongside "
        "DLFR-Gen results.",
        ["first", "second", "other"],
    )

    assert selection.paper_ids == ("first", "second")
    assert selection.expected_count == 2
    assert selection.reason == "default+method_owner"


def test_owner_count_does_not_expand_a_singular_paper_question(tmp_path):
    selector = OwnerAwarePaperSelector(
        method_owner_index_path=_method_owner_index(
            tmp_path,
            {"EasySpec": "easy", "AgentNet": "agent"},
        )
    )

    selection = selector.select(
        "In the EasySpec paper, how does it compare with AgentNet?",
        ["easy", "agent", "other"],
    )

    assert selection.paper_ids == ("easy",)
    assert selection.expected_count == 1


def test_a_single_paper_reference_does_not_fill_unconfirmed_slots(tmp_path):
    selector = OwnerAwarePaperSelector(
        method_owner_index_path=_method_owner_index(
            tmp_path, {"Dobi-SVD": "dobi"}
        )
    )

    selection = selector.select(
        "In the Dobi-SVD paper, how many parameters does Dobi-SVD use for "
        "Llama-7B, and how many parameters does LLM-Pruner require?",
        ["dobi", "unrelated"],
    )

    assert selection.paper_ids == ("dobi",)
    assert selection.expected_count == 1
    assert selection.reason.endswith("+single_paper_guard")


def test_explicit_paper_phrase_does_not_cross_another_method_name():
    question = "Compare EasySpec with the AgentNet paper."

    assert not explicitly_names_paper(question, "EasySpec")
    assert explicitly_names_paper(question, "AgentNet")


# --- the shipped select_style configurations -------------------------------


SELECT_STYLES = sorted(Path("configs/select_style").glob("*.yaml"))


def test_every_shipped_select_style_builds(tmp_path):
    assert [path.stem for path in SELECT_STYLES] == [
        "f1_balanced",
        "f1_method_owner",
        "high_precision",
        "high_recall",
    ]
    owner_index = _method_owner_index(tmp_path)
    for path in SELECT_STYLES:
        assert (
            build_paper_selector(
                yaml.safe_load(path.read_text()),
                method_owner_index_path=str(owner_index),
            )
            is not None
        )


def test_cardinality_f1_baseline_does_not_require_a_method_index():
    spec = yaml.safe_load(Path("configs/select_style/f1_balanced.yaml").read_text())

    assert build_paper_selector(spec) is not None


@pytest.mark.parametrize(
    ("style", "open_set", "default", "stated_two"),
    [
        ("high_precision", 1, 1, 2),
        ("f1_balanced", 3, 1, 2),
        ("f1_method_owner", 1, 1, 2),
        ("high_recall", 5, 2, 3),
    ],
)
def test_shipped_styles_bracket_the_tradeoff(
    tmp_path, style, open_set, default, stated_two
):
    selector = build_paper_selector(
        yaml.safe_load(Path(f"configs/select_style/{style}.yaml").read_text()),
        method_owner_index_path=str(_method_owner_index(tmp_path)),
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
