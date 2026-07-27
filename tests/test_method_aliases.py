"""Tests for conservative, gold-free method alias extraction."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.retrieve.method_aliases import (
    MethodAliasEvidence,
    extract_self_owned_method_aliases,
    has_standalone_exact_alias,
    method_aliases_equal,
    standalone_exact_alias_positions,
    text_before_references,
)


def _aliases(title: str, text: str, *, max_aliases: int = 6) -> list[str]:
    return [
        item.alias
        for item in extract_self_owned_method_aliases(
            title,
            text,
            max_aliases=max_aliases,
        )
    ]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "D-FINE: Redefine Regression Task as Distribution Refinement",
            ["D-FINE"],
        ),
        ("YOLOv12: Attention-Centric Object Detectors", ["YOLOv12"]),
        ("Mr. DETR: Instructive Multi-Route Training", ["Mr. DETR"]),
        ("RAG: Retrieval-Augmented Generation", []),
        ("Mixture of Decoding: Adaptive Decoding", []),
        ("An Ordinary Title Without a Colon", []),
    ],
)
def test_extracts_only_distinctive_title_prefixes(title, expected):
    assert _aliases(title, "") == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("D-FINE", "D FINE"),
        ("D.FINE", "d fine"),
        ("SCT", "sct"),
        ("ＦＩＮＥ", "fine"),
    ],
)
def test_method_alias_equality_normalizes_separators(left, right):
    assert method_aliases_equal(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("sCT", "SCT"),
        ("sCM", "SCM"),
        ("Dobi-SVD", "SVD"),
        ("retrieval augmented generation", "retrieval augmented generation model"),
    ],
)
def test_method_alias_equality_is_case_safe_and_never_partial(left, right):
    assert not method_aliases_equal(left, right)


def test_extracts_proposed_parenthetical_method_with_trace():
    text = (
        "To address hallucinations, we propose Mixture of Decoding (MoD), "
        "a new adaptive decoding strategy."
    )

    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == ["MoD"]
    assert evidence[0].source == "parenthetical_definition"
    assert evidence[0].long_name == "Mixture of Decoding"
    assert text[evidence[0].start : evidence[0].end] == "MoD"
    assert evidence[0].to_dict() == {
        "alias": "MoD",
        "source": "parenthetical_definition",
        "start": evidence[0].start,
        "end": evidence[0].end,
        "long_name": "Mixture of Decoding",
    }


@pytest.mark.parametrize(
    ("text", "expected_alias", "expected_long_name"),
    [
        (
            "In this paper, we propose a new training algorithm, termed "
            "Truncated Consistency Models (TCM).",
            "TCM",
            "Truncated Consistency Models",
        ),
        (
            "To close this gap, we introduce a method refered to as "
            "Pay Attention to Image (PAI).",
            "PAI",
            "Pay Attention to Image",
        ),
        (
            "We propose a simple method named "
            "$\\textit{Self-Introspective Decoding}$ (SID).",
            "SID",
            "Self-Introspective Decoding",
        ),
    ],
)
def test_handles_realistic_self_definition_forms(
    text,
    expected_alias,
    expected_long_name,
):
    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == [expected_alias]
    assert evidence[0].long_name == expected_long_name


def test_scans_late_body_for_referred_to_as_definition():
    text = (
        ("Background details without an alias. " * 600)
        + "\nTo test our improvements, we employ both consistency training "
        "(referred to as sCT) and consistency distillation "
        "(referred to as sCD)."
    )

    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == ["sCT", "sCD"]
    assert all(item.start > 10_000 for item in evidence)
    assert evidence[0].long_name == "consistency training"
    assert evidence[1].long_name == "consistency distillation"


def test_extracts_direct_owned_naming_statements():
    text = (
        "Our proposed method, PAI, reduces hallucinations. "
        "We call this framework SecEmb. "
        "We introduce D-FINE for accurate localization."
    )

    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == ["PAI", "SecEmb", "D-FINE"]
    assert all(item.source == "naming_statement" for item in evidence)


def test_extracts_explicit_secondary_name_for_an_owned_method():
    text = (
        "We introduce Easy Consistency Tuning (ECT) for efficient training. "
        "We compare models trained with ECT (denoted as ECM) against baselines."
    )

    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == ["ECT", "ECM"]
    assert evidence[1].source == "derived_naming_statement"


def test_extracts_method_marked_as_ours():
    text = (
        "The main comparison reports DPO, SimPO, and D2PO (ours). "
        "Later rows repeat D2PO (ours) with other base models."
    )

    evidence = extract_self_owned_method_aliases("A Paper", text)

    assert [item.alias for item in evidence] == ["D2PO"]
    assert evidence[0].source == "ours_label"
    assert text[evidence[0].start : evidence[0].end] == "D2PO"


def test_rejects_generic_method_marked_as_ours():
    assert _aliases("A Paper", "The table compares DPO (ours).") == []


def test_does_not_claim_secondary_name_of_an_unowned_baseline():
    text = (
        "We compare models trained with ECT (denoted as ECM) against "
        "our proposed framework, NEWM."
    )

    assert _aliases("A Paper", text) == ["NEWM"]


@pytest.mark.parametrize(
    "text",
    [
        "We propose a model based on BERT (BERT).",
        "We introduce a metric called FID.",
        "Our proposed framework, LLM, uses retrieval.",
        "We call this model COCO.",
        "We introduce CVPR for this example.",
        "We propose a detector named DETR.",
        "We introduce a training-free algorithm.",
        "We propose a two-stage training procedure.",
    ],
)
def test_rejects_generic_concepts_metrics_models_and_venues(text):
    assert _aliases("A Paper", text) == []


def test_does_not_claim_baseline_parenthetical_abbreviations():
    text = (
        "We use Random Forest (RF) and compare against Easy Consistency "
        "Tuning (ECT). Neither method is introduced by this paper."
    )

    assert _aliases("A Paper", text) == []


def test_references_heading_stops_alias_extraction():
    text = (
        "We discuss retrieval without naming a new method.\n"
        "8. References\n"
        "Smith et al. propose Mixture of Decoding (MoD).\n"
        "Jones et al. call their method SecEmb."
    )

    assert _aliases("A Paper", text) == []
    assert "MoD" not in text_before_references(text)


def test_markdown_bibliography_heading_stops_alias_extraction():
    text = (
        "We introduce D-FINE for localization.\n"
        "## Bibliography\n"
        "We propose Mixture of Decoding (MoD)."
    )

    assert _aliases("A Paper", text) == ["D-FINE"]


def test_prose_word_references_does_not_truncate_body():
    text = (
        "This paragraph references earlier work. "
        "We propose Mixture of Decoding (MoD)."
    )

    assert _aliases("A Paper", text) == ["MoD"]


def test_deduplicates_evidence_deterministically_and_obeys_limit():
    text = (
        "We propose AlphaNet (ANET). "
        "We call this method ANET. "
        "We propose BetaNet (BNET). "
        "We propose GammaNet (GNET)."
    )

    first = extract_self_owned_method_aliases(
        "D-FINE: A Detector",
        text,
        max_aliases=3,
    )
    second = extract_self_owned_method_aliases(
        "D-FINE: A Detector",
        text,
        max_aliases=3,
    )

    assert [item.alias for item in first] == ["D-FINE", "ANET", "BNET"]
    assert first == second


@pytest.mark.parametrize("max_aliases", [0, 7, -1])
def test_rejects_out_of_range_alias_limits(max_aliases):
    with pytest.raises(ValueError, match="max_aliases"):
        extract_self_owned_method_aliases("Title", "", max_aliases=max_aliases)


@pytest.mark.parametrize("max_aliases", [True, 1.5, "2"])
def test_rejects_non_integer_alias_limits(max_aliases):
    with pytest.raises(TypeError, match="max_aliases"):
        extract_self_owned_method_aliases("Title", "", max_aliases=max_aliases)


def test_malformed_title_and_text_fail_closed():
    assert extract_self_owned_method_aliases(None, None) == ()
    assert text_before_references(None) == ""


def test_standalone_mentions_are_exact_case_and_boundary_aware():
    text = "sCT improves results; SCT differs; xsCT and sCT2 are other tokens. sCT."

    positions = standalone_exact_alias_positions(text, "sCT")

    assert positions == (0, text.rfind("sCT"))
    assert has_standalone_exact_alias(text, "sCT")
    assert not has_standalone_exact_alias(text, "Sct")


def test_standalone_mentions_normalize_width_and_whitespace():
    text = "We compare Ｍｒ．   ＤＥＴＲ with another detector."

    positions = standalone_exact_alias_positions(text, "Mr. DETR")

    assert len(positions) == 1


def test_standalone_mentions_exclude_references_by_default():
    text = "No method is named here.\nReferences\nA citation to MoD."

    assert standalone_exact_alias_positions(text, "MoD") == ()
    assert standalone_exact_alias_positions(
        text,
        "MoD",
        exclude_references=False,
    )


def test_standalone_mentions_accept_non_generic_aliases_without_ownership_guessing():
    text = "The comparison includes D-FINE and MoD."

    assert has_standalone_exact_alias(text, "D-FINE")
    assert has_standalone_exact_alias(text, "MoD")


def test_empty_or_non_string_mention_inputs_fail_closed():
    assert standalone_exact_alias_positions(None, "MoD") == ()
    assert standalone_exact_alias_positions("MoD", None) == ()
    assert standalone_exact_alias_positions("MoD", "") == ()


def test_rejects_non_boolean_reference_option():
    with pytest.raises(TypeError, match="exclude_references"):
        standalone_exact_alias_positions(
            "MoD",
            "MoD",
            exclude_references=1,
        )


def test_evidence_record_is_frozen():
    evidence = MethodAliasEvidence(
        alias="MoD",
        source="naming_statement",
        start=0,
        end=3,
    )

    with pytest.raises((AttributeError, TypeError)):
        evidence.alias = "changed"
