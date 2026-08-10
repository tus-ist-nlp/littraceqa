from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import (
    PaperEvidenceDocument,
    PaperTable,
)
from littraceqa.di_pipeline.select.multi_paper_coverage import (
    MultiPaperCoverageRefiner,
)
from littraceqa.di_pipeline.select.two_slot_question import parse_two_slot_question
from littraceqa.di_pipeline.select.selector import PaperSelection
from littraceqa.di_pipeline.select.evidence_coverage import EvidenceCoverageRefiner


class StubEvidenceSource:
    def __init__(self, documents: dict[str, PaperEvidenceDocument]) -> None:
        self.documents = documents

    def document(self, paper_id: str) -> PaperEvidenceDocument:
        return self.documents.get(
            paper_id,
            PaperEvidenceDocument(paper_id, "", (), ()),
        )

    def tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        return self.document(paper_id).tables


def _table(paper_id: str, text: str) -> PaperTable:
    row = tuple(part.strip() for part in text.split("|"))
    return PaperTable(
        paper_id=paper_id,
        table_id="Table 1",
        caption="",
        rows=(row,),
        text=text,
    )


def _document(
    paper_id: str,
    *,
    title: str,
    body: str,
    tables: tuple[PaperTable, ...] = (),
) -> PaperEvidenceDocument:
    return PaperEvidenceDocument(
        paper_id=paper_id,
        title=title,
        text_blocks=tuple(body.split("\n")),
        tables=tables,
    )


def _query(question: str, answer_types: list[str] | None = None) -> Query:
    return Query(
        query_id="q",
        question=question,
        answer_types=answer_types if answer_types is not None else ["multiple_choice"],
    )


def _selection(paper_id: str = "first") -> PaperSelection:
    return PaperSelection((paper_id,), expected_count=1, reason="default")


def test_parses_paired_achievement_slots():
    slots = parse_two_slot_question(
        "What COCO val2017 mAP does DEIM-D-FINE-X achieve in 50 epochs, "
        "and what mAP does Mr. DETR with Swin-L achieve in 12 epochs?"
    )

    assert slots is not None
    assert [slot.target for slot in slots] == ["DEIM-D-FINE-X", "Mr. DETR"]
    assert "50" in slots[0].terms
    assert "swin" in slots[1].terms


def test_adds_two_uniquely_owned_table_reporters_in_rank_order():
    question = (
        "What COCO val2017 mAP does DEIM-D-FINE-X achieve in 50 epochs, "
        "and what mAP does Mr. DETR with Swin-L achieve in 12 epochs?"
    )
    deim = _document(
        "deim",
        title="DEIM: DETR with Improved Matching",
        body="We introduce DEIM and build our DEIM-D-FINE-X.",
        tables=(
            _table(
                "deim",
                "COCO val2017 AP | DEIM-D-FINE-X | Epochs | 50 | 54.7",
            ),
        ),
    )
    mr_detr = _document(
        "mr-detr",
        title="Mr. DETR: Instructive Multi-Route Training",
        body="We introduce Mr. DETR.",
        tables=(
            _table(
                "mr-detr",
                "COCO val2017 AP | Mr. DETR | Swin-L | Epochs | 12 | 58.4",
            ),
        ),
    )

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"deim": deim, "mr-detr": mr_detr})
    ).refine(_query(question), ["mr-detr", "deim"], _selection("mr-detr"))

    assert result.paper_ids == ("mr-detr", "deim")
    assert result.expected_count == 2
    assert result.reason == "default+multi_paper_coverage"


def test_adds_two_uniquely_owned_pope_table_reporters():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    header = ("Setting", "Method", "LLaVA-1.5", "Accuracy", "F1")
    vti = _document(
        "vti",
        title="Latent Space Steering",
        body="We introduce VTI.",
        tables=(
            PaperTable(
                "vti",
                "Table 4",
                "POPE results",
                (header, ("Adversarial", "VTI", "LLaVA-1.5", "82.5", "82.1")),
                "POPE results",
            ),
        ),
    )
    mod = _document(
        "mod",
        title="Mixture of Decoding",
        body="We call our method MoD.",
        tables=(
            PaperTable(
                "mod",
                "Table 1",
                "POPE results",
                (header, ("Adversarial", "MoD", "LLaVA-1.5", "79.7", "81.3")),
                "POPE results",
            ),
        ),
    )

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"vti": vti, "mod": mod})
    ).refine(_query(question), ["vti", "mod"], _selection("vti"))

    assert result.paper_ids == ("vti", "mod")


def test_adds_two_text_reporters_for_a_coordinated_use_question():
    question = (
        "What VAE latent channel mean normalization values do sCM and IMM use "
        "for their ImageNet experiments, and do they match?"
    )
    scm = _document(
        "scm",
        title="Scaling Consistency Models",
        body=(
            "We introduce sCM.\nImageNet preprocessing.\n"
            "Stable Diffusion VAE latents use channel mean [1.5, -0.6, 0.4, 0.7] "
            "for normalization in these experiments."
        ),
    )
    imm = _document(
        "imm",
        title="Inductive Moment Matching",
        body=(
            "We propose IMM.\nFor ImageNet, VAE latent normalization uses "
            "channel mean [0.8, -0.2, 0.2, 0.3] in these experiments."
        ),
    )

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"scm": scm, "imm": imm})
    ).refine(_query(question), ["imm", "scm"], _selection("imm"))

    assert result.paper_ids == ("imm", "scm")


def test_combined_evidence_refiner_runs_multi_paper_coverage_last():
    question = (
        "What VAE latent channel mean normalization values do sCM and IMM use "
        "for their ImageNet experiments, and do they match?"
    )
    scm = _document(
        "scm",
        title="Scaling Consistency Models",
        body=(
            "We introduce sCM.\nImageNet preprocessing.\n"
            "VAE latent normalization uses channel mean [1.5, -0.6, 0.4, 0.7] "
            "in these experiments."
        ),
    )
    imm = _document(
        "imm",
        title="Inductive Moment Matching",
        body=(
            "We propose IMM.\nFor ImageNet, VAE latent normalization uses "
            "channel mean [0.8, -0.2, 0.2, 0.3] in these experiments."
        ),
    )
    source = StubEvidenceSource({"scm": scm, "imm": imm})

    result = EvidenceCoverageRefiner(source, evidence_source=source).refine(
        _query(question), ["imm", "scm"], _selection("imm")
    )

    assert result.paper_ids == ("imm", "scm")


def test_combined_refiner_keeps_the_positional_candidate_limit_api():
    source = StubEvidenceSource({})

    refiner = EvidenceCoverageRefiner.from_evidence_source(source, 10)

    assert refiner.multi_paper is not None


def test_rejects_a_comparison_table_without_self_ownership():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    comparison = _document(
        "comparison",
        title="A Survey of Hallucination Mitigation",
        body="We compare VTI with other methods.",
        tables=(
            _table(
                "comparison",
                "POPE adversarial Acc | VTI | LLaVA-v1.5 | 82.5",
            ),
        ),
    )
    mod = _document(
        "mod",
        title="Mixture of Decoding",
        body="We call our method MoD.",
        tables=(_table("mod", "POPE adversarial Acc | MoD | LLaVA-v1.5 | 79.7"),),
    )
    original = _selection("mod")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"comparison": comparison, "mod": mod})
    ).refine(_query(question), ["mod", "comparison"], original)

    assert result is original


def test_rejects_conditions_that_only_appear_on_another_table_row():
    question = (
        "What COCO val2017 mAP does DEIM-D-FINE-X achieve in 50 epochs, "
        "and what mAP does Mr. DETR with Swin-L achieve in 12 epochs?"
    )
    deim = _document(
        "deim",
        title="DEIM: DETR with Improved Matching",
        body="We introduce DEIM and build our DEIM-D-FINE-X.",
        tables=(
            PaperTable(
                paper_id="deim",
                table_id="Table 1",
                caption="COCO val2017 AP",
                rows=(
                    ("Setting", "Model", "Epochs", "AP"),
                    ("same", "Baseline", "50", "54.7"),
                    ("same", "DEIM-D-FINE-X", "24", "52.0"),
                ),
                text="COCO val2017 AP DEIM-D-FINE-X 24 52.0 Baseline 50 54.7",
            ),
        ),
    )
    mr_detr = _document(
        "mr-detr",
        title="Mr. DETR: Instructive Multi-Route Training",
        body="We introduce Mr. DETR.",
        tables=(
            _table(
                "mr-detr",
                "COCO val2017 AP | Mr. DETR | Swin-L | 12 epochs | 58.4",
            ),
        ),
    )
    original = _selection("mr-detr")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"deim": deim, "mr-detr": mr_detr})
    ).refine(_query(question), ["mr-detr", "deim"], original)

    assert result is original


def test_rejects_context_from_a_leading_data_row():
    question = (
        "What COCO val2017 mAP does DEIM-D-FINE-X achieve in 50 epochs, "
        "and what mAP does Mr. DETR with Swin-L achieve in 12 epochs?"
    )
    deim = _document(
        "deim",
        title="DEIM: DETR with Improved Matching",
        body="We introduce DEIM and build our DEIM-D-FINE-X.",
        tables=(
            PaperTable(
                paper_id="deim",
                table_id="Table 1",
                caption="",
                rows=(
                    ("COCO val2017 AP", "Baseline", "54.7"),
                    ("DEIM-D-FINE-X", "50 epochs", "53.2", "60.1"),
                ),
                text=(
                    "COCO val2017 AP Baseline 54.7 "
                    "DEIM-D-FINE-X 50 epochs 53.2 60.1"
                ),
            ),
        ),
    )
    mr_detr = _document(
        "mr-detr",
        title="Mr. DETR: Instructive Multi-Route Training",
        body="We introduce Mr. DETR.",
        tables=(
            _table(
                "mr-detr",
                "COCO val2017 AP | Mr. DETR | Swin-L | 12 epochs | 58.4 | 60.0",
            ),
        ),
    )
    original = _selection("mr-detr")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"deim": deim, "mr-detr": mr_detr})
    ).refine(_query(question), ["mr-detr", "deim"], original)

    assert result is original


@pytest.mark.parametrize(
    "body",
    [
        "We introduce DINO and compare our DINO-X and DINO-I.",
        "We introduce DINO. Smith built DINO-I.",
        "We introduce DINO. Prior work proposed DINO-I.",
        "We introduce DINO.\nReferences\nSmith et al. We propose DINO-I.",
    ],
)
def test_rejects_a_title_owner_prefix_without_a_full_variant_claim(body: str):
    question = (
        "What COCO val2017 mAP does DINO-I achieve in 12 epochs, "
        "and what mAP does Mr. DETR with Swin-L achieve in 12 epochs?"
    )
    dino = _document(
        "dino",
        title="DINO: DETR with Improved DeNoising",
        body=body,
        tables=(
            _table("dino", "COCO val2017 AP | DINO-I | 12 epochs | 55.0"),
        ),
    )
    mr_detr = _document(
        "mr-detr",
        title="Mr. DETR: Instructive Multi-Route Training",
        body="We introduce Mr. DETR.",
        tables=(
            _table(
                "mr-detr",
                "COCO val2017 AP | Mr. DETR | Swin-L | 12 epochs | 58.4",
            ),
        ),
    )
    original = _selection("mr-detr")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"dino": dino, "mr-detr": mr_detr})
    ).refine(_query(question), ["mr-detr", "dino"], original)

    assert result is original


def test_rejects_a_numeric_vector_for_the_wrong_property():
    question = (
        "What VAE latent channel mean normalization values do sCM and IMM use "
        "for their ImageNet experiments, and do they match?"
    )
    scm = _document(
        "scm",
        title="Scaling Consistency Models",
        body=(
            "We introduce sCM. ImageNet experiments use VAE latent normalization.\n"
            "The channel standard deviation is [1.0, 2.0, 3.0, 4.0].\n"
            "The mean score is reported elsewhere."
        ),
    )
    imm = _document(
        "imm",
        title="Inductive Moment Matching",
        body=(
            "We propose IMM. For ImageNet experiments, VAE latent normalization "
            "uses channel mean [0.8, -0.2, 0.2, 0.3]."
        ),
    )
    original = _selection("imm")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"scm": scm, "imm": imm})
    ).refine(_query(question), ["imm", "scm"], original)

    assert result is original


def test_falls_back_for_ambiguous_or_same_paper_coverage():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    both = _document(
        "both",
        title="VTI: A Method",
        body="We introduce VTI. We call our second method MoD.",
        tables=(
            _table("both", "POPE adversarial Acc | VTI MoD | LLaVA-v1.5 | 82.5"),
        ),
    )
    original = _selection("both")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"both": both})
    ).refine(_query(question), ["both"], original)

    assert result is original


def test_falls_back_when_current_selection_is_not_a_verified_reporter():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    vti = _document(
        "vti",
        title="VTI: Steering",
        body="We introduce VTI.",
        tables=(_table("vti", "POPE adversarial Acc | VTI | LLaVA-v1.5 | 82.5"),),
    )
    mod = _document(
        "mod",
        title="MoD: Decoding",
        body="We introduce MoD.",
        tables=(_table("mod", "POPE adversarial Acc | MoD | LLaVA-v1.5 | 79.7"),),
    )
    original = _selection("wrong")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"vti": vti, "mod": mod})
    ).refine(_query(question), ["wrong", "vti", "mod"], original)

    assert result is original


def test_ignores_candidates_beyond_the_configured_limit():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    vti = _document(
        "vti",
        title="VTI: Steering",
        body="We introduce VTI.",
        tables=(_table("vti", "POPE adversarial Acc | VTI | LLaVA-v1.5 | 82.5"),),
    )
    mod = _document(
        "mod",
        title="MoD: Decoding",
        body="We introduce MoD.",
        tables=(_table("mod", "POPE adversarial Acc | MoD | LLaVA-v1.5 | 79.7"),),
    )
    original = _selection("vti")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"vti": vti, "mod": mod}), candidate_limit=1
    ).refine(_query(question), ["vti", "mod"], original)

    assert result is original


def test_does_not_handle_table_answers_or_unrelated_question_forms():
    question = (
        "What are the scores for TCM, ECM-XL, iCT-deep, and SiD as reported "
        "in their respective papers?"
    )
    original = _selection()
    refiner = MultiPaperCoverageRefiner(StubEvidenceSource({}))

    assert refiner.refine(_query(question, ["table"]), ["first"], original) is original
    assert parse_two_slot_question(question) is None


def test_mixed_case_aliases_do_not_match_a_different_case():
    question = (
        "What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, "
        "and what adversarial accuracy does MoD achieve on LLaVA-1.5?"
    )
    vti = _document(
        "vti",
        title="VTI: Steering",
        body="We introduce VTI.",
        tables=(_table("vti", "POPE adversarial Acc | VTI | LLaVA-v1.5 | 82.5"),),
    )
    wrong_mod = _document(
        "mod",
        title="MOD: A Different Method",
        body="We introduce MOD.",
        tables=(_table("mod", "POPE adversarial Acc | MOD | LLaVA-v1.5 | 79.7"),),
    )
    original = _selection("vti")

    result = MultiPaperCoverageRefiner(
        StubEvidenceSource({"vti": vti, "mod": wrong_mod})
    ).refine(_query(question), ["vti", "mod"], original)

    assert result is original
