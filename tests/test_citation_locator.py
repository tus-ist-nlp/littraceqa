from __future__ import annotations

from typing import Any

from littraceqa.citation_locator import infer_citation_locator_overrides
from littraceqa.di_pipeline.contracts import Query


def _record(
    chunk_id: str,
    text: str,
    *,
    paper_id: str = "p1",
    page: int = 10,
    section: str = "References",
    citation_id: str | int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"page": page, "section": section}
    if citation_id is not None:
        metadata["citation_id"] = citation_id
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "text": text,
        "chunk_type": "text_span",
        "metadata": metadata,
    }


def _fact(value: Any, chunk_id: str, *, paper_id: str = "p1") -> dict[str, Any]:
    return {
        "id": "f1",
        "name": "answer-bearing bibliography fact",
        "value": value,
        "value_kind": "reported",
        "paper_id": paper_id,
        "chunk_ids": [chunk_id],
    }


def _derivation(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts": [fact],
        "operations": [],
        "answer_bindings": [
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": fact["id"],
                "answer_fragment": str(fact["value"]),
            }
        ],
        "final_semantic_answer": str(fact["value"]),
    }


def test_numeric_nth_reference_recovers_explicit_id_only_from_matching_entry():
    record = _record(
        "p1#refs",
        "[NeurIPS 2025] Synthetic\n"
        "[23] Jeffrey Zhou. Prior work, 2023.\n"
        "[24] Freda Shi, Mirac Suzgun, et al. Multilingual reasoning, 2022.\n"
        "[25] Heming Xia. Later work, 2024.",
        page=12,
    )
    query = Query(
        query_id="q",
        question="Who is the first author of the 24th reference cited in EasySpec?",
        answer_types=["freeform"],
    )

    assert infer_citation_locator_overrides(
        query,
        derivation=_derivation(_fact("Freda Shi", "p1#refs")),
        answer={"freeform": {"text": "Freda Shi"}},
        support_records=[record],
        paper_records=[record],
    ) == {"p1#refs": ("24",)}

    assert infer_citation_locator_overrides(
        query,
        derivation=_derivation(_fact("Heming Xia", "p1#refs")),
        answer={"freeform": {"text": "Heming Xia"}},
        support_records=[record],
        paper_records=[record],
    ) == {}


def test_word_nth_reference_supports_combined_freeform_and_table_answer():
    record = _record(
        "p1#refs",
        "[CVPR 2025] Synthetic\n"
        "[1] Chaitanya Ahuja. First work, 2019.\n"
        "[2] Nikos Athanasiou. Second work, 2022.\n"
        "[3] Nikos Athanasiou. SINC, 2023.\n"
        "[4] Michael Ayers. Fourth work, 1991.",
        page=9,
    )
    query = Query(
        query_id="q",
        question="Who is the first author of the third reference cited in the paper?",
        answer_types=["freeform", "table"],
        table_schema=[{"name": "Author", "type": "string", "is_row_key": True}],
    )

    assert infer_citation_locator_overrides(
        query,
        derivation=_derivation(_fact("Nikos Athanasiou", "p1#refs")),
        answer={
            "freeform": {"text": "The first author is Nikos Athanasiou."},
            "table": {"rows": [{"Author": "Nikos Athanasiou"}]},
        },
        support_records=[record],
        paper_records=[record],
    ) == {"p1#refs": ("3",)}


def test_last_reference_index_requires_complete_gap_free_explicit_bibliography():
    first = _record(
        "p1#refs-a",
        "[NeurIPS 2025] Synthetic\n"
        + "\n".join(f"[{index}] Author {index}. Work {index}, 2020." for index in range(1, 61)),
        page=11,
    )
    last = _record(
        "p1#refs-b",
        "[NeurIPS 2025] Synthetic\n"
        + "\n".join(f"[{index}] Author {index}. Work {index}, 2020." for index in range(61, 68)),
        page=14,
    )
    query = Query(
        query_id="q",
        question="What is the index of the last reference in FedRACE?",
        answer_types=["freeform"],
    )
    derivation = _derivation(_fact("67", "p1#refs-b"))
    answer = {"freeform": {"text": "67"}}

    assert infer_citation_locator_overrides(
        query,
        derivation=derivation,
        answer=answer,
        support_records=[last],
        paper_records=[first, last],
    ) == {"p1#refs-b": ("67",)}

    first_with_gap = {
        **first,
        "text": first["text"].replace("[60] Author 60.", "[59] Author 60."),
    }
    assert infer_citation_locator_overrides(
        query,
        derivation=derivation,
        answer=answer,
        support_records=[last],
        paper_records=[first_with_gap, last],
    ) == {}


def _author_count_derivation(chunk_id: str = "p1#refs") -> dict[str, Any]:
    items = ["Bell (2020)", "Bonawitz (2017)", "Bonawitz (2019)"]
    fact = _fact(items, chunk_id)
    return {
        "facts": [fact],
        "operations": [
            {
                "id": "op1",
                "kind": "count",
                "fact_ids": ["f1"],
                "items": items,
                "result": 3,
                "answer_binding": {
                    "answer_path": "answer.freeform.text",
                    "expected": 3,
                    "answer_fragment": "3",
                },
            }
        ],
        "answer_bindings": [
            {
                "answer_path": "answer.freeform.text",
                "source_type": "operation",
                "source_id": "op1",
                "answer_fragment": "3",
            }
        ],
        "final_semantic_answer": "3",
    }


def _secemb_reference_record(*, leading_continuation: bool = False) -> dict[str, Any]:
    prefix = "continued text from a missing earlier entry\n" if leading_continuation else ""
    return _record(
        "p1#refs",
        "[ICML 2025] SecEmb\n"
        + prefix
        + "Abadi, M., Chu, A., and Zhang, L. Differential privacy. CCS, 2016.\n"
        "Addanki, S., Garbe, K., and Jaffe, E. Prio plus. SCN, 2022.\n"
        "Aji, A. F. and Heafield, K. Sparse communication. EMNLP, 2017.\n"
        "Ammad-Ud-Din, M., Khan, S. A., and Flanagan, A. Federated CF, 2019.\n"
        "Bell, J. H., Bonawitz, K. A., Gascon, A., and Raykova, M. Secure aggregation. CCS, 2020.\n"
        "Bonawitz, K., Ivanov, V., and McMahan, H. Practical secure "
        "aggregation. In proceedings of the\n"
        "2017 ACM SIGSAC Conference, 2017.\n"
        "Bonawitz, K., Eichner, H., and McMahan, B. Federated learning at scale. MLSys, 2019.\n"
        "Boneh, D., Boyle, E., and Ishai, Y. Private heavy hitters. IEEE, 2021.",
        page=10,
    )


def test_author_filtered_unnumbered_bibliography_expands_one_chunk_to_ids_5_6_7():
    record = _secemb_reference_record()
    query = Query(
        query_id="q",
        question="How many references in the SecEmb paper include Bonawitz as an author?",
        answer_types=["freeform", "multiple_choice"],
        options={"A": "2", "B": "3", "C": "4", "D": "8"},
    )

    assert infer_citation_locator_overrides(
        query,
        derivation=_author_count_derivation(),
        answer={"freeform": {"text": "3"}, "multiple_choice": {"label": "B"}},
        support_records=[record],
        paper_records=[record],
    ) == {"p1#refs": ("5", "6", "7")}


def test_author_filtered_count_fails_closed_for_incomplete_prefix_or_ambiguous_item():
    query = Query(
        query_id="q",
        question="How many references in the SecEmb paper include Bonawitz as an author?",
        answer_types=["freeform"],
    )
    incomplete = _secemb_reference_record(leading_continuation=True)
    assert infer_citation_locator_overrides(
        query,
        derivation=_author_count_derivation(),
        answer={"freeform": {"text": "3"}},
        support_records=[incomplete],
        paper_records=[incomplete],
    ) == {}

    record = _secemb_reference_record()
    ambiguous = {
        **record,
        "text": record["text"]
        + "\nBell, J. H., Bonawitz, K. A. Another Bell paper. CCS, 2020.",
    }
    assert infer_citation_locator_overrides(
        query,
        derivation=_author_count_derivation(),
        answer={"freeform": {"text": "3"}},
        support_records=[ambiguous],
        paper_records=[ambiguous],
    ) == {}


def test_existing_conflicting_citation_metadata_forces_no_op():
    record = _record(
        "p1#refs",
        "[NeurIPS 2025] Synthetic\n[24] Freda Shi. Work, 2022.",
        citation_id="23",
    )
    query = Query(
        query_id="q",
        question="Who is the first author of the 24th reference?",
        answer_types=["freeform"],
    )

    assert infer_citation_locator_overrides(
        query,
        derivation=_derivation(_fact("Freda Shi", "p1#refs")),
        answer={"freeform": {"text": "Freda Shi"}},
        support_records=[record],
        paper_records=[record],
    ) == {}


def test_author_filter_does_not_count_a_surname_mentioned_only_in_title():
    record = _record(
        "p1#refs",
        "[ICML 2025] Synthetic\n"
        "Smith, J. A critique of Bonawitz. Privacy Journal, 2020.",
    )
    query = Query(
        query_id="q",
        question="How many references in the paper include Bonawitz as an author?",
        answer_types=["freeform"],
    )
    derivation = _author_count_derivation()
    derivation["facts"][0]["value"] = ["Smith (2020)"]
    derivation["operations"][0]["items"] = ["Smith (2020)"]
    derivation["operations"][0]["result"] = 1

    assert infer_citation_locator_overrides(
        query,
        derivation=derivation,
        answer={"freeform": {"text": "1"}},
        support_records=[record],
        paper_records=[record],
    ) == {}


def test_publisher_continuation_is_not_invented_as_an_unnumbered_author_entry():
    record = _record(
        "p1#refs",
        "[ICML 2025] Synthetic\n"
        "Smith, J. First work. Journal, 2015.\n"
        "Springer, Cham, 2016.\n"
        "Bonawitz, K. Secure aggregation. Journal, 2017.",
    )
    query = Query(
        query_id="q",
        question="How many references in the paper include Springer as an author?",
        answer_types=["freeform"],
    )
    derivation = _author_count_derivation()
    derivation["facts"][0]["value"] = ["Springer (2016)"]
    derivation["operations"][0]["items"] = ["Springer (2016)"]
    derivation["operations"][0]["result"] = 1

    assert infer_citation_locator_overrides(
        query,
        derivation=derivation,
        answer={"freeform": {"text": "1"}},
        support_records=[record],
        paper_records=[record],
    ) == {}


def test_conflicting_duplicate_or_support_corpus_record_forces_no_op():
    record = _record(
        "p1#refs",
        "[NeurIPS 2025] Synthetic\n[24] Freda Shi. Work, 2022.",
    )
    conflicting = {**record, "text": str(record["text"]).replace("Freda", "Heming")}
    query = Query(
        query_id="q",
        question="Who is the first author of the 24th reference?",
        answer_types=["freeform"],
    )
    kwargs = {
        "derivation": _derivation(_fact("Freda Shi", "p1#refs")),
        "answer": {"freeform": {"text": "Freda Shi"}},
    }

    assert infer_citation_locator_overrides(
        query,
        **kwargs,
        support_records=[record, conflicting],
        paper_records=[record],
    ) == {}
    assert infer_citation_locator_overrides(
        query,
        **kwargs,
        support_records=[record],
        paper_records=[record, conflicting],
    ) == {}
    assert infer_citation_locator_overrides(
        query,
        **kwargs,
        support_records=[record],
        paper_records=[conflicting],
    ) == {}
    assert infer_citation_locator_overrides(
        query,
        **kwargs,
        support_records=[record, dict(record)],
        paper_records=[record, dict(record)],
    ) == {"p1#refs": ("24",)}


def test_author_filtered_count_requires_the_submitted_answer_to_equal_result():
    record = _secemb_reference_record()
    query = Query(
        query_id="q",
        question="How many references in the SecEmb paper include Bonawitz as an author?",
        answer_types=["freeform"],
    )

    assert infer_citation_locator_overrides(
        query,
        derivation=_author_count_derivation(),
        answer={"freeform": {"text": "4"}},
        support_records=[record],
        paper_records=[record],
    ) == {}
