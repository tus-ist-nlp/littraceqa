from __future__ import annotations

from typing import Any

import pytest

from littraceqa.answer_derivation import (
    DerivationValidationError,
    validate_answer_semantics,
    validate_table_rows,
)
from littraceqa.di_pipeline.contracts import Query


def _fact(
    fact_id: str = "f1",
    value: Any = "value",
    *,
    name: str | None = None,
    value_kind: str = "reported",
) -> dict[str, Any]:
    return {
        "id": fact_id,
        "name": name or f"source fact {fact_id}",
        "value": value,
        "value_kind": value_kind,
        "paper_id": "p1",
        "chunk_ids": [f"p1#{fact_id}"],
    }


def _binding(
    answer_path: str,
    expected: Any,
    answer_fragment: str | None = None,
) -> dict[str, Any]:
    output = {"answer_path": answer_path, "expected": expected}
    if answer_fragment is not None:
        output["answer_fragment"] = answer_fragment
    return output


def _derivation(
    facts: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    final_answer: str,
) -> dict[str, Any]:
    return {
        "facts": facts,
        "operations": operations,
        "final_semantic_answer": final_answer,
    }


def test_compare_recomputes_yes_no_polarity() -> None:
    query = Query("q", "Does A have more than B?", ["freeform"])
    facts = [_fact("left", 30), _fact("right", 21)]
    operation = {
        "id": "compare_ab",
        "kind": "compare",
        "fact_ids": ["left", "right"],
        "left": 30,
        "operator": ">",
        "right": 21,
        "result": True,
        "answer_binding": _binding("answer.freeform.text", True, "No"),
    }

    with pytest.raises(DerivationValidationError, match="does not express expected"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "No"),
            answer={"freeform": {"text": "No"}},
        )

    operation["answer_binding"] = _binding("answer.freeform.text", True, "Yes")
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "Yes"),
        answer={"freeform": {"text": "Yes"}},
    )
    assert validated["operations"][0]["result"] is True


def test_compare_rejects_incorrect_boolean_before_answer_binding() -> None:
    query = Query("q", "Does A have more than B?", ["freeform"])
    operation = {
        "id": "compare_ab",
        "kind": "compare",
        "fact_ids": ["left", "right"],
        "left": 30,
        "operator": ">",
        "right": 21,
        "result": False,
        "answer_binding": _binding("answer.freeform.text", False, "No"),
    }
    with pytest.raises(DerivationValidationError, match="30 > 21 is True"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("left", 30), _fact("right", 21)],
                [operation],
                "No",
            ),
            answer={"freeform": {"text": "No"}},
        )


def test_operation_rejects_invented_operands_not_present_in_facts() -> None:
    query = Query("q", "What is the difference?", ["freeform"])
    operation = {
        "id": "difference",
        "kind": "subtract",
        "fact_ids": ["left", "right"],
        "operands": [10, 3],
        "result": 7,
        "answer_binding": _binding("answer.freeform.text", 7, "7"),
    }
    with pytest.raises(DerivationValidationError, match="referenced fact values"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("left", 10), _fact("right", 2)], [operation], "7"
            ),
            answer={"freeform": {"text": "7"}},
        )


def test_count_requires_distinct_fact_grounded_items_and_matching_result() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    facts = [_fact("panels", ["(a)", "(b)", "(c)"], value_kind="visual")]
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": ["(a)", "(b)", "(c)"],
        "result": 2,
        "answer_binding": _binding("answer.freeform.text", 2, "2"),
    }
    with pytest.raises(DerivationValidationError, match="contain 3 entries"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "2"),
            answer={"freeform": {"text": "2"}},
        )

    operation.update(
        {
            "items": ["(a)", "(b)", "(b)"],
            "result": 3,
            "answer_binding": _binding("answer.freeform.text", 3, "3"),
        }
    )
    with pytest.raises(DerivationValidationError, match="distinct items"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "3"),
            answer={"freeform": {"text": "3"}},
        )


def test_count_flattens_list_fact_values() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["row1", "row2"],
        "items": ["(a)", "(b)", "(c)", "(d)"],
        "result": 4,
        "answer_binding": _binding("answer.freeform.text", 4, "four"),
    }
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [
                _fact("row1", [["(a)", "(b)"]], value_kind="visual"),
                _fact("row2", ["(c)", "(d)"], value_kind="visual"),
            ],
            [operation],
            "four",
        ),
        answer={"freeform": {"text": "four"}},
    )
    assert validated["operations"][0]["result"] == 4


def test_count_rejects_eight_vs_seven_word_answer() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    items = list("abcdefgh")
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": items,
        "result": 8,
        "answer_binding": _binding("answer.freeform.text", 8, "seven"),
    }
    with pytest.raises(DerivationValidationError, match="does not express expected"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("panels", items, value_kind="visual")],
                [operation],
                "seven",
            ),
            answer={"freeform": {"text": "seven"}},
        )


def test_multi_operation_contradictory_multiple_choice_is_rejected() -> None:
    query = Query(
        "q",
        "Which count is correct?",
        ["multiple_choice"],
        options={"A": "Eight panels", "B": "Seven panels"},
    )
    eight_items = list("abcdefgh")
    seven_items = list("abcdefg")
    operations = [
        {
            "id": "first_count",
            "kind": "count",
            "fact_ids": ["eight_panels"],
            "items": eight_items,
            "result": 8,
            "answer_binding": _binding(
                "answer.multiple_choice.selected_option_text", 8, "Seven"
            ),
        },
        {
            "id": "second_count",
            "kind": "count",
            "fact_ids": ["seven_panels"],
            "items": seven_items,
            "result": 7,
            "answer_binding": _binding("answer.multiple_choice", 7, "Seven"),
        },
    ]
    with pytest.raises(DerivationValidationError, match="expected result"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [
                    _fact("eight_panels", eight_items, value_kind="visual"),
                    _fact("seven_panels", seven_items, value_kind="visual"),
                ],
                operations,
                "Seven panels",
            ),
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Seven panels",
                }
            },
        )


def test_decimal_arithmetic_is_recomputed_exactly() -> None:
    query = Query("q", "What is the difference?", ["freeform"])
    facts = [_fact("left", "53.46"), _fact("right", "41.15")]
    operation = {
        "id": "difference",
        "kind": "subtract",
        "fact_ids": ["left", "right"],
        "operands": ["53.46", "41.15"],
        "result": "12.30",
        "answer_binding": _binding("answer.freeform.text", "12.30", "12.30"),
    }
    with pytest.raises(DerivationValidationError, match="gives 12.31"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "12.30"),
            answer={"freeform": {"text": "12.30"}},
        )

    # A directly reported lookup remains valid with no fake arithmetic.
    validated = validate_answer_semantics(
        query,
        derivation=_derivation([_fact("reported", "12.30")], [], "12.30"),
        answer={"freeform": {"text": "12.30"}},
    )
    assert validated["operations"] == []


def test_divide_requires_explicit_rounding_or_exact_contract() -> None:
    query = Query("q", "What is one third?", ["freeform"])
    facts = [_fact("numerator", 1), _fact("denominator", 3)]
    operation = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 3],
        "result": "0.333",
        "answer_binding": _binding("answer.freeform.text", "0.333", "0.333"),
    }
    with pytest.raises(DerivationValidationError, match="requires exact=true or a rounding"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "0.333"),
            answer={"freeform": {"text": "0.333"}},
        )

    operation["rounding"] = {"decimal_places": 3, "mode": "half_up"}
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "0.333"),
        answer={"freeform": {"text": "0.333"}},
    )
    assert validated["operations"][0]["result"] == "0.333"


def test_divide_rounding_mode_and_exact_terminating_result() -> None:
    query = Query("q", "What is the quotient?", ["freeform"])
    facts = [_fact("numerator", 1), _fact("denominator", 8)]
    half_even = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 8],
        "result": "0.12",
        "rounding": {"decimal_places": 2, "mode": "half_even"},
        "answer_binding": _binding("answer.freeform.text", "0.12", "0.12"),
    }
    assert validate_answer_semantics(
        query,
        derivation=_derivation(facts, [half_even], "0.12"),
        answer={"freeform": {"text": "0.12"}},
    )

    exact = {
        **half_even,
        "result": "0.125",
        "exact": True,
        "answer_binding": _binding("answer.freeform.text", "0.125", "0.125"),
    }
    exact.pop("rounding")
    assert validate_answer_semantics(
        query,
        derivation=_derivation(facts, [exact], "0.125"),
        answer={"freeform": {"text": "0.125"}},
    )


def test_nonterminating_exact_divide_is_rejected() -> None:
    query = Query("q", "What is one third?", ["freeform"])
    operation = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 3],
        "result": "0.333",
        "exact": True,
        "answer_binding": _binding("answer.freeform.text", "0.333", "0.333"),
    }
    with pytest.raises(DerivationValidationError, match="non-terminating"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("numerator", 1), _fact("denominator", 3)],
                [operation],
                "0.333",
            ),
            answer={"freeform": {"text": "0.333"}},
        )


def test_variable_mc_label_and_exact_option_text_are_required() -> None:
    query = Query(
        "q",
        "Which option?",
        ["multiple_choice"],
        options={"A": "Alpha", "B": "Beta", "E": "Epsilon"},
    )
    derivation = _derivation([_fact()], [], "Epsilon")
    answer = {
        "multiple_choice": {
            "label": "E",
            "selected_option_text": "Epsilon",
        }
    }
    assert validate_answer_semantics(query, derivation=derivation, answer=answer)

    answer["multiple_choice"]["selected_option_text"] = "Epsilon "
    with pytest.raises(DerivationValidationError, match="exactly equal"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_argmax_candidates_must_match_fact_label_values() -> None:
    query = Query("q", "Which is largest?", ["freeform"])
    facts = [
        _fact("a", {"label": "A", "value": 29}),
        _fact("b", {"label": "B", "value": 32}),
        _fact("c", {"label": "C", "value": 30}),
    ]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["a", "b", "c"],
        "candidates": [
            {"label": "A", "value": 29},
            {"label": "B", "value": 31},
            {"label": "C", "value": 30},
        ],
        "result": "B",
        "answer_binding": _binding("answer.freeform.text", "B", "B"),
    }
    with pytest.raises(DerivationValidationError, match="referenced facts"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "B"),
            answer={"freeform": {"text": "B"}},
        )


def test_argmax_result_must_match_candidate_values_and_answer() -> None:
    query = Query("q", "Which is largest?", ["freeform"])
    facts = [
        _fact("a", {"label": "A", "value": 29}),
        _fact("b", {"label": "B", "value": 32}),
        _fact("c", {"label": "C", "value": 30}),
    ]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["a", "b", "c"],
        "candidates": [
            {"label": "A", "value": 29},
            {"label": "B", "value": 32},
            {"label": "C", "value": 30},
        ],
        "result": "A",
        "answer_binding": _binding("answer.freeform.text", "A", "A"),
    }
    with pytest.raises(DerivationValidationError, match="winner"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "A"),
            answer={"freeform": {"text": "A"}},
        )

    operation.update(
        {
            "result": "B",
            "answer_binding": _binding("answer.freeform.text", "B", "Backbone-B"),
        }
    )
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "Backbone-B"),
        answer={"freeform": {"text": "Backbone-B"}},
    )
    assert validated["operations"][0]["result"] == "B"


def test_fact_contract_requires_unique_ids_names_and_typed_value() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    invalid = _fact()
    invalid.pop("value_kind")
    with pytest.raises(DerivationValidationError, match="missing required fields"):
        validate_answer_semantics(
            query,
            derivation=_derivation([invalid], [], "value"),
            answer={"freeform": {"text": "value"}},
        )

    with pytest.raises(DerivationValidationError, match="duplicate derivation fact id"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("same"), _fact("same")], [], "value"),
            answer={"freeform": {"text": "value"}},
        )

    with pytest.raises(DerivationValidationError, match="duplicate derivation fact name"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("a", name="Repeated"), _fact("b", name=" repeated ")],
                [],
                "value",
            ),
            answer={"freeform": {"text": "value"}},
        )


def test_operation_contract_requires_unique_id_fact_ids_and_binding() -> None:
    query = Query("q", "What is the sum?", ["freeform"])
    operation = {
        "id": "sum",
        "kind": "add",
        "fact_ids": ["a", "b"],
        "operands": [1, 2],
        "result": 3,
        "answer_binding": _binding("answer.freeform.text", 3, "3"),
    }
    duplicate = {**operation}
    with pytest.raises(DerivationValidationError, match="duplicate derivation operation id"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("a", 1), _fact("b", 2)],
                [operation, duplicate],
                "3",
            ),
            answer={"freeform": {"text": "3"}},
        )

    operation["fact_ids"] = ["a", "invented"]
    with pytest.raises(DerivationValidationError, match="unknown facts"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("a", 1)], [operation], "3"),
            answer={"freeform": {"text": "3"}},
        )


def test_typed_table_answer_binding_requires_exact_value() -> None:
    query = Query(
        "q",
        "How many panels?",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Panel Count", "type": "number", "is_row_key": False},
        ],
    )
    items = ["a", "b", "c"]
    operation = {
        "id": "count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": items,
        "result": 3,
        "answer_binding": _binding("answer.table.rows[0].Panel Count", 3),
    }
    derivation = _derivation(
        [_fact("panels", items, value_kind="visual")], [operation], "3"
    )
    answer = {"table": {"rows": [{"Method": "Nova", "Panel Count": 3}]}}
    assert validate_answer_semantics(query, derivation=derivation, answer=answer)

    answer["table"]["rows"][0]["Panel Count"] = 2
    with pytest.raises(DerivationValidationError, match="does not exactly equal"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_table_schema_requires_native_types_and_exact_columns() -> None:
    query = Query(
        "q",
        "Return rows.",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
            {"name": "Passed", "type": "boolean", "is_row_key": False},
        ],
    )
    rows = [{"Method": "Nova", "Score": 0.9, "Passed": True}]
    assert validate_table_rows(query, rows) == rows

    with pytest.raises(DerivationValidationError, match="must be a JSON number"):
        validate_table_rows(
            query, [{"Method": "Nova", "Score": ".9", "Passed": True}]
        )
    with pytest.raises(DerivationValidationError, match="contain exactly"):
        validate_table_rows(
            query,
            [{"Method": "Nova", "Score": 0.9, "Passed": True, "Extra": 1}],
        )
    with pytest.raises(DerivationValidationError, match="duplicate table row key"):
        validate_table_rows(query, [rows[0], {**rows[0], "Score": 1.0}])


def test_lookup_final_answer_must_equal_freeform() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    with pytest.raises(DerivationValidationError, match="freeform.text"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("reported", "42")], [], "42"),
            answer={"freeform": {"text": "41"}},
        )
