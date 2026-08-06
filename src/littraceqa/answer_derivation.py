"""Deterministic checks for LLM-produced LitTraceQA answer derivations.

The model is responsible for reading scientific prose and figures.  Arithmetic,
counting, comparisons, option mapping, and output types are not reading tasks,
so this module verifies those parts without another probabilistic judgment.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import (
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    localcontext,
)
from fractions import Fraction
from typing import Any

from littraceqa.di_pipeline.contracts import Query


class DerivationValidationError(ValueError):
    """A structured derivation contradicts its inputs or final answer."""


_COMPARE_OPERATORS = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}
_NUMBER_IN_TEXT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
# Facts must be values read directly from the supplied source.  Derived values
# belong in ``operations``; accepting a free-standing ``computed`` fact would
# let a model attach an arbitrary result to an otherwise valid chunk.
_FACT_VALUE_KINDS = frozenset({"reported", "visual", "text"})
_ROUNDING_MODES = {
    "half_up": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
)


def validate_answer_semantics(
    query: Query,
    *,
    derivation: Any,
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize a Stage-2 derivation and answer.

    The returned mapping is safe to keep in the audit record.  Any mismatch is
    raised so the caller can request exactly one evidence-aware repair.
    """

    if not isinstance(derivation, dict):
        raise DerivationValidationError("derivation must be an object")
    facts, facts_by_id = _validate_facts(derivation.get("facts"))

    _validate_answer_types(query, answer)

    raw_operations = derivation.get("operations")
    if not isinstance(raw_operations, list):
        raise DerivationValidationError("derivation.operations must be a list")
    operations: list[dict[str, Any]] = []
    operation_results: dict[str, tuple[Any, str]] = {}
    seen_operation_ids: set[str] = set()
    for index, operation in enumerate(raw_operations):
        validated_operation, computed = _validate_operation(
            operation,
            index,
            facts_by_id=facts_by_id,
            answer=answer,
        )
        operation_id = str(validated_operation["id"])
        if operation_id in seen_operation_ids:
            raise DerivationValidationError(
                f"duplicate derivation operation id: {operation_id!r}"
            )
        seen_operation_ids.add(operation_id)
        operations.append(validated_operation)
        operation_results[operation_id] = (
            computed,
            str(validated_operation["kind"]),
        )

    answer_bindings = _validate_final_answer_bindings(
        derivation.get("answer_bindings"),
        query=query,
        answer=answer,
        facts_by_id=facts_by_id,
        operation_results=operation_results,
    )

    final_answer = derivation.get("final_semantic_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise DerivationValidationError(
            "derivation.final_semantic_answer must be a non-empty string"
        )
    final_answer = final_answer.strip()

    freeform = answer.get("freeform")
    if isinstance(freeform, dict) and final_answer != freeform["text"].strip():
        raise DerivationValidationError(
            "final_semantic_answer must exactly equal freeform.text"
        )
    # A combined freeform + multiple-choice record commonly uses a descriptive
    # freeform sentence and a shorter option text for the same semantic answer.
    # The bindings above validate both output components independently.  Only an
    # MC-only answer uses the option text as its canonical final string.
    _validate_multiple_choice(
        query,
        answer,
        final_answer,
        require_final_match="freeform" not in query.answer_types,
    )

    return {
        **derivation,
        "facts": facts,
        "operations": operations,
        "answer_bindings": answer_bindings,
        "final_semantic_answer": final_answer,
    }


def validate_table_rows(query: Query, rows: Any) -> list[dict[str, Any]]:
    """Require exact schema columns, native JSON types, and unique row keys."""

    if not isinstance(rows, list) or not rows:
        raise DerivationValidationError("table answer must contain at least one row")
    schema = query.table_schema or []
    columns = [
        str(column.get("name"))
        for column in schema
        if isinstance(column, dict) and column.get("name")
    ]
    if not columns:
        raise DerivationValidationError("table_schema has no columns")
    column_types = {
        str(column["name"]): str(column.get("type") or "string").lower()
        for column in schema
        if isinstance(column, dict) and column.get("name")
    }
    row_keys = [
        str(column["name"])
        for column in schema
        if isinstance(column, dict)
        and column.get("name")
        and column.get("is_row_key") is True
    ]
    # Match the official evaluator: when no explicit key exists, only the first
    # schema column is the implicit row key (not the entire row).
    dedupe_columns = row_keys or columns[:1]
    seen_keys: set[tuple[str, ...]] = set()
    validated: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(columns):
            raise DerivationValidationError(
                f"table row {row_index} must contain exactly {columns}"
            )
        for column in columns:
            _validate_cell_type(
                row[column],
                column_types[column],
                path=f"table.rows[{row_index}].{column}",
            )
        # The pinned evaluator enforces non-empty cells only for explicitly
        # declared row keys.  Its implicit first-column fallback still produces
        # a valid (possibly empty-string) tuple key.
        if any(row[column] in (None, "") for column in row_keys):
            raise DerivationValidationError(
                f"table row {row_index} has an empty row-key cell"
            )
        key = tuple(_normalize_row_key(row[column]) for column in dedupe_columns)
        if key in seen_keys:
            raise DerivationValidationError(f"duplicate table row key: {key}")
        seen_keys.add(key)
        validated.append(dict(row))
    return validated


def _validate_facts(
    raw_facts: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(raw_facts, list) or not raw_facts:
        raise DerivationValidationError("derivation.facts must be a non-empty list")

    facts: list[dict[str, Any]] = []
    facts_by_id: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    for index, fact in enumerate(raw_facts):
        path = f"derivation.facts[{index}]"
        if not isinstance(fact, dict):
            raise DerivationValidationError(f"{path} must be an object")
        missing = sorted(
            {"id", "name", "value", "value_kind", "paper_id", "chunk_ids"}
            - set(fact)
        )
        if missing:
            raise DerivationValidationError(
                f"{path} is missing required fields: {missing}"
            )

        fact_id = _non_empty_string(fact.get("id"), f"{path}.id")
        if fact_id in facts_by_id:
            raise DerivationValidationError(f"duplicate derivation fact id: {fact_id!r}")
        name = _non_empty_string(fact.get("name"), f"{path}.name")
        normalized_name = _normalize_item(name)
        if normalized_name in seen_names:
            raise DerivationValidationError(
                f"duplicate derivation fact name: {name!r}"
            )
        seen_names.add(normalized_name)

        value_kind = str(fact.get("value_kind") or "").strip()
        if value_kind not in _FACT_VALUE_KINDS:
            raise DerivationValidationError(
                f"{path}.value_kind must be one of {sorted(_FACT_VALUE_KINDS)}"
            )
        paper_id = _non_empty_string(fact.get("paper_id"), f"{path}.paper_id")
        raw_chunk_ids = fact.get("chunk_ids")
        if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
            raise DerivationValidationError(f"{path}.chunk_ids must be non-empty")
        chunk_ids = [
            _non_empty_string(value, f"{path}.chunk_ids[{chunk_index}]")
            for chunk_index, value in enumerate(raw_chunk_ids)
        ]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise DerivationValidationError(f"{path}.chunk_ids must be distinct")

        normalized = {
            **fact,
            "id": fact_id,
            "name": name,
            "value_kind": value_kind,
            "paper_id": paper_id,
            "chunk_ids": chunk_ids,
        }
        facts.append(normalized)
        facts_by_id[fact_id] = normalized
    return facts, facts_by_id


def _validate_operation(
    operation: Any,
    index: int,
    *,
    facts_by_id: dict[str, dict[str, Any]],
    answer: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    if not isinstance(operation, dict):
        raise DerivationValidationError(
            f"derivation.operations[{index}] must be an object"
        )
    path = f"derivation.operations[{index}]"
    operation_id = _non_empty_string(operation.get("id"), f"{path}.id")
    kind = str(operation.get("kind") or "").strip()
    raw_fact_ids = operation.get("fact_ids")
    if not isinstance(raw_fact_ids, list) or not raw_fact_ids:
        raise DerivationValidationError(f"{path}.fact_ids must be non-empty")
    fact_ids = [
        _non_empty_string(value, f"{path}.fact_ids[{fact_index}]")
        for fact_index, value in enumerate(raw_fact_ids)
    ]
    if len(set(fact_ids)) != len(fact_ids):
        raise DerivationValidationError(f"{path}.fact_ids must be distinct")
    unknown_fact_ids = [fact_id for fact_id in fact_ids if fact_id not in facts_by_id]
    if unknown_fact_ids:
        raise DerivationValidationError(
            f"{path}.fact_ids reference unknown facts: {unknown_fact_ids}"
        )
    referenced_facts = [facts_by_id[fact_id] for fact_id in fact_ids]

    computed: Any
    if kind in {"add", "subtract", "multiply", "divide"}:
        operands = operation.get("operands")
        if not isinstance(operands, list) or len(operands) < 2:
            raise DerivationValidationError(f"{path}.operands needs at least two numbers")
        numbers = [_decimal(value, f"{path}.operands") for value in operands]
        fact_numbers = [
            _decimal(fact["value"], f"{path}.fact_ids[{fact_index}].value")
            for fact_index, fact in enumerate(referenced_facts)
        ]
        _require_numeric_sequence_match(
            numbers,
            fact_numbers,
            path=f"{path}.operands",
        )
        computed = _compute_arithmetic(kind, numbers, operation, path)
        reported = _decimal(operation.get("result"), f"{path}.result")
        if reported != computed:
            raise DerivationValidationError(
                f"{path}.result={reported} but deterministic {kind} gives {computed}"
            )
    elif kind == "count":
        items = operation.get("items")
        if not isinstance(items, list) or not items:
            raise DerivationValidationError(f"{path}.items must be non-empty")
        normalized_items = [_normalize_item(item) for item in items]
        if any(not item for item in normalized_items):
            raise DerivationValidationError(f"{path}.items contains an empty item")
        if len(set(normalized_items)) != len(normalized_items):
            raise DerivationValidationError(f"{path}.items must contain distinct items")
        fact_items: list[Any] = []
        for fact in referenced_facts:
            fact_items.extend(_flatten_count_value(fact["value"]))
        if Counter(normalized_items) != Counter(
            _normalize_item(item) for item in fact_items
        ):
            raise DerivationValidationError(
                f"{path}.items do not match values in referenced facts"
            )
        result = operation.get("result")
        if isinstance(result, bool) or not isinstance(result, int):
            raise DerivationValidationError(f"{path}.result must be an integer")
        if result != len(items):
            raise DerivationValidationError(
                f"{path}.result={result} but items contain {len(items)} entries"
            )
        computed = result
    elif kind in {"argmax", "argmin"}:
        candidates = operation.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise DerivationValidationError(f"{path}.candidates needs at least two rows")
        parsed: list[tuple[str, Decimal]] = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or set(candidate) != {"label", "value"}:
                raise DerivationValidationError(
                    f"{path}.candidates[{candidate_index}] must contain label and value"
                )
            label = str(candidate.get("label") or "").strip()
            if not label:
                raise DerivationValidationError(f"{path}.candidate label is empty")
            parsed.append((label, _decimal(candidate.get("value"), f"{path}.value")))
        normalized_labels = [_normalize_item(label) for label, _ in parsed]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise DerivationValidationError(f"{path}.candidate labels must be distinct")
        fact_candidates = _fact_candidates(referenced_facts, path)
        if _candidate_map(parsed) != _candidate_map(fact_candidates):
            raise DerivationValidationError(
                f"{path}.candidates do not match label/value pairs in referenced facts"
            )
        target_value = (max if kind == "argmax" else min)(value for _, value in parsed)
        winners = {label for label, value in parsed if value == target_value}
        result = str(operation.get("result") or "").strip()
        if result not in winners:
            raise DerivationValidationError(
                f"{path}.result={result!r} is not a deterministic {kind} winner {sorted(winners)}"
            )
        computed = result
    elif kind == "compare":
        left = _decimal(operation.get("left"), f"{path}.left")
        right = _decimal(operation.get("right"), f"{path}.right")
        fact_numbers = [
            _decimal(fact["value"], f"{path}.fact_ids[{fact_index}].value")
            for fact_index, fact in enumerate(referenced_facts)
        ]
        _require_numeric_sequence_match(
            [left, right],
            fact_numbers,
            path=f"{path}.left/right",
        )
        operator = str(operation.get("operator") or "")
        if operator not in _COMPARE_OPERATORS:
            raise DerivationValidationError(f"{path}.operator is invalid: {operator!r}")
        result = operation.get("result")
        if not isinstance(result, bool):
            raise DerivationValidationError(f"{path}.result must be boolean")
        computed = _COMPARE_OPERATORS[operator](left, right)
        if result is not computed:
            raise DerivationValidationError(
                f"{path}.result={result} but {left} {operator} {right} is {computed}"
            )
    else:
        raise DerivationValidationError(f"{path}.kind is unsupported: {kind!r}")

    binding = _validate_answer_binding(
        operation.get("answer_binding"),
        computed=computed,
        kind=kind,
        answer=answer,
        path=f"{path}.answer_binding",
    )
    return (
        {
            **operation,
            "id": operation_id,
            "fact_ids": fact_ids,
            "answer_binding": binding,
        },
        computed,
    )


def _compute_arithmetic(
    kind: str,
    numbers: list[Decimal],
    operation: dict[str, Any],
    path: str,
) -> Decimal:
    if kind != "divide":
        if "rounding" in operation or "exact" in operation:
            raise DerivationValidationError(
                f"{path}.rounding/exact is only valid for divide"
            )
        computed = numbers[0]
        for number in numbers[1:]:
            if kind == "add":
                computed += number
            elif kind == "subtract":
                computed -= number
            else:
                computed *= number
        return computed

    if any(number == 0 for number in numbers[1:]):
        raise DerivationValidationError(f"{path} divides by zero")
    exact = operation.get("exact")
    rounding = operation.get("rounding")
    if exact is True and rounding is not None:
        raise DerivationValidationError(
            f"{path} divide must use either exact=true or rounding, not both"
        )
    if exact is not True and rounding is None:
        raise DerivationValidationError(
            f"{path} divide requires exact=true or a rounding contract"
        )
    if exact not in (None, True):
        raise DerivationValidationError(f"{path}.exact must be true when present")

    rational = Fraction(numbers[0])
    for number in numbers[1:]:
        rational /= Fraction(number)
    if exact is True:
        decimal_places = _terminating_decimal_places(rational.denominator)
        if decimal_places is None:
            raise DerivationValidationError(
                f"{path} exact division is non-terminating; provide rounding"
            )
        with localcontext() as context:
            context.prec = max(
                50,
                len(str(abs(rational.numerator))) + decimal_places + 20,
            )
            return Decimal(rational.numerator) / Decimal(rational.denominator)

    if not isinstance(rounding, dict) or set(rounding) != {
        "decimal_places",
        "mode",
    }:
        raise DerivationValidationError(
            f"{path}.rounding must contain decimal_places and mode"
        )
    decimal_places = rounding.get("decimal_places")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise DerivationValidationError(
            f"{path}.rounding.decimal_places must be an integer"
        )
    if decimal_places < 0:
        raise DerivationValidationError(
            f"{path}.rounding.decimal_places must be non-negative"
        )
    mode = str(rounding.get("mode") or "")
    if mode not in _ROUNDING_MODES:
        raise DerivationValidationError(
            f"{path}.rounding.mode must be one of {sorted(_ROUNDING_MODES)}"
        )
    with localcontext() as context:
        context.prec = max(
            50,
            decimal_places
            + len(str(abs(rational.numerator)))
            + len(str(abs(rational.denominator)))
            + 20,
        )
        quotient = Decimal(rational.numerator) / Decimal(rational.denominator)
        quantum = Decimal(1).scaleb(-decimal_places)
        return quotient.quantize(quantum, rounding=_ROUNDING_MODES[mode])


def _validate_answer_binding(
    binding: Any,
    *,
    computed: Any,
    kind: str,
    answer: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise DerivationValidationError(f"{path} must be an object")
    required = {"answer_path", "expected"}
    missing = sorted(required - set(binding))
    extra = sorted(set(binding) - required - {"answer_fragment"})
    if missing or extra:
        raise DerivationValidationError(
            f"{path} requires answer_path/expected and optional answer_fragment; "
            f"missing={missing}, extra={extra}"
        )
    answer_path = _non_empty_string(binding.get("answer_path"), f"{path}.answer_path")
    expected = binding["expected"]
    if not _operation_values_equal(kind, expected, computed):
        raise DerivationValidationError(
            f"{path}.expected={expected!r} does not equal computed result {computed!r}"
        )

    answer_value = _resolve_answer_path(answer, answer_path, path=f"{path}.answer_path")
    fragment = binding.get("answer_fragment")
    if fragment is not None and not isinstance(fragment, str):
        raise DerivationValidationError(f"{path}.answer_fragment must be a string")
    if isinstance(answer_value, str):
        if not isinstance(fragment, str) or not fragment:
            raise DerivationValidationError(
                f"{path}.answer_fragment is required for a string answer"
            )
        if fragment not in answer_value:
            raise DerivationValidationError(
                f"{path}.answer_fragment is not an exact substring of {answer_path}"
            )
        if not _fragment_contains_expected(fragment, computed, kind):
            raise DerivationValidationError(
                f"{path}.answer_fragment={fragment!r} does not express expected "
                f"result {computed!r}"
            )
    elif not _typed_answer_matches(answer_value, computed, kind):
        raise DerivationValidationError(
            f"{answer_path}={answer_value!r} does not exactly equal expected "
            f"result {computed!r}"
        )

    normalized = {
        "answer_path": answer_path,
        "expected": expected,
    }
    if fragment is not None:
        normalized["answer_fragment"] = fragment
    return normalized


def _validate_final_answer_bindings(
    raw_bindings: Any,
    *,
    query: Query,
    answer: dict[str, Any],
    facts_by_id: dict[str, dict[str, Any]],
    operation_results: dict[str, tuple[Any, str]],
) -> list[dict[str, Any]]:
    """Bind every emitted answer component to a sourced fact or operation.

    Operation-local ``answer_binding`` proves that an operation was computed
    correctly.  This top-level contract is deliberately separate: it also
    covers pure lookups (``operations=[]``), combined answer types, and table
    cells.  Consequently a syntactically valid final string cannot float free
    of the evidence-backed derivation.
    """

    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise DerivationValidationError(
            "derivation.answer_bindings must be a non-empty list"
        )

    validated: list[dict[str, Any]] = []
    for index, binding in enumerate(raw_bindings):
        path = f"derivation.answer_bindings[{index}]"
        if not isinstance(binding, dict):
            raise DerivationValidationError(f"{path} must be an object")
        required = {"answer_path", "source_type", "source_id"}
        missing = sorted(required - set(binding))
        extra = sorted(set(binding) - required - {"answer_fragment"})
        if missing or extra:
            raise DerivationValidationError(
                f"{path} requires answer_path/source_type/source_id and optional "
                f"answer_fragment; missing={missing}, extra={extra}"
            )

        answer_path = _non_empty_string(
            binding.get("answer_path"), f"{path}.answer_path"
        )
        source_type = _non_empty_string(
            binding.get("source_type"), f"{path}.source_type"
        )
        source_id = _non_empty_string(binding.get("source_id"), f"{path}.source_id")
        if source_type == "fact":
            fact = facts_by_id.get(source_id)
            if fact is None:
                raise DerivationValidationError(
                    f"{path} references unknown fact {source_id!r}"
                )
            source_value = fact["value"]
            source_kind = "fact"
        elif source_type == "operation":
            operation_result = operation_results.get(source_id)
            if operation_result is None:
                raise DerivationValidationError(
                    f"{path} references unknown operation {source_id!r}"
                )
            source_value, source_kind = operation_result
        else:
            raise DerivationValidationError(
                f"{path}.source_type must be 'fact' or 'operation'"
            )

        answer_value = _resolve_answer_path(
            answer, answer_path, path=f"{path}.answer_path"
        )
        fragment = binding.get("answer_fragment")
        if fragment is not None and not isinstance(fragment, str):
            raise DerivationValidationError(f"{path}.answer_fragment must be a string")
        _validate_source_value_in_answer(
            answer_value,
            source_value=source_value,
            source_kind=source_kind,
            answer_fragment=fragment,
            answer_path=answer_path,
            path=path,
        )
        normalized = {
            "answer_path": answer_path,
            "source_type": source_type,
            "source_id": source_id,
        }
        if fragment is not None:
            normalized["answer_fragment"] = fragment
        validated.append(normalized)

    _validate_answer_binding_coverage(query, answer, validated)
    return validated


def _validate_source_value_in_answer(
    answer_value: Any,
    *,
    source_value: Any,
    source_kind: str,
    answer_fragment: str | None,
    answer_path: str,
    path: str,
) -> None:
    if isinstance(answer_value, str):
        if not answer_fragment:
            raise DerivationValidationError(
                f"{path}.answer_fragment is required for a string answer"
            )
        if answer_fragment not in answer_value:
            raise DerivationValidationError(
                f"{path}.answer_fragment is not an exact substring of {answer_path}"
            )
        if source_kind == "fact":
            if not _fragment_contains_fact(answer_fragment, source_value):
                raise DerivationValidationError(
                    f"{path}.answer_fragment={answer_fragment!r} does not express "
                    f"sourced fact value {source_value!r}"
                )
        elif not _fragment_contains_expected(
            answer_fragment, source_value, source_kind
        ):
            raise DerivationValidationError(
                f"{path}.answer_fragment={answer_fragment!r} does not express "
                f"operation result {source_value!r}"
            )
        return

    if source_kind == "fact":
        matches = _fact_values_equal(answer_value, source_value)
    else:
        matches = _typed_answer_matches(answer_value, source_value, source_kind)
    if not matches:
        raise DerivationValidationError(
            f"{answer_path}={answer_value!r} does not exactly equal sourced "
            f"value {source_value!r}"
        )


def _fragment_contains_fact(fragment: str, value: Any) -> bool:
    if isinstance(value, bool):
        return _yes_no_polarity(fragment) is value
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return _decimal(value, "fact value") in _numbers_in_text(fragment)
    if isinstance(value, str):
        expected = _normalize_item(value)
        actual = _normalize_item(fragment)
        return _contains_normalized_token(actual, expected)
    return False


def _fact_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float, Decimal)) and isinstance(
        right, (int, float, Decimal)
    ):
        try:
            return _decimal(left, "answer value") == _decimal(right, "fact value")
        except DerivationValidationError:
            return False
    return left == right


def _validate_answer_binding_coverage(
    query: Query,
    answer: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> None:
    paths = {str(item["answer_path"]) for item in bindings}
    allowed_paths: set[str] = set()
    required_paths: set[str] = set()
    if "freeform" in query.answer_types:
        required_paths.add("answer.freeform.text")
    if "multiple_choice" in query.answer_types:
        multiple_choice_paths = {
            "answer.multiple_choice",
            "answer.multiple_choice.selected_option_text",
        }
        allowed_paths.update(multiple_choice_paths)
        if not multiple_choice_paths.intersection(paths):
            raise DerivationValidationError(
                "derivation.answer_bindings is missing the multiple-choice answer"
            )
        if "freeform" in query.answer_types:
            freeform_sources = {
                (str(item["source_type"]), str(item["source_id"]))
                for item in bindings
                if item["answer_path"] == "answer.freeform.text"
            }
            multiple_choice_sources = {
                (str(item["source_type"]), str(item["source_id"]))
                for item in bindings
                if item["answer_path"] in multiple_choice_paths
            }
            if not freeform_sources.intersection(multiple_choice_sources):
                raise DerivationValidationError(
                    "combined freeform and multiple-choice answers must share "
                    "at least one derivation source"
                )
    if "table" in query.answer_types:
        rows = answer["table"]["rows"]
        for row_index, row in enumerate(rows):
            row_path = f"answer.table.rows[{row_index}]"
            cell_paths = {f"{row_path}.{column}" for column in row}
            allowed_paths.add(row_path)
            allowed_paths.update(cell_paths)
            if row_path not in paths and not cell_paths.issubset(paths):
                missing = sorted(cell_paths - paths)
                raise DerivationValidationError(
                    f"derivation.answer_bindings for table row {row_index} need "
                    f"a row binding or every cell; missing={missing}"
                )
    allowed_paths.update(required_paths)
    missing_required = sorted(required_paths - paths)
    if missing_required:
        raise DerivationValidationError(
            "derivation.answer_bindings is missing required answer paths: "
            f"{missing_required}"
        )
    unknown_paths = sorted(paths - allowed_paths)
    if unknown_paths:
        raise DerivationValidationError(
            "derivation.answer_bindings contains unsupported answer paths: "
            f"{unknown_paths}"
        )


def _resolve_answer_path(
    answer: dict[str, Any], answer_path: str, *, path: str
) -> Any:
    if not answer_path.startswith("answer."):
        raise DerivationValidationError(f"{path} must start with 'answer.'")
    relative = answer_path.removeprefix("answer.")
    if relative == "multiple_choice":
        multiple_choice = answer.get("multiple_choice")
        if not isinstance(multiple_choice, dict):
            raise DerivationValidationError(f"{answer_path} does not exist")
        if "selected_option_text" not in multiple_choice:
            raise DerivationValidationError(
                f"{answer_path}.selected_option_text does not exist"
            )
        return multiple_choice["selected_option_text"]

    table_match = re.fullmatch(r"table\.rows\[(\d+)\](?:\.(.+))?", relative)
    if table_match:
        table = answer.get("table")
        rows = table.get("rows") if isinstance(table, dict) else None
        row_index = int(table_match.group(1))
        if not isinstance(rows, list) or row_index >= len(rows):
            raise DerivationValidationError(f"{answer_path} does not exist")
        row = rows[row_index]
        if not isinstance(row, dict):
            raise DerivationValidationError(f"{answer_path} does not exist")
        column = table_match.group(2)
        if column is None:
            return row
        if column not in row:
            raise DerivationValidationError(f"{answer_path} does not exist")
        return row[column]

    current: Any = answer
    for component in relative.split("."):
        if not isinstance(current, dict) or component not in current:
            raise DerivationValidationError(f"{answer_path} does not exist")
        current = current[component]
    return current


def _operation_values_equal(kind: str, left: Any, right: Any) -> bool:
    if kind in {"add", "subtract", "multiply", "divide"}:
        try:
            return _decimal(left, "answer_binding.expected") == _decimal(
                right, "operation.result"
            )
        except DerivationValidationError:
            return False
    if kind == "count":
        return (
            not isinstance(left, bool)
            and isinstance(left, int)
            and left == right
        )
    if kind == "compare":
        return isinstance(left, bool) and left is right
    return isinstance(left, str) and left.strip() == str(right).strip()


def _typed_answer_matches(answer_value: Any, computed: Any, kind: str) -> bool:
    if kind in {"add", "subtract", "multiply", "divide"}:
        if isinstance(answer_value, bool):
            return False
        try:
            return _decimal(answer_value, "answer value") == computed
        except DerivationValidationError:
            return False
    if kind == "count":
        return (
            not isinstance(answer_value, bool)
            and isinstance(answer_value, int)
            and answer_value == computed
        )
    if kind == "compare":
        return isinstance(answer_value, bool) and answer_value is computed
    return answer_value == computed


def _fragment_contains_expected(fragment: str, computed: Any, kind: str) -> bool:
    if kind in {"add", "subtract", "multiply", "divide", "count"}:
        expected_number = _decimal(computed, "computed result")
        return expected_number in _numbers_in_text(fragment)
    if kind == "compare":
        return _yes_no_polarity(fragment) is computed

    expected_label = _normalize_item(computed)
    normalized_fragment = _normalize_item(fragment)
    return _contains_normalized_token(normalized_fragment, expected_label)


def _contains_normalized_token(text: str, token: str) -> bool:
    """Match a normalized value as a token/phrase, never inside another word.

    Bindings may point at a concise fragment inside a longer answer sentence,
    but a source value such as ``42`` must not validate ``142`` and an argmax
    label such as ``Flint`` must not validate ``SuperFlint``.
    """

    if not token:
        return False
    return bool(re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", text))


def _numbers_in_text(text: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for match in _NUMBER_IN_TEXT.findall(text.replace(",", "")):
        try:
            numbers.add(Decimal(match))
        except InvalidOperation:
            continue
    for match in _NUMBER_WORD_RE.findall(text):
        numbers.add(Decimal(_NUMBER_WORDS[match.casefold()]))
    return numbers


def _require_numeric_sequence_match(
    actual: list[Decimal], expected: list[Decimal], *, path: str
) -> None:
    if actual != expected:
        raise DerivationValidationError(
            f"{path}={actual} do not match referenced fact values {expected}"
        )


def _flatten_count_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten_count_value(item))
        return output
    return [value]


def _fact_candidates(
    facts: list[dict[str, Any]], path: str
) -> list[tuple[str, Decimal]]:
    candidates: list[tuple[str, Decimal]] = []
    for fact_index, fact in enumerate(facts):
        value = fact["value"]
        if not isinstance(value, dict) or set(value) != {"label", "value"}:
            raise DerivationValidationError(
                f"{path}.fact_ids[{fact_index}].value must contain label and value"
            )
        label = _non_empty_string(
            value.get("label"), f"{path}.fact_ids[{fact_index}].value.label"
        )
        candidates.append(
            (
                label,
                _decimal(
                    value.get("value"),
                    f"{path}.fact_ids[{fact_index}].value.value",
                ),
            )
        )
    normalized_labels = [_normalize_item(label) for label, _ in candidates]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise DerivationValidationError(
            f"{path}.fact_ids contain duplicate candidate labels"
        )
    return candidates


def _candidate_map(
    candidates: list[tuple[str, Decimal]],
) -> dict[str, Decimal]:
    return {_normalize_item(label): value for label, value in candidates}


def _terminating_decimal_places(denominator: int) -> int | None:
    denominator = abs(denominator)
    powers: list[int] = []
    for factor in (2, 5):
        power = 0
        while denominator and denominator % factor == 0:
            denominator //= factor
            power += 1
        powers.append(power)
    return max(powers) if denominator == 1 else None


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DerivationValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _validate_answer_types(query: Query, answer: dict[str, Any]) -> None:
    if set(answer) != set(query.answer_types):
        raise DerivationValidationError(
            "answer keys must exactly match answer_types: "
            f"expected {sorted(query.answer_types)}, got {sorted(answer)}"
        )
    if "freeform" in query.answer_types:
        freeform = answer.get("freeform")
        if not isinstance(freeform, dict) or set(freeform) != {"text"}:
            raise DerivationValidationError("freeform must contain only text")
        if not isinstance(freeform["text"], str) or not freeform["text"].strip():
            raise DerivationValidationError("freeform.text must be non-empty")
    if "multiple_choice" in query.answer_types:
        multiple_choice = answer.get("multiple_choice")
        expected = {"label", "selected_option_text"}
        if not isinstance(multiple_choice, dict) or set(multiple_choice) != expected:
            raise DerivationValidationError(
                "multiple_choice must contain exactly label and selected_option_text"
            )
    if "table" in query.answer_types:
        table = answer.get("table")
        if not isinstance(table, dict) or set(table) != {"rows"}:
            raise DerivationValidationError("table must contain only rows")
        table["rows"] = validate_table_rows(query, table.get("rows"))


def _validate_multiple_choice(
    query: Query,
    answer: dict[str, Any],
    final_answer: str,
    *,
    require_final_match: bool,
) -> None:
    if "multiple_choice" not in query.answer_types:
        return
    if not query.options:
        raise DerivationValidationError(
            "multiple-choice query has no released option mapping"
        )
    multiple_choice = answer["multiple_choice"]
    label = str(multiple_choice["label"] or "").strip().upper()
    selected_text = multiple_choice["selected_option_text"]
    if label not in query.options:
        raise DerivationValidationError(
            f"multiple-choice label {label!r} is not one of {list(query.options)}"
        )
    if not isinstance(selected_text, str) or selected_text != query.options[label]:
        raise DerivationValidationError(
            "selected_option_text must exactly equal the released text for "
            f"label {label}"
        )
    if require_final_match and final_answer != selected_text:
        raise DerivationValidationError(
            "final_semantic_answer must exactly equal selected_option_text"
        )
    multiple_choice["label"] = label


def _validate_cell_type(value: Any, declared_type: str, *, path: str) -> None:
    if value is None:
        return
    if declared_type == "string":
        if not isinstance(value, str):
            raise DerivationValidationError(f"{path} must be a JSON string")
        return
    if declared_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DerivationValidationError(f"{path} must be a JSON number")
        if isinstance(value, float) and not math.isfinite(value):
            raise DerivationValidationError(f"{path} must be finite")
        return
    if declared_type == "boolean":
        if not isinstance(value, bool):
            raise DerivationValidationError(f"{path} must be JSON true or false")
        return
    if declared_type == "null":
        if value is not None:
            raise DerivationValidationError(f"{path} must be JSON null")
        return
    raise DerivationValidationError(f"{path} has unsupported schema type {declared_type!r}")


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float, str)):
        raise DerivationValidationError(f"{path} must be numeric")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise DerivationValidationError(f"{path} is not a decimal number: {value!r}") from None
    if not number.is_finite():
        raise DerivationValidationError(f"{path} must be finite")
    return number


def _yes_no_polarity(text: str) -> bool | None:
    normalized = _normalize_item(text)
    positive = bool(re.search(r"\b(?:yes|true)\b", normalized))
    negative = bool(re.search(r"\b(?:no|false)\b", normalized))
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    return None


def _normalize_item(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _normalize_row_key(value: Any) -> str:
    # This is the pinned official evaluator's row-alignment normalization.
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    return re.sub(r"\s+", " ", text)
