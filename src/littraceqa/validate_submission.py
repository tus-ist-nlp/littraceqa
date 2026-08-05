#!/usr/bin/env python3
"""Gold-free lint for LitTraceQA submission files.

scripts/evaluate.py needs gold labels, so it cannot check the hidden-test
submission. This linter validates a prediction file against only the released
inputs file and converts silent zero-score failure modes into pre-submission
errors: unparsable lines, missing/duplicate query_ids, answer objects that do
not match the declared answer_types, out-of-range multiple-choice letters
(strict mode requires A-D when no option mapping is known),
empty freeform answers, empty table rows on table-typed questions (guaranteed
zero table metrics), evidence items missing the fields the evaluator keys on
(it silently drops items without paper_id/source_type/page, and table or
figure items without their object id), and empty paper lists.

The default mode remains backwards compatible with existing repository output
aliases and richer locator objects. ``--strict-official-shape`` additionally
requires the exact, gold-free shape emitted by the pairwise reader and checks
paper IDs against the released metadata.

Run this as the mandatory final gate before submitting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from .common import ROOT, Record, read_jsonl


OFFICIAL_SOURCE_TYPES = {
    "text_span",
    "table",
    "figure",
    "citation_context",
    "equation_algorithm",
}
FREEFORM_WARN_CHARS = 200
TOP_LEVEL_KEYS = {"query_id", "gold_papers", "evidence", "answer"}
SINGLE_LETTER_RE = re.compile(r"[A-Z]")


class CheckCounter:
    """Named failure/warning counters with example query_ids."""

    def __init__(self) -> None:
        self.failures: Counter[str] = Counter()
        self.warnings: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}
        self.order: list[str] = []

    def register(self, name: str) -> None:
        if name not in self.order:
            self.order.append(name)
        self.failures.setdefault(name, 0)

    def fail(self, name: str, example: str = "") -> None:
        self.register(name)
        self.failures[name] += 1
        if example:
            examples = self.examples.setdefault(name, [])
            if len(examples) < 5:
                examples.append(example)

    def warn(self, name: str, example: str = "") -> None:
        if name not in self.order:
            self.order.append(name)
        self.warnings[name] += 1
        if example:
            examples = self.examples.setdefault(name, [])
            if len(examples) < 5:
                examples.append(example)

    @property
    def failed(self) -> bool:
        return any(count for count in self.failures.values())


def read_predictions(path: Path, checks: CheckCounter) -> list[Record]:
    """Parse the prediction file line by line, counting bad lines."""
    records: list[Record] = []
    checks.register("prediction line is valid JSON object")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                checks.fail("prediction line is valid JSON object", f"line {line_number}")
                continue
            if not isinstance(record, dict):
                checks.fail("prediction line is valid JSON object", f"line {line_number}")
                continue
            records.append(record)
    return records


def load_option_keys(path: Optional[Path]) -> dict[str, tuple[str, ...]]:
    """Map query_id -> valid multiple-choice letter keys.

    Accepts either an options sidecar with a top-level ``options`` object or a
    gold-shaped file with ``answer.multiple_choice.options``.
    """
    if path is None:
        return {}
    keys: dict[str, tuple[str, ...]] = {}
    for record in read_jsonl(path):
        query_id = str(record.get("query_id") or "")
        if not query_id:
            continue
        options = record.get("options")
        if not isinstance(options, dict):
            answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
            mc = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
            options = mc.get("options")
        if isinstance(options, dict) and options:
            keys[query_id] = tuple(str(key).upper() for key in options)
    return keys


def predicted_letter(record: Record, *, strict: bool = False) -> str:
    """Read the letter the official evaluator reads."""
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    mc = answer.get("multiple_choice")
    if isinstance(mc, dict):
        raw = mc.get("gold")
        if not strict:
            raw = raw or mc.get("answer") or mc.get("predicted_answer_id")
        return str(raw or "").strip().upper()
    return ""


def check_answer(
    record: Record,
    answer_types: set[str],
    table_schema: list[Record],
    option_keys: dict[str, tuple[str, ...]],
    checks: CheckCounter,
    *,
    strict: bool,
) -> None:
    query_id = str(record.get("query_id") or "")
    answer = record.get("answer")
    if not isinstance(answer, dict):
        checks.fail("answer object present", query_id)
        return
    if set(answer) != answer_types:
        checks.fail("answer keys match declared answer_types", query_id)

    if "multiple_choice" in answer_types:
        mc = answer.get("multiple_choice")
        if strict and (not isinstance(mc, dict) or set(mc) != {"gold"}):
            checks.fail("multiple_choice object exact", query_id)
        letter = predicted_letter(record, strict=strict)
        valid = option_keys.get(query_id)
        if strict:
            letter_valid = letter in {"A", "B", "C", "D"}
        else:
            letter_valid = bool(SINGLE_LETTER_RE.fullmatch(letter))
        if not letter_valid or (valid is not None and letter not in valid):
            checks.fail("multiple_choice letter within valid keys", query_id)

    if "freeform" in answer_types:
        freeform = answer.get("freeform")
        if strict and (not isinstance(freeform, dict) or set(freeform) != {"text"}):
            checks.fail("freeform object exact", query_id)
        raw_text = freeform.get("text") if isinstance(freeform, dict) else None
        if strict and not isinstance(raw_text, str):
            checks.fail("freeform text is string", query_id)
        if strict:
            text = raw_text if isinstance(raw_text, str) else ""
        else:
            text = str(raw_text or "")
        if not text.strip():
            checks.fail("freeform text non-empty", query_id)
        elif len(text) > FREEFORM_WARN_CHARS:
            checks.warn(
                f"freeform text longer than {FREEFORM_WARN_CHARS} chars (gold is short-extractive)",
                query_id,
            )

    if "table" in answer_types:
        table = answer.get("table")
        table_keys = set(table) if isinstance(table, dict) else set()
        if strict and table_keys not in ({"rows"}, {"rows", "schema"}):
            checks.fail("table object exact", query_id)
        if strict and isinstance(table, dict) and "schema" in table:
            if table.get("schema") != table_schema:
                checks.fail("table schema matches input", query_id)
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(rows, list) or not rows:
            # Empty rows guarantee zero table metrics, so this is an error.
            checks.fail("table rows non-empty", query_id)
        elif strict:
            check_table_rows(query_id, rows, table_schema, checks)


def check_table_rows(
    query_id: str,
    rows: list[Any],
    table_schema: list[Record],
    checks: CheckCounter,
) -> None:
    columns = [
        str(column.get("name"))
        for column in table_schema
        if isinstance(column, dict) and column.get("name")
    ]
    row_keys = [
        str(column.get("name"))
        for column in table_schema
        if isinstance(column, dict)
        and column.get("name")
        and column.get("is_row_key")
    ]
    types = {
        str(column.get("name")): str(column.get("type") or "")
        for column in table_schema
        if isinstance(column, dict) and column.get("name")
    }
    seen_keys: set[tuple[str, ...]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(columns):
            checks.fail("table rows match schema columns", query_id)
            continue
        valid_types = True
        for column in columns:
            value = row.get(column)
            declared_type = types.get(column)
            if value is None:
                continue
            if declared_type == "string":
                valid = isinstance(value, str)
            elif declared_type == "number":
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                )
            elif declared_type == "boolean":
                valid = isinstance(value, bool)
            else:
                valid = False
            valid_types = valid_types and valid
        if not valid_types:
            checks.fail("table cell types match schema", query_id)
        if any(row.get(column) in (None, "") for column in row_keys):
            checks.fail("table row keys non-empty", query_id)
            continue
        key = tuple(str(row.get(column)) for column in (row_keys or columns))
        if key in seen_keys:
            checks.fail("table row keys duplicate-free", query_id)
        seen_keys.add(key)


def check_evidence(
    record: Record,
    submitted_papers: set[str],
    canonical_papers: set[str],
    checks: CheckCounter,
    *,
    strict: bool,
) -> None:
    query_id = str(record.get("query_id") or "")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        checks.fail("evidence list non-empty", query_id)
        return
    for item in evidence:
        if not isinstance(item, dict):
            checks.fail("evidence item is an object", query_id)
            continue
        if strict and set(item) != {"paper_id", "source_type", "locator"}:
            checks.fail("evidence item keys exact", query_id)
        if not str(item.get("paper_id") or "").strip():
            checks.fail("evidence item has paper_id", query_id)
        paper_id = str(item.get("paper_id") or "").strip()
        if strict and paper_id and paper_id not in submitted_papers:
            checks.fail("evidence paper is submitted", query_id)
        if strict and canonical_papers and paper_id and paper_id not in canonical_papers:
            checks.fail("paper ids are canonical", query_id)
        source_type = str(item.get("source_type") or "").strip()
        if source_type not in OFFICIAL_SOURCE_TYPES:
            checks.fail("evidence source_type in official values", query_id)
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        page = locator.get("page")
        if strict:
            page_valid = not isinstance(page, bool) and isinstance(page, int) and page >= 1
        else:
            page_valid = bool(str(page if page is not None else "").strip())
        if not page_valid:
            checks.fail("evidence locator has page", query_id)
        allowed_locator_keys = {"page"}
        if source_type == "table":
            table_id = locator.get("table_id")
            if not str(table_id or "").strip():
                checks.fail("table evidence has locator.table_id", query_id)
            allowed_locator_keys.add("table_id")
        if source_type == "figure":
            figure_id = locator.get("figure_id")
            if not str(figure_id or "").strip():
                checks.fail("figure evidence has locator.figure_id", query_id)
            allowed_locator_keys.add("figure_id")
        if strict and set(locator) != allowed_locator_keys:
            checks.fail("evidence locator is coarse official shape", query_id)


def check_papers(
    record: Record,
    canonical_papers: set[str],
    checks: CheckCounter,
    *,
    strict: bool,
) -> set[str]:
    query_id = str(record.get("query_id") or "")
    papers = record.get("gold_papers")
    if not strict:
        papers = papers or record.get("papers")
    output: set[str] = set()
    if not isinstance(papers, list) or not papers:
        checks.fail("papers list non-empty", query_id)
        return output
    for item in papers:
        if strict and (not isinstance(item, dict) or set(item) != {"paper_id"}):
            checks.fail("paper entry shape exact", query_id)
            continue
        paper_id = item.get("paper_id") if isinstance(item, dict) else item
        if not str(paper_id or "").strip():
            checks.fail("paper entry has paper_id", query_id)
            continue
        paper_id = str(paper_id).strip()
        if strict and paper_id in output:
            checks.fail("paper entries duplicate-free", query_id)
        output.add(paper_id)
        if strict and canonical_papers and paper_id not in canonical_papers:
            checks.fail("paper ids are canonical", query_id)
    return output


def validate(
    inputs: list[Record],
    predictions: list[Record],
    option_keys: dict[str, tuple[str, ...]],
    checks: CheckCounter,
    *,
    strict: bool = False,
    canonical_papers: set[str] | None = None,
) -> None:
    canonical_papers = canonical_papers or set()
    base_checks = (
        "query_id sets identical",
        "query_ids duplicate-free",
        "answer object present",
        "answer keys match declared answer_types",
        "multiple_choice letter within valid keys",
        "freeform text non-empty",
        "table rows non-empty",
        "evidence list non-empty",
        "evidence item is an object",
        "evidence item has paper_id",
        "evidence source_type in official values",
        "evidence locator has page",
        "table evidence has locator.table_id",
        "figure evidence has locator.figure_id",
        "papers list non-empty",
        "paper entry has paper_id",
    )
    strict_checks = (
        "top-level keys exact",
        "multiple_choice object exact",
        "freeform object exact",
        "freeform text is string",
        "table object exact",
        "table schema matches input",
        "table rows match schema columns",
        "table cell types match schema",
        "table row keys non-empty",
        "table row keys duplicate-free",
        "evidence item keys exact",
        "evidence locator is coarse official shape",
        "evidence paper is submitted",
        "paper entry shape exact",
        "paper entries duplicate-free",
        "paper ids are canonical",
    )
    for name in base_checks + (strict_checks if strict else ()):
        checks.register(name)

    if strict and not canonical_papers:
        raise ValueError("strict validation requires canonical paper metadata")

    input_by_id = {str(record.get("query_id") or ""): record for record in inputs}
    input_by_id.pop("", None)

    pred_ids = [str(record.get("query_id") or "") for record in predictions]
    for query_id, count in Counter(pred_ids).items():
        if count > 1:
            checks.fail("query_ids duplicate-free", f"{query_id} x{count}")
    missing = sorted(set(input_by_id) - set(pred_ids))
    extra = sorted(set(pred_ids) - set(input_by_id) - {""})
    for query_id in missing:
        checks.fail("query_id sets identical", f"missing {query_id}")
    for query_id in extra:
        checks.fail("query_id sets identical", f"extra {query_id}")

    for record in predictions:
        query_id = str(record.get("query_id") or "")
        if strict and set(record) != TOP_LEVEL_KEYS:
            checks.fail("top-level keys exact", query_id)
        sample = input_by_id.get(query_id)
        if sample is None:
            continue
        answer_types = set(sample.get("answer_types") or [])
        table_schema = sample.get("table_schema") or []
        check_answer(
            record,
            answer_types,
            table_schema,
            option_keys,
            checks,
            strict=strict,
        )
        submitted_papers = check_papers(
            record, canonical_papers, checks, strict=strict
        )
        check_evidence(
            record,
            submitted_papers,
            canonical_papers,
            checks,
            strict=strict,
        )


def print_summary(checks: CheckCounter) -> None:
    width = max(len(name) for name in checks.order) if checks.order else 20
    print(f"{'check':<{width}}  {'count':>5}  status")
    print("-" * (width + 15))
    for name in checks.order:
        if name in checks.warnings:
            count = checks.warnings[name]
            status = "WARN" if count else "OK"
        else:
            count = checks.failures.get(name, 0)
            status = "FAIL" if count else "OK"
        print(f"{name:<{width}}  {count:>5}  {status}")
        if count and name in checks.examples:
            print(f"{'':<{width}}         e.g. {', '.join(checks.examples[name])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gold-free format lint for LitTraceQA prediction files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "data" / "validation_inputs.jsonl",
        help="Released inputs JSONL (test or validation inputs).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction JSONL file to lint.",
    )
    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=ROOT / "data" / "paper_metadata.jsonl",
        help="Canonical paper metadata JSONL used by --strict-official-shape.",
    )
    parser.add_argument(
        "--strict-official-shape",
        action="store_true",
        help=(
            "Require the exact pairwise-reader submission schema in addition "
            "to the backwards-compatible zero-score checks"
        ),
    )
    parser.add_argument(
        "--options-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with per-question multiple-choice options "
            "(top-level 'options' or gold-shaped answer.multiple_choice.options). "
            "Questions without known options accept A-D only."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checks = CheckCounter()

    inputs = read_jsonl(args.inputs)
    predictions = read_predictions(args.predictions, checks)
    option_keys = load_option_keys(args.options_file)
    canonical_papers: set[str] = set()
    if args.strict_official_shape:
        canonical_papers = {
            str(record.get("paper_id") or "")
            for record in read_jsonl(args.paper_metadata)
            if record.get("paper_id")
        }
        if not canonical_papers:
            print(
                f"canonical paper metadata is empty: {args.paper_metadata}",
                file=sys.stderr,
            )
            return 1

    validate(
        inputs,
        predictions,
        option_keys,
        checks,
        strict=args.strict_official_shape,
        canonical_papers=canonical_papers,
    )
    print(f"inputs: {len(inputs)} questions ({args.inputs})")
    print(f"predictions: {len(predictions)} parsed lines ({args.predictions})")
    print_summary(checks)

    if checks.failed:
        print("RESULT: FAIL - fix the issues above before submitting", file=sys.stderr)
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
