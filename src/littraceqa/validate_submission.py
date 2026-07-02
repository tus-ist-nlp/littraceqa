#!/usr/bin/env python3
"""Gold-free lint for LitTraceQA submission files.

scripts/evaluate.py needs gold labels, so it cannot check the hidden-test
submission. This linter validates a prediction file against only the released
inputs file and converts silent zero-score failure modes into pre-submission
errors: unparsable lines, missing/duplicate query_ids, answer objects that do
not match the declared answer_types, out-of-range multiple-choice letters
(when no options are known for a question, any single A-Z letter is accepted,
matching run_rag's pass-through, but empty/multi-character strings still fail),
empty freeform answers, empty table rows on table-typed questions (guaranteed
zero table metrics), evidence items missing the fields the evaluator keys on
(it silently drops items without paper_id/source_type/page, and table or
figure items without their object id), and empty paper lists.

Run this as the mandatory final gate before submitting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from .common import ROOT, Record, read_jsonl


OFFICIAL_SOURCE_TYPES = {
    "text_span",
    "table",
    "figure",
    "citation_context",
    "equation_algorithm",
}
SINGLE_LETTER_RE = re.compile(r"[A-Z]")
FREEFORM_WARN_CHARS = 200


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


def predicted_letter(record: Record) -> str:
    """Read the letter the official evaluator reads."""
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    mc = answer.get("multiple_choice")
    if isinstance(mc, dict):
        raw = mc.get("gold") or mc.get("answer") or mc.get("predicted_answer_id")
        return str(raw or "").strip().upper()
    return ""


def check_answer(
    record: Record,
    answer_types: set[str],
    option_keys: dict[str, tuple[str, ...]],
    checks: CheckCounter,
) -> None:
    query_id = str(record.get("query_id") or "")
    answer = record.get("answer")
    if not isinstance(answer, dict):
        checks.fail("answer object present", query_id)
        return
    if set(answer) != answer_types:
        checks.fail("answer keys match declared answer_types", query_id)

    if "multiple_choice" in answer_types:
        letter = predicted_letter(record)
        valid = option_keys.get(query_id)
        if valid is not None:
            if letter not in valid:
                checks.fail("multiple_choice letter within valid keys", query_id)
        elif not SINGLE_LETTER_RE.fullmatch(letter):
            # No options known for this question (e.g. hidden test set without
            # an options file): mirror run_rag's pass-through and accept any
            # single A-Z letter, but reject empty/multi-character strings.
            checks.fail("multiple_choice letter within valid keys", query_id)

    if "freeform" in answer_types:
        freeform = answer.get("freeform")
        text = str(freeform.get("text") or "") if isinstance(freeform, dict) else ""
        if not text.strip():
            checks.fail("freeform text non-empty", query_id)
        elif len(text) > FREEFORM_WARN_CHARS:
            checks.warn(
                f"freeform text longer than {FREEFORM_WARN_CHARS} chars (gold is short-extractive)",
                query_id,
            )

    if "table" in answer_types:
        table = answer.get("table")
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(rows, list) or not rows:
            # Empty rows guarantee zero table metrics, so this is an error.
            checks.fail("table rows non-empty", query_id)


def check_evidence(record: Record, checks: CheckCounter) -> None:
    query_id = str(record.get("query_id") or "")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        checks.fail("evidence list non-empty", query_id)
        return
    for item in evidence:
        if not isinstance(item, dict):
            checks.fail("evidence item is an object", query_id)
            continue
        if not str(item.get("paper_id") or "").strip():
            checks.fail("evidence item has paper_id", query_id)
        source_type = str(item.get("source_type") or "").strip()
        if source_type not in OFFICIAL_SOURCE_TYPES:
            checks.fail("evidence source_type in official values", query_id)
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        if not str(locator.get("page", "")).strip():
            checks.fail("evidence locator has page", query_id)
        if source_type == "table" and not str(locator.get("table_id", "")).strip():
            checks.fail("table evidence has locator.table_id", query_id)
        if source_type == "figure" and not str(locator.get("figure_id", "")).strip():
            checks.fail("figure evidence has locator.figure_id", query_id)


def check_papers(record: Record, checks: CheckCounter) -> None:
    query_id = str(record.get("query_id") or "")
    papers = record.get("gold_papers") or record.get("papers")
    if not isinstance(papers, list) or not papers:
        checks.fail("papers list non-empty", query_id)
        return
    for item in papers:
        paper_id = item.get("paper_id") if isinstance(item, dict) else item
        if not str(paper_id or "").strip():
            checks.fail("paper entry has paper_id", query_id)


def validate(
    inputs: list[Record],
    predictions: list[Record],
    option_keys: dict[str, tuple[str, ...]],
    checks: CheckCounter,
) -> None:
    for name in (
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
    ):
        checks.register(name)

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
        sample = input_by_id.get(query_id)
        if sample is None:
            continue
        answer_types = set(sample.get("answer_types") or [])
        check_answer(record, answer_types, option_keys, checks)
        check_evidence(record, checks)
        check_papers(record, checks)


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
        "--options-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with per-question multiple-choice options "
            "(top-level 'options' or gold-shaped answer.multiple_choice.options). "
            "Questions without known options accept any single A-Z letter."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checks = CheckCounter()

    inputs = read_jsonl(args.inputs)
    predictions = read_predictions(args.predictions, checks)
    option_keys = load_option_keys(args.options_file)

    validate(inputs, predictions, option_keys, checks)
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
