#!/usr/bin/env python3
"""Per-question diff of two LitTraceQA prediction files.

With only 55 dev questions, aggregate metric moves smaller than ~2 questions
are noise. This tool scores both runs with the official scripts/evaluate.py
logic (imported, not copied), prints per-metric aggregates and deltas, and
lists exactly which query_ids changed on each metric, so regressions hidden
inside a flat aggregate become visible.

Metric families per query: paper_f1, evidence_f1, mc_correct (multiple-choice
questions only), freeform_correct (freeform questions only), table_row_f1
(table questions only).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from .common import ROOT, Record, read_jsonl


METRIC_NAMES = (
    "paper_f1",
    "evidence_f1",
    "mc_correct",
    "freeform_correct",
    "table_row_f1",
)
DEFAULT_EVALUATE_PATH = ROOT / "scripts" / "evaluate.py"


def load_evaluate_module(path: Path) -> Any:
    """Import the official evaluator via importlib without copying its logic."""
    if not path.is_file():
        raise FileNotFoundError(f"official evaluator not found: {path}")
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("littraceqa_official_evaluate", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load evaluator from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def per_query_metrics(evaluate: Any, gold: Record, pred: Record) -> dict[str, float]:
    """Score one prediction against one gold record with official functions."""
    metrics: dict[str, float] = {}
    metrics["paper_f1"] = evaluate.prf(
        evaluate.paper_id_set(gold), evaluate.paper_id_set(pred)
    )[2]
    metrics["evidence_f1"] = evaluate.prf(
        evaluate.evidence_set(gold), evaluate.evidence_set(pred)
    )[2]
    answer_types = set(gold.get("answer_types") or [])
    if "multiple_choice" in answer_types:
        metrics["mc_correct"] = float(
            evaluate.multiple_choice_prediction(pred) == evaluate.multiple_choice_gold(gold)
        )
    if "freeform" in answer_types:
        metrics["freeform_correct"] = float(
            evaluate.freeform_prediction(pred) == evaluate.freeform_gold(gold)
        )
    if "table" in answer_types:
        metrics["table_row_f1"] = float(evaluate.table_metrics(gold, pred)["row_f1"])
    return metrics


def score_run(
    evaluate: Any,
    gold_by_id: dict[str, Record],
    predictions: list[Record],
) -> dict[str, dict[str, float]]:
    pred_by_id = {str(record.get("query_id") or ""): record for record in predictions}
    pred_by_id.pop("", None)
    scores: dict[str, dict[str, float]] = {}
    for query_id, gold in gold_by_id.items():
        scores[query_id] = per_query_metrics(evaluate, gold, pred_by_id.get(query_id, {}))
    return scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_value(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def report(
    gold_by_id: dict[str, Record],
    scores_a: dict[str, dict[str, float]],
    scores_b: dict[str, dict[str, float]],
    *,
    label_a: str,
    label_b: str,
) -> None:
    print(f"run A: {label_a}")
    print(f"run B: {label_b}")
    print(f"questions: {len(gold_by_id)}")
    print()
    print(f"{'metric':<18} {'run A':>7} {'run B':>7} {'delta':>8}  changed")
    print("-" * 62)

    for metric in METRIC_NAMES:
        query_ids = [
            query_id for query_id in gold_by_id if metric in scores_a[query_id]
        ]
        if not query_ids:
            print(f"{metric:<18} {'n/a':>7} {'n/a':>7} {'':>8}  0 questions")
            continue
        values_a = [scores_a[query_id][metric] for query_id in query_ids]
        values_b = [scores_b[query_id][metric] for query_id in query_ids]
        changed = [
            query_id
            for query_id in query_ids
            if abs(scores_a[query_id][metric] - scores_b[query_id][metric]) > 1e-9
        ]
        delta = mean(values_b) - mean(values_a)
        print(
            f"{metric:<18} {mean(values_a):>7.3f} {mean(values_b):>7.3f} "
            f"{delta:>+8.3f}  {len(changed)}/{len(query_ids)} questions"
        )
        for query_id in changed:
            value_a = scores_a[query_id][metric]
            value_b = scores_b[query_id][metric]
            direction = "+" if value_b > value_a else "-"
            print(f"    {direction} {query_id}: {value_a:.3f} -> {value_b:.3f}")

    print()
    print(
        "note: with this few questions, treat aggregate moves worth fewer than "
        "~2 flipped questions as noise; judge changes by the per-query flips above."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-question metric diff of two prediction files (official scoring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "validation.jsonl",
        help="Gold JSONL with answers and evidence.",
    )
    parser.add_argument("--pred-a", type=Path, required=True, help="Baseline prediction JSONL.")
    parser.add_argument("--pred-b", type=Path, required=True, help="Candidate prediction JSONL.")
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=DEFAULT_EVALUATE_PATH,
        help="Path to the official evaluate.py.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate = load_evaluate_module(args.evaluate_script)

    gold_records = read_jsonl(args.gold)
    gold_by_id = {
        str(record.get("query_id") or ""): record for record in gold_records
    }
    gold_by_id.pop("", None)

    scores_a = score_run(evaluate, gold_by_id, read_jsonl(args.pred_a))
    scores_b = score_run(evaluate, gold_by_id, read_jsonl(args.pred_b))
    report(
        gold_by_id,
        scores_a,
        scores_b,
        label_a=str(args.pred_a),
        label_b=str(args.pred_b),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
