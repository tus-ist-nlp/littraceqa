#!/usr/bin/env python3
"""Score submitted paper sets against the official gold, without running the agent.

``scripts/eval_retrieval.py`` reports Recall@k, which measures whether the gold
papers reached the reading agent. It does not measure the submitted score: the
official metric compares the submitted set with the gold set and macro-averages
F1, so the number of papers submitted dominates it.

This script takes a retrieval output, applies a selection strategy to each
ranked candidate list, and reports paper precision/recall/F1 macro. Because it
reuses a finished retrieval run, a strategy can be evaluated in a second with
no GPU and no LLM.

Example:
    uv run python scripts/eval_paper_selection.py \\
      --retrieval evaluations/post_lane_removal.json \\
      --gold data/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from littraceqa.di_pipeline.evaluation.paper_selection import (
    load_gold_paper_sets,
    score_selection,
)
from littraceqa.di_pipeline.select.cardinality import (
    expected_paper_count,
    is_open_ended_enumeration,
)
from littraceqa.di_pipeline.select.selector import CardinalityPaperSelector

Strategy = Callable[[str, Sequence[str]], Sequence[str]]


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Read ``query_id -> ranked paper ids`` from an eval_retrieval output."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries") or []
    if not queries:
        raise ValueError(f"{path} contains no queries")
    return {
        str(entry["query_id"]): list(entry.get("ranked_papers") or [])
        for entry in queries
    }


def load_questions(path: Path) -> dict[str, str]:
    """Read ``query_id -> question`` from a gold or input JSONL file."""

    questions: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                questions[str(record["query_id"])] = str(record.get("question", ""))
    return questions


def build_strategies(questions: dict[str, str]) -> dict[str, Strategy]:
    """The fixed cutoffs to compare against, plus the selector itself."""

    strategies: dict[str, Strategy] = {
        f"fixed top{k}": (lambda q, r, k=k: r[:k]) for k in (1, 2, 3, 4, 5, 10, 20)
    }
    selector = CardinalityPaperSelector()
    strategies["cardinality selector"] = (
        lambda q, r: selector.select(questions.get(q, ""), r).paper_ids
    )
    for count in (4, 8, 10):
        open_selector = CardinalityPaperSelector(open_set_count=count)
        strategies[f"cardinality selector, open-set {count}"] = (
            lambda q, r, s=open_selector: s.select(questions.get(q, ""), r).paper_ids
        )
    return strategies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval",
        type=Path,
        required=True,
        help="Output of scripts/eval_retrieval.py.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/validation.jsonl"),
        help="Official gold JSONL. Its gold_papers field is the scored set.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Where to read question text from. Defaults to the gold file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rankings = load_rankings(args.retrieval)
    gold = load_gold_paper_sets(args.gold)
    questions = load_questions(args.questions or args.gold)

    missing = [query_id for query_id in gold if query_id not in rankings]
    if missing:
        raise SystemExit(
            f"{len(missing)} gold queries are absent from {args.retrieval}, "
            f"first: {missing[0]}"
        )

    print(f"{'strategy':40s} {'P':>8} {'R':>8} {'F1':>8} {'papers/q':>9}")
    rows: list[tuple[str, float]] = []
    for name, strategy in build_strategies(questions).items():
        selected = {
            query_id: list(strategy(query_id, rankings[query_id])) for query_id in gold
        }
        metrics = score_selection(gold, selected)
        per_query = sum(len(v) for v in selected.values()) / len(selected)
        print(
            f"{name:40s} {metrics.paper_precision_macro:8.4f} "
            f"{metrics.paper_recall_macro:8.4f} {metrics.paper_f1_macro:8.4f} "
            f"{per_query:9.2f}"
        )
        rows.append((name, metrics.paper_f1_macro))

    ceiling = score_selection(
        gold, {q: set(rankings[q][:50]) & gold[q] for q in gold}
    )
    print(
        f"{'ceiling: perfect pick from top 50':40s} "
        f"{ceiling.paper_precision_macro:8.4f} {ceiling.paper_recall_macro:8.4f} "
        f"{ceiling.paper_f1_macro:8.4f}"
    )
    best = max(rows, key=lambda row: row[1])
    print(f"\nbest strategy: {best[0]} (F1 {best[1]:.4f})")

    stated = sum(
        1 for q in gold if expected_paper_count(questions.get(q, ""), default=0) >= 2
    )
    open_set = sum(1 for q in gold if is_open_ended_enumeration(questions.get(q, "")))
    print(
        f"questions stating a count: {stated}/{len(gold)}; "
        f"open-set enumerations: {open_set}/{len(gold)}"
    )


if __name__ == "__main__":
    main()
