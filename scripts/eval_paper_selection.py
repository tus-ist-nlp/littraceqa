#!/usr/bin/env python3
"""Score paper-selection styles on a completed retrieval run.

The script reports the official macro paper precision, recall, and F1 without
running a GPU model or reading agent. Evidence-based filtering is inactive
because a retrieval-only result contains no evidence decisions.

Example:
    uv run python scripts/eval_paper_selection.py \\
      --retrieval evaluations/post_lane_removal.json \\
      --gold data/validation.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from littraceqa.di_pipeline.evaluation.submission_scoring import (
    load_gold_paper_sets,
    score_selection,
)
from littraceqa.di_pipeline.evaluation.selection_input import (
    load_queries,
    load_rankings as load_rankings,
    load_retrieval_run,
)
from littraceqa.di_pipeline.select import build_paper_selector
from littraceqa.di_pipeline.select.cardinality import (
    expected_paper_count,
    is_open_ended_enumeration,
)

SELECT_STYLE_DIR = Path("configs/select_style")


def load_questions(path: Path) -> dict[str, str]:
    """Read ``query_id -> question`` from a gold or input JSONL file."""

    return {
        str(record["query_id"]): str(record.get("question", ""))
        for record in load_queries(path)
    }


def report(
    name: str,
    gold: dict[str, set[str]],
    selected: dict[str, Sequence[str]],
) -> None:
    metrics = score_selection(gold, selected)
    per_query = sum(len(value) for value in selected.values()) / len(selected)
    print(
        f"{name:34s} {metrics.paper_precision_macro:8.4f} "
        f"{metrics.paper_recall_macro:8.4f} {metrics.paper_f1_macro:8.4f} "
        f"{per_query:9.2f}"
    )


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
    parser.add_argument(
        "--select-style-dir",
        type=Path,
        default=SELECT_STYLE_DIR,
        help="Directory of select_style YAML files to score.",
    )
    parser.add_argument(
        "--method-owner-index",
        type=Path,
        default=None,
        help=(
            "Override method_alias_graph.json. By default it is read from "
            "the retrieval checkpoint."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retrieval = load_retrieval_run(args.retrieval)
    rankings = retrieval.rankings
    method_owner_index = (
        args.method_owner_index or retrieval.method_owner_index_path
    )
    gold = load_gold_paper_sets(args.gold)
    questions = load_questions(args.questions or args.gold)

    missing = [query_id for query_id in gold if query_id not in rankings]
    if missing:
        raise SystemExit(
            f"{len(missing)} gold queries are absent from {args.retrieval}, "
            f"first: {missing[0]}"
        )

    print(f"{'strategy':34s} {'P':>8} {'R':>8} {'F1':>8} {'papers/q':>9}")
    for k in (1, 2, 4, 10, 20):
        report(f"fixed top{k}", gold, {q: rankings[q][:k] for q in gold})

    print()
    for config_path in sorted(args.select_style_dir.glob("*.yaml")):
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if spec.get("name") == "owner_aware" and method_owner_index is None:
            print(f"{config_path.stem:34s} skipped (no method-owner index)")
            continue
        selector = build_paper_selector(
            spec,
            method_owner_index_path=(
                str(method_owner_index) if method_owner_index is not None else None
            ),
        )
        report(
            config_path.stem,
            gold,
            {
                q: selector.select(questions.get(q, ""), rankings[q]).paper_ids
                for q in gold
            },
        )

    print()
    report(
        "ceiling: perfect pick from top 50",
        gold,
        {q: set(rankings[q][:50]) & gold[q] for q in gold},
    )

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
