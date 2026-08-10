#!/usr/bin/env python3
"""Score paper-selection styles on a completed retrieval run.

The script reports the official macro paper precision, recall, and F1 without
running a GPU model or reading agent. Reading-agent evidence filtering remains
inactive; an optional evidence refiner can inspect existing MinerU output.

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
from littraceqa.di_pipeline.evaluation.evidence_coverage_input import (
    MissingPaperMetadataError,
    prepare_evidence_coverage,
)
from littraceqa.di_pipeline.evaluation.selection_input import (
    load_queries,
    load_retrieval_run,
)
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.select import build_paper_selector
from littraceqa.di_pipeline.select.cardinality import (
    expected_paper_count,
    is_open_ended_enumeration,
)

SELECT_STYLE_DIR = Path("configs/select_style")


def load_query_records(path: Path) -> dict[str, Query]:
    """Read production query fields keyed by query id."""

    return {
        str(record["query_id"]): Query.from_dict(record)
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
    parser.add_argument(
        "--evidence-coverage-mineru-dir",
        type=Path,
        default=None,
        help=(
            "Also report each strategy after high-confidence MinerU "
            "coverage refinement."
        ),
    )
    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=Path("data/paper_metadata.jsonl"),
        help="Paper metadata used by evidence conditions.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.evidence_coverage_mineru_dir is not None and args.questions is None:
        parser.error("--questions is required with evidence coverage")
    retrieval = load_retrieval_run(args.retrieval)
    rankings = retrieval.rankings
    method_owner_index = (
        args.method_owner_index or retrieval.method_owner_index_path
    )
    gold = load_gold_paper_sets(args.gold)
    query_path = args.questions or args.gold
    query_records = load_query_records(query_path)
    questions = {query_id: query.question for query_id, query in query_records.items()}
    evidence_refiner = None
    if args.evidence_coverage_mineru_dir is not None:
        if not args.evidence_coverage_mineru_dir.is_dir():
            raise SystemExit(
                "--evidence-coverage-mineru-dir must be an existing directory"
            )
        try:
            setup = prepare_evidence_coverage(
                args.evidence_coverage_mineru_dir,
                args.paper_metadata,
                query_records,
                rankings,
            )
        except MissingPaperMetadataError as exc:
            parser.error(str(exc))
        evidence_refiner = setup.refiner

    missing = [query_id for query_id in gold if query_id not in rankings]
    if missing:
        raise SystemExit(
            f"{len(missing)} gold queries are absent from {args.retrieval}, "
            f"first: {missing[0]}"
        )
    missing_questions = [query_id for query_id in gold if query_id not in query_records]
    if missing_questions:
        raise SystemExit(
            f"{len(missing_questions)} gold queries are absent from {query_path}, "
            f"first: {missing_questions[0]}"
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
        selections = {
            q: selector.select(questions.get(q, ""), rankings[q]) for q in gold
        }
        report(
            config_path.stem,
            gold,
            {q: selection.paper_ids for q, selection in selections.items()},
        )
        if evidence_refiner is not None:
            refined = {
                q: evidence_refiner.refine(query_records[q], rankings[q], selection)
                for q, selection in selections.items()
            }
            report(
                f"{config_path.stem}+evidence_coverage",
                gold,
                {q: selection.paper_ids for q, selection in refined.items()},
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
