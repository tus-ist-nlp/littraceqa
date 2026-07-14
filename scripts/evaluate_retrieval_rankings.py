#!/usr/bin/env python3
"""Evaluate saved paper rankings separately from retrieval.

The retrieval process must not read gold annotations. This script is the only
stage that joins saved rankings with the development gold file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from evaluate import evaluate_rankings, read_jsonl


DEFAULT_BASELINE = "mineru_v1_paper_rank_rrf_fill20_d100"
COMPARISON_METRICS = (
    "paper_precision_at_5_macro",
    "paper_f1_at_5_macro",
    "paper_recall_at_5_macro",
    "paper_precision_at_10_macro",
    "paper_f1_at_10_macro",
    "paper_recall_at_10_macro",
    "paper_precision_at_20_macro",
    "paper_f1_at_20_macro",
    "paper_recall_at_20_macro",
    "mrr",
    "ndcg_at_10_macro",
    "all_gold_at_5_rate",
    "all_gold_at_10_rate",
    "all_gold_at_20_rate",
)
GOLD_COUNT_GROUPS = ("single_gold_paper", "multiple_gold_papers")
RANK_DIAGNOSTIC_CUTOFF = 20


def _outcome(value: float, baseline_value: float) -> str:
    difference = value - baseline_value
    if difference > 1e-12:
        return "improved"
    if difference < -1e-12:
        return "worsened"
    return "unchanged"


def _empty_outcome_ids() -> dict[str, list[str]]:
    return {"improved": [], "worsened": [], "unchanged": []}


def _counts(values: dict[str, list[str]]) -> dict[str, int]:
    return {name: len(query_ids) for name, query_ids in values.items()}


def _queries_by_id(
    evaluation: dict[str, Any],
    *,
    method: str,
) -> dict[str, dict[str, Any]]:
    """Validate and index per-query evaluation rows for paired comparison."""
    queries = evaluation.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"evaluation for {method!r} has no query list")
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, query in enumerate(queries, start=1):
        if not isinstance(query, dict):
            raise ValueError(
                f"evaluation for {method!r} query row {row_number} is not an object"
            )
        query_id = str(query.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(
                f"evaluation for {method!r} query row {row_number} has no query_id"
            )
        if query_id in indexed:
            raise ValueError(
                f"evaluation for {method!r} contains duplicate query_id: {query_id}"
            )
        indexed[query_id] = query
    return indexed


def _gold_paper_ids(query: dict[str, Any]) -> tuple[str, ...]:
    values = query.get("gold_paper_ids")
    if not isinstance(values, list):
        return ()
    normalized: set[str] = set()
    for value in values:
        paper_id = str(value or "").strip()
        if paper_id:
            normalized.add(paper_id)
    return tuple(sorted(normalized))


def _validate_paired_query_coverage(
    method: str,
    method_queries: dict[str, dict[str, Any]],
    baseline: str,
    baseline_queries: dict[str, dict[str, Any]],
) -> None:
    """Reject comparisons across different query or gold-paper sets."""
    method_ids = set(method_queries)
    baseline_ids = set(baseline_queries)
    if method_ids != baseline_ids:
        missing = sorted(baseline_ids - method_ids)
        extra = sorted(method_ids - baseline_ids)
        raise ValueError(
            f"evaluation for {method!r} is not paired with baseline {baseline!r}; "
            f"missing query IDs: {missing}; extra query IDs: {extra}"
        )
    mismatched_gold = sorted(
        query_id
        for query_id in method_ids
        if _gold_paper_ids(method_queries[query_id])
        != _gold_paper_ids(baseline_queries[query_id])
    )
    if mismatched_gold:
        raise ValueError(
            f"evaluation for {method!r} uses different gold papers from baseline "
            f"{baseline!r} for query IDs: {mismatched_gold}"
        )


def _paired_query_metric_diagnostics(
    queries: list[dict[str, Any]],
    baseline_queries: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Compare one higher-is-better per-query metric, including gold-count groups."""
    query_ids = {
        "overall": _empty_outcome_ids(),
        **{group: _empty_outcome_ids() for group in GOLD_COUNT_GROUPS},
    }
    for query in queries:
        baseline_query = baseline_queries.get(query["query_id"])
        if baseline_query is None or not query["gold_paper_ids"]:
            continue
        outcome = _outcome(
            float(query["metrics"][metric]),
            float(baseline_query["metrics"][metric]),
        )
        query_ids["overall"][outcome].append(query["query_id"])
        group = query["gold_count_group"]
        if group in GOLD_COUNT_GROUPS:
            query_ids[group][outcome].append(query["query_id"])
    for group in query_ids.values():
        for ids in group.values():
            ids.sort()
    return {
        group: {"counts": _counts(ids), "query_ids": ids}
        for group, ids in query_ids.items()
    }


def _all_gold_transition_diagnostics(
    queries: list[dict[str, Any]],
    baseline_queries: dict[str, dict[str, Any]],
    cutoff: int,
) -> dict[str, Any]:
    """Report whether reranking gains or loses complete gold-paper recovery."""
    transition_names = ("gained", "lost", "retained", "neither")
    query_ids = {
        "overall": {name: [] for name in transition_names},
        **{
            group: {name: [] for name in transition_names}
            for group in GOLD_COUNT_GROUPS
        },
    }
    metric = f"all_gold_at_{cutoff}"
    for query in queries:
        baseline_query = baseline_queries.get(query["query_id"])
        if baseline_query is None or not query["gold_paper_ids"]:
            continue
        baseline_value = bool(baseline_query["metrics"][metric])
        value = bool(query["metrics"][metric])
        if value and not baseline_value:
            transition = "gained"
        elif baseline_value and not value:
            transition = "lost"
        elif value:
            transition = "retained"
        else:
            transition = "neither"
        query_ids["overall"][transition].append(query["query_id"])
        group = query["gold_count_group"]
        if group in GOLD_COUNT_GROUPS:
            query_ids[group][transition].append(query["query_id"])
    for group in query_ids.values():
        for ids in group.values():
            ids.sort()
    return {
        group: {
            "counts": {name: len(ids) for name, ids in transitions.items()},
            "query_ids": transitions,
        }
        for group, transitions in query_ids.items()
    }


def _capped_relevant_rank_diagnostics(
    queries: list[dict[str, Any]],
    baseline_queries: dict[str, dict[str, Any]],
    *,
    cutoff: int,
) -> dict[str, Any]:
    """Compare gold-paper ranks, assigning rank ``cutoff + 1`` when absent.

    Positive mean rank change means that a relevant paper moved upward. Capping
    absent papers makes the diagnostic consistent with Recall@cutoff and avoids
    pretending that an unobserved paper has a known rank below the saved list.
    """
    missing_rank = cutoff + 1
    accumulators: dict[str, dict[str, Any]] = {
        group: {
            "baseline_ranks": [],
            "method_ranks": [],
            "paper_outcomes": _empty_outcome_ids(),
            "query_outcomes": _empty_outcome_ids(),
        }
        for group in ("overall", *GOLD_COUNT_GROUPS)
    }

    for query in queries:
        baseline_query = baseline_queries.get(query["query_id"])
        if baseline_query is None or not query["gold_paper_ids"]:
            continue
        groups = ["overall"]
        if query["gold_count_group"] in GOLD_COUNT_GROUPS:
            groups.append(query["gold_count_group"])

        baseline_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(
                baseline_query["ranked_paper_ids"][:cutoff], start=1
            )
        }
        method_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(query["ranked_paper_ids"][:cutoff], start=1)
        }
        query_changes: list[float] = []
        for paper_id in sorted(query["gold_paper_ids"]):
            baseline_rank = baseline_ranks.get(paper_id, missing_rank)
            method_rank = method_ranks.get(paper_id, missing_rank)
            rank_change = float(baseline_rank - method_rank)
            paper_outcome = _outcome(rank_change, 0.0)
            query_changes.append(rank_change)
            paper_key = f"{query['query_id']}:{paper_id}"
            for group in groups:
                accumulators[group]["baseline_ranks"].append(baseline_rank)
                accumulators[group]["method_ranks"].append(method_rank)
                accumulators[group]["paper_outcomes"][paper_outcome].append(paper_key)

        query_outcome = _outcome(sum(query_changes) / len(query_changes), 0.0)
        for group in groups:
            accumulators[group]["query_outcomes"][query_outcome].append(
                query["query_id"]
            )

    diagnostics: dict[str, Any] = {
        "cutoff": cutoff,
        "missing_rank": missing_rank,
        "positive_change_means": "relevant papers moved upward",
    }
    for group, accumulator in accumulators.items():
        baseline_ranks = accumulator["baseline_ranks"]
        method_ranks = accumulator["method_ranks"]
        paper_outcomes = accumulator["paper_outcomes"]
        query_outcomes = accumulator["query_outcomes"]
        for ids in (*paper_outcomes.values(), *query_outcomes.values()):
            ids.sort()
        baseline_mean = (
            sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else 0.0
        )
        method_mean = sum(method_ranks) / len(method_ranks) if method_ranks else 0.0
        diagnostics[group] = {
            "relevant_paper_count": len(baseline_ranks),
            "baseline_mean_rank": baseline_mean,
            "method_mean_rank": method_mean,
            "mean_rank_change": baseline_mean - method_mean,
            "paper_outcome_counts": _counts(paper_outcomes),
            "paper_outcome_ids": paper_outcomes,
            "query_outcome_counts": _counts(query_outcomes),
            "query_outcome_ids": query_outcomes,
        }
    return diagnostics


def discover_ranking_files(rankings_dir: Path) -> dict[str, Path]:
    """Map deterministic method names to top-level ranking JSONL files."""
    if not rankings_dir.is_dir():
        raise ValueError(f"rankings directory does not exist: {rankings_dir}")
    files = {
        path.stem: path
        for path in sorted(rankings_dir.glob("*.jsonl"))
        if path.is_file() and not path.name.startswith(".")
    }
    if not files:
        raise ValueError(f"no ranking JSONL files found in: {rankings_dir}")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evaluation_manifest(
    gold_path: Path,
    ranking_files: dict[str, Path],
    *,
    baseline: str,
    gold_query_count: int,
) -> dict[str, Any]:
    """Record every input and evaluator source needed to reproduce metrics."""
    repo_root = Path(__file__).resolve().parents[1]
    evaluator_sources = (
        repo_root / "scripts" / "evaluate.py",
        Path(__file__).resolve(),
    )
    return {
        "schema_version": 1,
        "gold_used_only_by_evaluation": True,
        "gold": {
            "path": str(gold_path),
            "sha256": _sha256_file(gold_path),
            "query_count": gold_query_count,
        },
        "rankings": {
            method: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for method, path in sorted(ranking_files.items())
        },
        "protocol": {
            "baseline": baseline,
            "cutoffs": [5, 10, 20],
            "primary_metric": "paper_f1_at_5_macro",
            "single_multiple_split": "number of gold papers",
            "rank_diagnostic_cutoff": RANK_DIAGNOSTIC_CUTOFF,
            "rank_diagnostic_missing_rank": RANK_DIAGNOSTIC_CUTOFF + 1,
        },
        "environment": {
            "python": platform.python_version(),
            "code_sha256": {
                str(path.relative_to(repo_root)): _sha256_file(path)
                for path in evaluator_sources
            },
        },
    }


def evaluate_ranking_files(
    gold_records: list[dict[str, Any]],
    ranking_files: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Evaluate every method using the same fixed cutoffs."""
    return {
        method: evaluate_rankings(gold_records, read_jsonl(path))
        for method, path in sorted(ranking_files.items())
    }


def build_comparison_summary(
    evaluations: dict[str, dict[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    """Build method-level deltas and paired query counts against a baseline."""
    if baseline not in evaluations:
        available = ", ".join(sorted(evaluations))
        raise ValueError(
            f"baseline method {baseline!r} is missing; available methods: {available}"
        )
    baseline_metrics = evaluations[baseline]["metrics"]
    baseline_queries = _queries_by_id(evaluations[baseline], method=baseline)

    methods: dict[str, Any] = {}
    for method, evaluation in sorted(evaluations.items()):
        method_queries = _queries_by_id(evaluation, method=method)
        _validate_paired_query_coverage(
            method,
            method_queries,
            baseline,
            baseline_queries,
        )
        metrics = evaluation["metrics"]
        deltas = {
            metric: float(metrics[metric]) - float(baseline_metrics[metric])
            for metric in COMPARISON_METRICS
        }
        paired_query_ids = _paired_query_metric_diagnostics(
            list(method_queries.values()), baseline_queries, "paper_f1_at_5"
        )["overall"]["query_ids"]
        paired_recall = {
            f"at_{cutoff}": _paired_query_metric_diagnostics(
                list(method_queries.values()),
                baseline_queries,
                f"paper_recall_at_{cutoff}",
            )
            for cutoff in (5, 10, 20)
        }
        all_gold_transitions = {
            f"at_{cutoff}": _all_gold_transition_diagnostics(
                list(method_queries.values()), baseline_queries, cutoff
            )
            for cutoff in (5, 10, 20)
        }
        methods[method] = {
            "metrics": metrics,
            "delta_from_baseline": deltas,
            "paired_query_f1_at_5": {
                outcome: len(query_ids)
                for outcome, query_ids in paired_query_ids.items()
            },
            "paired_query_f1_at_5_query_ids": paired_query_ids,
            "paired_query_recall": paired_recall,
            "paired_all_gold_recovery": all_gold_transitions,
            "capped_relevant_paper_rank_at_20": _capped_relevant_rank_diagnostics(
                list(method_queries.values()),
                baseline_queries,
                cutoff=RANK_DIAGNOSTIC_CUTOFF,
            ),
            "paper_metrics_by_gold_count": evaluation[
                "paper_metrics_by_gold_count"
            ],
            "details": evaluation["details"],
        }

    return {
        "baseline": baseline,
        "primary_metric": "paper_f1_at_5_macro",
        "cutoffs": [5, 10, 20],
        "methods": methods,
    }


def build_question_comparison(
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join per-query metrics from all methods for failure analysis."""
    joined: dict[str, dict[str, Any]] = {}
    for method, evaluation in sorted(evaluations.items()):
        for query in evaluation["queries"]:
            query_id = query["query_id"]
            row = joined.setdefault(
                query_id,
                {
                    "query_id": query_id,
                    "gold_paper_ids": query["gold_paper_ids"],
                    "gold_paper_count": query["gold_paper_count"],
                    "gold_count_group": query["gold_count_group"],
                    "methods": {},
                },
            )
            row["methods"][method] = {
                "ranked_paper_ids": query["ranked_paper_ids"],
                "ranking_missing": query["ranking_missing"],
                "metrics": query["metrics"],
            }
    return [joined[query_id] for query_id in sorted(joined)]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_evaluation_outputs(
    output_dir: Path,
    evaluations: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    question_comparison: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Write evaluation artifacts while avoiding accidental replacement."""
    output_dir = output_dir.expanduser().resolve()
    protected_root = Path("/data2/iseakira").resolve()
    if output_dir == protected_root or output_dir.is_relative_to(protected_root):
        raise ValueError("evaluation output must not be under /data2/iseakira")

    intended_files = [
        output_dir / "comparison_summary.json",
        output_dir / "question_comparison.jsonl",
        *[
            output_dir / "methods" / f"{method}.json"
            for method in evaluations
        ],
    ]
    if manifest is not None:
        intended_files.append(output_dir / "evaluation_manifest.json")
    existing = [path for path in intended_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    methods_dir = output_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)
    for method, evaluation in sorted(evaluations.items()):
        _write_json(methods_dir / f"{method}.json", evaluation)
    _write_json(output_dir / "comparison_summary.json", comparison)
    _write_jsonl(output_dir / "question_comparison.jsonl", question_comparison)
    if manifest is not None:
        _write_json(output_dir / "evaluation_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved paper rankings without exposing gold to retrieval."
    )
    parser.add_argument(
        "--gold",
        default="data/validation.jsonl",
        help="Gold development JSONL read only by this evaluation stage.",
    )
    parser.add_argument(
        "--rankings-dir",
        required=True,
        help="Directory containing one <method>.jsonl ranking file per method.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for method metrics and comparison artifacts.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Method name used for paired deltas.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the evaluator artifacts that this command owns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rankings_dir = Path(args.rankings_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rankings_dir == output_dir:
        raise ValueError("output directory must differ from rankings directory")

    ranking_files = discover_ranking_files(rankings_dir)
    gold_path = Path(args.gold).expanduser().resolve()
    gold_records = read_jsonl(gold_path)
    evaluations = evaluate_ranking_files(gold_records, ranking_files)
    comparison = build_comparison_summary(evaluations, baseline=args.baseline)
    questions = build_question_comparison(evaluations)
    manifest = build_evaluation_manifest(
        gold_path,
        ranking_files,
        baseline=args.baseline,
        gold_query_count=len(gold_records),
    )
    write_evaluation_outputs(
        output_dir,
        evaluations,
        comparison,
        questions,
        manifest=manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
