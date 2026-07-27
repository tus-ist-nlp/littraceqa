#!/usr/bin/env python3
"""Measure fixed-cutoff paper retrieval metrics without running an agent.

The end-to-end evaluator mixes retrieval quality with an agent's variable-length
paper selection. This script instead evaluates the raw retriever ranking at
fixed cutoffs. It reports all queries together and also splits them by the
actual number of gold papers:

* ``single``: exactly one gold paper
* ``multi``: more than one gold paper

All groups are accumulated during the same retrieval pass, so a large index is
loaded only once.

Example:
    uv run python scripts/eval_retrieval.py \
      --paths configs/paths/default.yaml \
      --process configs/process_style/mineru.yaml \
      --search configs/search_style/bm25.yaml \
      --allow-shared-index-load \
      --ks 1,5,10,20,50
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCENARIOS = ("total", "single", "multi")
METRIC_NAMES = ("recall", "precision", "hit_rate", "all_gold")

Ranking = tuple[set[str], list[str]]
MetricValues = dict[str, float | int | None]
RetrievalMetrics = dict[str, dict[int, MetricValues]]
_CHECKPOINT_SCHEMA_VERSION = 1
_MAX_INDEX_STATE_ENTRIES = 2048


def load_gold(path: Path) -> list[dict]:
    """Load JSONL records containing questions and gold papers."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_records(records: Iterable[dict], query_ids: Sequence[str]) -> list[dict]:
    """Select explicit query IDs in request order for bounded evaluations."""
    requested = tuple(dict.fromkeys(query_ids))
    available: dict[str, dict] = {}
    for record in records:
        query_id = str(record.get("query_id") or "")
        if not query_id:
            raise ValueError("evaluation input contains an empty query_id")
        if query_id in available:
            raise ValueError(f"duplicate query_id in evaluation input: {query_id}")
        available[query_id] = record

    if not requested:
        return list(available.values())

    missing = [query_id for query_id in requested if query_id not in available]
    if missing:
        raise ValueError(f"query_id not found in evaluation input: {', '.join(missing)}")
    return [available[query_id] for query_id in requested]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    """Fingerprint retrieval source and dependency declarations without data files."""
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src" / "littraceqa" / "di_pipeline"
    paths = [
        Path(__file__).resolve(),
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        *sorted(source_root.rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        try:
            relative_path = path.relative_to(project_root)
        except ValueError:
            relative_path = path
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_state() -> dict:
    """Record the interpreter and retrieval package versions used by a run."""
    package_versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "faiss-cpu", "numpy", "torch", "transformers"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": package_versions,
    }


def _index_state(cfg: dict) -> list[dict]:
    """Snapshot shallow index metadata without hashing large index contents."""
    snapshots: list[dict] = []
    indexers = cfg.get("retriever", {}).get("indexers", [])
    for position, indexer in enumerate(indexers):
        raw_path = (indexer.get("params") or {}).get("index_dir")
        snapshot: dict = {
            "position": position,
            "name": str(indexer.get("name") or ""),
            "path": str(Path(raw_path).expanduser().resolve()) if raw_path else None,
        }
        if not raw_path:
            snapshot["state"] = "unspecified"
            snapshots.append(snapshot)
            continue

        path = Path(raw_path).expanduser().resolve()
        try:
            root_stat = path.stat()
        except FileNotFoundError:
            snapshot["state"] = "missing"
            snapshots.append(snapshot)
            continue
        except OSError as exc:
            snapshot.update({"state": "unreadable", "error_type": type(exc).__name__})
            snapshots.append(snapshot)
            continue

        snapshot.update(
            {
                "state": "directory" if path.is_dir() else "file",
                "size": root_stat.st_size,
                "mtime_ns": root_stat.st_mtime_ns,
            }
        )
        if path.is_dir():
            entries = []
            truncated = False
            try:
                for entry in path.iterdir():
                    if len(entries) >= _MAX_INDEX_STATE_ENTRIES:
                        truncated = True
                        break
                    entries.append(entry)
                children = []
                for entry in sorted(entries, key=lambda item: item.name):
                    entry_stat = entry.stat()
                    children.append(
                        {
                            "name": entry.name,
                            "kind": "directory" if entry.is_dir() else "file",
                            "size": entry_stat.st_size,
                            "mtime_ns": entry_stat.st_mtime_ns,
                        }
                    )
            except OSError as exc:
                snapshot.update(
                    {"state": "unreadable", "error_type": type(exc).__name__}
                )
            else:
                snapshot["children"] = children
                snapshot["children_truncated"] = truncated
        snapshots.append(snapshot)
    return snapshots


def build_checkpoint(
    cfg: dict,
    ks: Sequence[int],
    query_path: Path,
    records: Sequence[dict],
) -> dict:
    """Build a deterministic signature that prevents incompatible resume runs."""
    resolved_query_path = query_path.expanduser().resolve()
    run_spec = {
        "config": cfg,
        "ks": list(ks),
        "queries_path": str(resolved_query_path),
        "queries_sha256": _file_sha256(resolved_query_path),
        "requested_query_ids": [str(record["query_id"]) for record in records],
        "source_sha256": _source_sha256(),
        "runtime": _runtime_state(),
        "index_state": _index_state(cfg),
    }
    encoded = json.dumps(
        run_spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "run_signature": hashlib.sha256(encoded).hexdigest(),
        "run_spec": run_spec,
    }


def load_resume_state(
    output: Path,
    expected_checkpoint: dict,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load successful diagnostics and retryable failures from a checkpoint."""
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read resume checkpoint: {output}") from exc
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a JSON object")

    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("resume output does not contain checkpoint metadata")
    if checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema version is incompatible")
    if checkpoint.get("run_signature") != expected_checkpoint["run_signature"]:
        raise ValueError("resume checkpoint was created with different inputs or settings")

    requested_ids = set(expected_checkpoint["run_spec"]["requested_query_ids"])
    diagnostics: dict[str, dict] = {}
    raw_diagnostics = payload.get("queries", [])
    if not isinstance(raw_diagnostics, list):
        raise ValueError("resume checkpoint queries must be a list")
    for diagnostic in raw_diagnostics:
        if not isinstance(diagnostic, dict):
            raise ValueError("resume checkpoint contains an invalid query diagnostic")
        query_id = str(diagnostic.get("query_id") or "")
        if not query_id or query_id not in requested_ids:
            raise ValueError(f"resume checkpoint contains unexpected query_id: {query_id}")
        if query_id in diagnostics:
            raise ValueError(f"resume checkpoint contains duplicate query_id: {query_id}")
        if not isinstance(diagnostic.get("gold_papers"), list) or not isinstance(
            diagnostic.get("ranked_papers"), list
        ):
            raise ValueError(
                f"resume checkpoint diagnostic is incomplete for query_id: {query_id}"
            )
        diagnostics[query_id] = diagnostic

    failures: dict[str, dict] = {}
    raw_failures = payload.get("failures", [])
    if not isinstance(raw_failures, list):
        raise ValueError("resume checkpoint failures must be a list")
    for failure in raw_failures:
        if not isinstance(failure, dict):
            raise ValueError("resume checkpoint contains an invalid failure")
        query_id = str(failure.get("query_id") or "")
        if not query_id or query_id not in requested_ids:
            raise ValueError(
                f"resume checkpoint contains unexpected failed query_id: {query_id}"
            )
        if query_id in failures:
            raise ValueError(
                f"resume checkpoint contains duplicate failed query_id: {query_id}"
            )
        if query_id in diagnostics:
            continue
        attempts = failure.get("attempts", 1)
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts <= 0
        ):
            raise ValueError(
                f"resume checkpoint has invalid attempts for query_id: {query_id}"
            )
        failures[query_id] = {
            "query_id": query_id,
            "error_type": str(failure.get("error_type") or "UnknownError"),
            "error": str(failure.get("error") or "")[:2000],
            "attempts": attempts,
        }
    return diagnostics, failures


def _final_rerank_status_counts(
    diagnostics: Iterable[dict],
) -> dict[str, int]:
    """Count at most one typed final-rerank status per query diagnostic."""

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        details = diagnostic.get("ranking_details")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            status = detail.get("final_rerank_status")
            if not isinstance(status, str) or not status:
                continue
            counts[status] = counts.get(status, 0) + 1
            break
    return dict(sorted(counts.items()))


def build_output_payload(
    records: Sequence[dict],
    diagnostics: dict[str, dict],
    failures: dict[str, dict],
    checkpoint: dict,
    ks: Sequence[int],
) -> dict:
    """Recompute metrics from successful diagnostics and preserve input order."""
    ordered_diagnostics = [
        diagnostics[str(record["query_id"])]
        for record in records
        if str(record["query_id"]) in diagnostics
    ]
    rankings = [
        (
            {str(paper_id) for paper_id in diagnostic["gold_papers"]},
            [str(paper_id) for paper_id in diagnostic["ranked_papers"]],
        )
        for diagnostic in ordered_diagnostics
    ]
    ordered_failures = [
        failures[str(record["query_id"])]
        for record in records
        if str(record["query_id"]) in failures
    ]
    pending_query_count = (
        len(records) - len(ordered_diagnostics) - len(ordered_failures)
    )
    summary = {
        "requested_query_count": len(records),
        "successful_query_count": len(ordered_diagnostics),
        "failed_query_count": len(ordered_failures),
        "pending_query_count": pending_query_count,
        "completed": len(ordered_diagnostics) == len(records),
        "metrics_include_successful_queries_only": (
            len(ordered_diagnostics) != len(records)
        ),
    }
    final_rerank_status_counts = _final_rerank_status_counts(
        ordered_diagnostics
    )
    if final_rerank_status_counts:
        summary["final_rerank_status_counts"] = final_rerank_status_counts
    return {
        "metrics": aggregate_rankings(rankings, ks),
        "queries": ordered_diagnostics,
        "failures": ordered_failures,
        "summary": summary,
        "_checkpoint": checkpoint,
    }


def write_output_atomic(output: Path, payload: dict) -> None:
    """Atomically replace an evaluation checkpoint in its target directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def gold_paper_ids(record: dict) -> set[str]:
    """Return the non-empty paper IDs in a validation record."""
    return {
        str(paper["paper_id"])
        for paper in record.get("gold_papers", [])
        if isinstance(paper, dict) and paper.get("paper_id")
    }


_METHOD_RANKING_INT_FIELDS = (
    "qwen3_rank",
    "method_dense_tail_baseline_rank",
    "method_dense_tail_rank",
    "method_dense_tail_best_neighbor_rank",
    "paper_dense_tail_baseline_rank",
    "paper_dense_tail_rank",
    "paper_dense_tail_best_neighbor_rank",
    "paper_dense_consensus_support",
    "paper_dense_consensus_best_neighbor_rank",
    "paper_dense_reciprocal_seed_count",
    "paper_dense_reciprocal_discovered_candidates",
    "paper_dense_reciprocal_examined_candidates",
    "paper_dense_reciprocal_support",
    "paper_dense_reciprocal_forward_support",
    "paper_dense_reciprocal_best_forward_rank",
    "paper_dense_reciprocal_best_reverse_rank",
    "method_bridge_topic_rank",
    "method_bridge_strength",
    "method_relation_baseline_rank",
    "method_owner_rank",
    "method_relation_rank",
    "method_relation_strength",
    "method_topic_rank",
    "method_topic_search_rank",
    "output_order_rank",
)
_METHOD_RANKING_LIST_FIELDS = (
    "method_dense_tail_via_papers",
    "paper_dense_tail_via_papers",
    "paper_dense_consensus_via_papers",
    "paper_dense_reciprocal_forward_via_papers",
    "paper_dense_reciprocal_reverse_via_papers",
    "method_bridge_owner_papers",
    "method_bridge_via_papers",
    "method_bridge_aliases",
    "method_owner_aliases",
    "method_relation_aliases",
    "method_relation_via_papers",
    "method_topic_via_papers",
)
_METHOD_RANKING_FLOAT_FIELDS = (
    "qwen3_score",
    "rank_fusion_base_weight",
    "rank_fusion_k",
    "method_dense_tail_best_similarity",
    "method_dense_tail_rrf_score",
    "paper_dense_tail_best_similarity",
    "paper_dense_tail_rrf_score",
    "paper_dense_consensus_best_similarity",
    "paper_dense_consensus_rrf_score",
    "paper_dense_reciprocal_best_similarity",
    "paper_dense_reciprocal_forward_rrf_score",
    "paper_dense_reciprocal_reverse_rrf_score",
    "pre_output_order_score",
)
_METHOD_RANKING_BOOL_FIELDS = (
    "final_rerank_candidate_set_preserved",
    "method_dense_tail_is_new",
    "paper_dense_tail_is_new",
    "paper_dense_consensus_is_new",
    "paper_dense_reciprocal_is_new",
    "method_bridge_is_new",
)
_METHOD_RANKING_STR_FIELDS = (
    "final_rerank_status",
    "final_rerank_error_type",
    "paper_dense_consensus_replaced_paper_id",
    "paper_dense_reciprocal_replaced_paper_id",
    "method_bridge_replaced_paper_id",
)


def _method_ranking_metadata(
    metadata: Any,
) -> dict[str, bool | float | int | str | list[str] | None]:
    """Copy only typed, JSON-safe method-ranking provenance fields."""
    if not isinstance(metadata, dict):
        return {}

    copied: dict[str, bool | float | int | str | list[str] | None] = {}
    for key in _METHOD_RANKING_INT_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            value
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )

    for key in _METHOD_RANKING_LIST_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            list(value)
            if isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            else []
        )
    for key in _METHOD_RANKING_FLOAT_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )
    for key in _METHOD_RANKING_BOOL_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = value if isinstance(value, bool) else None
    for key in _METHOD_RANKING_STR_FIELDS:
        if key not in metadata:
            continue
        value = metadata[key]
        copied[key] = value if isinstance(value, str) and value else None
    return copied


def paper_ranking_details(
    results: Sequence[Any],
    max_papers: int | None = None,
) -> list[dict]:
    """Return JSON-safe paper scores and reranker provenance without document text."""
    best_by_paper: dict[str, tuple[float, int, Any]] = {}
    for result_index, result in enumerate(results):
        paper_id = str(result.paper_id)
        score = float(result.score)
        previous = best_by_paper.get(paper_id)
        if previous is None or score > previous[0]:
            best_by_paper[paper_id] = (score, result_index, result)

    ranked = sorted(
        best_by_paper.values(),
        key=lambda item: (-item[0], item[1]),
    )
    if max_papers is not None:
        ranked = ranked[:max_papers]

    details = []
    for score, _, result in ranked:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        pre_rerank_rank = metadata.get("pre_rerank_rank")
        pre_rerank_score = metadata.get("pre_rerank_score")
        detail = {
            "paper_id": str(result.paper_id),
            "score": score,
            "source": str(result.source),
            "representative_chunk_id": str(result.chunk_id),
            "chunk_type": str(result.chunk_type),
            "pre_rerank_rank": (
                pre_rerank_rank
                if isinstance(pre_rerank_rank, int)
                and not isinstance(pre_rerank_rank, bool)
                else None
            ),
            "pre_rerank_score": (
                float(pre_rerank_score)
                if isinstance(pre_rerank_score, (int, float))
                and not isinstance(pre_rerank_score, bool)
                else None
            ),
        }
        detail.update(_method_ranking_metadata(metadata))
        details.append(detail)
    return details


def pre_rerank_papers(results: Sequence[Any]) -> list[str] | None:
    """Recover the full paper order recorded by a provenance-aware reranker."""
    if not results:
        return None

    for result in results:
        recorded = (result.metadata or {}).get("pre_rerank_candidate_papers")
        if recorded is None:
            continue
        if not isinstance(recorded, list) or not recorded:
            return None
        normalized = [str(paper_id).strip() for paper_id in recorded]
        if any(not paper_id for paper_id in normalized):
            return None
        if len(set(normalized)) != len(normalized):
            return None
        return normalized

    first_rank_by_paper: dict[str, int] = {}
    for result in results:
        rank = (result.metadata or {}).get("pre_rerank_rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank <= 0
        ):
            return None
        paper_id = str(result.paper_id)
        first_rank_by_paper[paper_id] = min(
            rank,
            first_rank_by_paper.get(paper_id, rank),
        )
    return sorted(first_rank_by_paper, key=first_rank_by_paper.__getitem__)


def parse_ks(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive, unique retrieval cutoffs."""
    try:
        ks = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must contain integers") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks


def _scenario_for_gold(gold: set[str]) -> str | None:
    if len(gold) == 1:
        return "single"
    if len(gold) > 1:
        return "multi"
    return None


def aggregate_rankings(
    rankings: Iterable[Ranking], ks: Sequence[int]
) -> RetrievalMetrics:
    """Aggregate paper-ranking metrics for total, single, and multi groups.

    Precision uses ``k`` as its denominator, matching conventional
    ``precision@k`` and the previous version of this script. ``hit_rate`` is the
    fraction of queries with at least one gold paper in the top k, while
    ``all_gold`` is the fraction whose complete gold set is in the top k.

    Queries without gold papers remain in ``total`` but are not classified as
    ``single`` or ``multi``. Empty groups report ``None`` instead of a misleading
    zero score.
    """
    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain at least one positive integer")

    counts = {scenario: 0 for scenario in SCENARIOS}
    sums = {
        scenario: {
            k: {metric: 0.0 for metric in METRIC_NAMES}
            for k in normalized_ks
        }
        for scenario in SCENARIOS
    }

    for gold, ranked_papers in rankings:
        scenarios = ["total"]
        specific_scenario = _scenario_for_gold(gold)
        if specific_scenario is not None:
            scenarios.append(specific_scenario)
        for scenario in scenarios:
            counts[scenario] += 1

        for k in normalized_ks:
            top_k = set(ranked_papers[:k])
            hit_count = len(gold & top_k)
            values = {
                "recall": hit_count / len(gold) if gold else 1.0,
                "precision": hit_count / k,
                "hit_rate": float(hit_count > 0),
                "all_gold": float(gold.issubset(top_k)),
            }
            for scenario in scenarios:
                for metric, value in values.items():
                    sums[scenario][k][metric] += value

    metrics: RetrievalMetrics = {}
    for scenario in SCENARIOS:
        query_count = counts[scenario]
        metrics[scenario] = {}
        for k in normalized_ks:
            metrics[scenario][k] = {"query_count": query_count}
            for metric in METRIC_NAMES:
                metrics[scenario][k][metric] = (
                    sums[scenario][k][metric] / query_count
                    if query_count
                    else None
                )
    return metrics


def query_diagnostic(
    record: dict,
    ranked_papers: list[str],
    ks: Sequence[int],
    *,
    pre_rerank_ranked_papers: list[str] | None = None,
    ranking_details: list[dict] | None = None,
    elapsed_seconds: float | None = None,
) -> dict:
    """Describe exact gold ranks for one query without using task labels."""
    gold = gold_paper_ids(record)
    first_ranks: dict[str, int] = {}
    for rank, paper_id in enumerate(ranked_papers, start=1):
        first_ranks.setdefault(paper_id, rank)
    diagnostic = {
        "query_id": str(record.get("query_id") or ""),
        "question": record.get("question"),
        "gold_count": len(gold),
        "scenario": _scenario_for_gold(gold),
        "gold_papers": sorted(gold),
        "gold_ranks": {
            paper_id: first_ranks.get(paper_id) for paper_id in sorted(gold)
        },
        "ranked_papers": ranked_papers,
        "all_gold_at_k": {
            str(k): gold.issubset(set(ranked_papers[:k])) for k in ks
        },
    }
    if pre_rerank_ranked_papers is not None:
        pre_rerank_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(pre_rerank_ranked_papers, start=1)
        }
        diagnostic.update(
            {
                "pre_rerank_papers": pre_rerank_ranked_papers,
                "pre_rerank_gold_ranks": {
                    paper_id: pre_rerank_ranks.get(paper_id)
                    for paper_id in sorted(gold)
                },
                "pre_rerank_all_gold_at_k": {
                    str(k): gold.issubset(set(pre_rerank_ranked_papers[:k]))
                    for k in ks
                },
            }
        )
    if ranking_details is not None:
        diagnostic["ranking_details"] = ranking_details
    if elapsed_seconds is not None:
        diagnostic["elapsed_seconds"] = elapsed_seconds
    return diagnostic


def validate_output_path(output: Path, read_only_root: Path) -> Path:
    """Resolve an evaluation output path and reject shared input locations."""
    resolved = output.expanduser().resolve()
    shared = read_only_root.expanduser().resolve()
    try:
        resolved.relative_to(shared)
    except ValueError:
        return resolved
    raise ValueError(f"refusing to write evaluation output under {shared}")


def validate_shared_index_load(
    index_dirs: Iterable[str | Path],
    read_only_root: Path,
    *,
    allow: bool,
) -> None:
    """Require explicit opt-in before loading indexes from a shared data root."""
    if allow:
        return
    shared = read_only_root.expanduser().resolve()
    for index_dir in index_dirs:
        resolved = Path(index_dir).expanduser().resolve()
        try:
            resolved.relative_to(shared)
        except ValueError:
            continue
        raise ValueError(
            "refusing to load a shared index without --allow-shared-index-load: "
            f"{resolved}"
        )


def validate_retrieval_cutoffs(
    retriever_config: dict,
    ks: Sequence[int],
) -> None:
    """Reject metrics beyond a configured wrapper's result limit."""

    wrapper = retriever_config.get("retriever_wrapper")
    if not isinstance(wrapper, dict):
        return
    params = wrapper.get("params")
    if not isinstance(params, dict):
        return
    max_results = params.get("max_results")
    if (
        isinstance(max_results, int)
        and not isinstance(max_results, bool)
        and max_results > 0
        and max(ks) > max_results
    ):
        raise ValueError(
            f"largest --ks value ({max(ks)}) exceeds retriever wrapper "
            f"max_results ({max_results})"
        )


def print_metrics(metrics: RetrievalMetrics, ks: Sequence[int]) -> None:
    """Print one compact metric table per gold-paper-count group."""
    header = (
        f"{'k':>5} | {'recall@k':>10} | {'precision@k':>12} | "
        f"{'hit_rate@k':>10} | {'all_gold@k':>10}"
    )
    for scenario in SCENARIOS:
        query_count = metrics[scenario][ks[0]]["query_count"]
        print(f"\n[{scenario}] {query_count} queries")
        print(header)
        print("-" * len(header))
        for k in ks:
            row = metrics[scenario][k]
            if query_count == 0:
                rendered = ["-", "-", "-", "-"]
            else:
                rendered = [
                    f"{row['recall']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['hit_rate']:.4f}",
                    f"{row['all_gold']:.4f}",
                ]
            print(
                f"{k:>5} | {rendered[0]:>10} | {rendered[1]:>12} | "
                f"{rendered[2]:>10} | {rendered[3]:>10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retriever-only paper metrics at fixed cutoffs"
    )
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument(
        "--queries",
        default="data/validation.jsonl",
        help="JSONL containing question and gold_papers fields",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Evaluate only this query ID; repeat the option to select several.",
    )
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("5,10,20,50"))
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Override the enabled reranker's candidate pool (1-1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for metrics and per-query gold-rank diagnostics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible --output checkpoint and retry failed queries.",
    )
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="Shared input root that must never receive evaluation output.",
    )
    parser.add_argument(
        "--allow-shared-index-load",
        action="store_true",
        help=(
            "Explicitly allow loading indexes under --read-only-root. Full-corpus "
            "BM25 metadata can require several gigabytes of memory."
        ),
    )
    args = parser.parse_args()
    ks = args.ks

    # Keep heavyweight optional dependencies out of pure aggregation tests.
    from littraceqa.di_pipeline.config import (
        build_pipeline,
        compose_config,
        load_config,
        override_rerank_pool,
    )
    from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers

    search_cfg = load_config(args.search)
    try:
        search_cfg = override_rerank_pool(search_cfg, args.rerank_pool_k)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=search_cfg,
        agent={"name": "simple", "params": {}},
    )
    try:
        validate_retrieval_cutoffs(cfg["retriever"], ks)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_shared_index_load(
            (item["params"]["index_dir"] for item in cfg["retriever"]["indexers"]),
            args.read_only_root,
            allow=args.allow_shared_index_load,
        )
    except ValueError as exc:
        parser.error(str(exc))
    max_k = max(ks)
    reranker_name = cfg["retriever"].get("reranker", {}).get("name", "none")
    if reranker_name != "none":
        pool_k = cfg["retriever"].get("pool_k")
        if pool_k is not None and pool_k < max_k:
            parser.error(
                f"largest --ks value ({max_k}) exceeds reranker pool_k ({pool_k}); "
                "use cutoffs within the candidate pool"
            )
        # With a paper-level fuser, this makes pool_k count actual paper
        # candidates and prevents an evaluator request from silently expanding it.
        # Request the complete reranker pool so pre-rerank diagnostics can be
        # reconstructed even when the reported cutoffs are smaller.
        request_k = pool_k if pool_k is not None else max_k
    else:
        # Chunk-level fusion may return several chunks from one paper, so request
        # a larger pool before converting the output to unique paper IDs.
        request_k = cfg["retriever"]["per_index_k"] * max(
            1, len(cfg["retriever"]["indexers"])
        )

    query_path = Path(args.queries)
    try:
        records = select_records(load_gold(query_path), args.query_id)
    except ValueError as exc:
        parser.error(str(exc))

    output: Path | None = None
    if args.output is not None:
        try:
            output = validate_output_path(args.output, args.read_only_root)
        except ValueError as exc:
            parser.error(str(exc))
    if args.resume and output is None:
        parser.error("--resume requires --output")

    checkpoint = build_checkpoint(cfg, ks, query_path, records)
    diagnostics: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    if args.resume:
        if not output.is_file():
            parser.error(f"resume checkpoint does not exist: {output}")
        try:
            diagnostics, failures = load_resume_state(output, checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
    pending_records = [
        record
        for record in records
        if str(record["query_id"]) not in diagnostics
    ]
    if pending_records:
        _, retriever, _ = build_pipeline(
            cfg,
            build_agent=False,
            build_preprocessor=False,
        )

        print("Loading existing indexes...")
        for indexer in retriever.indexers:
            indexer.load()
        print("Indexes loaded.")

        # Preserve any existing output until every required index loads.
        if not args.resume and output is not None:
            write_output_atomic(
                output,
                build_output_payload(
                    records,
                    diagnostics,
                    failures,
                    checkpoint,
                    ks,
                ),
            )

        for record in pending_records:
            query_id = str(record["query_id"])
            started_at = time.perf_counter()
            try:
                # Gold labels are intentionally excluded from the retrieval call.
                results = retriever.retrieve(record["question"], request_k)
                ranked_papers = to_gold_papers(results, max_papers=max_k)
                diagnostics[query_id] = query_diagnostic(
                    record,
                    ranked_papers,
                    ks,
                    pre_rerank_ranked_papers=pre_rerank_papers(results),
                    ranking_details=paper_ranking_details(
                        results,
                        max_papers=request_k,
                    ),
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                failures.pop(query_id, None)
            except Exception as exc:  # Continue after one failed query.
                previous_attempts = failures.get(query_id, {}).get("attempts", 0)
                failures[query_id] = {
                    "query_id": query_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "attempts": previous_attempts + 1,
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
                print(
                    f"Query {query_id} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            if output is not None:
                write_output_atomic(
                    output,
                    build_output_payload(
                        records,
                        diagnostics,
                        failures,
                        checkpoint,
                        ks,
                    ),
                )
    else:
        print("All selected queries are already complete; indexes were not loaded.")

    payload = build_output_payload(
        records,
        diagnostics,
        failures,
        checkpoint,
        ks,
    )
    metrics = payload["metrics"]
    print(
        f"\nEvaluated {len(diagnostics)} of {len(records)} queries using gold papers from "
        f"{args.queries}."
    )
    print_metrics(metrics, ks)
    if output is not None:
        write_output_atomic(output, payload)
        print(f"Detailed diagnostics written to {output}.")
    if failures:
        print(
            f"{len(failures)} queries failed; successful-query metrics are partial.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
