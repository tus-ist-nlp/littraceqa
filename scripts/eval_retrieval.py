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
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from littraceqa.di_pipeline.evaluation.checkpoint import (
    build_checkpoint,
    load_resume_state,
)
from littraceqa.di_pipeline.evaluation.diagnostics import (
    paper_ranking_details,
    pre_rerank_papers,
    query_diagnostic,
)
from littraceqa.di_pipeline.evaluation.gold import (
    load_gold,
    parse_ks,
    select_records,
)
from littraceqa.di_pipeline.evaluation.metrics import SCENARIOS, RetrievalMetrics
from littraceqa.di_pipeline.evaluation.output import (
    build_output_payload,
    validate_output_path,
    write_output_atomic,
)


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


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line surface."""

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
    return parser


@dataclass
class _EvaluationState:
    """Results accumulated so far, plus everything needed to persist them.

    The payload is rewritten after every query so an interrupted run can be
    resumed from the last completed one instead of restarting the corpus load.
    """

    records: list[dict]
    ks: Sequence[int]
    checkpoint: dict
    output: Path | None
    diagnostics: dict[str, dict]
    failures: dict[str, dict]

    def payload(self) -> dict:
        return build_output_payload(
            self.records,
            self.diagnostics,
            self.failures,
            self.checkpoint,
            self.ks,
        )

    def persist(self) -> None:
        if self.output is not None:
            write_output_atomic(self.output, self.payload())


def resolve_retrieval_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    ks: Sequence[int],
) -> dict:
    """Compose the retrieval configuration and reject unusable combinations."""

    # Keep heavyweight optional dependencies out of pure aggregation tests.
    from littraceqa.di_pipeline.config import (
        compose_config,
        load_config,
        override_rerank_pool,
    )

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
    return cfg


def resolve_request_k(
    parser: argparse.ArgumentParser,
    cfg: dict,
    ks: Sequence[int],
) -> int:
    """Decide how many results to request so every cutoff stays reconstructable."""

    max_k = max(ks)
    reranker_name = cfg["retriever"].get("reranker", {}).get("name", "none")
    if reranker_name == "none":
        # Chunk-level fusion may return several chunks from one paper, so request
        # a larger pool before converting the output to unique paper IDs.
        return cfg["retriever"]["per_index_k"] * max(
            1, len(cfg["retriever"]["indexers"])
        )
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
    return pool_k if pool_k is not None else max_k


def load_retriever(cfg: dict):
    """Build the retriever and load every index it needs."""

    from littraceqa.di_pipeline.config import build_pipeline

    _, retriever, _ = build_pipeline(
        cfg,
        build_agent=False,
        build_preprocessor=False,
    )
    print("Loading existing indexes...")
    for indexer in retriever.indexers:
        indexer.load()
    print("Indexes loaded.")
    return retriever


def evaluate_pending_records(
    retriever,
    pending_records: Sequence[dict],
    state: _EvaluationState,
    *,
    request_k: int,
    max_k: int,
) -> None:
    """Retrieve for each pending query, recording failures instead of aborting."""

    from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers

    for record in pending_records:
        query_id = str(record["query_id"])
        started_at = time.perf_counter()
        try:
            # Gold labels are intentionally excluded from the retrieval call.
            results = retriever.retrieve(record["question"], request_k)
            ranked_papers = to_gold_papers(results, max_papers=max_k)
            state.diagnostics[query_id] = query_diagnostic(
                record,
                ranked_papers,
                state.ks,
                pre_rerank_ranked_papers=pre_rerank_papers(results),
                ranking_details=paper_ranking_details(
                    results,
                    max_papers=request_k,
                ),
                elapsed_seconds=time.perf_counter() - started_at,
            )
            state.failures.pop(query_id, None)
        except Exception as exc:  # Continue after one failed query.
            previous_attempts = state.failures.get(query_id, {}).get("attempts", 0)
            state.failures[query_id] = {
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
        state.persist()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ks = args.ks

    cfg = resolve_retrieval_config(parser, args, ks)
    request_k = resolve_request_k(parser, cfg, ks)

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

    state = _EvaluationState(
        records=records,
        ks=ks,
        checkpoint=build_checkpoint(cfg, ks, query_path, records),
        output=output,
        diagnostics={},
        failures={},
    )
    if args.resume:
        if not output.is_file():
            parser.error(f"resume checkpoint does not exist: {output}")
        try:
            state.diagnostics, state.failures = load_resume_state(
                output,
                state.checkpoint,
            )
        except ValueError as exc:
            parser.error(str(exc))

    pending_records = [
        record
        for record in records
        if str(record["query_id"]) not in state.diagnostics
    ]
    if pending_records:
        retriever = load_retriever(cfg)
        # Preserve any existing output until every required index loads.
        if not args.resume:
            state.persist()
        evaluate_pending_records(
            retriever,
            pending_records,
            state,
            request_k=request_k,
            max_k=max(ks),
        )
    else:
        print("All selected queries are already complete; indexes were not loaded.")

    payload = state.payload()
    print(
        f"\nEvaluated {len(state.diagnostics)} of {len(records)} queries using gold papers from "
        f"{args.queries}."
    )
    print_metrics(payload["metrics"], ks)
    if output is not None:
        write_output_atomic(output, payload)
        print(f"Detailed diagnostics written to {output}.")
    if state.failures:
        print(
            f"{len(state.failures)} queries failed; successful-query metrics are partial.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
