#!/usr/bin/env python3
"""Merge per-shard retrieval outputs back into one evaluation document.

Sharding a large query set across GPUs produces one output per shard. Only the
per-query records need to survive the merge: metrics are recomputed downstream
and the checkpoints describe their own shard, so keeping one of them would
misrepresent the whole run.

The merged order follows the original query file, not the shard order, so the
result is identical to what a single unsharded run would have produced.

Example:
    uv run python scripts/merge_retrieval_shards.py \\
      --queries data/test_extra.jsonl \\
      --shard shards/test_extra_0_retrieval.json \\
      --shard shards/test_extra_1_retrieval.json \\
      --output test_extra_retrieval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from littraceqa.di_pipeline.evaluation.output import validate_output_path


def query_order(path: Path) -> list[str]:
    """Return the query IDs in the order the original input file lists them."""

    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                order.append(str(json.loads(line)["query_id"]))
    if not order:
        raise ValueError(f"{path} contains no queries")
    return order


def collect(shards: list[Path]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Gather per-query records and failures from every shard."""

    queries: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        for entry in payload.get("queries") or []:
            query_id = str(entry["query_id"])
            if query_id in queries:
                raise ValueError(f"{query_id} appears in more than one shard")
            queries[query_id] = entry
        for query_id, record in (payload.get("failures") or {}).items():
            failures[str(query_id)] = record
    return queries, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="Shared input root that must never receive output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = validate_output_path(args.output, args.read_only_root)
    except ValueError as exc:
        parser.error(str(exc))

    order = query_order(args.queries)
    queries, failures = collect(args.shard)

    missing = [query_id for query_id in order if query_id not in queries]
    if missing:
        parser.error(
            f"{len(missing)} queries are in no shard, first: {missing[0]}"
        )
    extra = set(queries) - set(order)
    if extra:
        parser.error(f"{len(extra)} shard queries are not in {args.queries}")

    payload = {
        "queries": [queries[query_id] for query_id in order],
        "failures": failures,
        "summary": {
            "requested_query_count": len(order),
            "successful_query_count": len(order) - len(failures),
            "failed_query_count": len(failures),
            "completed": not failures,
            "merged_from_shards": [shard.name for shard in args.shard],
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(order)} queries merged from {len(args.shard)} shards "
        f"into {output}"
    )
    if failures:
        print(f"{len(failures)} queries failed and are recorded as such")


if __name__ == "__main__":
    main()
