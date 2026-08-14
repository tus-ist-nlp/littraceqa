"""Build a validation-only paper-ID sidecar without answer/evidence leakage.

This diagnostic intentionally reads validation gold in a separate process and
projects it to exactly ``query_id`` plus a lexicographically sorted list of
paper-ID strings. Inference must receive the clean organizer inputs and this
sidecar; it must never receive the gold file itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from littraceqa.candidate_handoff import production_query_from_record, read_jsonl

_CLEAN_QUERY_FIELDS = frozenset(
    {
        "query_id",
        "benchmark",
        "question",
        "answer_types",
        "multiple_choice_options",
        "table_schema",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_oracle_paper_records(
    *,
    gold_records: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project validation gold to paper IDs after strict clean-input checks."""

    clean_query_ids: list[str] = []
    seen_queries: set[str] = set()
    for index, record in enumerate(query_records, start=1):
        unexpected = sorted(set(record) - _CLEAN_QUERY_FIELDS)
        if unexpected:
            raise ValueError(
                f"clean query record {index} contains forbidden/unexpected fields: "
                f"{unexpected}"
            )
        if "options" in record:
            raise ValueError("legacy options field is forbidden; use official inputs")
        query = production_query_from_record(record)
        if query.query_id in seen_queries:
            raise ValueError(f"duplicate clean query_id: {query.query_id}")
        seen_queries.add(query.query_id)
        clean_query_ids.append(query.query_id)

    metadata_ids: set[str] = set()
    for index, record in enumerate(metadata_records, start=1):
        paper_id = str(record.get("paper_id") or "")
        if not paper_id or paper_id in metadata_ids:
            raise ValueError(
                f"invalid or duplicate paper metadata ID at row {index}: {paper_id!r}"
            )
        metadata_ids.add(paper_id)

    gold_by_id: dict[str, list[str]] = {}
    for index, record in enumerate(gold_records, start=1):
        query_id = str(record.get("query_id") or "")
        if not query_id or query_id in gold_by_id:
            raise ValueError(
                f"invalid or duplicate gold query_id at row {index}: {query_id!r}"
            )
        raw_papers = record.get("gold_papers")
        if not isinstance(raw_papers, list) or not raw_papers:
            raise ValueError(f"{query_id}: gold_papers must be a non-empty list")
        paper_ids: list[str] = []
        for position, item in enumerate(raw_papers, start=1):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{query_id}: gold_papers[{position}] must be an object"
                )
            paper_id = str(item.get("paper_id") or "")
            if not paper_id:
                raise ValueError(f"{query_id}: blank gold paper ID")
            paper_ids.append(paper_id)
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError(f"{query_id}: duplicate gold paper ID")
        unknown = sorted(set(paper_ids) - metadata_ids)
        if unknown:
            raise ValueError(f"{query_id}: unknown paper IDs: {unknown}")
        gold_by_id[query_id] = sorted(paper_ids)

    missing = sorted(set(clean_query_ids) - set(gold_by_id))
    extra = sorted(set(gold_by_id) - set(clean_query_ids))
    if missing or extra:
        raise ValueError(f"gold/query coverage mismatch: missing={missing}, extra={extra}")

    return [
        {
            "query_id": query_id,
            "candidate_papers": gold_by_id[query_id],
        }
        for query_id in clean_query_ids
    ]


def _write_new_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Create one new sidecar and remove it if the write cannot complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    created = False
    try:
        with path.open("x", encoding="utf-8") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a strict validation-only selected-paper sidecar."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--paper-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--confirm-validation-only",
        action="store_true",
        help="Acknowledge that this diagnostic must never be used for test inference.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.confirm_validation_only:
        raise SystemExit("refusing without --confirm-validation-only")
    gold_path = Path(args.gold).resolve()
    query_path = Path(args.queries).resolve()
    metadata_path = Path(args.paper_metadata).resolve()
    output_path = Path(args.output).resolve()
    if output_path in {gold_path, query_path, metadata_path}:
        raise SystemExit("output must be a new sidecar path")

    records = build_oracle_paper_records(
        gold_records=read_jsonl(gold_path),
        query_records=read_jsonl(query_path),
        metadata_records=read_jsonl(metadata_path),
    )
    if len(records) != 55:
        raise SystemExit(
            f"expected the 55-question validation split, found {len(records)}"
        )
    _write_new_jsonl(output_path, records)
    pair_count = sum(len(record["candidate_papers"]) for record in records)
    unique_papers = {
        paper_id
        for record in records
        for paper_id in record["candidate_papers"]
    }
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(output_path),
                "queries": len(records),
                "query_paper_pairs": pair_count,
                "unique_papers": len(unique_papers),
                "gold_sha256": _sha256(gold_path),
                "queries_sha256": _sha256(query_path),
                "paper_metadata_sha256": _sha256(metadata_path),
                "output_sha256": _sha256(output_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
