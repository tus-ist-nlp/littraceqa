#!/usr/bin/env python3
"""Turn a retrieval-only evaluation into per-query candidate paper lists.

The submission task is gold-paper retrieval, so this joins the ranked papers
that ``scripts/eval_retrieval.py`` recorded for each query with the corpus
metadata a reader needs. No agent runs and no answer is produced.

Input records are copied through unchanged, so a downstream consumer still sees
the original question, answer types and option list; ``candidate_papers`` and
``_meta`` are the only added keys.

Example:
    uv run python scripts/build_candidate_lists.py \\
      --queries data/test.jsonl \\
      --retrieval <retrieval-output>.json \\
      --paper-metadata data/paper_metadata.jsonl \\
      --search configs/search_style/bm25_paper_rank_seed_expansion_qwen3_reranker.yaml \\
      --output <user-owned>/test_with_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from littraceqa.di_pipeline.evaluation.output import validate_output_path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON") from exc
    return records


def load_paper_metadata(path: Path) -> dict[str, dict]:
    """Index the corpus by paper ID, keeping only the displayed fields."""

    papers: dict[str, dict] = {}
    for record in load_jsonl(path):
        paper_id = record.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            papers[paper_id] = {
                "title": record.get("title"),
                "venue": record.get("venue"),
                "year": record.get("year"),
            }
    if not papers:
        raise ValueError(f"{path} contains no paper metadata")
    return papers


def ranked_papers_by_query(path: Path) -> dict[str, list[str]]:
    """Read the per-query ranking the retrieval evaluation recorded."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"{path} has no 'queries' list")
    failures = payload.get("failures") or {}
    if failures:
        raise ValueError(
            f"{path} records {len(failures)} failed queries; rerun the "
            "evaluation with --resume before building candidate lists"
        )
    rankings: dict[str, list[str]] = {}
    for entry in queries:
        query_id = str(entry.get("query_id", ""))
        ranked = entry.get("ranked_papers")
        if not query_id or not isinstance(ranked, list):
            raise ValueError(f"{path} has an unusable entry for {query_id!r}")
        rankings[query_id] = [str(paper_id) for paper_id in ranked]
    return rankings


def build_records(
    queries: list[dict],
    rankings: dict[str, list[str]],
    papers: dict[str, dict],
    *,
    n_candidates: int,
    meta: dict,
) -> list[dict]:
    """Attach the top ``n_candidates`` papers to every input record, in order."""

    output: list[dict] = []
    for record in queries:
        query_id = str(record.get("query_id", ""))
        if query_id not in rankings:
            raise ValueError(f"no retrieval result for query {query_id!r}")
        candidates = []
        for rank, paper_id in enumerate(rankings[query_id][:n_candidates], start=1):
            if paper_id not in papers:
                raise ValueError(
                    f"query {query_id!r} ranked unknown paper {paper_id!r}"
                )
            candidates.append(
                {"rank": rank, "paper_id": paper_id, **papers[paper_id]}
            )
        output.append({**record, "candidate_papers": candidates, "_meta": meta})
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--paper-metadata", type=Path, required=True)
    parser.add_argument(
        "--search",
        required=True,
        help="Search style the retrieval used; recorded in _meta.",
    )
    parser.add_argument("--n-candidates", type=int, default=50)
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
    if not 1 <= args.n_candidates <= 1000:
        parser.error("--n-candidates must be between 1 and 1000")
    try:
        output = validate_output_path(args.output, args.read_only_root)
    except ValueError as exc:
        parser.error(str(exc))

    queries = load_jsonl(args.queries)
    rankings = ranked_papers_by_query(args.retrieval)
    papers = load_paper_metadata(args.paper_metadata)
    meta = {
        "source_retrieval": args.retrieval.name,
        "search": args.search,
        "agent": None,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_candidates": args.n_candidates,
    }
    try:
        records = build_records(
            queries,
            rankings,
            papers,
            n_candidates=args.n_candidates,
            meta=meta,
        )
    except ValueError as exc:
        parser.error(str(exc))

    short = sum(
        1 for record in records if len(record["candidate_papers"]) < args.n_candidates
    )
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} queries written to {output}")
    if short:
        print(f"{short} queries returned fewer than {args.n_candidates} candidates")


if __name__ == "__main__":
    main()
