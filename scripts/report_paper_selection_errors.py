#!/usr/bin/env python3
"""Write a compact review file for incorrect paper-selection outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from littraceqa.common import read_json
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.evaluation.output import (
    validate_output_path,
    write_output_atomic,
)
from littraceqa.di_pipeline.evaluation.evidence_coverage_input import (
    MissingPaperMetadataError,
    prepare_evidence_coverage,
)
from littraceqa.di_pipeline.evaluation.paper_selection_report import (
    build_report,
    collect_review_cases,
    index_retrieval_entries,
)
from littraceqa.di_pipeline.evaluation.selection_input import (
    load_paper_metadata,
    load_queries,
    load_retrieval_run,
)
from littraceqa.di_pipeline.select import build_paper_selector

_PRODUCTION_QUERY_FIELDS = ("question", "answer_types", "table_schema")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact review file for paper-selection errors."
    )
    parser.add_argument(
        "--retrieval", type=Path, required=True, help="eval_retrieval.py JSON"
    )
    parser.add_argument(
        "--gold", type=Path, required=True, help="JSONL containing scored gold_papers"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="optional production query JSONL, required for evidence coverage",
    )
    parser.add_argument(
        "--paper-metadata", type=Path, required=True, help="paper metadata JSONL"
    )
    parser.add_argument(
        "--select", type=Path, required=True, help="paper select_style YAML"
    )
    parser.add_argument("--output", type=Path, required=True, help="report JSON")
    parser.add_argument(
        "--analysis-cutoff",
        dest="cutoff",
        type=int,
        default=20,
        help="diagnostic rank boundary; does not truncate selector input",
    )
    parser.add_argument(
        "--top-candidates", type=int, default=10, help="candidates shown per query"
    )
    parser.add_argument(
        "--abstract-chars", type=int, default=800, help="maximum abstract length"
    )
    parser.add_argument(
        "--method-owner-index",
        type=Path,
        default=None,
        help="optional method_alias_graph.json override",
    )
    parser.add_argument(
        "--evidence-coverage-mineru-dir",
        type=Path,
        default=None,
        help="optionally apply high-confidence MinerU evidence coverage",
    )
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="path below which report writes are rejected",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.cutoff <= 1000:
        parser.error("--analysis-cutoff must be between 1 and 1000")
    if not 1 <= args.top_candidates <= 100:
        parser.error("--top-candidates must be between 1 and 100")
    if not 0 <= args.abstract_chars <= 5000:
        parser.error("--abstract-chars must be between 0 and 5000")
    try:
        output = validate_output_path(args.output, args.read_only_root)
    except ValueError as exc:
        parser.error(str(exc))

    retrieval_payload = read_json(args.retrieval)
    retrieval_run = load_retrieval_run(args.retrieval)
    method_owner_index = (
        args.method_owner_index or retrieval_run.method_owner_index_path
    )
    select_spec = yaml.safe_load(args.select.read_text(encoding="utf-8"))
    selector = build_paper_selector(
        select_spec,
        method_owner_index_path=(
            str(method_owner_index) if method_owner_index is not None else None
        ),
    )
    gold_records = load_queries(args.gold)
    if args.questions is not None:
        query_records = {
            str(record["query_id"]): record for record in load_queries(args.questions)
        }
        for record in gold_records:
            query = query_records.get(str(record["query_id"]))
            if query is None:
                parser.error(f"query {record['query_id']} is missing from --questions")
            for field in _PRODUCTION_QUERY_FIELDS:
                record[field] = query.get(field)
    query_objects = {
        str(record["query_id"]): Query.from_dict(record)
        for record in gold_records
    }

    selection_refiner = None
    selection_metadata: dict[str, dict] = {}
    if args.evidence_coverage_mineru_dir is not None:
        if args.questions is None:
            parser.error("--questions is required with evidence coverage")
        if not args.evidence_coverage_mineru_dir.is_dir():
            parser.error(
                "--evidence-coverage-mineru-dir must be an existing directory"
            )
        try:
            setup = prepare_evidence_coverage(
                args.evidence_coverage_mineru_dir,
                args.paper_metadata,
                query_objects,
                retrieval_run.rankings,
                abstract_chars=args.abstract_chars,
            )
        except MissingPaperMetadataError as exc:
            parser.error(str(exc))
        selection_refiner = setup.refiner
        selection_metadata = setup.paper_metadata

    cases, wanted_ids = collect_review_cases(
        gold_records,
        index_retrieval_entries(retrieval_payload),
        selector,
        top_candidates=args.top_candidates,
        selection_refiner=selection_refiner,
    )
    metadata = dict(selection_metadata)
    metadata.update(
        load_paper_metadata(
            args.paper_metadata,
            wanted_ids - metadata.keys(),
            abstract_chars=args.abstract_chars,
        )
    )
    checkpoint = retrieval_payload.get("_checkpoint", {})
    report = build_report(
        cases,
        metadata,
        analysis_cutoff=args.cutoff,
        top_candidates=args.top_candidates,
        sources={
            "retrieval": args.retrieval.name,
            "gold": args.gold.name,
            "questions": args.questions.name if args.questions else None,
            "paper_metadata": args.paper_metadata.name,
            "select_style": args.select.name,
            "select_config": select_spec,
            "retrieval_run_signature": checkpoint.get("run_signature"),
            "evidence_coverage": selection_refiner is not None,
        },
    )
    write_output_atomic(output, report)
    summary = report["summary"]
    print(
        f"{summary['queries_with_selection_errors']} incorrect queries, "
        f"{summary['missed_gold_papers']} missed gold papers, "
        f"{summary['false_positive_papers']} false positives written to {output}"
    )


if __name__ == "__main__":
    main()
