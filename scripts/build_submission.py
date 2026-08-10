#!/usr/bin/env python3
"""Write a paper-only submission from a completed retrieval run.

Every input query receives a record. Evidence and answers stay empty so the
paper-selection stage can be run without an LLM.

Example:
    uv run python scripts/build_submission.py \\
      --retrieval test_retrieval_4b.json \\
      --queries data/test.jsonl \\
      --select configs/select_style/f1_method_owner.yaml \\
      --output submissions/test_f1_method_owner.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from littraceqa.di_pipeline.evaluation.output import validate_output_path
from littraceqa.di_pipeline.evaluation.selection_input import (
    load_paper_metadata,
    load_queries,
    load_rankings as load_rankings,
    load_retrieval_run,
)
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import MinerUPaperTableSource
from littraceqa.di_pipeline.select import build_paper_selector
from littraceqa.di_pipeline.select.citation_table_coverage import (
    citation_table_candidate_ids,
)
from littraceqa.di_pipeline.select.table_coverage import EvidenceCoverageRefiner

_ANSWER_SKELETONS: dict[str, dict[str, Any]] = {
    "freeform": {"text": ""},
    "multiple_choice": {"gold": ""},
    "table": {"rows": []},
}


def empty_answer(answer_types: Any) -> dict[str, Any]:
    """Build the empty answer shape the sample submission uses."""

    if not isinstance(answer_types, list):
        return {}
    # Deep copy: the table skeleton holds a list, and a shallow copy would
    # hand every record the same one.
    return {
        name: copy.deepcopy(skeleton)
        for name in answer_types
        if (skeleton := _ANSWER_SKELETONS.get(name)) is not None
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--select", type=Path, required=True, help="configs/select_style/*.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
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
            "Optionally refine paper selection with high-confidence MinerU "
            "evidence."
        ),
    )
    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=Path("data/paper_metadata.jsonl"),
        help="Paper metadata used by evidence conditions.",
    )
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

    retrieval = load_retrieval_run(args.retrieval)
    rankings = retrieval.rankings
    queries = load_queries(args.queries)
    query_records = {
        str(record["query_id"]): Query.from_dict(record) for record in queries
    }
    method_owner_index = (
        args.method_owner_index or retrieval.method_owner_index_path
    )
    selector = build_paper_selector(
        yaml.safe_load(args.select.read_text(encoding="utf-8")),
        method_owner_index_path=(
            str(method_owner_index) if method_owner_index is not None else None
        ),
    )
    table_refiner = None
    if args.evidence_coverage_mineru_dir is not None:
        if not args.evidence_coverage_mineru_dir.is_dir():
            parser.error(
                "--evidence-coverage-mineru-dir must be an existing directory"
            )
        evidence_source = MinerUPaperTableSource(
            args.evidence_coverage_mineru_dir
        )
        wanted_metadata = citation_table_candidate_ids(query_records, rankings)
        paper_metadata = load_paper_metadata(
            args.paper_metadata,
            wanted_metadata,
            abstract_chars=0,
        )
        missing_metadata = wanted_metadata - paper_metadata.keys()
        if missing_metadata:
            parser.error(
                f"paper metadata is missing {next(iter(sorted(missing_metadata)))}"
            )
        table_refiner = EvidenceCoverageRefiner(
            evidence_source,
            evidence_source=evidence_source,
            paper_metadata=paper_metadata,
        )

    missing = [
        str(record["query_id"])
        for record in queries
        if str(record["query_id"]) not in rankings
    ]
    if missing:
        parser.error(
            f"{len(missing)} queries have no retrieval result, first: {missing[0]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    paper_total = 0
    refined_total = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in queries:
            query_id = str(record["query_id"])
            selection = selector.select(
                record.get("question", ""), rankings[query_id]
            )
            if table_refiner is not None:
                refined = table_refiner.refine(
                    query_records[query_id], rankings[query_id], selection
                )
                refined_total += refined.paper_ids != selection.paper_ids
                selection = refined
            paper_total += len(selection.paper_ids)
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "gold_papers": [
                            {"paper_id": paper_id}
                            for paper_id in selection.paper_ids
                        ],
                        "evidence": [],
                        "answer": empty_answer(record.get("answer_types")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"{len(queries)} records written to {output} "
        f"({paper_total} papers, {paper_total / len(queries):.2f} per query)"
    )
    if table_refiner is not None:
        print(f"evidence coverage changed {refined_total} queries")
    print("evidence and answer are empty; those metrics will score zero")


if __name__ == "__main__":
    main()
