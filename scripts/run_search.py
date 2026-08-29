#!/usr/bin/env python3
"""Run the search and write the prediction jsonl.

**The system itself is `littraceqa.di_pipeline.pipeline`.** What this takes is the
locations for this machine (paths) and the input and output — none of the method's
knobs appear as arguments.

Usage:
    # first time: preprocess and build the indexes, then search
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # afterwards: load the existing indexes and search
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --production-input

**Scoring is a separate step**, scripts/evaluate.py, called by hand:
    uv run python scripts/evaluate.py --gold data/validation.jsonl --pred predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.di_pipeline.contracts import Chunk, Query
from littraceqa.di_pipeline.pipeline import (
    Paths,
    build_agent,
    build_expander_index,
    build_indexers,
    build_preprocessor,
)


def load_papers(path: Path) -> list[dict]:
    papers = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


# The gold that scoring goes against; also what a split run's coverage is measured on.
GOLD_PATH = Path("data/validation.jsonl")


def read_predictions(path: Path) -> dict[str, dict]:
    """Read a prediction jsonl as query_id -> record."""
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record.get("query_id", ""))] = record
    return records


def merge_predictions(output_path: Path, others: list[str]) -> tuple[Path, list[str]]:
    """Join other split runs into this one; returns the path of the joined file.

    Running in halves (val_a, 28 queries; val_b, 27) and scoring one of them against
    the 55-query gold **dilutes every macro metric by the coverage** — roughly half,
    for val_a alone. Comparing that number against another configuration leads
    straight to a wrong conclusion, so the halves are joined before scoring. On a
    query_id collision this run wins.
    """
    merged = read_predictions(output_path)
    for other in others:
        other_path = Path(other)
        if not other_path.exists():
            print(f"error: {other_path}, given to --merge-with, does not exist", file=sys.stderr)
            sys.exit(1)
        records = read_predictions(other_path)
        overlap = sorted(set(records) & set(merged))
        if overlap:
            print(
                f"warning: {other_path} shares {len(overlap)} query_id(s) with this "
                f"run (e.g. {', '.join(overlap[:3])}); this run's predictions win.",
                file=sys.stderr,
            )
        for query_id, record in records.items():
            merged.setdefault(query_id, record)

    merged_path = output_path.with_name(f"{output_path.stem}_merged{output_path.suffix}")
    with merged_path.open("w", encoding="utf-8") as f:
        for _, record in sorted(merged.items()):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"joined into {len(merged)} queries and wrote {merged_path} (score this one)")
    return merged_path, others


def dump_runs(handle, query_id: str, runs: list) -> None:
    """One line per subquery: the ranking and scores retrieval returned.

    **No text** — it can be looked up from the chunk_id through chunk_store, and
    including it would make the file hundreds of MB. Replaying the candidate
    assembly offline needs only the ranks and the scores.
    """
    for run in runs:
        handle.write(
            json.dumps(
                {
                    "query_id": query_id,
                    "step": run.step,
                    "subquery": run.subquery,
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "paper_id": r.paper_id,
                            "rank": rank,
                            "score": r.score,
                            "source": r.source,
                        }
                        for rank, r in enumerate(run.results, 1)
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def check_coverage(scored_path: Path) -> dict[str, Any]:
    """How many of the gold queries the file to be scored covers; warns if any are missing.

    **No overlap at all means this was a production run.** Production input
    (data/test_inputs.jsonl) uses `ltqa_*` query_ids, which share nothing with
    validation's `q_*`. Scoring the two against each other raises no exception — it
    returns 0.0 for every metric, perfectly normally — so scoring must not be
    suggested.
    """
    if not GOLD_PATH.exists():
        return {}
    gold_ids = set(read_predictions(GOLD_PATH))
    pred_ids = set(read_predictions(scored_path))
    covered = len(gold_ids & pred_ids)
    total = len(gold_ids)
    if covered == 0:
        print(
            f"\nNo query_id here appears in {GOLD_PATH}. Treating this as a run on "
            "production input, so no scoring is suggested.\n"
            "        Next, build the handoff file with "
            "scripts/build_candidate_handoff.py --no-gold\n"
        )
        return {"covered": 0, "gold_total": total}
    if covered < total:
        print(
            f"\nwarning: only {covered} of the {total} gold queries have a "
            f"prediction, so every macro metric is diluted to about "
            f"{covered / total:.0%}.\n"
            f"        Run the remaining split with --merge-with {scored_path} to "
            f"score all {total}. Do not compare this run's numbers against another "
            f"configuration.\n",
            file=sys.stderr,
        )
    return {"covered": covered, "gold_total": total}


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


# The fields production input actually carries — this is the settled spec.
# **multiple_choice's `options` is not given in production**, so it is not here.
# `multiple_choice_options` does exist in production input (50 of the 71 queries in
# data/test_inputs.jsonl), but `Query.from_dict` reads `options` (the oracle field
# joined in from validation gold), so listing it changes nothing on the Query. It is
# named here only to keep this an honest description of the production input.
_PRODUCTION_FIELDS = (
    "query_id",
    "question",
    "answer_types",
    "multiple_choice_options",
    "table_schema",
)


def load_queries(path: Path, production_input: bool = False) -> list[Query]:
    """Load the queries.

    With production_input=True the fields production input does not carry
    (task_family / primary_evidence_type / benchmark) are dropped before the Query
    is built. The local validation_inputs.jsonl has task_family on all 55 queries
    and production has none, so **leaving it in scores the system as though it had
    been told part of the answer**, and the number drifts from what production
    would give. Comparison runs use this.
    """
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if production_input:
                record = {k: v for k, v in record.items() if k in _PRODUCTION_FIELDS}
            queries.append(Query.from_dict(record))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", required=True,
        help="configs/paths/*.yaml (this machine's pdf_dir / chunks_dir / index_dir)",
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--build", action="store_true", help="preprocess and build the indexes (first run only)"
    )
    parser.add_argument(
        "--production-input",
        action="store_true",
        help="drop task_family / primary_evidence_type and run on the same fields "
        "production gives (use this for comparison runs)",
    )
    parser.add_argument(
        "--dump-runs",
        default=None,
        metavar="RUNS.JSONL",
        help="write out each subquery's results (step / subquery / the top chunks' "
        "ranks and scores), so the candidate assembly can be replayed offline",
    )
    parser.add_argument(
        "--merge-with",
        nargs="+",
        default=[],
        metavar="PREDICTIONS.JSONL",
        help="join other split runs' prediction jsonl into one file. Scoring one "
        "half (val_a, 28 queries; val_b, 27) against the 55-query gold dilutes the "
        "macro metrics by the coverage, so they are joined before scoring.",
    )
    args = parser.parse_args()

    paths = Paths.load(args.paths)

    if args.build:
        # Preprocessing and index building only. The chunks do not exist yet, so no
        # agent is assembled here.
        papers = load_papers(paths.paper_metadata)
        chunks = []
        for paper in tqdm(papers, desc="preprocessing"):
            chunks.extend(build_preprocessor(paths).process(paper))

        paths.chunks.parent.mkdir(parents=True, exist_ok=True)
        with paths.chunks.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        print(f"saved {len(chunks)} chunks to {paths.chunks}")

        # The three search indexes, plus the SPECTER2 index ranking B reads. The
        # latter never reaches the fuser, but without building it here there would be
        # no way to rebuild it at all.
        for indexer in [*build_indexers(paths), build_expander_index(paths)]:
            print(f"  building {indexer.name}...")
            indexer.build(chunks)
        print("indexes built")

    agent = build_agent(paths)
    print("loading the existing indexes...")
    for indexer in agent.retriever.indexers:
        try:
            indexer.load()
        except Exception as exc:
            print(
                f"error: could not load the {indexer.name} index: {exc}\n"
                f"Build the indexes first, with --build.",
                file=sys.stderr,
            )
            sys.exit(1)
    print("indexes loaded")

    queries = load_queries(Path(args.queries), production_input=args.production_input)
    if args.production_input:
        print(
            "running on the same fields production gives "
            "(query_id / question / answer_types / table_schema)"
        )
    print(f"searching for {len(queries)} questions...")

    # --dump-runs writes each subquery's results to a separate file. Deliberately
    # not in Prediction.trace, which would bloat the submission — this is the basis
    # for replaying just the candidate assembly offline.
    runs_file = open(args.dump_runs, "w", encoding="utf-8") if args.dump_runs else None

    predictions = []
    for i, query in enumerate(queries):
        pred = agent.run(query)
        predictions.append(pred.to_dict())
        if runs_file is not None:
            dump_runs(runs_file, query.query_id, getattr(agent, "last_runs", []))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} done")

    if runs_file is not None:
        runs_file.close()
        print(f"wrote the per-subquery results to {args.dump_runs}")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"wrote the predictions to {output_path}")

    # Scoring a split run on its own dilutes the macro metrics by the coverage.
    scored_path = output_path
    if args.merge_with:
        scored_path, _ = merge_predictions(output_path, args.merge_with)
    coverage = check_coverage(scored_path)

    # On a production run (no overlap) scoring is itself the mistake; do not suggest it.
    if coverage.get("covered", 0):
        print(
            "\nTo score:\n"
            f"  uv run python scripts/evaluate.py --gold {GOLD_PATH} --pred {scored_path}"
        )


if __name__ == "__main__":
    main()
