"""Run the reading-only, pairwise Azure OpenAI workflow.

Each candidate paper receives its own durable judgment before the answer stage
starts.  The command never opens a validation gold file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from littraceqa.aoai_pairwise_reader import PairwiseAOAIReader
from littraceqa.candidate_handoff import (
    CandidateHandoff,
    load_candidate_handoffs,
    read_jsonl,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.corpus_preflight import inspect_corpus
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.pairwise_run_store import (
    QueryRunPaths,
    atomic_write_json,
    ensure_manifest,
    invalidate_aggregate_query,
    load_judgments,
    materialize_run_outputs,
    record_error,
    validate_judgment_checkpoint,
    write_judgments,
)
from littraceqa.submission import prediction_to_submission

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

_RUNTIME_FILES = (
    Path(__file__),
    ROOT / "src/littraceqa/__init__.py",
    ROOT / "src/littraceqa/aoai_pairwise_reader.py",
    ROOT / "src/littraceqa/candidate_handoff.py",
    ROOT / "src/littraceqa/chunk_store.py",
    ROOT / "src/littraceqa/corpus_preflight.py",
    ROOT / "src/littraceqa/mineru_record.py",
    ROOT / "src/littraceqa/pairwise_run_store.py",
    ROOT / "src/littraceqa/submission.py",
    ROOT / "src/littraceqa/di_pipeline/__init__.py",
    ROOT / "src/littraceqa/di_pipeline/agent/__init__.py",
    ROOT / "src/littraceqa/di_pipeline/agent/evidence.py",
    ROOT / "src/littraceqa/di_pipeline/agent/json_utils.py",
    ROOT / "src/littraceqa/di_pipeline/contracts.py",
    ROOT / "src/littraceqa/di_pipeline/llm/azure_openai.py",
    ROOT / "src/littraceqa/di_pipeline/llm/base.py",
    ROOT / "src/littraceqa/di_pipeline/llm/fake.py",
    ROOT / "src/littraceqa/di_pipeline/llm/__init__.py",
    ROOT / "src/littraceqa/di_pipeline/registry.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
)
_SAFE_QUERY_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
)
_PAIRWISE_SYSTEM = (
    "あなたは科学論文QAの読解コンポーネントとして動作しています。"
    "与えられた候補論文と根拠だけを読み、検索や外部知識を使わないでください。"
    "指示された出力フォーマットに厳密に従ってください。"
    "JSON を求められたら、前置きや説明を付けずに JSON だけを出力してください。"
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"reader config is not an object: {path}")
    if config.get("name") != "aoai_pairwise_reader":
        raise ValueError("reader config name must be aoai_pairwise_reader")
    return config


def build_llm(config: dict[str, Any]):
    llm_config = config.get("llm")
    if not isinstance(llm_config, dict):
        raise TypeError("reader config must contain an llm object")
    name = str(llm_config.get("name") or "")
    params = llm_config.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("llm.params must be an object")
    if name == "azure_openai":
        # This runner is deliberately reading-only.  Keep its grounding policy
        # local instead of changing the shared Azure adapter's generic default,
        # and do not allow an incidental config value to weaken the policy.
        return AzureOpenAILLM(**{**params, "system": _PAIRWISE_SYSTEM})
    if name == "fake":
        return FakeLLM(**params)
    raise ValueError("pairwise reader supports only azure_openai (or fake in tests)")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": _sha256(resolved),
    }


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    store: ChunkStore,
) -> dict[str, Any]:
    chunks = Path(args.chunks).resolve()
    chunks_stat = chunks.stat()
    llm_config = config.get("llm") or {}
    llm_params = llm_config.get("params") or {}
    return {
        "schema_version": 1,
        "workflow": "fixed_candidates_pairwise_reading_only",
        "inputs": {
            "queries": _file_fingerprint(args.queries),
            "candidates": _file_fingerprint(args.candidates),
            "paper_metadata": _file_fingerprint(args.paper_metadata),
            # Hashing a 3.8GB corpus on every resume is needlessly expensive.
            # Every paper judgment also contains its own content SHA256, while
            # this run-level guard detects replacement via size + nanosecond mtime.
            "chunks": {
                "path": str(chunks),
                "size": chunks_stat.st_size,
                "mtime_ns": chunks_stat.st_mtime_ns,
            },
            "reader_config": _file_fingerprint(args.reader),
        },
        "runtime": {
            str(path.relative_to(ROOT)): _file_fingerprint(path)
            for path in _RUNTIME_FILES
        },
        "llm": {
            "adapter": llm_config.get("name"),
            "endpoint": (
                llm_params.get("endpoint")
                or os.environ.get("AZURE_OPENAI_ENDPOINT")
            ),
            "deployment": (
                llm_params.get("deployment")
                or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
            ),
            "api_version": (
                llm_params.get("api_version")
                or os.environ.get("AZURE_OPENAI_API_VERSION")
            ),
        },
        "reader": {
            "max_candidates": args.max_candidates,
            "allow_missing_figure_images": args.allow_missing_figure_images,
            "image_root": (
                str(Path(args.image_root).resolve()) if args.image_root else None
            ),
            "chunk_index": (
                str(Path(args.chunk_index).resolve()) if args.chunk_index else None
            ),
            "params": config.get("params") or {},
        },
        "corpus_paper_count": len(store),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Judge each fixed candidate paper independently with Azure OpenAI, "
            "then answer from accepted MinerU evidence."
        )
    )
    parser.add_argument("--queries", required=True, help="Organizer four-field JSONL")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Sanitized query_id + candidate_papers JSONL (never the _gold file)",
    )
    parser.add_argument("--chunks", required=True, help="Student MinerU chunks JSONL")
    parser.add_argument(
        "--paper-metadata", default="data/paper_metadata.jsonl"
    )
    parser.add_argument("--chunk-index", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument(
        "--allow-missing-figure-images",
        action="store_true",
        help=(
            "Explicitly allow figure queries with no readable candidate image; "
            "all other corpus preflight errors remain fatal"
        ),
    )
    parser.add_argument(
        "--reader",
        default="configs/agent_style/aoai_pairwise_reader.yaml",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Run only this query; repeat for several. Default is all 55.",
    )
    parser.add_argument(
        "--paper-id",
        default=None,
        help="Rejudge one paper (requires one --query-id and --stage judge)",
    )
    parser.add_argument(
        "--stage", choices=("all", "judge", "answer"), default="all"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Use all 50 in the real run; lower only for a smoke test.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected query/paper; requires --query-id.",
    )
    return parser


def _select_handoffs(
    all_handoffs: list[CandidateHandoff], query_ids: list[str]
) -> list[CandidateHandoff]:
    if not query_ids:
        return all_handoffs
    duplicates = sorted(
        query_id for query_id in set(query_ids) if query_ids.count(query_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate --query-id values: {duplicates}")
    by_id = {handoff.query.query_id: handoff for handoff in all_handoffs}
    missing = sorted(set(query_ids) - set(by_id))
    if missing:
        raise ValueError(f"unknown --query-id values: {missing}")
    requested = set(query_ids)
    return [handoff for handoff in all_handoffs if handoff.query.query_id in requested]


def main() -> None:
    args = build_parser().parse_args()
    if args.max_candidates < 1 or args.max_candidates > 50:
        raise SystemExit("--max-candidates must be between 1 and 50")
    if args.force and not args.query_id:
        raise SystemExit("--force requires at least one --query-id")
    if args.paper_id and (
        len(args.query_id) != 1 or args.stage != "judge"
    ):
        raise SystemExit("--paper-id requires exactly one --query-id and --stage judge")
    for query_id in args.query_id:
        if not _SAFE_QUERY_ID.fullmatch(query_id):
            raise SystemExit(f"unsafe query id: {query_id!r}")

    config = load_config(args.reader)
    # Fail before corpus scanning or writing a run manifest when AOAI
    # credentials/deployment are absent.
    llm = build_llm(config)
    all_handoffs = load_candidate_handoffs(
        args.queries,
        args.candidates,
        paper_metadata_path=args.paper_metadata,
    )
    unsafe_input_ids = sorted(
        handoff.query.query_id
        for handoff in all_handoffs
        if not _SAFE_QUERY_ID.fullmatch(handoff.query.query_id)
    )
    if unsafe_input_ids:
        raise ValueError(f"unsafe query_ids in input: {unsafe_input_ids}")
    all_handoffs = [
        CandidateHandoff(
            query=handoff.query,
            candidate_papers=handoff.candidate_papers[: args.max_candidates],
        )
        for handoff in all_handoffs
    ]
    selected_handoffs = _select_handoffs(all_handoffs, args.query_id)
    if args.chunk_index:
        Path(args.chunk_index).parent.mkdir(parents=True, exist_ok=True)
    store = ChunkStore(
        args.chunks,
        index_path=args.chunk_index,
        image_root=args.image_root,
    )
    canonical_ids = {
        str(record.get("paper_id") or "")
        for record in read_jsonl(args.paper_metadata)
        if record.get("paper_id")
    }
    preflight, errors = inspect_corpus(
        selected_handoffs,
        store,
        canonical_ids,
        allow_missing_figure_images=args.allow_missing_figure_images,
    )
    print(
        "preflight: "
        f"{preflight['queries']} queries, "
        f"{preflight['candidate_entries']} query-paper pairs, "
        f"{preflight['image_paths']['existing']} readable images"
    )
    for warning in preflight["warnings"]:
        print(f"preflight warning: {warning}")
    if errors:
        raise RuntimeError(
            "pairwise reader preflight failed:\n"
            + json.dumps(preflight, ensure_ascii=False, indent=2)
        )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, config, store)
    ensure_manifest(run_dir / "manifest.json", manifest, args.resume)
    atomic_write_json(run_dir / "preflight.json", preflight)

    reader = PairwiseAOAIReader(store, llm, **(config.get("params") or {}))
    # Validate every previously materialized query against current chunks and
    # image bytes before making another API call. This catches an image repaired
    # or replaced between incremental runs, even when that query is not selected.
    materialize_run_outputs(run_dir, all_handoffs, reader)
    for handoff in selected_handoffs:
        query = handoff.query
        paths = QueryRunPaths.under(run_dir, query.query_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        judgments = load_judgments(paths.judgments, query.query_id)

        target_candidates = list(handoff.candidate_papers)
        if args.paper_id:
            target_candidates = [
                candidate
                for candidate in target_candidates
                if candidate.paper_id == args.paper_id
            ]
            if not target_candidates:
                raise ValueError(
                    f"{query.query_id}: --paper-id is not in the candidate ranking: "
                    f"{args.paper_id}"
                )

        if args.stage in {"all", "judge"}:
            total = len(target_candidates)
            for index, candidate in enumerate(target_candidates, start=1):
                existing = judgments.get(candidate.paper_id)
                records = store.load_paper(candidate.paper_id)
                expected_cache_key = reader.judgment_cache_key(
                    query, candidate, records
                )
                if existing and not args.force:
                    if existing.get("cache_key") != expected_cache_key:
                        raise ValueError(
                            f"{query.query_id}/{candidate.paper_id}: cached judgment "
                            "does not match current query/corpus/config; use a new run or --force"
                        )
                    print(
                        f"[{query.query_id} {index}/{total}] rank={candidate.rank} "
                        f"{candidate.paper_id} cached"
                    )
                    continue
                try:
                    judgment = reader.judge_candidate(query, candidate)
                except Exception as exc:
                    record_error(
                        paths.errors,
                        stage="judge",
                        query_id=query.query_id,
                        paper_id=candidate.paper_id,
                        error=exc,
                    )
                    raise
                judgments[candidate.paper_id] = judgment
                write_judgments(
                    paths.judgments, judgments, handoff.candidate_papers
                )
                invalidate_aggregate_query(run_dir, query.query_id)
                print(
                    f"[{query.query_id} {index}/{total}] rank={candidate.rank} "
                    f"{candidate.paper_id} -> {judgment['label']}"
                )

        if args.stage in {"all", "answer"}:
            checkpoint = validate_judgment_checkpoint(handoff, judgments, reader)
            if not checkpoint.complete:
                raise RuntimeError(
                    f"{query.query_id}: answer stage requires all "
                    f"{len(handoff.candidate_papers)} pair judgments; missing "
                    f"{len(checkpoint.missing_paper_ids)}: "
                    f"{list(checkpoint.missing_paper_ids[:5])}"
                )
            if checkpoint.stale_paper_ids:
                raise ValueError(
                    f"{query.query_id}/{checkpoint.stale_paper_ids[0]}: "
                    "stale judgment checkpoint"
                )
            expected_answer_key = reader.answer_cache_key(
                query, list(judgments.values())
            )
            if paths.answer.exists() and paths.submission.exists() and not args.force:
                cached_answer = json.loads(paths.answer.read_text(encoding="utf-8"))
                if cached_answer.get("cache_key") != expected_answer_key:
                    raise ValueError(
                        f"{query.query_id}: cached answer is stale; rerun with --force "
                        "and this --query-id"
                    )
                print(f"[{query.query_id}] answer cached")
            else:
                try:
                    prediction, answer_record = reader.answer_from_judgments(
                        query,
                        handoff.candidate_papers,
                        list(judgments.values()),
                    )
                    submission = prediction_to_submission(query, prediction)
                except Exception as exc:
                    record_error(
                        paths.errors,
                        stage="answer",
                        query_id=query.query_id,
                        error=exc,
                    )
                    raise
                atomic_write_json(paths.answer, answer_record)
                atomic_write_json(paths.submission, submission)
                print(f"[{query.query_id}] answer complete")
    trace_count, submission_count = materialize_run_outputs(
        run_dir, all_handoffs, reader
    )
    print(f"wrote {trace_count} traces to {run_dir / 'reading_traces.jsonl'}")
    print(f"wrote {submission_count} submissions to {run_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
