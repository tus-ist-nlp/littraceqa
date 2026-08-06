"""Run the reading-only, pairwise Azure OpenAI workflow.

Each query-paper pair receives one base AOAI judgment call and its own durable
checkpoint before the answer stage starts. JSON repair, an image-policy
text-only fallback, and provider retry are exceptional recovery requests,
never paper partitions. The command never opens a validation gold file.
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
    ensure_submission_from_answer_checkpoint,
    ensure_manifest,
    invalidate_aggregate_query,
    load_judgments,
    materialize_run_outputs,
    record_error,
    validate_judgment_checkpoint,
    write_judgments,
)
from littraceqa.pairwise_prompts import PAIRWISE_SYSTEM_PROMPT
from littraceqa.submission import prediction_to_submission

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

_RUNTIME_FILES = (
    Path(__file__),
    ROOT / "src/littraceqa/__init__.py",
    ROOT / "src/littraceqa/aoai_pairwise_reader.py",
    ROOT / "src/littraceqa/answer_derivation.py",
    ROOT / "src/littraceqa/candidate_handoff.py",
    ROOT / "src/littraceqa/chunk_store.py",
    ROOT / "src/littraceqa/corpus_preflight.py",
    ROOT / "src/littraceqa/mineru_record.py",
    ROOT / "src/littraceqa/pairwise_run_store.py",
    ROOT / "src/littraceqa/pairwise_prompts.py",
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
_PAIRWISE_SYSTEM = PAIRWISE_SYSTEM_PROMPT


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"reader config is not an object: {path}")
    if config.get("name") != "aoai_pairwise_reader":
        raise ValueError("reader config name must be aoai_pairwise_reader")
    return config


def resolve_image_root(
    cli_image_root: str | Path | None,
    config: dict[str, Any],
    *,
    repo_root: Path = ROOT,
) -> tuple[str | None, str]:
    """Resolve and validate the CLI/config image root before corpus scanning."""

    source = "cli" if cli_image_root is not None else "config"
    raw = cli_image_root if cli_image_root is not None else config.get("image_root")
    if raw is None:
        return None, "disabled"
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError("image_root must be a non-empty filesystem path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        # Config paths are repository-relative and therefore independent of the
        # shell's current directory. Explicit CLI paths retain normal cwd-relative
        # command-line semantics.
        base = Path.cwd() if source == "cli" else repo_root
        path = base / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(
            f"{source} image root is not a directory: {path}. "
            "Point it at the MinerU directory containing paper_id/auto/images."
        )
    return str(path), source


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
            "evidence_policy": args.evidence_policy,
            "require_evidence": args.evidence_policy == "required",
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
            "Judge each fixed query-paper pair with one base Azure OpenAI call "
            "(never by paper partition), then answer from accepted MinerU "
            "evidence. The required challenge test has 71 questions; "
            "test_extra's 4,901 questions are optional."
        )
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Current official input JSONL, including conditional answer schemas",
    )
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
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "MinerU image directory containing paper_id/auto/images. Overrides "
            "reader config image_root; relative CLI paths use the current directory."
        ),
    )
    parser.add_argument(
        "--allow-missing-required-visual-images",
        "--allow-missing-figure-images",
        dest="allow_missing_figure_images",
        action="store_true",
        help=(
            "Allow an isolated explicit Figure/chart/image/panel query to proceed "
            "without a readable candidate image for diagnostic judgment only; "
            "requires --stage judge. This never bypasses a globally wrong "
            "--image-root where every declared image is unavailable."
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
        help=(
            "Run only this query; repeat for several. Default is every input row. "
            "Do not concatenate the 71-question main test with the optional "
            "4,901-question test_extra split."
        ),
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
        default=None,
        help=(
            "Optional positive cap for a smoke test. By default every candidate "
            "in each sidecar row is judged; scored runs should normally omit it."
        ),
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("required", "optional"),
        default="required",
        help=(
            "Official test requires evidence; test_extra does not score it. "
            "The safe default is required. Pass optional explicitly only for "
            "test_extra; filenames are never used to weaken this policy."
        ),
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help=(
            "Confirm an unfiltered all-query AOAI run after its exact minimum "
            "query/pair/call summary is printed. This does not mean test_extra is "
            "required: the main challenge test has 71 questions and test_extra's "
            "4,901 questions are optional. Runs with explicit --query-id values "
            "do not require this flag."
        ),
    )
    parser.add_argument(
        "--confirm-optional-test-extra",
        action="store_true",
        help=(
            "Second cost gate required whenever more than 71 queries are "
            "selected. The 4,901-question test_extra split is optional and must "
            "never be included merely to complete the main leaderboard run."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected query/paper; requires --query-id.",
    )
    return parser


def require_evidence_for_policy(policy: str) -> bool:
    """Resolve an explicit evidence policy without trusting a filename."""

    if policy == "required":
        return True
    if policy == "optional":
        return False
    raise ValueError(f"unknown evidence policy: {policy!r}")


def _planned_aoai_calls(
    handoffs: list[CandidateHandoff],
    *,
    stage: str,
    paper_id: str | None,
) -> tuple[int, int, int]:
    """Return query count, pair count, and exact fresh minimum AOAI calls.

    Stage 1 makes exactly one base call per selected query-paper pair. Stage 2
    makes exactly one base call per selected question. JSON repair attempts,
    image-policy text-only fallbacks, and provider retries can still increase
    the actual request count.
    """

    candidate_pairs = sum(
        1
        for handoff in handoffs
        for candidate in handoff.candidate_papers
        if paper_id is None or candidate.paper_id == paper_id
    )
    judgment_calls = candidate_pairs if stage in {"all", "judge"} else 0
    answer_calls = len(handoffs) if stage in {"all", "answer"} else 0
    return len(handoffs), candidate_pairs, judgment_calls + answer_calls


def _print_and_confirm_run_plan(
    args: argparse.Namespace,
    selected_handoffs: list[CandidateHandoff],
) -> None:
    """Print AOAI scope and require confirmation for an implicit full run."""

    query_count, candidate_pairs, minimum_calls = _planned_aoai_calls(
        selected_handoffs,
        stage=args.stage,
        paper_id=args.paper_id,
    )
    print(
        "AOAI run plan: "
        f"queries={query_count}, candidate_pairs={candidate_pairs}, "
        f"minimum_calls_without_cache={minimum_calls}, stage={args.stage}"
    )
    if query_count > 71:
        print(
            "AOAI scope warning: the required challenge test contains 71 "
            "questions. test_extra contains 4,901 optional diagnostic questions; "
            "do not run or concatenate it unless that optional experiment is "
            "intentional."
        )
        if not getattr(args, "confirm_optional_test_extra", False):
            raise SystemExit(
                "refusing a run with more than 71 queries without "
                "--confirm-optional-test-extra; the 4,901-question test_extra "
                "split is optional and has a separate AOAI budget"
            )
    if not args.query_id and not args.confirm_full_run:
        raise SystemExit(
            "refusing an unfiltered all-query AOAI run without "
            "--confirm-full-run; review the run plan above, then pass the flag"
        )


def checkpoint_judgment_update(
    *,
    run_dir: Path,
    query_id: str,
    judgments_path: Path,
    judgments: dict[str, dict[str, Any]],
    candidates: tuple[Any, ...],
) -> None:
    """Invalidate aggregate output before durably replacing judgments."""

    invalidate_aggregate_query(run_dir, query_id)
    write_judgments(judgments_path, judgments, candidates)


def checkpoint_answer_update(
    *,
    run_dir: Path,
    query_id: str,
    answer_path: Path,
    answer_record: dict[str, Any],
    submission_path: Path,
    submission: dict[str, Any],
) -> None:
    """Invalidate aggregate output before replacing an answer checkpoint."""

    # ``answer.json`` and ``submission.json`` are two separate atomic files.
    # Removing this query from the aggregate first ensures that interruption
    # between their writes cannot leave an older, apparently complete aggregate
    # row available for upload. ``materialize_run_outputs`` can reconstruct a
    # missing submission from the current answer on the next invocation.
    invalidate_aggregate_query(run_dir, query_id)
    atomic_write_json(answer_path, answer_record)
    atomic_write_json(submission_path, submission)


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
    if args.max_candidates is not None and args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive")
    if args.force and not args.query_id:
        raise SystemExit("--force requires at least one --query-id")
    if args.paper_id and (
        len(args.query_id) != 1 or args.stage != "judge"
    ):
        raise SystemExit("--paper-id requires exactly one --query-id and --stage judge")
    if args.allow_missing_figure_images and args.stage != "judge":
        raise SystemExit(
            "--allow-missing-required-visual-images is a diagnostic --stage judge "
            "override; it cannot be used with answer or all"
        )
    for query_id in args.query_id:
        if not _SAFE_QUERY_ID.fullmatch(query_id):
            raise SystemExit(f"unsafe query id: {query_id!r}")

    config = load_config(args.reader)
    require_evidence = require_evidence_for_policy(args.evidence_policy)
    print(
        "submission evidence: "
        + ("required" if require_evidence else "optional (explicit policy)")
    )
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
    _print_and_confirm_run_plan(args, selected_handoffs)

    try:
        args.image_root, image_root_source = resolve_image_root(
            args.image_root, config
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.image_root:
        print(f"image root ({image_root_source}): {args.image_root}")
    else:
        print(
            "image paths are disabled; configure --image-root to enable visual input"
        )
    # Fail before corpus scanning or writing a run manifest when AOAI
    # credentials/deployment are absent. The full-run confirmation gate above
    # intentionally runs first so an accidental all-query invocation cannot even
    # instantiate the provider client.
    llm = build_llm(config)
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
        f"{preflight['image_paths']['existing']}/"
        f"{preflight['image_paths']['unique_declared']} readable declared images"
    )
    print(
        "visual image gate: "
        f"{len(preflight['visual_image_required_queries'])} explicitly visual "
        f"queries, {len(preflight['queries_without_required_visual_images'])} "
        "without a readable candidate figure/chart image"
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
    materialize_run_outputs(
        run_dir,
        all_handoffs,
        reader,
        require_evidence=require_evidence,
    )
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
                checkpoint_judgment_update(
                    run_dir=run_dir,
                    query_id=query.query_id,
                    judgments_path=paths.judgments,
                    judgments=judgments,
                    candidates=handoff.candidate_papers,
                )
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
            if paths.answer.exists() and not args.force:
                cached_answer = json.loads(paths.answer.read_text(encoding="utf-8"))
                if not isinstance(cached_answer, dict):
                    raise ValueError(f"invalid answer checkpoint: {paths.answer}")
                if cached_answer.get("cache_key") != expected_answer_key:
                    raise ValueError(
                        f"{query.query_id}: cached answer is stale; rerun with --force "
                        "and this --query-id"
                    )
                ensure_submission_from_answer_checkpoint(
                    query,
                    cached_answer,
                    paths.submission,
                    require_evidence=require_evidence,
                )
                print(f"[{query.query_id}] answer cached")
            else:
                try:
                    prediction, answer_record = reader.answer_from_judgments(
                        query,
                        handoff.candidate_papers,
                        list(judgments.values()),
                    )
                    submission = prediction_to_submission(
                        query,
                        prediction,
                        require_evidence=require_evidence,
                    )
                except Exception as exc:
                    record_error(
                        paths.errors,
                        stage="answer",
                        query_id=query.query_id,
                        error=exc,
                    )
                    raise
                checkpoint_answer_update(
                    run_dir=run_dir,
                    query_id=query.query_id,
                    answer_path=paths.answer,
                    answer_record=answer_record,
                    submission_path=paths.submission,
                    submission=submission,
                )
                print(f"[{query.query_id}] answer complete")
    trace_count, submission_count = materialize_run_outputs(
        run_dir,
        all_handoffs,
        reader,
        require_evidence=require_evidence,
    )
    print(f"wrote {trace_count} traces to {run_dir / 'reading_traces.jsonl'}")
    print(f"wrote {submission_count} submissions to {run_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
