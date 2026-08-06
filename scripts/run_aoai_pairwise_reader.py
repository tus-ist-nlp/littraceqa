"""Run the reading-only, pairwise Azure OpenAI workflow.

Each query-paper pair receives its own durable checkpoint before the answer
stage starts. A narrow canonical-owner mismatch is decided without AOAI; every
other pair receives one base judgment call. JSON repair, an image-policy
text-only fallback, and provider retry are exceptional recovery requests,
never paper partitions. The command never opens a validation gold file.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, NamedTuple

import openai
import yaml
from dotenv import load_dotenv

from littraceqa.aoai_pairwise_reader import (
    JudgmentResponseExhaustedError,
    NAMED_OWNER_RESOLVER_VERSION,
    PairwiseAOAIReader,
    resolve_named_owner,
)
from littraceqa.candidate_handoff import (
    CandidateHandoff,
    CandidatePaper,
    load_candidate_handoffs,
    read_jsonl,
)
from littraceqa.chunk_store import ChunkStore
from littraceqa.corpus_preflight import inspect_corpus
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.pairwise_run_store import (
    PROVIDER_ATTEMPT_LEDGER_VERSION,
    QueryRunPaths,
    atomic_write_json,
    ensure_submission_from_answer_checkpoint,
    ensure_manifest,
    invalidate_aggregate_queries,
    load_judgments,
    materialize_run_outputs,
    provider_attempt_summary,
    record_answer_attempt,
    record_error,
    record_provider_attempt_event,
    run_directory_lock,
    validate_judgment_checkpoint,
    write_judgments,
)
from littraceqa.pairwise_prompts import PAIRWISE_SYSTEM_PROMPT
from littraceqa.submission import prediction_to_submission

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MAX_AOAI_WORKERS = 100
AOAI_RATE_LIMIT_COOLDOWN_SECONDS = 60.0
MAX_AOAI_RETRY_AFTER_SECONDS = 600.0
MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS = 16
AOAI_RATE_LIMIT_CONCURRENCY_FACTOR = 0.75
AOAI_CONCURRENCY_INCREASE_FRACTION = 0.05
AOAI_CONCURRENCY_MIN_SUCCESS_WINDOW = 20
MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS = 4
AOAI_TRANSIENT_COOLDOWN_SECONDS = 5.0
_TRANSIENT_NON_5XX_STATUS_CODES = frozenset({408, 409, 425})

_RUNTIME_FILES = (
    Path(__file__),
    ROOT / "src/littraceqa/__init__.py",
    ROOT / "src/littraceqa/aoai_pairwise_reader.py",
    ROOT / "src/littraceqa/answer_derivation.py",
    ROOT / "src/littraceqa/candidate_handoff.py",
    ROOT / "src/littraceqa/citation_locator.py",
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


def _normalized_status_code(value: Any) -> int | None:
    """Return an integer HTTP status without interpreting error messages."""

    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Return structured wrappers/groups once, without following error text."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    output: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        output.append(current)
        for linked in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(linked, BaseException):
                pending.append(linked)
        grouped = getattr(current, "exceptions", ())
        if isinstance(grouped, (list, tuple)):
            pending.extend(
                item for item in grouped if isinstance(item, BaseException)
            )
    return output


def _exception_status_codes(error: BaseException) -> set[int]:
    """Collect explicit status codes from exceptions and HTTP responses."""

    statuses: set[int] = set()
    for current in _exception_chain(error):
        direct = _normalized_status_code(getattr(current, "status_code", None))
        if direct is not None:
            statuses.add(direct)
        response = getattr(current, "response", None)
        response_status = (
            response.get("status_code")
            if isinstance(response, dict)
            else getattr(response, "status_code", None)
        )
        normalized = _normalized_status_code(response_status)
        if normalized is not None:
            statuses.add(normalized)
    return statuses


def is_rate_limit_error(error: BaseException) -> bool:
    """Find an HTTP 429 on an exception, response, or wrapped exception.

    Azure/OpenAI exceptions normally expose ``status_code`` directly and also
    carry an HTTP response. Adapters may wrap the provider exception, while an
    ``ExceptionGroup`` may contain several request failures. Traverse all of
    those structured links, but deliberately do not guess from error text.
    """

    return 429 in _exception_status_codes(error)


def is_transient_provider_error(error: BaseException) -> bool:
    """Recognize retryable transport/server failures using structured data."""

    statuses = _exception_status_codes(error)
    if statuses.intersection(_TRANSIENT_NON_5XX_STATUS_CODES) or any(
        500 <= status <= 599 for status in statuses
    ):
        return True
    return any(
        isinstance(current, (openai.APIConnectionError, openai.APITimeoutError))
        for current in _exception_chain(error)
    )


def _retry_after_seconds(error: BaseException) -> float | None:
    """Read Azure retry advice from a structured exception response."""

    candidates: list[float] = []
    for current in _exception_chain(error):
        response = getattr(current, "response", None)
        headers = (
            response.get("headers")
            if isinstance(response, dict)
            else getattr(response, "headers", None)
        )
        if headers is None:
            continue
        try:
            normalized = {
                str(key).lower(): str(value).strip()
                for key, value in headers.items()
            }
        except (AttributeError, TypeError, ValueError):
            continue

        raw_milliseconds = normalized.get("retry-after-ms")
        if raw_milliseconds is not None:
            try:
                seconds = float(raw_milliseconds) / 1000.0
            except ValueError:
                pass
            else:
                if math.isfinite(seconds) and seconds >= 0:
                    candidates.append(seconds)
                    continue

        raw_retry_after = normalized.get("retry-after")
        if raw_retry_after is None:
            continue
        try:
            seconds = float(raw_retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw_retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = max(
                    0.0,
                    (retry_at - datetime.now(UTC)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            candidates.append(seconds)

    if not candidates:
        return None
    return min(MAX_AOAI_RETRY_AFTER_SECONDS, max(candidates))


def _recover_from_transient_error(
    *,
    stage: str,
    round_number: int,
    transient_jobs: int,
    errors: Iterable[BaseException] = (),
) -> None:
    """Apply bounded exponential cooldown for a provider transport/5xx wave."""

    exponential_cooldown = min(
        30.0,
        AOAI_TRANSIENT_COOLDOWN_SECONDS * (2 ** max(0, round_number - 1)),
    )
    provider_delays = [
        delay
        for error in errors
        if (delay := _retry_after_seconds(error)) is not None
    ]
    cooldown = max([exponential_cooldown, *provider_delays])
    cooldown_source = (
        "provider"
        if provider_delays and cooldown in provider_delays
        else "local"
    )
    print(
        "AOAI transient recovery: "
        f"stage={stage}, round={round_number}/"
        f"{MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS}, "
        f"transient_jobs={transient_jobs}, cooldown_seconds={cooldown:g}, "
        f"cooldown_source={cooldown_source}"
    )
    if cooldown > 0:
        time.sleep(cooldown)


def _reduced_aoai_concurrency(
    current: int,
    *,
    reduction_step: int = 1,
    aggressive_backoff: bool = False,
) -> int:
    """Return the deterministic effective cap after one 429 wave."""

    factor = (
        0.5
        if aggressive_backoff and reduction_step <= 2
        else AOAI_RATE_LIMIT_CONCURRENCY_FACTOR
    )
    proportional = math.ceil(current * factor)
    return max(1, min(current - 1, proportional)) if current > 1 else 1


class _AdaptiveAOAIConcurrency:
    """Share one AIMD concurrency cap across every paid AOAI stage.

    HTTP 429 is the only signal that lowers the cap.  A clean success window
    raises it additively toward the configured maximum.  The cap deliberately
    remains independent of the number of jobs left in a particular stage, so a
    two-job tail cannot teach the following stage that the provider limit is
    one or two workers.
    """

    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("maximum concurrency must be positive")
        self.maximum = maximum
        self.limit = maximum
        self.clean_success_credit = 0
        self.congestion_streak = 0

    @property
    def increase_step(self) -> int:
        return max(
            1,
            math.ceil(self.maximum * AOAI_CONCURRENCY_INCREASE_FRACTION),
        )

    def record_successes(self, count: int) -> tuple[int, int] | None:
        """Credit durable successes and return a cap change, if any."""

        if count < 0:
            raise ValueError("success count must not be negative")
        if count == 0 or self.limit >= self.maximum:
            # Never bank successes while already at the ceiling.  Otherwise a
            # later 429 would be undone immediately by old, pre-congestion work.
            if self.limit >= self.maximum:
                self.clean_success_credit = 0
            return None

        old_limit = self.limit
        self.clean_success_credit += count
        while self.limit < self.maximum:
            success_window = max(
                self.limit,
                AOAI_CONCURRENCY_MIN_SUCCESS_WINDOW,
            )
            if self.clean_success_credit < success_window:
                break
            self.clean_success_credit -= success_window
            self.limit = min(
                self.maximum,
                self.limit + self.increase_step,
            )

        if self.limit == old_limit:
            return None
        # A complete clean window separates a later 429 from the previous
        # congestion burst; it should again receive the initial strong backoff.
        self.congestion_streak = 0
        if self.limit >= self.maximum:
            self.clean_success_credit = 0
        return old_limit, self.limit

    def record_rate_limit(self) -> tuple[int, int, int]:
        """Apply multiplicative decrease and clear all clean-success credit."""

        self.congestion_streak += 1
        old_limit = self.limit
        self.limit = _reduced_aoai_concurrency(
            old_limit,
            reduction_step=self.congestion_streak,
            aggressive_backoff=self.maximum > 50,
        )
        self.clean_success_credit = 0
        return old_limit, self.limit, self.congestion_streak


def _concurrency_controller(
    workers: int,
    controller: _AdaptiveAOAIConcurrency | None,
) -> _AdaptiveAOAIConcurrency:
    """Create a stage-local controller or validate a shared one."""

    if controller is None:
        return _AdaptiveAOAIConcurrency(workers)
    if controller.maximum != workers:
        raise ValueError(
            "shared concurrency maximum does not match --workers: "
            f"{controller.maximum} != {workers}"
        )
    return controller


def _record_concurrency_successes(
    *,
    controller: _AdaptiveAOAIConcurrency,
    stage: str,
    successes: int,
) -> None:
    change = controller.record_successes(successes)
    if change is None:
        return
    old_limit, new_limit = change
    print(
        "AOAI concurrency recovery: "
        f"stage={stage}, durable_successes={successes}, "
        f"effective_workers={old_limit}->{new_limit}, "
        f"maximum_workers={controller.maximum}"
    )


def _recover_from_rate_limit(
    *,
    controller: _AdaptiveAOAIConcurrency,
    stage: str,
    round_number: int,
    rate_limited_jobs: int,
    errors: Iterable[BaseException] = (),
) -> None:
    """Apply shared multiplicative decrease, report it, and cool down."""

    current_workers, reduced_workers, reduction_step = (
        controller.record_rate_limit()
    )
    provider_delays = [
        delay
        for error in errors
        if (delay := _retry_after_seconds(error)) is not None
    ]
    cooldown = (
        max(provider_delays)
        if provider_delays
        else AOAI_RATE_LIMIT_COOLDOWN_SECONDS
    )
    cooldown_source = "provider" if provider_delays else "fallback"
    print(
        "AOAI 429 recovery: "
        f"stage={stage}, round={round_number}/"
        f"{MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS}, "
        f"backoff_step={reduction_step}, "
        f"rate_limited_jobs={rate_limited_jobs}, "
        f"cooldown_seconds={cooldown:g}, "
        f"cooldown_source={cooldown_source}, "
        f"effective_workers={current_workers}->{reduced_workers}"
    )
    if cooldown > 0:
        time.sleep(cooldown)


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
    handoffs: Iterable[CandidateHandoff] | None = None,
) -> dict[str, Any]:
    chunks = Path(args.chunks).resolve()
    chunks_stat = chunks.stat()
    llm_config = config.get("llm") or {}
    llm_params = llm_config.get("params") or {}
    return {
        "schema_version": 2,
        "workflow": "fixed_candidates_pairwise_reading_only",
        "provider_attempt_ledger": PROVIDER_ATTEMPT_LEDGER_VERSION,
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
            "named_owner_resolution": (
                _named_owner_audit(handoffs)
                if handoffs is not None
                else {
                    "version": NAMED_OWNER_RESOLVER_VERSION,
                    "scope": (
                        "unique_literal_grammatical_owner_of_local_object_fuzzy_soft_only"
                    ),
                }
            ),
        },
        "corpus_paper_count": len(store),
    }


def _worker_count(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be an integer") from exc
    if not 1 <= value <= MAX_AOAI_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be between 1 and {MAX_AOAI_WORKERS}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint each fixed query-paper pair, using a zero-call named-owner "
            "gate only for decisive local-object mismatches and one base Azure "
            "OpenAI call for every other pair (never by paper partition), then "
            "answer from accepted MinerU evidence. The required challenge test "
            "has 71 questions; "
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
        "--workers",
        type=_worker_count,
        default=1,
        help=(
            "Starting maximum concurrent AOAI calls across all selected queries in one "
            f"process and one run directory (1-{MAX_AOAI_WORKERS}, default: 1). "
            "Stage 1 query-paper judgments share one global pool; Stage 2 "
            "query answers share another bounded pool. HTTP 429 responses trigger "
            "a global cooldown and a lower effective cap."
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


def _named_owner_audit(
    handoffs: Iterable[CandidateHandoff],
    *,
    paper_id: str | None = None,
) -> dict[str, Any]:
    """Return an auditable per-query owner resolution and zero-call count."""

    queries: list[dict[str, Any]] = []
    total_rejections = 0
    for handoff in handoffs:
        resolution = resolve_named_owner(
            handoff.query, handoff.candidate_papers
        )
        selected_candidates = [
            candidate
            for candidate in handoff.candidate_papers
            if paper_id is None or candidate.paper_id == paper_id
        ]
        rejected = (
            sum(
                candidate.paper_id != resolution["paper_id"]
                for candidate in selected_candidates
            )
            if resolution["status"] == "resolved" and resolution["hard_gate"]
            else 0
        )
        total_rejections += rejected
        queries.append(
            {
                "query_id": handoff.query.query_id,
                **resolution,
                "candidate_pairs": len(selected_candidates),
                "deterministic_owner_rejections": rejected,
            }
        )
    return {
        "version": NAMED_OWNER_RESOLVER_VERSION,
        "scope": (
            "unique_literal_grammatical_owner_of_local_object_fuzzy_soft_only"
        ),
        "deterministic_owner_rejections": total_rejections,
        "queries": queries,
    }


def _planned_aoai_calls(
    handoffs: list[CandidateHandoff],
    *,
    stage: str,
    paper_id: str | None,
) -> tuple[int, int, int, int]:
    """Return query/pair/rejection counts and exact fresh minimum AOAI calls.

    A uniquely resolved wrong owner of a paper-local object is checkpointed
    without a provider call. Every other Stage-1 pair and every Stage-2 query
    makes one base call. JSON repairs, fallbacks, and retries can increase the
    actual request count.
    """

    candidate_pairs = sum(
        1
        for handoff in handoffs
        for candidate in handoff.candidate_papers
        if paper_id is None or candidate.paper_id == paper_id
    )
    owner_audit = _named_owner_audit(handoffs, paper_id=paper_id)
    deterministic_rejections = (
        int(owner_audit["deterministic_owner_rejections"])
        if stage in {"all", "judge"}
        else 0
    )
    judgment_calls = (
        candidate_pairs - deterministic_rejections
        if stage in {"all", "judge"}
        else 0
    )
    answer_calls = len(handoffs) if stage in {"all", "answer"} else 0
    return (
        len(handoffs),
        candidate_pairs,
        deterministic_rejections,
        judgment_calls + answer_calls,
    )


def _print_and_confirm_run_plan(
    args: argparse.Namespace,
    selected_handoffs: list[CandidateHandoff],
) -> None:
    """Print AOAI scope and require confirmation for an implicit full run."""

    (
        query_count,
        candidate_pairs,
        deterministic_rejections,
        minimum_calls,
    ) = _planned_aoai_calls(
        selected_handoffs,
        stage=args.stage,
        paper_id=args.paper_id,
    )
    print(
        "AOAI run plan: "
        f"queries={query_count}, candidate_pairs={candidate_pairs}, "
        f"deterministic_owner_rejections={deterministic_rejections}, "
        f"minimum_calls_without_cache={minimum_calls}, stage={args.stage}, "
        f"aoai_workers={getattr(args, 'workers', 1)}"
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
    """Durably replace judgments after the coordinator's bulk invalidation."""

    del run_dir, query_id
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
    """Replace answer checkpoints after coordinator bulk invalidation."""

    # ``answer.json`` and ``submission.json`` are two separate atomic files.
    # The coordinator removed every pending query from both root aggregates
    # before starting any paid call, so interruption between these writes cannot
    # leave an older uploadable row. ``materialize_run_outputs`` reconstructs a
    # missing submission from the current answer on the next invocation.
    del run_dir, query_id
    atomic_write_json(answer_path, answer_record)
    atomic_write_json(submission_path, submission)


def invalidate_forced_checkpoints(
    *,
    run_dir: Path,
    query_id: str,
    paths: QueryRunPaths,
    judgments: dict[str, dict[str, Any]],
    target_candidates: list[CandidatePaper],
    all_candidates: tuple[CandidatePaper, ...],
) -> None:
    """Durably remove forced targets and any answer derived from them."""

    target_ids = {candidate.paper_id for candidate in target_candidates}
    removed_ids = target_ids.intersection(judgments)
    if not removed_ids and not paths.answer.exists() and not paths.submission.exists():
        return

    # The coordinator bulk-invalidates every forced query before reaching this
    # mutation. A forced pair stays missing until its new paid call succeeds.
    del run_dir, query_id
    paths.answer.unlink(missing_ok=True)
    paths.submission.unlink(missing_ok=True)
    for paper_id in removed_ids:
        judgments.pop(paper_id)
    write_judgments(paths.judgments, judgments, all_candidates)


def invalidate_forced_answer(
    *,
    run_dir: Path,
    query_id: str,
    paths: QueryRunPaths,
) -> None:
    """Remove only the derived answer for ``--stage answer --force``."""

    # Root aggregates were bulk-invalidated before any forced mutation.
    del run_dir, query_id
    paths.answer.unlink(missing_ok=True)
    paths.submission.unlink(missing_ok=True)


class QueryExecutionState(NamedTuple):
    """Coordinator-owned mutable state for one selected query."""

    handoff: CandidateHandoff
    paths: QueryRunPaths
    judgments: dict[str, dict[str, Any]]
    target_candidates: list[CandidatePaper]


class _JudgmentJob(NamedTuple):
    sequence: int
    state: QueryExecutionState
    index: int
    total: int
    candidate: CandidatePaper


class _AnswerJob(NamedTuple):
    sequence: int
    state: QueryExecutionState
    judgments: tuple[dict[str, Any], ...]


class _AnswerWorkerResult(NamedTuple):
    prediction: Any | None
    answer_record: dict[str, Any] | None
    error: Exception | None


class _CachedJudgmentValidation(NamedTuple):
    state: QueryExecutionState
    index: int
    total: int
    candidate: CandidatePaper
    existing: dict[str, Any]


class _ProviderAttemptEnvelope(NamedTuple):
    """Worker-to-coordinator handoff with a durability acknowledgement."""

    path: Path
    stage: str
    query_id: str
    paper_id: str | None
    whole_job_attempt_index: int
    event: dict[str, Any]
    acknowledgement: threading.Event
    errors: list[Exception]


def _provider_attempt_callback(
    queue: SimpleQueue[_ProviderAttemptEnvelope],
    *,
    path: Path,
    stage: str,
    query_id: str,
    paper_id: str | None,
    whole_job_attempt_index: int,
):
    """Return a worker callback that waits for coordinator fsync."""

    def callback(event: dict[str, Any]) -> None:
        acknowledgement = threading.Event()
        errors: list[Exception] = []
        queue.put(
            _ProviderAttemptEnvelope(
                path=path,
                stage=stage,
                query_id=query_id,
                paper_id=paper_id,
                whole_job_attempt_index=whole_job_attempt_index,
                event=dict(event),
                acknowledgement=acknowledgement,
                errors=errors,
            )
        )
        if not acknowledgement.wait(timeout=30):
            raise TimeoutError("provider attempt ledger acknowledgement timed out")
        if errors:
            raise RuntimeError("provider attempt ledger write failed") from errors[0]

    return callback


def _normalized_provider_attempt_event(
    envelope: _ProviderAttemptEnvelope,
) -> dict[str, Any]:
    """Strip exception objects and retain only structured, non-secret fields."""

    event = dict(envelope.event)
    error = event.pop("_exception", None)
    event.update(
        {
            "stage": envelope.stage,
            "query_id": envelope.query_id,
            "paper_id": envelope.paper_id,
            "whole_job_attempt_index": envelope.whole_job_attempt_index,
        }
    )
    if isinstance(error, BaseException):
        statuses = sorted(_exception_status_codes(error))
        if 429 in statuses:
            retry_category = "rate_limit"
            recovery_round = envelope.whole_job_attempt_index
        elif is_transient_provider_error(error):
            retry_category = "transient_provider"
            recovery_round = envelope.whole_job_attempt_index
        elif statuses == [400]:
            retry_category = "request_rejected"
            recovery_round = 0
        else:
            retry_category = "terminal"
            recovery_round = 0
        event.update(
            {
                "exception_type": type(error).__name__,
                "status_codes": statuses,
                "retry_category": retry_category,
                "recovery_round": recovery_round,
                "retry_after_seconds": _retry_after_seconds(error),
            }
        )
    return event


def _drain_provider_attempts(
    queue: SimpleQueue[_ProviderAttemptEnvelope],
) -> None:
    """Persist all queued provider events from the coordinator thread only."""

    while True:
        try:
            envelope = queue.get_nowait()
        except Empty:
            return
        try:
            record_provider_attempt_event(
                envelope.path,
                event=_normalized_provider_attempt_event(envelope),
            )
        except Exception as exc:
            envelope.errors.append(exc)
        finally:
            envelope.acknowledgement.set()


def _record_uninstrumented_provider_error(
    *,
    path: Path,
    stage: str,
    query_id: str,
    paper_id: str | None,
    whole_job_attempt_index: int,
    error: Exception,
) -> None:
    """Account for retryable fake/legacy readers without callback support."""

    if getattr(error, "_littraceqa_provider_attempt_id", None):
        return
    attempt_id = str(uuid.uuid4())
    common = {
        "attempt_id": attempt_id,
        "semantic_phase": f"{stage}_legacy_uninstrumented",
        "provider_invocation_index": 1,
        "provider_invocation_count": 1,
    }
    for event in (
        {**common, "event_kind": "prepare"},
        {
            **common,
            "event_kind": "finalize",
            "outcome": "provider_error",
            "_exception": error,
        },
    ):
        envelope = _ProviderAttemptEnvelope(
            path=path,
            stage=stage,
            query_id=query_id,
            paper_id=paper_id,
            whole_job_attempt_index=whole_job_attempt_index,
            event=event,
            acknowledgement=threading.Event(),
            errors=[],
        )
        record_provider_attempt_event(
            path,
            event=_normalized_provider_attempt_event(envelope),
        )
    error._littraceqa_provider_attempt_id = attempt_id  # type: ignore[attr-defined]


def _apply_provider_ledger_accounting(
    record: dict[str, Any],
    *,
    path: Path,
    stage: str,
    paper_id: str | None = None,
) -> None:
    """Make the durable ledger authoritative in a completed checkpoint."""

    summary = provider_attempt_summary(path, stage=stage, paper_id=paper_id)
    count = int(summary["provider_invocation_count"])
    record["provider_attempt_ledger"] = summary
    record["provider_invocation_count"] = count
    record["provider_attempt_ids"] = summary["attempt_ids"]
    record["provider_request_ids"] = summary["request_ids"]
    record["provider_uncertain_invocation_count"] = summary[
        "uncertain_provider_invocation_count"
    ]
    record["provider_usage"] = summary["usage"]
    if stage == "judge":
        record["judgment_call_count"] = count
    elif stage == "answer":
        record["answer_call_count"] = count


def _round_robin_jobs(
    jobs_by_query: list[list[tuple[int, CandidatePaper]]],
    states: list[QueryExecutionState],
) -> deque[_JudgmentJob]:
    """Interleave query queues so one long query cannot monopolize the pool."""

    queues = [deque(items) for items in jobs_by_query]
    jobs: deque[_JudgmentJob] = deque()
    sequence = 0
    while any(queues):
        for state, queue in zip(states, queues, strict=True):
            if not queue:
                continue
            index, candidate = queue.popleft()
            jobs.append(
                _JudgmentJob(
                    sequence=sequence,
                    state=state,
                    index=index,
                    total=len(state.target_candidates),
                    candidate=candidate,
                )
            )
            sequence += 1
    return jobs


def run_candidate_judgments_globally(
    *,
    reader: PairwiseAOAIReader,
    states: list[QueryExecutionState],
    run_dir: Path,
    workers: int,
    force: bool,
    preinvalidated_query_ids: Iterable[str] = (),
    concurrency: _AdaptiveAOAIConcurrency | None = None,
) -> None:
    """Run every selected query-paper job through one bounded worker pool.

    Worker threads make AOAI calls only. The coordinator thread owns every
    judgment mapping, aggregate invalidation, per-query checkpoint, error log,
    and progress line. Pending jobs are interleaved by query. Authentication,
    configuration, corpus, and unexpected provider failures
    still stop new work after the current in-flight set is checkpointed. A
    narrowly typed exhausted model-response repair is instead isolated: its
    pair remains missing for ``--resume`` while all other pending pairs finish.

    Retryable 429, HTTP 5xx, timeout, and connection failures restart the whole
    query-paper job. If a base response succeeded but a later internal repair
    request failed transiently, the base AOAI call is made again; resuming inside
    a reader call would require an LLM-layer protocol.
    """

    concurrency = _concurrency_controller(workers, concurrency)
    owner_resolutions = {
        state.handoff.query.query_id: resolve_named_owner(
            state.handoff.query,
            state.handoff.candidate_papers,
        )
        for state in states
    }
    invalidated_query_ids = set(preinvalidated_query_ids)
    if force:
        forced_query_ids = {
            state.handoff.query.query_id for state in states
        }
        invalidate_aggregate_queries(
            run_dir, forced_query_ids - invalidated_query_ids
        )
        invalidated_query_ids.update(forced_query_ids)
        for state in states:
            query = state.handoff.query
            invalidate_forced_checkpoints(
                run_dir=run_dir,
                query_id=query.query_id,
                paths=state.paths,
                judgments=state.judgments,
                target_candidates=state.target_candidates,
                all_candidates=state.handoff.candidate_papers,
            )

    jobs_by_query: list[list[tuple[int, CandidatePaper]]] = []
    cached_validations: list[_CachedJudgmentValidation] = []
    for state in states:
        query = state.handoff.query
        pending: list[tuple[int, CandidatePaper]] = []
        total = len(state.target_candidates)
        for index, candidate in enumerate(state.target_candidates, start=1):
            existing = state.judgments.get(candidate.paper_id)
            if existing and not force:
                cached_validations.append(
                    _CachedJudgmentValidation(
                        state=state,
                        index=index,
                        total=total,
                        candidate=candidate,
                        existing=existing,
                    )
                )
                continue
            pending.append((index, candidate))
        jobs_by_query.append(pending)

    # Cached checkpoints still need content-sensitive validation, but their
    # independent paper/image hashes should not serialize every resume before
    # the paid global pool can start. Workers return keys only; printing and all
    # checkpoint state remain coordinator-owned and deterministic.
    def expected_cached_key(item: _CachedJudgmentValidation) -> str:
        records = reader.chunk_store.load_paper(item.candidate.paper_id)
        if getattr(reader, "supports_named_owner_resolution", False):
            return reader.judgment_cache_key(
                item.state.handoff.query,
                item.candidate,
                records,
                owner_resolution=owner_resolutions[
                    item.state.handoff.query.query_id
                ],
            )
        return reader.judgment_cache_key(
            item.state.handoff.query, item.candidate, records
        )

    if cached_validations:
        with ThreadPoolExecutor(
            max_workers=min(workers, 64, len(cached_validations)),
            thread_name_prefix="judgment-cache-validate",
        ) as cache_executor:
            expected_keys = cache_executor.map(
                expected_cached_key, cached_validations
            )
            for item, expected_cache_key in zip(
                cached_validations, expected_keys, strict=True
            ):
                query = item.state.handoff.query
                candidate = item.candidate
                if item.existing.get("cache_key") != expected_cache_key:
                    raise ValueError(
                        f"{query.query_id}/{candidate.paper_id}: cached judgment "
                        "does not match current query/corpus/config; use a new "
                        "run or --force"
                    )
                print(
                    f"[{query.query_id} {item.index}/{item.total}] "
                    f"rank={candidate.rank} {candidate.paper_id} cached"
                )

    pending_jobs = _round_robin_jobs(jobs_by_query, states)
    if not pending_jobs:
        return
    pending_query_ids = {
        job.state.handoff.query.query_id for job in pending_jobs
    }
    invalidate_aggregate_queries(
        run_dir, pending_query_ids - invalidated_query_ids
    )

    in_flight: dict[Future[dict[str, Any]], _JudgmentJob] = {}
    provider_attempt_queue: SimpleQueue[_ProviderAttemptEnvelope] = SimpleQueue()
    response_failures: list[tuple[_JudgmentJob, JudgmentResponseExhaustedError]] = []
    rate_limit_retries: dict[int, int] = {}
    transient_retries: dict[int, int] = {}
    whole_job_attempts: dict[int, int] = {}
    wave_width = min(concurrency.limit, len(pending_jobs))

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        if not pending_jobs:
            return False
        job = pending_jobs.popleft()
        whole_job_attempt_index = whole_job_attempts.get(job.sequence, 0) + 1
        whole_job_attempts[job.sequence] = whole_job_attempt_index
        kwargs: dict[str, Any] = {}
        if getattr(reader, "supports_named_owner_resolution", False):
            kwargs["owner_resolution"] = owner_resolutions[
                job.state.handoff.query.query_id
            ]
        if getattr(reader, "supports_provider_attempt_ledger", False):
            kwargs["provider_attempt_callback"] = _provider_attempt_callback(
                provider_attempt_queue,
                path=job.state.paths.provider_attempts,
                stage="judge",
                query_id=job.state.handoff.query.query_id,
                paper_id=job.candidate.paper_id,
                whole_job_attempt_index=whole_job_attempt_index,
            )
        future = executor.submit(
            reader.judge_candidate,
            job.state.handoff.query,
            job.candidate,
            **kwargs,
        )
        in_flight[future] = job
        return True

    with ThreadPoolExecutor(
        max_workers=min(workers, len(pending_jobs)),
        thread_name_prefix="aoai-paper-judge",
    ) as executor:
        for _ in range(wave_width):
            submit_one(executor)

        while in_flight or pending_jobs:
            rate_limited: list[tuple[_JudgmentJob, Exception]] = []
            transient_failed: list[tuple[_JudgmentJob, Exception]] = []
            terminal_error: Exception | None = None

            while in_flight:
                completed, _ = wait(
                    in_flight,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                _drain_provider_attempts(provider_attempt_queue)
                durable_successes = 0
                for future in sorted(
                    completed, key=lambda item: in_flight[item].sequence
                ):
                    job = in_flight.pop(future)
                    state = job.state
                    query = state.handoff.query
                    candidate = job.candidate
                    try:
                        judgment = future.result()
                    except Exception as exc:
                        if is_rate_limit_error(exc):
                            _record_uninstrumented_provider_error(
                                path=state.paths.provider_attempts,
                                stage="judge",
                                query_id=query.query_id,
                                paper_id=candidate.paper_id,
                                whole_job_attempt_index=whole_job_attempts[
                                    job.sequence
                                ],
                                error=exc,
                            )
                            rate_limited.append((job, exc))
                            outcome = "RATE_LIMITED"
                        elif is_transient_provider_error(exc):
                            _record_uninstrumented_provider_error(
                                path=state.paths.provider_attempts,
                                stage="judge",
                                query_id=query.query_id,
                                paper_id=candidate.paper_id,
                                whole_job_attempt_index=whole_job_attempts[
                                    job.sequence
                                ],
                                error=exc,
                            )
                            transient_failed.append((job, exc))
                            outcome = "TRANSIENT"
                        elif isinstance(exc, JudgmentResponseExhaustedError):
                            record_error(
                                state.paths.errors,
                                stage="judge",
                                query_id=query.query_id,
                                paper_id=candidate.paper_id,
                                error=exc,
                                details={"calls": exc.calls},
                            )
                            response_failures.append((job, exc))
                            outcome = "RESPONSE_INVALID"
                        else:
                            record_error(
                                state.paths.errors,
                                stage="judge",
                                query_id=query.query_id,
                                paper_id=candidate.paper_id,
                                error=exc,
                            )
                            if terminal_error is None:
                                terminal_error = exc
                            outcome = "ERROR"
                        print(
                            f"[{query.query_id} {job.index}/{job.total}] "
                            f"rank={candidate.rank} {candidate.paper_id} -> "
                            f"{outcome} {type(exc).__name__}: {exc}"
                        )
                        continue

                    current_job_provider_invocations = int(
                        judgment.get("provider_invocation_count") or 0
                    )
                    if getattr(reader, "supports_provider_attempt_ledger", False):
                        _apply_provider_ledger_accounting(
                            judgment,
                            path=state.paths.provider_attempts,
                            stage="judge",
                            paper_id=candidate.paper_id,
                        )
                    state.judgments[candidate.paper_id] = judgment
                    checkpoint_judgment_update(
                        run_dir=run_dir,
                        query_id=query.query_id,
                        judgments_path=state.paths.judgments,
                        judgments=state.judgments,
                        candidates=state.handoff.candidate_papers,
                    )
                    # A deterministic owner rejection is durable progress but
                    # did not exercise AOAI.  It must not earn clean-provider
                    # credit or raise a cap that was reduced after a 429.
                    if current_job_provider_invocations > 0:
                        durable_successes += 1
                    print(
                        f"[{query.query_id} {job.index}/{job.total}] "
                        f"rank={candidate.rank} {candidate.paper_id} -> "
                        f"{judgment['label']}"
                    )

                # A 429 in this recovery wave invalidates its clean-success
                # window.  Successes completed before the first 429 may already
                # have grown the cap, but work draining after it cannot undo the
                # imminent multiplicative decrease.
                if terminal_error is None and not rate_limited:
                    _record_concurrency_successes(
                        controller=concurrency,
                        stage="judge",
                        successes=durable_successes,
                    )
                    wave_width = min(
                        concurrency.limit,
                        len(in_flight) + len(pending_jobs),
                    )

                if (
                    terminal_error is None
                    and not rate_limited
                    and not transient_failed
                ):
                    while (
                        len(in_flight) < wave_width
                        and submit_one(executor)
                    ):
                        pass

            if terminal_error is not None:
                raise terminal_error
            retry_jobs: list[_JudgmentJob] = []
            rate_limit_jobs: list[_JudgmentJob] = []
            rate_limit_round = 0
            if rate_limited:
                # Retry only 429 jobs; successes are already durable and the
                # remaining queue has never been submitted. Stable sequence
                # order keeps retries deterministic across runs.
                rate_limit_jobs = [
                    job
                    for job, _ in sorted(
                        rate_limited, key=lambda item: item[0].sequence
                    )
                ]
                exhausted_pair: tuple[_JudgmentJob, Exception] | None = None
                for job, error in rate_limited:
                    retries = rate_limit_retries.get(job.sequence, 0) + 1
                    rate_limit_retries[job.sequence] = retries
                    if retries > MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS:
                        exhausted_pair = (job, error)
                        break
                if exhausted_pair is not None:
                    exhausted_job, exhausted = exhausted_pair
                    exhausted_state = exhausted_job.state
                    exhausted_query = exhausted_state.handoff.query
                    record_error(
                        exhausted_state.paths.errors,
                        stage="judge",
                        query_id=exhausted_query.query_id,
                        paper_id=exhausted_job.candidate.paper_id,
                        error=exhausted,
                    )
                    print(
                        "AOAI 429 recovery exhausted: stage=judge, "
                        f"paper={exhausted_job.candidate.paper_id}, "
                        f"rounds={MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS}"
                    )
                    raise exhausted
                rate_limit_round = max(
                    rate_limit_retries[job.sequence] for job in rate_limit_jobs
                )
                retry_jobs.extend(rate_limit_jobs)

            if transient_failed:
                transient_jobs = [
                    job
                    for job, _ in sorted(
                        transient_failed, key=lambda item: item[0].sequence
                    )
                ]
                exhausted_pair: tuple[_JudgmentJob, Exception] | None = None
                for job, error in transient_failed:
                    retries = transient_retries.get(job.sequence, 0) + 1
                    transient_retries[job.sequence] = retries
                    if retries > MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS:
                        exhausted_pair = (job, error)
                        break
                if exhausted_pair is not None:
                    exhausted_job, exhausted = exhausted_pair
                    exhausted_state = exhausted_job.state
                    exhausted_query = exhausted_state.handoff.query
                    record_error(
                        exhausted_state.paths.errors,
                        stage="judge",
                        query_id=exhausted_query.query_id,
                        paper_id=exhausted_job.candidate.paper_id,
                        error=exhausted,
                    )
                    print(
                        "AOAI transient recovery exhausted: stage=judge, "
                        f"paper={exhausted_job.candidate.paper_id}, "
                        f"rounds={MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS}"
                    )
                    raise exhausted
                transient_round = max(
                    transient_retries[job.sequence] for job in transient_jobs
                )
                if rate_limited:
                    print(
                        "AOAI transient recovery: stage=judge, "
                        f"round={transient_round}/"
                        f"{MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS}, "
                        f"transient_jobs={len(transient_jobs)}, "
                        "cooldown=covered_by_429_recovery"
                    )
                else:
                    _recover_from_transient_error(
                        stage="judge",
                        round_number=transient_round,
                        transient_jobs=len(transient_jobs),
                        errors=[error for _, error in transient_failed],
                    )
                retry_jobs.extend(transient_jobs)

            if rate_limit_jobs:
                # Account for every retry budget before the long 429 cooldown;
                # a mixed wave must fail immediately if a transient job is
                # already exhausted rather than sleeping for no useful retry.
                _recover_from_rate_limit(
                    controller=concurrency,
                    stage="judge",
                    round_number=rate_limit_round,
                    rate_limited_jobs=len(rate_limit_jobs),
                    errors=[error for _, error in rate_limited],
                )

            if retry_jobs:
                pending_jobs = deque(
                    [
                        *sorted(retry_jobs, key=lambda job: job.sequence),
                        *pending_jobs,
                    ]
                )

            wave_width = min(concurrency.limit, len(pending_jobs))
            while len(in_flight) < wave_width and submit_one(executor):
                pass

        _drain_provider_attempts(provider_attempt_queue)

    if response_failures:
        examples = ", ".join(
            f"{job.state.handoff.query.query_id}/{job.candidate.paper_id}"
            for job, _ in response_failures[:10]
        )
        suffix = "" if len(response_failures) <= 10 else ", ..."
        raise RuntimeError(
            f"{len(response_failures)} candidate judgment response(s) remained "
            "invalid after repair; every other candidate was checkpointed. "
            f"Resume the same run to retry only the missing pairs: {examples}{suffix}"
        )


def run_candidate_judgments(
    *,
    reader: PairwiseAOAIReader,
    query: Any,
    candidates: list[CandidatePaper],
    all_candidates: tuple[CandidatePaper, ...],
    judgments: dict[str, dict[str, Any]],
    paths: QueryRunPaths,
    run_dir: Path,
    workers: int,
    force: bool,
    concurrency: _AdaptiveAOAIConcurrency | None = None,
) -> None:
    """Backward-compatible one-query wrapper around the global pool."""

    handoff = CandidateHandoff(query=query, candidate_papers=all_candidates)
    run_candidate_judgments_globally(
        reader=reader,
        states=[
            QueryExecutionState(
                handoff=handoff,
                paths=paths,
                judgments=judgments,
                target_candidates=candidates,
            )
        ],
        run_dir=run_dir,
        workers=workers,
        force=force,
        concurrency=concurrency,
    )


def _answer_in_worker(
    reader: PairwiseAOAIReader,
    job: _AnswerJob,
    attempt_queue: SimpleQueue[tuple[_AnswerJob, dict[str, Any]]],
    provider_attempt_queue: SimpleQueue[_ProviderAttemptEnvelope],
    whole_job_attempt_index: int,
) -> _AnswerWorkerResult:
    """Make one Stage-2 AOAI call without touching any checkpoint files."""

    try:
        kwargs: dict[str, Any] = {
            "attempt_callback": lambda attempt: attempt_queue.put((job, attempt))
        }
        if getattr(reader, "supports_provider_attempt_ledger", False):
            kwargs["provider_attempt_callback"] = _provider_attempt_callback(
                provider_attempt_queue,
                path=job.state.paths.provider_attempts,
                stage="answer",
                query_id=job.state.handoff.query.query_id,
                paper_id=None,
                whole_job_attempt_index=whole_job_attempt_index,
            )
        prediction, answer_record = reader.answer_from_judgments(
            job.state.handoff.query,
            job.state.handoff.candidate_papers,
            list(job.judgments),
            **kwargs,
        )
    except Exception as exc:
        return _AnswerWorkerResult(None, None, exc)
    return _AnswerWorkerResult(
        prediction,
        answer_record,
        None,
    )


def run_answers_globally(
    *,
    reader: PairwiseAOAIReader,
    states: list[QueryExecutionState],
    run_dir: Path,
    workers: int,
    force: bool,
    require_evidence: bool,
    preinvalidated_query_ids: Iterable[str] = (),
    concurrency: _AdaptiveAOAIConcurrency | None = None,
) -> None:
    """Answer selected queries concurrently with coordinator-only writes.

    A retryable 429, HTTP 5xx, timeout, or connection failure restarts the whole
    query answer, so a successful base call followed by a transient repair can
    be paid twice. Every response emitted through the attempt callback remains
    durable before that whole-job retry.
    """

    concurrency = _concurrency_controller(workers, concurrency)
    invalidated_query_ids = set(preinvalidated_query_ids)
    if force:
        forced_query_ids = {
            state.handoff.query.query_id for state in states
        }
        invalidate_aggregate_queries(
            run_dir, forced_query_ids - invalidated_query_ids
        )
        invalidated_query_ids.update(forced_query_ids)
        for state in states:
            invalidate_forced_answer(
                run_dir=run_dir,
                query_id=state.handoff.query.query_id,
                paths=state.paths,
            )

    pending_jobs: deque[_AnswerJob] = deque()
    for state in states:
        query = state.handoff.query
        checkpoint = validate_judgment_checkpoint(
            state.handoff, state.judgments, reader
        )
        if not checkpoint.complete:
            raise RuntimeError(
                f"{query.query_id}: answer stage requires all "
                f"{len(state.handoff.candidate_papers)} pair judgments; missing "
                f"{len(checkpoint.missing_paper_ids)}: "
                f"{list(checkpoint.missing_paper_ids[:5])}"
            )
        if checkpoint.stale_paper_ids:
            raise ValueError(
                f"{query.query_id}/{checkpoint.stale_paper_ids[0]}: "
                "stale judgment checkpoint"
            )

        judgments = tuple(state.judgments.values())
        expected_answer_key = reader.answer_cache_key(query, list(judgments))
        if state.paths.answer.exists() and not force:
            cached_answer = json.loads(
                state.paths.answer.read_text(encoding="utf-8")
            )
            if not isinstance(cached_answer, dict):
                raise ValueError(
                    f"invalid answer checkpoint: {state.paths.answer}"
                )
            if cached_answer.get("cache_key") != expected_answer_key:
                raise ValueError(
                    f"{query.query_id}: cached answer is stale; rerun with "
                    "--force and this --query-id"
                )
            ensure_submission_from_answer_checkpoint(
                query,
                cached_answer,
                state.paths.submission,
                require_evidence=require_evidence,
            )
            print(f"[{query.query_id}] answer cached")
            continue
        pending_jobs.append(
            _AnswerJob(
                sequence=len(pending_jobs),
                state=state,
                judgments=judgments,
            )
        )

    if not pending_jobs:
        return
    pending_query_ids = {
        job.state.handoff.query.query_id for job in pending_jobs
    }
    invalidate_aggregate_queries(
        run_dir, pending_query_ids - invalidated_query_ids
    )

    in_flight: dict[Future[_AnswerWorkerResult], _AnswerJob] = {}
    attempt_queue: SimpleQueue[tuple[_AnswerJob, dict[str, Any]]] = SimpleQueue()
    provider_attempt_queue: SimpleQueue[_ProviderAttemptEnvelope] = SimpleQueue()
    rate_limit_retries: dict[int, int] = {}
    transient_retries: dict[int, int] = {}
    whole_job_attempts: dict[int, int] = {}
    wave_width = min(concurrency.limit, len(pending_jobs))

    def drain_answer_attempts() -> None:
        # The callback in a worker only enqueues an immutable handoff. Durable
        # logging remains coordinator-owned, and the bounded wait below ensures
        # a repair attempt is saved even while its later AOAI request is slow.
        while True:
            try:
                job, attempt = attempt_queue.get_nowait()
            except Empty:
                return
            state = job.state
            record_answer_attempt(
                state.paths.answer_attempts,
                query_id=state.handoff.query.query_id,
                attempt=attempt,
            )

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        if not pending_jobs:
            return False
        job = pending_jobs.popleft()
        whole_job_attempt_index = whole_job_attempts.get(job.sequence, 0) + 1
        whole_job_attempts[job.sequence] = whole_job_attempt_index
        in_flight[
            executor.submit(
                _answer_in_worker,
                reader,
                job,
                attempt_queue,
                provider_attempt_queue,
                whole_job_attempt_index,
            )
        ] = job
        return True

    with ThreadPoolExecutor(
        max_workers=min(workers, len(pending_jobs)),
        thread_name_prefix="aoai-query-answer",
    ) as executor:
        for _ in range(wave_width):
            submit_one(executor)

        while in_flight or pending_jobs:
            rate_limited: list[tuple[_AnswerJob, Exception]] = []
            transient_failed: list[tuple[_AnswerJob, Exception]] = []
            terminal_error: Exception | None = None

            while in_flight:
                completed, _ = wait(
                    in_flight,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                drain_answer_attempts()
                _drain_provider_attempts(provider_attempt_queue)
                durable_successes = 0
                for future in sorted(
                    completed, key=lambda item: in_flight[item].sequence
                ):
                    job = in_flight.pop(future)
                    state = job.state
                    query = state.handoff.query
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = _AnswerWorkerResult(None, None, exc)

                    if result.error is not None:
                        if is_rate_limit_error(result.error):
                            _record_uninstrumented_provider_error(
                                path=state.paths.provider_attempts,
                                stage="answer",
                                query_id=query.query_id,
                                paper_id=None,
                                whole_job_attempt_index=whole_job_attempts[
                                    job.sequence
                                ],
                                error=result.error,
                            )
                            rate_limited.append((job, result.error))
                            outcome = "RATE_LIMITED"
                        elif is_transient_provider_error(result.error):
                            _record_uninstrumented_provider_error(
                                path=state.paths.provider_attempts,
                                stage="answer",
                                query_id=query.query_id,
                                paper_id=None,
                                whole_job_attempt_index=whole_job_attempts[
                                    job.sequence
                                ],
                                error=result.error,
                            )
                            transient_failed.append((job, result.error))
                            outcome = "TRANSIENT"
                        else:
                            record_error(
                                state.paths.errors,
                                stage="answer",
                                query_id=query.query_id,
                                error=result.error,
                            )
                            if terminal_error is None:
                                terminal_error = result.error
                            outcome = "ERROR"
                        print(
                            f"[{query.query_id}] answer -> {outcome} "
                            f"{type(result.error).__name__}: {result.error}"
                        )
                        continue

                    if result.prediction is None or result.answer_record is None:
                        raise AssertionError("answer worker returned no result")
                    provider_ledger_supported = bool(
                        getattr(reader, "supports_provider_attempt_ledger", False)
                    )
                    current_job_provider_invocations = int(
                        result.answer_record.get("provider_invocation_count")
                        or (1 if not provider_ledger_supported else 0)
                    )
                    if provider_ledger_supported:
                        _apply_provider_ledger_accounting(
                            result.answer_record,
                            path=state.paths.provider_attempts,
                            stage="answer",
                        )
                    try:
                        submission = prediction_to_submission(
                            query,
                            result.prediction,
                            require_evidence=require_evidence,
                        )
                    except Exception as exc:
                        record_error(
                            state.paths.errors,
                            stage="answer",
                            query_id=query.query_id,
                            error=exc,
                        )
                        if terminal_error is None:
                            terminal_error = exc
                        print(
                            f"[{query.query_id}] answer -> ERROR "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    checkpoint_answer_update(
                        run_dir=run_dir,
                        query_id=query.query_id,
                        answer_path=state.paths.answer,
                        answer_record=result.answer_record,
                        submission_path=state.paths.submission,
                        submission=submission,
                    )
                    if current_job_provider_invocations > 0:
                        durable_successes += 1
                    print(f"[{query.query_id}] answer complete")

                if terminal_error is None and not rate_limited:
                    _record_concurrency_successes(
                        controller=concurrency,
                        stage="answer",
                        successes=durable_successes,
                    )
                    wave_width = min(
                        concurrency.limit,
                        len(in_flight) + len(pending_jobs),
                    )

                if (
                    terminal_error is None
                    and not rate_limited
                    and not transient_failed
                ):
                    while (
                        len(in_flight) < wave_width
                        and submit_one(executor)
                    ):
                        pass

            drain_answer_attempts()
            _drain_provider_attempts(provider_attempt_queue)
            if terminal_error is not None:
                raise terminal_error
            retry_jobs: list[_AnswerJob] = []
            rate_limit_jobs: list[_AnswerJob] = []
            rate_limit_round = 0
            if rate_limited:
                rate_limit_jobs = [
                    job
                    for job, _ in sorted(
                        rate_limited, key=lambda item: item[0].sequence
                    )
                ]
                exhausted_pair: tuple[_AnswerJob, Exception] | None = None
                for job, error in rate_limited:
                    retries = rate_limit_retries.get(job.sequence, 0) + 1
                    rate_limit_retries[job.sequence] = retries
                    if retries > MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS:
                        exhausted_pair = (job, error)
                        break
                if exhausted_pair is not None:
                    exhausted_job, exhausted = exhausted_pair
                    exhausted_state = exhausted_job.state
                    exhausted_query = exhausted_state.handoff.query
                    record_error(
                        exhausted_state.paths.errors,
                        stage="answer",
                        query_id=exhausted_query.query_id,
                        error=exhausted,
                    )
                    print(
                        "AOAI 429 recovery exhausted: stage=answer, "
                        f"query={exhausted_query.query_id}, "
                        f"rounds={MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS}"
                    )
                    raise exhausted
                rate_limit_round = max(
                    rate_limit_retries[job.sequence] for job in rate_limit_jobs
                )
                retry_jobs.extend(rate_limit_jobs)

            if transient_failed:
                transient_jobs = [
                    job
                    for job, _ in sorted(
                        transient_failed, key=lambda item: item[0].sequence
                    )
                ]
                exhausted_pair: tuple[_AnswerJob, Exception] | None = None
                for job, error in transient_failed:
                    retries = transient_retries.get(job.sequence, 0) + 1
                    transient_retries[job.sequence] = retries
                    if retries > MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS:
                        exhausted_pair = (job, error)
                        break
                if exhausted_pair is not None:
                    exhausted_job, exhausted = exhausted_pair
                    exhausted_state = exhausted_job.state
                    exhausted_query = exhausted_state.handoff.query
                    record_error(
                        exhausted_state.paths.errors,
                        stage="answer",
                        query_id=exhausted_query.query_id,
                        error=exhausted,
                    )
                    print(
                        "AOAI transient recovery exhausted: stage=answer, "
                        f"query={exhausted_query.query_id}, "
                        f"rounds={MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS}"
                    )
                    raise exhausted
                transient_round = max(
                    transient_retries[job.sequence] for job in transient_jobs
                )
                if rate_limited:
                    print(
                        "AOAI transient recovery: stage=answer, "
                        f"round={transient_round}/"
                        f"{MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS}, "
                        f"transient_jobs={len(transient_jobs)}, "
                        "cooldown=covered_by_429_recovery"
                    )
                else:
                    _recover_from_transient_error(
                        stage="answer",
                        round_number=transient_round,
                        transient_jobs=len(transient_jobs),
                        errors=[error for _, error in transient_failed],
                    )
                retry_jobs.extend(transient_jobs)

            if rate_limit_jobs:
                _recover_from_rate_limit(
                    controller=concurrency,
                    stage="answer",
                    round_number=rate_limit_round,
                    rate_limited_jobs=len(rate_limit_jobs),
                    errors=[error for _, error in rate_limited],
                )

            if retry_jobs:
                pending_jobs = deque(
                    [
                        *sorted(retry_jobs, key=lambda job: job.sequence),
                        *pending_jobs,
                    ]
                )

            wave_width = min(concurrency.limit, len(pending_jobs))
            while len(in_flight) < wave_width and submit_one(executor):
                pass

        drain_answer_attempts()
        _drain_provider_attempts(provider_attempt_queue)


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


def execute_locked_run(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    store: ChunkStore,
    llm: Any,
    preflight: dict[str, Any],
    run_dir: Path,
    all_handoffs: list[CandidateHandoff],
    selected_handoffs: list[CandidateHandoff],
    require_evidence: bool,
) -> None:
    """Run every paid call and checkpoint while holding the run-dir lock."""

    manifest = build_manifest(args, config, store, all_handoffs)
    ensure_manifest(run_dir / "manifest.json", manifest, args.resume)
    atomic_write_json(run_dir / "preflight.json", preflight)

    reader = PairwiseAOAIReader(store, llm, **(config.get("params") or {}))
    # A previous forced run may have stopped after making only some target
    # judgments missing.  Normalize that state before aggregate validation so
    # --resume can continue instead of being blocked by an obsolete answer.
    preinvalidated_query_ids: set[str] = set()
    if args.force:
        preinvalidated_query_ids = {
            handoff.query.query_id for handoff in selected_handoffs
        }
        invalidate_aggregate_queries(run_dir, preinvalidated_query_ids)
        for handoff in selected_handoffs:
            paths = QueryRunPaths.under(run_dir, handoff.query.query_id)
            paths.directory.mkdir(parents=True, exist_ok=True)
            if args.stage in {"all", "judge"}:
                target_candidates = list(handoff.candidate_papers)
                if args.paper_id:
                    target_candidates = [
                        candidate
                        for candidate in target_candidates
                        if candidate.paper_id == args.paper_id
                    ]
                    if not target_candidates:
                        raise ValueError(
                            f"{handoff.query.query_id}: --paper-id is not in the "
                            f"candidate ranking: {args.paper_id}"
                        )
                judgments = load_judgments(
                    paths.judgments, handoff.query.query_id
                )
                invalidate_forced_checkpoints(
                    run_dir=run_dir,
                    query_id=handoff.query.query_id,
                    paths=paths,
                    judgments=judgments,
                    target_candidates=target_candidates,
                    all_candidates=handoff.candidate_papers,
                )
            else:
                invalidate_forced_answer(
                    run_dir=run_dir,
                    query_id=handoff.query.query_id,
                    paths=paths,
                )
    # Validate every previously materialized query against current chunks and
    # image bytes before making another API call. This catches an image repaired
    # or replaced between incremental runs, even when that query is not selected.
    materialize_run_outputs(
        run_dir,
        all_handoffs,
        reader,
        require_evidence=require_evidence,
    )
    # Materialization may recreate a diagnostic trace for a forced query from
    # its surviving per-query checkpoints. Treat the aggregates as current
    # again so each global stage removes all of its pending queries in one
    # coordinator-owned rewrite immediately before any API work is submitted.
    preinvalidated_query_ids = set()
    states: list[QueryExecutionState] = []
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
        states.append(
            QueryExecutionState(
                handoff=handoff,
                paths=paths,
                judgments=judgments,
                target_candidates=target_candidates,
            )
        )

    # One provider deployment and one launch pacer serve both paid stages, so
    # keep the learned concurrency cap as well.  Stage 2 must not jump back to
    # the CLI maximum immediately after Stage 1 observed a 429.
    concurrency = _AdaptiveAOAIConcurrency(args.workers)
    if args.stage in {"all", "judge"}:
        run_candidate_judgments_globally(
            reader=reader,
            states=states,
            run_dir=run_dir,
            workers=args.workers,
            force=args.force,
            preinvalidated_query_ids=preinvalidated_query_ids,
            concurrency=concurrency,
        )

    if args.stage in {"all", "answer"}:
        run_answers_globally(
            reader=reader,
            states=states,
            run_dir=run_dir,
            workers=args.workers,
            force=args.force,
            require_evidence=require_evidence,
            preinvalidated_query_ids=preinvalidated_query_ids,
            concurrency=concurrency,
        )
    trace_count, submission_count = materialize_run_outputs(
        run_dir,
        all_handoffs,
        reader,
        require_evidence=require_evidence,
    )
    print(f"wrote {trace_count} traces to {run_dir / 'reading_traces.jsonl'}")
    print(f"wrote {submission_count} submissions to {run_dir / 'submission.jsonl'}")


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
        image_workers=min(args.workers, 64),
    )
    preflight["named_owner_resolution"] = _named_owner_audit(
        selected_handoffs,
        paper_id=args.paper_id,
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
    print(
        "named owner gate: "
        f"{preflight['named_owner_resolution']['deterministic_owner_rejections']} "
        "candidate pairs will be checkpointed without AOAI"
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
    with run_directory_lock(run_dir):
        execute_locked_run(
            args=args,
            config=config,
            store=store,
            llm=llm,
            preflight=preflight,
            run_dir=run_dir,
            all_handoffs=all_handoffs,
            selected_handoffs=selected_handoffs,
            require_evidence=require_evidence,
        )


if __name__ == "__main__":
    main()
