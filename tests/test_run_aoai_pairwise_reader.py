from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from littraceqa import pairwise_run_store as pairwise_run_store_module
from littraceqa.aoai_pairwise_reader import JudgmentResponseExhaustedError
from littraceqa.candidate_handoff import CandidatePaper
from littraceqa.di_pipeline.contracts import (
    Answer,
    Evidence,
    EvidenceLocator,
    Prediction,
    Query,
)
from littraceqa.di_pipeline.llm import azure_openai as azure_openai_module
from littraceqa.mineru_record import (
    MAX_IMAGE_BYTES,
    ImageValidationError,
    readable_image_path,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_aoai_pairwise_reader", ROOT / "scripts/run_aoai_pairwise_reader.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
_SAFE_QUERY_ID = _RUNNER._SAFE_QUERY_ID
invalidate_aggregate_query = pairwise_run_store_module.invalidate_aggregate_query
invalidate_aggregate_queries = pairwise_run_store_module.invalidate_aggregate_queries

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
VALID_BMP = base64.b64decode(
    "Qk06AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABABgAAAAAAAQAAADEDgAAxA4AA"
    "AAAAAAAAAAAAAD/AA=="
)


def test_shared_azure_client_keeps_generic_default_system(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage=None,
                model="test-model",
                id="test-request",
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(
        azure_openai_module,
        "AzureOpenAI",
        lambda **_kwargs: client,
    )
    llm = azure_openai_module.AzureOpenAILLM(
        endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="test-version",
        deployment="test-deployment",
    )

    assert llm("Return an empty JSON object.") == "{}"
    system = captured["messages"][0]["content"]
    assert "科学論文の検索システムの一部" in system
    assert "与えられた候補論文と根拠だけ" not in system
    assert "検索や外部知識を使わない" not in system
    with pytest.raises(ValueError, match="at most 10 images"):
        llm.complete_with_metadata("Too many images", ["unused.jpg"] * 11)


def test_azure_image_payload_rejects_corrupt_file_before_upload(tmp_path):
    image = tmp_path / "corrupt.png"
    image.write_bytes(b"not-an-image")

    with pytest.raises(ImageValidationError, match="unsupported/corrupt"):
        azure_openai_module._image_data_url(image)


def test_azure_upload_revalidates_bytes_after_readability_cache(tmp_path):
    image = tmp_path / "changed-after-check.png"
    image.write_bytes(VALID_PNG)
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#fig1",
        "chunk_type": "figure",
        "metadata": {"image_path": str(image)},
    }
    assert readable_image_path(record) == str(image)

    # A prior readability result is never sufficient for upload. The adapter
    # must validate the bytes it is about to base64-encode, even when the path
    # and byte count still look plausible.
    image.write_bytes(b"x" * len(VALID_PNG))
    with pytest.raises(ImageValidationError, match="unsupported/corrupt"):
        azure_openai_module._image_data_url(image)


def test_azure_upload_rechecks_size_after_readability_cache(tmp_path):
    image = tmp_path / "grown-after-check.png"
    image.write_bytes(VALID_PNG)
    record = {
        "paper_id": "p1",
        "chunk_id": "p1#fig1",
        "chunk_type": "figure",
        "metadata": {"image_path": str(image)},
    }
    assert readable_image_path(record) == str(image)

    with image.open("r+b") as handle:
        handle.truncate(MAX_IMAGE_BYTES + 1)
    with pytest.raises(ImageValidationError, match="exceeds"):
        azure_openai_module._image_data_url(image)


def test_azure_image_payload_rejects_oversized_file_before_reading(tmp_path):
    image = tmp_path / "oversized.jpg"
    with image.open("wb") as handle:
        handle.truncate(MAX_IMAGE_BYTES + 1)

    with pytest.raises(ImageValidationError, match="exceeds"):
        azure_openai_module._image_data_url(image)


def test_azure_image_payload_uses_validated_content_mime(tmp_path):
    image = tmp_path / "misleading-extension.jpg"
    image.write_bytes(VALID_PNG)

    data_url = azure_openai_module._image_data_url(image)
    header, encoded = data_url.split(",", 1)

    assert header == "data:image/png;base64"
    assert base64.b64decode(encoded) == VALID_PNG


def test_azure_image_payload_rejects_decodable_but_unsupported_format(tmp_path):
    image = tmp_path / "unsupported.bmp"
    image.write_bytes(VALID_BMP)

    with pytest.raises(ImageValidationError, match="unsupported image format BMP"):
        azure_openai_module._image_data_url(image)


def test_pairwise_build_llm_injects_fixed_grounding_system(monkeypatch):
    captured = {}

    class StubAzureOpenAILLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(_RUNNER, "AzureOpenAILLM", StubAzureOpenAILLM)

    _RUNNER.build_llm(
        {
            "llm": {
                "name": "azure_openai",
                "params": {
                    "json_mode": False,
                    "system": "This must not weaken the pairwise policy.",
                },
            }
        }
    )

    assert captured["json_mode"] is False
    assert captured["system"] == _RUNNER._PAIRWISE_SYSTEM
    assert "Use only the supplied official query" in captured["system"]
    assert "return JSON only" in captured["system"]
    assert "Do not browse, use external knowledge" in captured["system"]


def test_evidence_policy_is_explicit_and_never_inferred_from_filename() -> None:
    assert _RUNNER.require_evidence_for_policy("required") is True
    assert _RUNNER.require_evidence_for_policy("optional") is False
    with pytest.raises(ValueError, match="unknown evidence policy"):
        _RUNNER.require_evidence_for_policy("auto")

    parser = _RUNNER.build_parser()
    args = parser.parse_args(
        [
            "--queries",
            "queries.jsonl",
            "--candidates",
            "candidates.jsonl",
            "--chunks",
            "chunks.jsonl",
            "--run-dir",
            "run",
        ]
    )
    assert args.evidence_policy == "required"
    assert args.workers == 1


@pytest.mark.parametrize("workers", ["0", "-1", "101", "not-an-integer"])
def test_workers_reject_unsafe_values(workers: str) -> None:
    parser = _RUNNER.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--queries",
                "queries.jsonl",
                "--candidates",
                "candidates.jsonl",
                "--chunks",
                "chunks.jsonl",
                "--run-dir",
                "run",
                "--workers",
                workers,
            ]
        )


@pytest.mark.parametrize("workers", ["1", "8", "50", "100"])
def test_workers_accept_safe_boundaries(workers: str) -> None:
    parser = _RUNNER.build_parser()
    args = parser.parse_args(
        [
            "--queries",
            "queries.jsonl",
            "--candidates",
            "candidates.jsonl",
            "--chunks",
            "chunks.jsonl",
            "--run-dir",
            "run",
            "--workers",
            workers,
        ]
    )
    assert args.workers == int(workers)


def test_query_id_path_guard_rejects_dot_segments():
    assert _SAFE_QUERY_ID.fullmatch("q_001")
    assert not _SAFE_QUERY_ID.fullmatch(".")
    assert not _SAFE_QUERY_ID.fullmatch("..")
    assert not _SAFE_QUERY_ID.fullmatch("../escape")


def test_missing_figure_fallback_is_explicit_and_part_of_manifest(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "text",
                "metadata": {"page": 1},
            }
        ],
    )
    input_paths = {}
    for name in ("queries", "candidates", "paper_metadata", "reader"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        input_paths[name] = path
    base_args = [
        "--queries",
        str(input_paths["queries"]),
        "--candidates",
        str(input_paths["candidates"]),
        "--paper-metadata",
        str(input_paths["paper_metadata"]),
        "--chunks",
        str(chunks),
        "--reader",
        str(input_paths["reader"]),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    parser = _RUNNER.build_parser()
    strict_args = parser.parse_args(base_args)
    allowed_args = parser.parse_args(
        [*base_args, "--allow-missing-required-visual-images"]
    )
    store = _RUNNER.ChunkStore(chunks)
    config = {"llm": {"name": "fake", "params": {}}, "params": {}}

    strict_manifest = _RUNNER.build_manifest(strict_args, config, store)
    allowed_manifest = _RUNNER.build_manifest(allowed_args, config, store)
    owner_handoff = _RUNNER.CandidateHandoff(
        Query(
            "q_owner",
            "How many subfigures are in Figure 4 of the DynaPipe paper?",
            ["multiple_choice"],
            options={"A": "2", "B": "4"},
        ),
        (
            CandidatePaper("p1", 1, "DynaPipe: Dynamic Layer Redistribution"),
            CandidatePaper("p2", 2, "Another Pipeline Paper"),
        ),
    )
    audited_manifest = _RUNNER.build_manifest(
        strict_args,
        config,
        store,
        [owner_handoff],
    )

    assert strict_args.allow_missing_figure_images is False
    assert allowed_args.allow_missing_figure_images is True
    assert strict_manifest["reader"]["allow_missing_figure_images"] is False
    assert strict_manifest["reader"]["named_owner_resolution"] == {
        "version": _RUNNER.NAMED_OWNER_RESOLVER_VERSION,
        "scope": (
            "unique_literal_grammatical_owner_of_local_object_fuzzy_soft_only"
        ),
    }
    assert audited_manifest["reader"]["named_owner_resolution"][
        "deterministic_owner_rejections"
    ] == 1
    assert audited_manifest["reader"]["named_owner_resolution"]["queries"][0][
        "paper_id"
    ] == "p1"
    assert allowed_manifest["reader"]["allow_missing_figure_images"] is True
    assert strict_manifest != allowed_manifest


def test_evidence_policy_is_part_of_resume_manifest(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "text",
                "metadata": {"page": 1},
            }
        ],
    )
    input_paths = {}
    for name in ("queries", "candidates", "paper_metadata", "reader"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        input_paths[name] = path
    base_args = [
        "--queries",
        str(input_paths["queries"]),
        "--candidates",
        str(input_paths["candidates"]),
        "--paper-metadata",
        str(input_paths["paper_metadata"]),
        "--chunks",
        str(chunks),
        "--reader",
        str(input_paths["reader"]),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    parser = _RUNNER.build_parser()
    required_args = parser.parse_args(base_args)
    optional_args = parser.parse_args([*base_args, "--evidence-policy", "optional"])
    store = _RUNNER.ChunkStore(chunks)
    config = {"llm": {"name": "fake", "params": {}}, "params": {}}

    required = _RUNNER.build_manifest(required_args, config, store)
    optional = _RUNNER.build_manifest(optional_args, config, store)

    assert required["reader"]["evidence_policy"] == "required"
    assert required["reader"]["require_evidence"] is True
    assert optional["reader"]["evidence_policy"] == "optional"
    assert optional["reader"]["require_evidence"] is False
    assert required != optional


def test_missing_visual_override_is_rejected_outside_judge_stage(tmp_path):
    command = [
        sys.executable,
        "scripts/run_aoai_pairwise_reader.py",
        "--queries",
        str(tmp_path / "queries.jsonl"),
        "--candidates",
        str(tmp_path / "candidates.jsonl"),
        "--chunks",
        str(tmp_path / "chunks.jsonl"),
        "--run-dir",
        str(tmp_path / "run"),
        "--allow-missing-required-visual-images",
    ]

    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert "diagnostic --stage judge" in result.stderr


def test_legacy_missing_figure_flag_remains_a_cli_alias():
    parser = _RUNNER.build_parser()
    base_args = [
        "--queries",
        "queries.jsonl",
        "--candidates",
        "candidates.jsonl",
        "--chunks",
        "chunks.jsonl",
        "--run-dir",
        "run",
    ]

    preferred = parser.parse_args(
        [*base_args, "--allow-missing-required-visual-images"]
    )
    legacy = parser.parse_args([*base_args, "--allow-missing-figure-images"])

    assert preferred.allow_missing_figure_images is True
    assert legacy.allow_missing_figure_images is True


def test_resolve_image_root_prefers_cli_and_resolves_config_from_repo(tmp_path):
    repo_root = tmp_path / "repo"
    config_root = repo_root / "artifacts" / "images"
    cli_root = tmp_path / "cli-images"
    config_root.mkdir(parents=True)
    cli_root.mkdir()
    config = {"image_root": "artifacts/images"}

    resolved_config, config_source = _RUNNER.resolve_image_root(
        None, config, repo_root=repo_root
    )
    resolved_cli, cli_source = _RUNNER.resolve_image_root(
        cli_root, config, repo_root=repo_root
    )

    assert resolved_config == str(config_root.resolve())
    assert config_source == "config"
    assert resolved_cli == str(cli_root.resolve())
    assert cli_source == "cli"


def test_resolve_image_root_reports_disabled_when_unconfigured(tmp_path):
    resolved, source = _RUNNER.resolve_image_root(
        None,
        {},
        repo_root=tmp_path,
    )

    assert resolved is None
    assert source == "disabled"


def test_resolve_image_root_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="config image root is not a directory"):
        _RUNNER.resolve_image_root(
            None,
            {"image_root": "missing-images"},
            repo_root=tmp_path,
        )


def test_manifest_fingerprints_all_pairwise_runtime_dependencies(
    tmp_path, monkeypatch
):
    expected_dependencies = {
        "src/littraceqa/__init__.py",
        "src/littraceqa/di_pipeline/__init__.py",
        "src/littraceqa/di_pipeline/agent/__init__.py",
        "src/littraceqa/di_pipeline/agent/evidence.py",
        "src/littraceqa/di_pipeline/agent/json_utils.py",
        "src/littraceqa/di_pipeline/contracts.py",
        "src/littraceqa/di_pipeline/llm/azure_openai.py",
        "src/littraceqa/di_pipeline/llm/base.py",
        "src/littraceqa/di_pipeline/llm/fake.py",
        "src/littraceqa/di_pipeline/llm/__init__.py",
        "src/littraceqa/di_pipeline/registry.py",
    }
    actual_dependencies = {
        str(path.relative_to(ROOT)) for path in _RUNNER._RUNTIME_FILES
    }
    assert expected_dependencies <= actual_dependencies

    sandbox_root = tmp_path / "repo"
    sandbox_runtime_files = []
    for relative_path in sorted(actual_dependencies):
        path = sandbox_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"initial:{relative_path}\n", encoding="utf-8")
        sandbox_runtime_files.append(path)

    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "text",
                "metadata": {"page": 1},
            }
        ],
    )
    input_paths = {}
    for name in ("queries", "candidates", "paper_metadata", "reader"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        input_paths[name] = path
    args = _RUNNER.build_parser().parse_args(
        [
            "--queries",
            str(input_paths["queries"]),
            "--candidates",
            str(input_paths["candidates"]),
            "--paper-metadata",
            str(input_paths["paper_metadata"]),
            "--chunks",
            str(chunks),
            "--reader",
            str(input_paths["reader"]),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    config = {"llm": {"name": "fake", "params": {}}, "params": {}}
    store = _RUNNER.ChunkStore(chunks)
    monkeypatch.setattr(_RUNNER, "ROOT", sandbox_root)
    monkeypatch.setattr(
        _RUNNER, "_RUNTIME_FILES", tuple(sandbox_runtime_files)
    )

    before = _RUNNER.build_manifest(args, config, store)
    evidence_path = sandbox_root / "src/littraceqa/di_pipeline/agent/evidence.py"
    evidence_path.write_text("changed evidence serializer\n", encoding="utf-8")
    after = _RUNNER.build_manifest(args, config, store)

    relative_evidence_path = "src/littraceqa/di_pipeline/agent/evidence.py"
    assert (
        before["runtime"][relative_evidence_path]["sha256"]
        != after["runtime"][relative_evidence_path]["sha256"]
    )
    assert before != after


def test_mutated_query_is_removed_from_aggregate_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "reading_traces.jsonl",
        [{"query_id": "q1"}, {"query_id": "q2"}],
    )
    _write_jsonl(
        run_dir / "submission.jsonl",
        [{"query_id": "q1"}, {"query_id": "q2"}],
    )

    invalidate_aggregate_query(run_dir, "q1")

    assert json.loads((run_dir / "reading_traces.jsonl").read_text())["query_id"] == "q2"
    assert json.loads((run_dir / "submission.jsonl").read_text())["query_id"] == "q2"


def test_run_directory_lock_is_exclusive_and_reusable(tmp_path):
    run_dir = tmp_path / "run"

    with _RUNNER.run_directory_lock(run_dir):
        owner = json.loads((run_dir / ".run.lock").read_text(encoding="utf-8"))
        assert owner["pid"] > 0
        with pytest.raises(RuntimeError, match="run directory is already active"):
            with _RUNNER.run_directory_lock(run_dir):
                pass
        with pytest.raises(RuntimeError, match="run directory is already active"):
            with _RUNNER.run_directory_lock(run_dir):
                pass

    # The file remains for a stable inode, but the OS lock is released both on
    # normal exit and exception unwinding.
    assert (run_dir / ".run.lock").is_file()
    with pytest.raises(RuntimeError, match="sentinel"):
        with _RUNNER.run_directory_lock(run_dir):
            raise RuntimeError("sentinel")
    with _RUNNER.run_directory_lock(run_dir):
        pass


def test_aggregate_invalidation_removes_uploadable_submission_first(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "reading_traces.jsonl", [{"query_id": "q1"}])
    _write_jsonl(run_dir / "submission.jsonl", [{"query_id": "q1"}])
    writes = []
    real_atomic_write_jsonl = pairwise_run_store_module.atomic_write_jsonl

    def recording_write(path, records):
        writes.append(path.name)
        real_atomic_write_jsonl(path, records)

    monkeypatch.setattr(
        pairwise_run_store_module, "atomic_write_jsonl", recording_write
    )

    invalidate_aggregate_query(run_dir, "q1")

    assert writes == ["submission.jsonl", "reading_traces.jsonl"]


def test_bulk_aggregate_invalidation_parses_and_rewrites_each_root_once(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [
        {"query_id": "q1"},
        {"query_id": "q2"},
        {"query_id": "q3"},
    ]
    _write_jsonl(run_dir / "submission.jsonl", records)
    _write_jsonl(run_dir / "reading_traces.jsonl", records)
    reads = []
    writes = []
    real_read_jsonl = pairwise_run_store_module.read_jsonl
    real_atomic_write_jsonl = pairwise_run_store_module.atomic_write_jsonl

    def recording_read(path):
        reads.append(path.name)
        return real_read_jsonl(path)

    def recording_write(path, output_records):
        writes.append(path.name)
        real_atomic_write_jsonl(path, output_records)

    monkeypatch.setattr(pairwise_run_store_module, "read_jsonl", recording_read)
    monkeypatch.setattr(
        pairwise_run_store_module, "atomic_write_jsonl", recording_write
    )

    invalidate_aggregate_queries(run_dir, ["q1", "q2", "q1"])

    assert reads == ["submission.jsonl", "reading_traces.jsonl"]
    assert writes == ["submission.jsonl", "reading_traces.jsonl"]
    assert json.loads(
        (run_dir / "submission.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q3"
    assert json.loads(
        (run_dir / "reading_traces.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q3"


def test_judgment_update_only_writes_per_query_checkpoint(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        _RUNNER,
        "write_judgments",
        lambda path, judgments, candidates: events.append(
            ("checkpoint", path, judgments, candidates)
        ),
    )

    _RUNNER.checkpoint_judgment_update(
        run_dir=tmp_path / "run",
        query_id="q1",
        judgments_path=tmp_path / "judgments.jsonl",
        judgments={"p1": {"paper_id": "p1"}},
        candidates=(CandidatePaper("p1", 1),),
    )

    assert [event[0] for event in events] == ["checkpoint"]


class _HTTPStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str = "request failed") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_rate_limit_detection_uses_structured_status_across_wrappers():
    direct = _HTTPStatusError(429)
    response_error = RuntimeError("wrapped response")
    response_error.response = SimpleNamespace(status_code="429")
    outer = RuntimeError("adapter wrapper")
    outer.__cause__ = response_error

    assert _RUNNER.is_rate_limit_error(direct)
    assert _RUNNER.is_rate_limit_error(outer)
    assert _RUNNER.is_rate_limit_error(
        ExceptionGroup("requests", [RuntimeError("other"), direct])
    )
    assert not _RUNNER.is_rate_limit_error(_HTTPStatusError(503))
    assert not _RUNNER.is_rate_limit_error(RuntimeError("HTTP 429 in text only"))


def _rate_limit_with_headers(headers: dict[str, str]) -> _HTTPStatusError:
    error = _HTTPStatusError(429)
    error.response = SimpleNamespace(status_code=429, headers=headers)
    return error


def test_retry_after_prefers_provider_milliseconds_and_clamps():
    error = _rate_limit_with_headers(
        {"Retry-After-Ms": "250", "Retry-After": "9"}
    )
    huge = _rate_limit_with_headers({"retry-after": "99999"})

    assert _RUNNER._retry_after_seconds(error) == 0.25
    assert (
        _RUNNER._retry_after_seconds(huge)
        == _RUNNER.MAX_AOAI_RETRY_AFTER_SECONDS
    )
    assert _RUNNER._retry_after_seconds(_HTTPStatusError(429)) is None


def test_expired_retry_after_http_date_means_no_additional_wait():
    error = _rate_limit_with_headers(
        {"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"}
    )

    assert _RUNNER._retry_after_seconds(error) == 0.0


def test_rate_limit_recovery_uses_longest_provider_delay(monkeypatch, capsys):
    sleeps: list[float] = []
    monkeypatch.setattr(_RUNNER.time, "sleep", sleeps.append)
    controller = _RUNNER._AdaptiveAOAIConcurrency(4)

    _RUNNER._recover_from_rate_limit(
        controller=controller,
        stage="judge",
        round_number=1,
        rate_limited_jobs=2,
        errors=[
            _rate_limit_with_headers({"retry-after-ms": "250"}),
            _rate_limit_with_headers({"retry-after": "1.5"}),
        ],
    )

    assert sleeps == [1.5]
    assert controller.limit == 3
    output = capsys.readouterr().out
    assert "cooldown_seconds=1.5" in output
    assert "cooldown_source=provider" in output


def test_transient_recovery_honors_longer_provider_delay(monkeypatch, capsys):
    sleeps: list[float] = []
    monkeypatch.setattr(_RUNNER.time, "sleep", sleeps.append)
    error = _HTTPStatusError(503)
    error.response = SimpleNamespace(
        status_code=503, headers={"retry-after": "20"}
    )

    _RUNNER._recover_from_transient_error(
        stage="judge",
        round_number=2,
        transient_jobs=1,
        errors=[error],
    )

    assert sleeps == [20.0]
    assert "cooldown_source=provider" in capsys.readouterr().out


def test_transient_provider_detection_is_structured_and_excludes_fatal_4xx():
    direct = _HTTPStatusError(500)
    response_error = RuntimeError("wrapped response")
    response_error.response = {"status_code": "503"}
    outer = RuntimeError("adapter wrapper")
    outer.__cause__ = response_error

    assert _RUNNER.is_transient_provider_error(direct)
    assert all(
        _RUNNER.is_transient_provider_error(_HTTPStatusError(status))
        for status in (408, 409, 425, 501, 599)
    )
    assert _RUNNER.is_transient_provider_error(outer)
    assert _RUNNER.is_transient_provider_error(
        ExceptionGroup("requests", [RuntimeError("other"), direct])
    )
    assert all(
        not _RUNNER.is_transient_provider_error(_HTTPStatusError(status))
        for status in (400, 401, 403, 404, 422, 429)
    )
    assert not _RUNNER.is_transient_provider_error(_HTTPStatusError(429))
    assert not _RUNNER.is_transient_provider_error(
        RuntimeError("upstream connect error in text only")
    )
    request = httpx.Request("POST", "https://example.invalid")
    assert _RUNNER.is_transient_provider_error(
        _RUNNER.openai.APIConnectionError(request=request)
    )
    assert _RUNNER.is_transient_provider_error(
        _RUNNER.openai.APITimeoutError(request)
    )


def test_adaptive_concurrency_decreases_strictly_from_50_to_1():
    caps = [50]
    while caps[-1] > 1:
        caps.append(_RUNNER._reduced_aoai_concurrency(caps[-1]))

    assert caps == [50, 38, 29, 22, 17, 13, 10, 8, 6, 5, 4, 3, 2, 1]


def test_large_start_cap_halves_twice_before_proportional_backoff():
    first = _RUNNER._reduced_aoai_concurrency(
        100, reduction_step=1, aggressive_backoff=True
    )
    second = _RUNNER._reduced_aoai_concurrency(
        first, reduction_step=2, aggressive_backoff=True
    )
    third = _RUNNER._reduced_aoai_concurrency(
        second, reduction_step=3, aggressive_backoff=True
    )

    assert (first, second, third) == (50, 25, 19)
    assert _RUNNER._reduced_aoai_concurrency(
        100, reduction_step=1, aggressive_backoff=False
    ) == 75


def test_shared_aimd_controller_recovers_by_five_after_clean_windows():
    controller = _RUNNER._AdaptiveAOAIConcurrency(100)

    # Successes at the ceiling are deliberately not banked before congestion.
    assert controller.record_successes(500) is None
    assert controller.clean_success_credit == 0
    assert controller.record_rate_limit() == (100, 50, 1)
    assert controller.record_rate_limit() == (50, 25, 2)

    assert controller.record_successes(24) is None
    assert controller.limit == 25
    assert controller.record_successes(1) == (25, 30)
    assert controller.congestion_streak == 0

    transitions = []
    while controller.limit < controller.maximum:
        old_limit = controller.limit
        change = controller.record_successes(
            max(old_limit, _RUNNER.AOAI_CONCURRENCY_MIN_SUCCESS_WINDOW)
        )
        assert change is not None
        transitions.append(change)

    assert transitions[0] == (30, 35)
    assert transitions[-1] == (95, 100)
    assert all(new - old == 5 for old, new in transitions)
    assert controller.clean_success_credit == 0


def test_shared_aimd_controller_uses_cap_not_current_tail_size():
    controller = _RUNNER._AdaptiveAOAIConcurrency(50)

    # A stage may have only two jobs left, but that is not a provider cap of two.
    # The actual target is min(controller.limit, remaining jobs) at submission.
    assert controller.record_rate_limit() == (50, 38, 1)
    assert min(controller.limit, 2) == 2


def test_shared_aimd_controller_rejects_a_different_worker_ceiling():
    controller = _RUNNER._AdaptiveAOAIConcurrency(100)

    with pytest.raises(ValueError, match="does not match --workers"):
        _RUNNER._concurrency_controller(50, controller)


class _ConcurrentJudgmentReader:
    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        fail_paper_id: str | None = None,
        response_invalid_paper_id: str | None = None,
        rate_limit_failures: dict[str, int] | None = None,
        transient_failures: dict[str, int] | None = None,
        zero_provider_paper_ids: set[str] | None = None,
    ) -> None:
        self.chunk_store = self
        self.barrier = barrier
        self.fail_paper_id = fail_paper_id
        self.response_invalid_paper_id = response_invalid_paper_id
        self.rate_limit_failures = dict(rate_limit_failures or {})
        self.transient_failures = dict(transient_failures or {})
        self.zero_provider_paper_ids = set(zero_provider_paper_ids or set())
        self.calls: list[str] = []
        self.query_calls: list[tuple[str, str]] = []
        self.call_counts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0
        self.retry_active = 0
        self.max_retry_active = 0
        self.lock = threading.Lock()

    def load_paper(self, paper_id: str) -> list[dict]:
        return [{"paper_id": paper_id, "text": paper_id}]

    def judgment_cache_key(self, query, candidate, records) -> str:
        del query, records
        return f"cache:{candidate.paper_id}"

    def judge_candidate(self, query, candidate) -> dict:
        with self.lock:
            self.calls.append(candidate.paper_id)
            self.query_calls.append((query.query_id, candidate.paper_id))
            call_count = self.call_counts.get(candidate.paper_id, 0) + 1
            self.call_counts[candidate.paper_id] = call_count
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if call_count > 1:
                self.retry_active += 1
                self.max_retry_active = max(
                    self.max_retry_active, self.retry_active
                )
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            remaining_rate_limits = self.rate_limit_failures.get(
                candidate.paper_id, 0
            )
            if remaining_rate_limits:
                self.rate_limit_failures[candidate.paper_id] = (
                    remaining_rate_limits - 1
                )
                raise _HTTPStatusError(429, f"rate limit for {candidate.paper_id}")
            remaining_transients = self.transient_failures.get(
                candidate.paper_id, 0
            )
            if remaining_transients:
                self.transient_failures[candidate.paper_id] = (
                    remaining_transients - 1
                )
                raise _HTTPStatusError(
                    500, f"transient failure for {candidate.paper_id}"
                )
            if candidate.paper_id == self.fail_paper_id:
                raise RuntimeError(f"failure for {candidate.paper_id}")
            if candidate.paper_id == self.response_invalid_paper_id:
                raise JudgmentResponseExhaustedError(
                    f"invalid response for {candidate.paper_id}",
                    calls=[
                        {
                            "attempt": "initial",
                            "raw_response": "{}",
                            "parse_error": "invalid",
                        }
                    ],
                )
            if call_count > 1:
                time.sleep(0.03)
            # Reverse completion order without making concurrency assertions
            # depend on elapsed wall-clock time.
            time.sleep(max(0, 4 - candidate.rank) * 0.005)
            return {
                "query_id": query.query_id,
                "paper_id": candidate.paper_id,
                "rank": candidate.rank,
                "status": "complete",
                "cache_key": f"cache:{candidate.paper_id}",
                "label": "irrelevant",
                "provider_invocation_count": (
                    0 if candidate.paper_id in self.zero_provider_paper_ids else 1
                ),
            }
        finally:
            with self.lock:
                self.active -= 1
                if call_count > 1:
                    self.retry_active -= 1


def test_parallel_judgments_use_single_writer_and_keep_rank_order(
    monkeypatch, tmp_path
):
    query = Query("q1", "question", ["freeform"])
    candidates = tuple(
        CandidatePaper(f"p{rank}", rank) for rank in range(1, 4)
    )
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query.query_id)
    paths.directory.mkdir(parents=True)
    reader = _ConcurrentJudgmentReader(barrier=threading.Barrier(3))
    coordinator_thread = threading.get_ident()
    checkpoint_threads = []
    invalidations = []
    real_checkpoint = _RUNNER.checkpoint_judgment_update
    real_invalidate = _RUNNER.invalidate_aggregate_queries

    def recording_checkpoint(**kwargs):
        checkpoint_threads.append(threading.get_ident())
        real_checkpoint(**kwargs)

    def recording_invalidation(run_dir, query_ids):
        invalidations.append((threading.get_ident(), set(query_ids)))
        real_invalidate(run_dir, query_ids)

    monkeypatch.setattr(
        _RUNNER, "checkpoint_judgment_update", recording_checkpoint
    )
    monkeypatch.setattr(
        _RUNNER, "invalidate_aggregate_queries", recording_invalidation
    )
    judgments = {}

    _RUNNER.run_candidate_judgments(
        reader=reader,
        query=query,
        candidates=list(candidates),
        all_candidates=candidates,
        judgments=judgments,
        paths=paths,
        run_dir=tmp_path / "run",
        workers=3,
        force=False,
    )

    assert reader.max_active == 3
    assert sorted(reader.calls) == ["p1", "p2", "p3"]
    assert checkpoint_threads == [coordinator_thread] * 3
    assert invalidations == [(coordinator_thread, {"q1"})]
    persisted = [
        json.loads(line)
        for line in paths.judgments.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["paper_id"] for item in persisted] == ["p1", "p2", "p3"]


def _empty_execution_state(
    tmp_path: Path,
    query_id: str,
    candidate_count: int,
) -> _RUNNER.QueryExecutionState:
    query = Query(query_id, f"question {query_id}", ["freeform"])
    candidates = tuple(
        CandidatePaper(f"{query_id}_p{rank}", rank)
        for rank in range(1, candidate_count + 1)
    )
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query_id)
    paths.directory.mkdir(parents=True, exist_ok=True)
    return _RUNNER.QueryExecutionState(
        handoff=_RUNNER.CandidateHandoff(query, candidates),
        paths=paths,
        judgments={},
        target_candidates=list(candidates),
    )


def test_execute_shares_one_adaptive_controller_across_paid_stages(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    handoff = _RUNNER.CandidateHandoff(
        Query("q1", "question", ["freeform"]),
        (CandidatePaper("p1", 1),),
    )
    args = SimpleNamespace(
        force=False,
        paper_id=None,
        resume=False,
        stage="all",
        workers=100,
    )
    seen_controllers = []

    monkeypatch.setattr(_RUNNER, "build_manifest", lambda *_args: {})
    monkeypatch.setattr(
        _RUNNER, "PairwiseAOAIReader", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        _RUNNER,
        "materialize_run_outputs",
        lambda *_args, **_kwargs: (0, 0),
    )

    def fake_judgments(**kwargs):
        controller = kwargs["concurrency"]
        seen_controllers.append(controller)
        assert controller.limit == 100
        assert controller.record_rate_limit() == (100, 50, 1)

    def fake_answers(**kwargs):
        controller = kwargs["concurrency"]
        seen_controllers.append(controller)
        assert controller.limit == 50

    monkeypatch.setattr(
        _RUNNER, "run_candidate_judgments_globally", fake_judgments
    )
    monkeypatch.setattr(_RUNNER, "run_answers_globally", fake_answers)

    _RUNNER.execute_locked_run(
        args=args,
        config={},
        store=object(),
        llm=object(),
        preflight={},
        run_dir=run_dir,
        all_handoffs=[handoff],
        selected_handoffs=[handoff],
        require_evidence=True,
    )

    assert len(seen_controllers) == 2
    assert seen_controllers[0] is seen_controllers[1]


def test_global_judgment_pool_spans_queries_and_keeps_writes_on_coordinator(
    monkeypatch, tmp_path
):
    states = [
        _empty_execution_state(tmp_path, "q1", 2),
        _empty_execution_state(tmp_path, "q2", 2),
    ]
    reader = _ConcurrentJudgmentReader(barrier=threading.Barrier(4))
    coordinator_thread = threading.get_ident()
    checkpoint_threads = []
    invalidations = []
    real_checkpoint = _RUNNER.checkpoint_judgment_update
    real_invalidate = _RUNNER.invalidate_aggregate_queries

    def recording_checkpoint(**kwargs):
        checkpoint_threads.append(threading.get_ident())
        real_checkpoint(**kwargs)

    def recording_invalidation(run_dir, query_ids):
        invalidations.append((threading.get_ident(), set(query_ids)))
        real_invalidate(run_dir, query_ids)

    monkeypatch.setattr(
        _RUNNER, "checkpoint_judgment_update", recording_checkpoint
    )
    monkeypatch.setattr(
        _RUNNER, "invalidate_aggregate_queries", recording_invalidation
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=4,
        force=False,
    )

    assert reader.max_active == 4
    assert set(reader.query_calls) == {
        ("q1", "q1_p1"),
        ("q1", "q1_p2"),
        ("q2", "q2_p1"),
        ("q2", "q2_p2"),
    }
    assert checkpoint_threads == [coordinator_thread] * 4
    assert invalidations == [(coordinator_thread, {"q1", "q2"})]
    for state in states:
        persisted = [
            json.loads(line)
            for line in state.paths.judgments.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [item["rank"] for item in persisted] == [1, 2]


def test_global_judgment_failure_stops_queue_and_saves_other_query_success(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    states = [
        _empty_execution_state(tmp_path, "q1", 2),
        _empty_execution_state(tmp_path, "q2", 2),
    ]
    aggregate_rows = [
        {"query_id": "q1"},
        {"query_id": "q2"},
        {"query_id": "q3"},
    ]
    _write_jsonl(run_dir / "submission.jsonl", aggregate_rows)
    _write_jsonl(run_dir / "reading_traces.jsonl", aggregate_rows)
    reader = _ConcurrentJudgmentReader(
        barrier=threading.Barrier(2), fail_paper_id="q1_p1"
    )
    coordinator_thread = threading.get_ident()
    write_threads = []
    real_checkpoint = _RUNNER.checkpoint_judgment_update
    real_record_error = _RUNNER.record_error

    def recording_checkpoint(**kwargs):
        write_threads.append(threading.get_ident())
        real_checkpoint(**kwargs)

    def recording_error(*args, **kwargs):
        write_threads.append(threading.get_ident())
        real_record_error(*args, **kwargs)

    monkeypatch.setattr(
        _RUNNER, "checkpoint_judgment_update", recording_checkpoint
    )
    monkeypatch.setattr(_RUNNER, "record_error", recording_error)

    with pytest.raises(RuntimeError, match="failure for q1_p1"):
        _RUNNER.run_candidate_judgments_globally(
            reader=reader,
            states=states,
            run_dir=run_dir,
            workers=2,
            force=False,
        )

    assert set(reader.query_calls) == {
        ("q1", "q1_p1"),
        ("q2", "q2_p1"),
    }
    assert write_threads == [coordinator_thread] * 2
    assert not states[0].paths.judgments.exists()
    persisted = [
        json.loads(line)
        for line in states[1].paths.judgments.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [item["paper_id"] for item in persisted] == ["q2_p1"]
    assert json.loads((run_dir / "submission.jsonl").read_text())["query_id"] == "q3"
    assert json.loads((run_dir / "reading_traces.jsonl").read_text())["query_id"] == "q3"


def test_global_pool_isolates_exhausted_response_and_finishes_other_pairs(
    tmp_path,
):
    states = [
        _empty_execution_state(tmp_path, "q1", 2),
        _empty_execution_state(tmp_path, "q2", 2),
    ]
    reader = _ConcurrentJudgmentReader(response_invalid_paper_id="q1_p1")

    with pytest.raises(RuntimeError, match="1 candidate judgment response"):
        _RUNNER.run_candidate_judgments_globally(
            reader=reader,
            states=states,
            run_dir=tmp_path / "run",
            workers=2,
            force=False,
        )

    assert set(reader.query_calls) == {
        ("q1", "q1_p1"),
        ("q1", "q1_p2"),
        ("q2", "q2_p1"),
        ("q2", "q2_p2"),
    }
    assert not states[0].paths.judgments.exists() or [
        json.loads(line)["paper_id"]
        for line in states[0].paths.judgments.read_text(
            encoding="utf-8"
        ).splitlines()
    ] == ["q1_p2"]
    assert [
        json.loads(line)["paper_id"]
        for line in states[1].paths.judgments.read_text(
            encoding="utf-8"
        ).splitlines()
    ] == ["q2_p1", "q2_p2"]
    errors = [
        json.loads(line)
        for line in states[0].paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert errors[0]["error_type"] == "JudgmentResponseExhaustedError"
    assert errors[0]["details"]["calls"][0]["raw_response"] == "{}"


def test_stage1_429_requeues_only_rate_limited_and_untouched_jobs(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    state = _empty_execution_state(tmp_path, "q1", 7)
    reader = _ConcurrentJudgmentReader(
        rate_limit_failures={"q1_p1": 1}
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=4,
        force=False,
    )

    assert reader.calls.count("q1_p1") == 2
    assert all(
        reader.calls.count(f"q1_p{rank}") == 1 for rank in range(2, 8)
    )
    persisted = [
        json.loads(line)
        for line in state.paths.judgments.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["rank"] for item in persisted] == list(range(1, 8))
    assert not state.paths.errors.exists()
    output = capsys.readouterr().out
    assert "RATE_LIMITED" in output
    assert "effective_workers=4->3" in output


def test_stage1_clean_success_window_restores_shared_concurrency(
    tmp_path, capsys
):
    state = _empty_execution_state(tmp_path, "q1", 25)
    reader = _ConcurrentJudgmentReader()
    controller = _RUNNER._AdaptiveAOAIConcurrency(100)
    assert controller.record_rate_limit() == (100, 50, 1)
    assert controller.record_rate_limit() == (50, 25, 2)

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=100,
        force=False,
        concurrency=controller,
    )

    assert controller.limit == 30
    assert controller.congestion_streak == 0
    assert len(reader.calls) == 25
    output = capsys.readouterr().out
    assert "AOAI concurrency recovery" in output
    assert "effective_workers=25->30" in output


def test_stage1_zero_provider_checkpoints_do_not_restore_concurrency(tmp_path):
    state = _empty_execution_state(tmp_path, "q1", 25)
    zero_provider_papers = {
        candidate.paper_id for candidate in state.target_candidates
    }
    reader = _ConcurrentJudgmentReader(
        zero_provider_paper_ids=zero_provider_papers
    )
    controller = _RUNNER._AdaptiveAOAIConcurrency(100)
    assert controller.record_rate_limit() == (100, 50, 1)
    assert controller.record_rate_limit() == (50, 25, 2)

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=100,
        force=False,
        concurrency=controller,
    )

    assert controller.limit == 25
    assert controller.clean_success_credit == 0
    assert len(state.judgments) == 25
    assert all(
        judgment["provider_invocation_count"] == 0
        for judgment in state.judgments.values()
    )


def test_stage1_transient_requeues_only_failed_job_without_reducing_pool(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    state = _empty_execution_state(tmp_path, "q1", 7)
    reader = _ConcurrentJudgmentReader(
        transient_failures={"q1_p1": 1}
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=4,
        force=False,
    )

    assert reader.calls.count("q1_p1") == 2
    assert all(
        reader.calls.count(f"q1_p{rank}") == 1 for rank in range(2, 8)
    )
    assert not state.paths.errors.exists()
    output = capsys.readouterr().out
    assert "TRANSIENT" in output
    assert "AOAI transient recovery" in output
    assert "effective_workers" not in output


def test_stage1_provider_ledger_counts_invalid_repair_500_and_whole_job_retry(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    valid_response = json.dumps(
        {
            "is_relevant_to_answer": False,
            "has_usable_answer_evidence": False,
            "evidence_chunk_ids": [],
        }
    )

    class SequenceLLM:
        def __init__(self):
            self.outcomes = [
                {
                    "text": "{}",
                    "request_id": "req-invalid-base",
                    "usage": {"total_tokens": 5},
                    "rate_limit": {"x-ratelimit-remaining-requests": "99"},
                },
                _HTTPStatusError(500, "repair transport failed"),
                {
                    "text": valid_response,
                    "request_id": "req-valid-retry",
                    "usage": {"total_tokens": 7},
                },
            ]
            self.calls = 0

        def complete_with_metadata(self, _prompt, image_paths=None):
            del image_paths
            outcome = self.outcomes[self.calls]
            self.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "Unrelated paper content.",
                "metadata": {"page": 1},
            }
        ],
    )
    llm = SequenceLLM()
    reader = _RUNNER.PairwiseAOAIReader(_RUNNER.ChunkStore(chunks), llm)
    query = Query("q1", "What value is reported?", ["freeform"])
    candidate = CandidatePaper("p1", 1, "Paper One")
    handoff = _RUNNER.CandidateHandoff(query, (candidate,))
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query.query_id)
    paths.directory.mkdir(parents=True)
    state = _RUNNER.QueryExecutionState(
        handoff=handoff,
        paths=paths,
        judgments={},
        target_candidates=[candidate],
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=1,
        force=False,
    )

    assert llm.calls == 3
    judgment = state.judgments["p1"]
    assert judgment["provider_invocation_count"] == 3
    assert judgment["judgment_call_count"] == 3
    assert judgment["provider_request_ids"] == [
        "req-invalid-base",
        "req-valid-retry",
    ]
    assert judgment["provider_usage"] == {"total_tokens": 12}
    ledger = [
        json.loads(line)
        for line in paths.provider_attempts.read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger) == 6
    assert len({row["attempt_id"] for row in ledger}) == 3
    finalized = [row for row in ledger if row["event_kind"] == "finalize"]
    assert [row["outcome"] for row in finalized] == [
        "response",
        "provider_error",
        "response",
    ]
    assert finalized[0]["parse_error"].startswith(
        "q1/p1: current Stage-1 response must use exactly"
    )
    assert finalized[1]["status_codes"] == [500]
    assert finalized[1]["retry_category"] == "transient_provider"
    assert finalized[1]["recovery_round"] == 1
    assert finalized[1]["semantic_phase"] == "judgment_evidence_repair"
    serialized_ledger = json.dumps(ledger).lower()
    assert "raw_response" not in serialized_ledger
    assert "api_key" not in serialized_ledger
    assert "endpoint" not in serialized_ledger

    # Resume validates the completed checkpoint without creating or duplicating
    # any provider event.
    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=1,
        force=False,
    )
    assert llm.calls == 3
    assert len(paths.provider_attempts.read_text(encoding="utf-8").splitlines()) == 6

    assert _RUNNER.materialize_run_outputs(
        tmp_path / "run", [handoff], reader
    ) == (1, 0)
    usage_summary = json.loads(
        (tmp_path / "run" / "provider_usage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert usage_summary["provider_invocation_count"] == 3
    assert usage_summary["stages"]["judge"]["provider_invocation_count"] == 3
    trace = json.loads(
        (tmp_path / "run" / "reading_traces.jsonl").read_text(encoding="utf-8")
    )
    assert trace["provider_attempts"]["all"]["provider_invocation_count"] == 3


def test_provider_ledger_writes_stay_on_coordinator_under_concurrency(
    monkeypatch, tmp_path
):
    response = json.dumps(
        {
            "is_relevant_to_answer": False,
            "has_usable_answer_evidence": False,
            "evidence_chunk_ids": [],
        }
    )

    class ConcurrentLLM:
        def __init__(self):
            self.barrier = threading.Barrier(2)
            self.lock = threading.Lock()
            self.calls = 0
            self.call_threads = []

        def complete_with_metadata(self, _prompt, image_paths=None):
            del image_paths
            with self.lock:
                self.calls += 1
                request_index = self.calls
                self.call_threads.append(threading.get_ident())
            self.barrier.wait(timeout=2)
            return {
                "text": response,
                "request_id": f"req-concurrent-{request_index}",
            }

    candidates = (
        CandidatePaper("p1", 1, "Paper One"),
        CandidatePaper("p2", 2, "Paper Two"),
    )
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": candidate.paper_id,
                "chunk_id": f"{candidate.paper_id}#1",
                "chunk_type": "text_span",
                "text": "Unrelated content.",
                "metadata": {"page": 1},
            }
            for candidate in candidates
        ],
    )
    llm = ConcurrentLLM()
    reader = _RUNNER.PairwiseAOAIReader(_RUNNER.ChunkStore(chunks), llm)
    query = Query("q1", "What value is reported?", ["freeform"])
    handoff = _RUNNER.CandidateHandoff(query, candidates)
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query.query_id)
    paths.directory.mkdir(parents=True)
    state = _RUNNER.QueryExecutionState(
        handoff=handoff,
        paths=paths,
        judgments={},
        target_candidates=list(candidates),
    )
    coordinator_thread = threading.get_ident()
    writer_threads = []
    real_record = _RUNNER.record_provider_attempt_event

    def recording_provider_event(*args, **kwargs):
        writer_threads.append(threading.get_ident())
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        _RUNNER, "record_provider_attempt_event", recording_provider_event
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=2,
        force=False,
    )

    assert llm.calls == 2
    assert all(thread_id != coordinator_thread for thread_id in llm.call_threads)
    assert writer_threads == [coordinator_thread] * 4
    summary = _RUNNER.provider_attempt_summary(paths.provider_attempts)
    assert summary["provider_invocation_count"] == 2
    assert summary["uncertain_provider_invocation_count"] == 0


def test_provider_ledger_is_idempotent_and_reports_crash_uncertainty(tmp_path):
    path = tmp_path / "q1" / "provider_attempts.jsonl"
    prepare = {
        "attempt_id": "stable-attempt-id",
        "event_kind": "prepare",
        "stage": "judge",
        "query_id": "q1",
        "paper_id": "p1",
        "whole_job_attempt_index": 1,
        "semantic_phase": "judgment_initial_full_context",
        "provider_invocation_count": 1,
    }
    _RUNNER.record_provider_attempt_event(path, event=prepare)
    _RUNNER.record_provider_attempt_event(path, event=prepare)

    uncertain = _RUNNER.provider_attempt_summary(path)
    assert uncertain["provider_invocation_count"] == 1
    assert uncertain["finalized_provider_invocation_count"] == 0
    assert uncertain["uncertain_provider_invocation_count"] == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    finalize = {
        **prepare,
        "event_kind": "finalize",
        "outcome": "response",
        "request_id": "req-stable",
        "parse_error": None,
    }
    _RUNNER.record_provider_attempt_event(path, event=finalize)
    _RUNNER.record_provider_attempt_event(path, event=finalize)

    complete = _RUNNER.provider_attempt_summary(path)
    assert complete["provider_invocation_count"] == 1
    assert complete["finalized_provider_invocation_count"] == 1
    assert complete["uncertain_provider_invocation_count"] == 0
    assert complete["request_ids"] == ["req-stable"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_stage1_transient_retry_exhaustion_is_bounded_and_logged_once(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS", 2)
    state = _empty_execution_state(tmp_path, "q1", 1)
    reader = _ConcurrentJudgmentReader(
        transient_failures={"q1_p1": 3}
    )

    with pytest.raises(_HTTPStatusError, match="transient failure for q1_p1"):
        _RUNNER.run_candidate_judgments_globally(
            reader=reader,
            states=[state],
            run_dir=tmp_path / "run",
            workers=4,
            force=False,
        )

    assert reader.calls == ["q1_p1"] * 3
    errors = [
        json.loads(line)
        for line in state.paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "_HTTPStatusError"


def test_stage1_mixed_429_and_transient_retries_only_failed_jobs(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    state = _empty_execution_state(tmp_path, "q1", 3)
    reader = _ConcurrentJudgmentReader(
        rate_limit_failures={"q1_p1": 1},
        transient_failures={"q1_p2": 1},
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=2,
        force=False,
    )

    assert reader.call_counts == {"q1_p1": 2, "q1_p2": 2, "q1_p3": 1}
    assert not state.paths.errors.exists()
    output = capsys.readouterr().out
    assert "RATE_LIMITED" in output
    assert "TRANSIENT" in output
    assert "cooldown=covered_by_429_recovery" in output
    assert "effective_workers=2->1" in output


def test_stage1_429_retry_budget_is_per_job(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS", 1)
    state = _empty_execution_state(tmp_path, "q1", 2)
    reader = _ConcurrentJudgmentReader(
        rate_limit_failures={"q1_p1": 1, "q1_p2": 1}
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=1,
        force=False,
    )

    assert reader.calls == ["q1_p1", "q1_p1", "q1_p2", "q1_p2"]
    assert capsys.readouterr().out.count("round=1/1") == 2
    assert not state.paths.errors.exists()


def test_stage1_429_keeps_provider_cap_independent_of_two_job_tail(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    state = _empty_execution_state(tmp_path, "q1", 2)
    reader = _ConcurrentJudgmentReader(
        rate_limit_failures={"q1_p1": 1, "q1_p2": 1}
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=50,
        force=False,
    )

    assert reader.max_retry_active == 2
    assert reader.call_counts == {"q1_p1": 2, "q1_p2": 2}
    assert "effective_workers=50->38" in capsys.readouterr().out


def test_stage1_429_exhaustion_is_bounded_and_logged_once(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS", 2)
    state = _empty_execution_state(tmp_path, "q1", 1)
    reader = _ConcurrentJudgmentReader(
        rate_limit_failures={"q1_p1": 3}
    )

    with pytest.raises(_HTTPStatusError, match="rate limit for q1_p1"):
        _RUNNER.run_candidate_judgments_globally(
            reader=reader,
            states=[state],
            run_dir=tmp_path / "run",
            workers=4,
            force=False,
        )

    assert reader.calls == ["q1_p1"] * 3
    assert not state.paths.judgments.exists()
    errors = [
        json.loads(line)
        for line in state.paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "_HTTPStatusError"


def test_parallel_failure_stops_new_calls_and_checkpoints_in_flight_success(
    tmp_path,
):
    query = Query("q1", "question", ["freeform"])
    candidates = tuple(
        CandidatePaper(f"p{rank}", rank) for rank in range(1, 5)
    )
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query.query_id)
    paths.directory.mkdir(parents=True)
    reader = _ConcurrentJudgmentReader(
        barrier=threading.Barrier(2), fail_paper_id="p1"
    )
    judgments = {}

    with pytest.raises(RuntimeError, match="failure for p1"):
        _RUNNER.run_candidate_judgments(
            reader=reader,
            query=query,
            candidates=list(candidates),
            all_candidates=candidates,
            judgments=judgments,
            paths=paths,
            run_dir=tmp_path / "run",
            workers=2,
            force=False,
        )

    assert sorted(reader.calls) == ["p1", "p2"]
    assert list(judgments) == ["p2"]
    persisted = [
        json.loads(line)
        for line in paths.judgments.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["paper_id"] for item in persisted] == ["p2"]
    errors = [
        json.loads(line)
        for line in paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["paper_id"], item["error_type"]) for item in errors] == [
        ("p1", "RuntimeError")
    ]


def test_forced_parallel_failure_never_restores_old_judgment(tmp_path):
    query = Query("q1", "question", ["freeform"])
    run_dir = tmp_path / "run"
    candidates = tuple(
        CandidatePaper(f"p{rank}", rank) for rank in range(1, 3)
    )
    paths = _RUNNER.QueryRunPaths.under(run_dir, query.query_id)
    paths.directory.mkdir(parents=True)
    aggregate_rows = [{"query_id": "q1"}, {"query_id": "q2"}]
    _write_jsonl(run_dir / "submission.jsonl", aggregate_rows)
    _write_jsonl(run_dir / "reading_traces.jsonl", aggregate_rows)
    judgments = {
        candidate.paper_id: {
            "query_id": query.query_id,
            "paper_id": candidate.paper_id,
            "rank": candidate.rank,
            "status": "complete",
            "cache_key": f"cache:{candidate.paper_id}",
            "label": "old",
        }
        for candidate in candidates
    }
    _write_jsonl(paths.judgments, list(judgments.values()))
    paths.answer.write_text("{}\n", encoding="utf-8")
    paths.submission.write_text("{}\n", encoding="utf-8")
    reader = _ConcurrentJudgmentReader(
        barrier=threading.Barrier(2), fail_paper_id="p1"
    )

    with pytest.raises(RuntimeError, match="failure for p1"):
        _RUNNER.run_candidate_judgments(
            reader=reader,
            query=query,
            candidates=list(candidates),
            all_candidates=candidates,
            judgments=judgments,
            paths=paths,
            run_dir=run_dir,
            workers=2,
            force=True,
        )

    assert not paths.answer.exists()
    assert not paths.submission.exists()
    assert "p1" not in judgments
    assert judgments["p2"]["label"] == "irrelevant"
    persisted = [
        json.loads(line)
        for line in paths.judgments.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["paper_id"], item["label"]) for item in persisted] == [
        ("p2", "irrelevant")
    ]
    assert json.loads(
        (run_dir / "submission.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q2"
    assert json.loads(
        (run_dir / "reading_traces.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q2"


def test_forced_answer_invalidation_keeps_candidate_judgments(tmp_path):
    run_dir = tmp_path / "run"
    paths = _RUNNER.QueryRunPaths.under(run_dir, "q1")
    paths.directory.mkdir(parents=True)
    judgment_text = '{"query_id":"q1","paper_id":"p1"}\n'
    paths.judgments.write_text(judgment_text, encoding="utf-8")
    paths.answer.write_text("{}\n", encoding="utf-8")
    paths.submission.write_text("{}\n", encoding="utf-8")

    _RUNNER.invalidate_forced_answer(
        run_dir=run_dir,
        query_id="q1",
        paths=paths,
    )

    assert paths.judgments.read_text(encoding="utf-8") == judgment_text
    assert not paths.answer.exists()
    assert not paths.submission.exists()


def test_answer_update_only_writes_per_query_checkpoints(
    monkeypatch, tmp_path
):
    events = []
    monkeypatch.setattr(
        _RUNNER,
        "atomic_write_json",
        lambda path, payload: events.append(("write", path, payload)),
    )

    _RUNNER.checkpoint_answer_update(
        run_dir=tmp_path / "run",
        query_id="q1",
        answer_path=tmp_path / "answer.json",
        answer_record={"query_id": "q1", "kind": "answer"},
        submission_path=tmp_path / "submission.json",
        submission={"query_id": "q1", "kind": "submission"},
    )

    assert [event[0] for event in events] == ["write", "write"]


class _ConcurrentAnswerReader:
    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        fail_query_id: str | None = None,
        rate_limit_failures: dict[str, int] | None = None,
        transient_failures: dict[str, int] | None = None,
        attempt_emitted: threading.Event | None = None,
        release_after_attempt: threading.Event | None = None,
    ) -> None:
        self.chunk_store = self
        self.barrier = barrier
        self.fail_query_id = fail_query_id
        self.rate_limit_failures = dict(rate_limit_failures or {})
        self.transient_failures = dict(transient_failures or {})
        self.attempt_emitted = attempt_emitted
        self.release_after_attempt = release_after_attempt
        self.calls: list[str] = []
        self.call_counts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0
        self.retry_active = 0
        self.max_retry_active = 0
        self.lock = threading.Lock()

    def load_paper(self, paper_id: str) -> list[dict]:
        return [
            {
                "paper_id": paper_id,
                "chunk_id": f"{paper_id}#1",
                "chunk_type": "text_span",
                "text": paper_id,
                "metadata": {"page": 1},
            }
        ]

    def judgment_cache_key(self, query, candidate, records) -> str:
        del query, records
        return f"judgment-cache:{candidate.paper_id}"

    def answer_cache_key(self, query, judgments) -> str:
        del judgments
        return f"answer-cache:{query.query_id}"

    def answer_from_judgments(
        self, query, candidates, judgments, *, attempt_callback
    ):
        del candidates
        accepted_paper_ids = [
            str(judgment["paper_id"])
            for judgment in judgments
            if judgment.get("send_to_answer_agent") is True
        ]
        stage1_relevant_paper_ids = [
            str(judgment["paper_id"])
            for judgment in judgments
            if judgment.get("is_relevant_to_answer") is True
        ]
        with self.lock:
            self.calls.append(query.query_id)
            call_count = self.call_counts.get(query.query_id, 0) + 1
            self.call_counts[query.query_id] = call_count
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if call_count > 1:
                self.retry_active += 1
                self.max_retry_active = max(
                    self.max_retry_active, self.retry_active
                )
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            attempt_callback(
                {
                    "raw_response": f"response {query.query_id}",
                    "parse_error": None,
                    "call": {"query_id": query.query_id},
                }
            )
            if self.attempt_emitted is not None:
                self.attempt_emitted.set()
            if self.release_after_attempt is not None:
                assert self.release_after_attempt.wait(timeout=2)
            remaining_rate_limits = self.rate_limit_failures.get(
                query.query_id, 0
            )
            if remaining_rate_limits:
                self.rate_limit_failures[query.query_id] = (
                    remaining_rate_limits - 1
                )
                raise _HTTPStatusError(429, f"rate limit for {query.query_id}")
            remaining_transients = self.transient_failures.get(
                query.query_id, 0
            )
            if remaining_transients:
                self.transient_failures[query.query_id] = (
                    remaining_transients - 1
                )
                raise _HTTPStatusError(
                    500, f"transient failure for {query.query_id}"
                )
            if query.query_id == self.fail_query_id:
                raise RuntimeError(f"answer failure for {query.query_id}")
            if call_count > 1:
                time.sleep(0.03)
            if self.fail_query_id is not None:
                time.sleep(0.02)
            return (
                {"query_id": query.query_id},
                {
                    "query_id": query.query_id,
                    "status": "complete",
                    "cache_key": f"answer-cache:{query.query_id}",
                    "accepted_paper_ids": accepted_paper_ids,
                    "stage1_relevant_paper_ids": stage1_relevant_paper_ids,
                },
            )
        finally:
            with self.lock:
                self.active -= 1
                if call_count > 1:
                    self.retry_active -= 1


def _complete_answer_state(
    tmp_path: Path,
    query_id: str,
) -> _RUNNER.QueryExecutionState:
    state = _empty_execution_state(tmp_path, query_id, 1)
    candidate = state.handoff.candidate_papers[0]
    chunk_id = f"{candidate.paper_id}#1"
    state.judgments[candidate.paper_id] = {
        "query_id": query_id,
        "paper_id": candidate.paper_id,
        "rank": candidate.rank,
        "status": "complete",
        "cache_key": f"judgment-cache:{candidate.paper_id}",
        "is_relevant_to_answer": True,
        "has_usable_answer_evidence": True,
        "send_to_answer_agent": True,
        "evidence_chunk_ids": [chunk_id],
        "context_chunk_ids": [chunk_id],
        "evidence": [
            {
                "chunk_id": chunk_id,
                "source_type": "text_span",
                "locator": {"page": 1},
                "purpose": "answer",
                "quote_or_value": "",
            }
        ],
        "attached_image_count": 0,
        "attached_image_chunk_ids": [],
    }
    return state


def test_stage2_pool_parallelizes_uncached_queries_and_keeps_writes_on_coordinator(
    monkeypatch, tmp_path
):
    states = [
        _complete_answer_state(tmp_path, "q1"),
        _complete_answer_state(tmp_path, "q2"),
        _complete_answer_state(tmp_path, "q3"),
    ]
    _RUNNER.atomic_write_json(
        states[2].paths.answer,
        {
            "query_id": "q3",
            "status": "complete",
            "cache_key": "answer-cache:q3",
            "accepted_paper_ids": ["q3_p1"],
            "stage1_relevant_paper_ids": ["q3_p1"],
        },
    )
    reader = _ConcurrentAnswerReader(barrier=threading.Barrier(2))
    coordinator_thread = threading.get_ident()
    write_threads = []
    cached_queries = []
    invalidations = []
    real_checkpoint = _RUNNER.checkpoint_answer_update
    real_attempt = _RUNNER.record_answer_attempt
    real_invalidate = _RUNNER.invalidate_aggregate_queries

    def recording_checkpoint(**kwargs):
        write_threads.append(threading.get_ident())
        real_checkpoint(**kwargs)

    def recording_attempt(*args, **kwargs):
        write_threads.append(threading.get_ident())
        real_attempt(*args, **kwargs)

    def recording_invalidation(run_dir, query_ids):
        invalidations.append((threading.get_ident(), set(query_ids)))
        real_invalidate(run_dir, query_ids)

    def ensure_cached(query, answer_record, submission_path, **kwargs):
        del answer_record, kwargs
        write_threads.append(threading.get_ident())
        cached_queries.append(query.query_id)
        _RUNNER.atomic_write_json(
            submission_path, {"query_id": query.query_id, "cached": True}
        )

    monkeypatch.setattr(
        _RUNNER, "checkpoint_answer_update", recording_checkpoint
    )
    monkeypatch.setattr(_RUNNER, "record_answer_attempt", recording_attempt)
    monkeypatch.setattr(
        _RUNNER, "invalidate_aggregate_queries", recording_invalidation
    )
    monkeypatch.setattr(
        _RUNNER, "ensure_submission_from_answer_checkpoint", ensure_cached
    )
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
            "require_evidence": kwargs["require_evidence"],
        },
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=2,
        force=False,
        require_evidence=True,
    )

    assert reader.max_active == 2
    assert set(reader.calls) == {"q1", "q2"}
    assert cached_queries == ["q3"]
    assert invalidations == [(coordinator_thread, {"q1", "q2"})]
    assert write_threads == [coordinator_thread] * 5
    for state in states:
        assert state.paths.answer.exists()
        assert state.paths.submission.exists()
    assert states[0].paths.answer_attempts.exists()
    assert states[1].paths.answer_attempts.exists()
    assert not states[2].paths.answer_attempts.exists()


def test_stage2_failure_stops_queue_and_checkpoints_in_flight_success(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    states = [
        _complete_answer_state(tmp_path, query_id)
        for query_id in ("q1", "q2", "q3", "q4")
    ]
    aggregate_rows = [
        {"query_id": query_id}
        for query_id in ("q1", "q2", "q3", "q4", "q5")
    ]
    _write_jsonl(run_dir / "submission.jsonl", aggregate_rows)
    _write_jsonl(run_dir / "reading_traces.jsonl", aggregate_rows)
    reader = _ConcurrentAnswerReader(
        barrier=threading.Barrier(2), fail_query_id="q1"
    )
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    with pytest.raises(RuntimeError, match="answer failure for q1"):
        _RUNNER.run_answers_globally(
            reader=reader,
            states=states,
            run_dir=run_dir,
            workers=2,
            force=False,
            require_evidence=True,
        )

    assert set(reader.calls) == {"q1", "q2"}
    assert states[0].paths.answer_attempts.exists()
    assert states[0].paths.errors.exists()
    assert not states[0].paths.answer.exists()
    assert states[1].paths.answer.exists()
    assert states[1].paths.submission.exists()
    assert not states[2].paths.answer.exists()
    assert not states[3].paths.answer.exists()
    assert json.loads(
        (run_dir / "submission.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q5"
    assert json.loads(
        (run_dir / "reading_traces.jsonl").read_text(encoding="utf-8")
    )["query_id"] == "q5"


def test_stage2_429_requeues_query_and_preserves_each_paid_attempt(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    states = [
        _complete_answer_state(tmp_path, query_id)
        for query_id in ("q1", "q2", "q3", "q4", "q5")
    ]
    reader = _ConcurrentAnswerReader(rate_limit_failures={"q1": 1})
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=4,
        force=False,
        require_evidence=True,
    )

    assert reader.calls.count("q1") == 2
    assert all(
        reader.calls.count(query_id) == 1
        for query_id in ("q2", "q3", "q4", "q5")
    )
    for state in states:
        attempts = [
            json.loads(line)
            for line in state.paths.answer_attempts.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        expected_attempts = 2 if state.handoff.query.query_id == "q1" else 1
        assert len(attempts) == expected_attempts
        assert state.paths.answer.exists()
        assert state.paths.submission.exists()
    assert not states[0].paths.errors.exists()
    output = capsys.readouterr().out
    assert "RATE_LIMITED" in output
    assert "effective_workers=4->3" in output


def test_stage2_clean_success_window_restores_shared_concurrency(
    monkeypatch, tmp_path, capsys
):
    states = [
        _complete_answer_state(tmp_path, f"q{index}")
        for index in range(1, 26)
    ]
    reader = _ConcurrentAnswerReader()
    controller = _RUNNER._AdaptiveAOAIConcurrency(100)
    assert controller.record_rate_limit() == (100, 50, 1)
    assert controller.record_rate_limit() == (50, 25, 2)
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=100,
        force=False,
        require_evidence=True,
        concurrency=controller,
    )

    assert controller.limit == 30
    assert controller.congestion_streak == 0
    assert len(reader.calls) == 25
    output = capsys.readouterr().out
    assert "AOAI concurrency recovery" in output
    assert "effective_workers=25->30" in output


def test_stage2_transient_requeues_query_and_preserves_each_paid_attempt(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    states = [
        _complete_answer_state(tmp_path, query_id)
        for query_id in ("q1", "q2", "q3")
    ]
    reader = _ConcurrentAnswerReader(transient_failures={"q1": 1})
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=3,
        force=False,
        require_evidence=True,
    )

    assert reader.calls.count("q1") == 2
    assert reader.calls.count("q2") == 1
    assert reader.calls.count("q3") == 1
    for state in states:
        attempts = [
            json.loads(line)
            for line in state.paths.answer_attempts.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        expected_attempts = 2 if state.handoff.query.query_id == "q1" else 1
        assert len(attempts) == expected_attempts
        assert state.paths.answer.exists()
        assert state.paths.submission.exists()
    assert not states[0].paths.errors.exists()
    output = capsys.readouterr().out
    assert "TRANSIENT" in output
    assert "AOAI transient recovery" in output
    assert "effective_workers" not in output


def test_stage2_provider_ledger_records_transport_failure_before_retry(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    answer_response = json.dumps(
        {
            "status": "ready",
            "papers": [
                {"paper_id": "p1", "evidence_chunk_ids": ["p1#1"]}
            ],
            "paper_relevance": [
                {
                    "paper_id": "p1",
                    "role": "target_owner",
                    "reason": "reports the requested value",
                }
            ],
            "derivation": {
                "facts": [
                    {
                        "id": "f_value",
                        "name": "reported value",
                        "value": "42",
                        "value_kind": "reported",
                        "paper_id": "p1",
                        "chunk_ids": ["p1#1"],
                    }
                ],
                "operations": [],
                "answer_bindings": [
                    {
                        "answer_path": "answer.freeform.text",
                        "source_type": "fact",
                        "source_id": "f_value",
                        "answer_fragment": "42",
                    }
                ],
                "final_semantic_answer": "42",
            },
            "answer": {"freeform": {"text": "42"}},
            "support": [
                {
                    "answer_path": "answer.freeform.text",
                    "paper_id": "p1",
                    "chunk_ids": ["p1#1"],
                }
            ],
            "completeness": {"answered_parts": ["value"], "missing": []},
        }
    )

    class SequenceLLM:
        def __init__(self):
            self.outcomes = [
                _HTTPStatusError(500, "answer transport failed"),
                {
                    "text": answer_response,
                    "request_id": "req-answer-retry",
                    "usage": {"prompt_tokens": 11, "completion_tokens": 13},
                },
            ]
            self.calls = 0

        def complete_with_metadata(self, _prompt, image_paths=None):
            del image_paths
            outcome = self.outcomes[self.calls]
            self.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "The reported value is 42.",
                "metadata": {"page": 1},
            }
        ],
    )
    llm = SequenceLLM()
    store = _RUNNER.ChunkStore(chunks)
    reader = _RUNNER.PairwiseAOAIReader(store, llm)
    query = Query("q1", "What value is reported?", ["freeform"])
    candidate = CandidatePaper("p1", 1, "Paper One")
    candidates = (candidate,)
    owner_resolution = _RUNNER.resolve_named_owner(query, candidates)
    records = store.load_paper(candidate.paper_id)
    judgment = {
        "query_id": query.query_id,
        "paper_id": candidate.paper_id,
        "rank": candidate.rank,
        "title": candidate.title,
        "status": "complete",
        "cache_key": reader.judgment_cache_key(
            query,
            candidate,
            records,
            owner_resolution=owner_resolution,
        ),
        "is_relevant_to_answer": True,
        "has_usable_answer_evidence": True,
        "send_to_answer_agent": True,
        "evidence_chunk_ids": ["p1#1"],
        "context_chunk_ids": ["p1#1"],
        "evidence": [
            {
                "chunk_id": "p1#1",
                "source_type": "text_span",
                "locator": {"page": 1},
                "purpose": "answer",
                "quote_or_value": "",
            }
        ],
        "attached_image_count": 0,
        "attached_image_chunk_ids": [],
    }
    handoff = _RUNNER.CandidateHandoff(query, candidates)
    paths = _RUNNER.QueryRunPaths.under(tmp_path / "run", query.query_id)
    paths.directory.mkdir(parents=True)
    state = _RUNNER.QueryExecutionState(
        handoff=handoff,
        paths=paths,
        judgments={candidate.paper_id: judgment},
        target_candidates=[candidate],
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=[state],
        run_dir=tmp_path / "run",
        workers=1,
        force=False,
        require_evidence=True,
    )

    assert llm.calls == 2
    answer_checkpoint = json.loads(paths.answer.read_text(encoding="utf-8"))
    assert answer_checkpoint["provider_invocation_count"] == 2
    assert answer_checkpoint["answer_call_count"] == 2
    assert answer_checkpoint["provider_request_ids"] == ["req-answer-retry"]
    ledger = [
        json.loads(line)
        for line in paths.provider_attempts.read_text(encoding="utf-8").splitlines()
    ]
    finalized = [row for row in ledger if row["event_kind"] == "finalize"]
    assert len(finalized) == 2
    assert finalized[0]["outcome"] == "provider_error"
    assert finalized[0]["status_codes"] == [500]
    assert finalized[0]["retry_category"] == "transient_provider"
    assert finalized[0]["semantic_phase"] == "answer_initial"
    assert finalized[1]["outcome"] == "response"
    assert finalized[1]["request_id"] == "req-answer-retry"


def test_stage2_transient_exhaustion_logs_once_after_persisting_attempts(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_TRANSIENT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "MAX_AOAI_TRANSIENT_RECOVERY_ROUNDS", 2)
    state = _complete_answer_state(tmp_path, "q1")
    reader = _ConcurrentAnswerReader(transient_failures={"q1": 3})
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    with pytest.raises(_HTTPStatusError, match="transient failure for q1"):
        _RUNNER.run_answers_globally(
            reader=reader,
            states=[state],
            run_dir=tmp_path / "run",
            workers=4,
            force=False,
            require_evidence=True,
        )

    assert reader.calls == ["q1"] * 3
    attempts = [
        json.loads(line)
        for line in state.paths.answer_attempts.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(attempts) == 3
    errors = [
        json.loads(line)
        for line in state.paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "_HTTPStatusError"


def test_stage2_429_keeps_provider_cap_independent_of_two_query_tail(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    states = [
        _complete_answer_state(tmp_path, query_id)
        for query_id in ("q1", "q2")
    ]
    reader = _ConcurrentAnswerReader(
        rate_limit_failures={"q1": 1, "q2": 1}
    )
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    _RUNNER.run_answers_globally(
        reader=reader,
        states=states,
        run_dir=tmp_path / "run",
        workers=50,
        force=False,
        require_evidence=True,
    )

    assert reader.max_retry_active == 2
    assert reader.call_counts == {"q1": 2, "q2": 2}
    assert "effective_workers=50->38" in capsys.readouterr().out


def test_stage2_429_exhaustion_logs_once_after_persisting_all_attempts(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_RUNNER, "AOAI_RATE_LIMIT_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(_RUNNER, "MAX_AOAI_RATE_LIMIT_RECOVERY_ROUNDS", 2)
    state = _complete_answer_state(tmp_path, "q1")
    reader = _ConcurrentAnswerReader(rate_limit_failures={"q1": 3})
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )

    with pytest.raises(_HTTPStatusError, match="rate limit for q1"):
        _RUNNER.run_answers_globally(
            reader=reader,
            states=[state],
            run_dir=tmp_path / "run",
            workers=4,
            force=False,
            require_evidence=True,
        )

    assert reader.calls == ["q1"] * 3
    attempts = [
        json.loads(line)
        for line in state.paths.answer_attempts.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(attempts) == 3
    errors = [
        json.loads(line)
        for line in state.paths.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "_HTTPStatusError"
    assert not state.paths.answer.exists()
    assert not state.paths.submission.exists()


def test_stage2_attempt_is_durable_before_the_worker_future_completes(
    monkeypatch, tmp_path
):
    state = _complete_answer_state(tmp_path, "q1")
    attempt_emitted = threading.Event()
    release_after_attempt = threading.Event()
    reader = _ConcurrentAnswerReader(
        attempt_emitted=attempt_emitted,
        release_after_attempt=release_after_attempt,
    )
    monkeypatch.setattr(
        _RUNNER,
        "prediction_to_submission",
        lambda query, prediction, **_kwargs: {
            "query_id": query.query_id,
            "prediction": prediction,
        },
    )
    coordinator_errors = []

    def run_coordinator():
        try:
            _RUNNER.run_answers_globally(
                reader=reader,
                states=[state],
                run_dir=tmp_path / "run",
                workers=1,
                force=False,
                require_evidence=True,
            )
        except Exception as exc:  # pragma: no cover - asserted after join
            coordinator_errors.append(exc)

    coordinator = threading.Thread(target=run_coordinator)
    coordinator.start()
    assert attempt_emitted.wait(timeout=1)
    deadline = time.monotonic() + 1
    while not state.paths.answer_attempts.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert state.paths.answer_attempts.exists()
    assert coordinator.is_alive()
    assert not state.paths.answer.exists()

    release_after_attempt.set()
    coordinator.join(timeout=2)
    assert not coordinator.is_alive()
    assert coordinator_errors == []
    assert state.paths.answer.exists()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _checkpointed_run(
    tmp_path: Path,
    *,
    candidate_count: int = 1,
    judgment_count: int | None = None,
    include_evidence: bool = True,
):
    judgment_count = candidate_count if judgment_count is None else judgment_count
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": f"p{index}",
                "chunk_id": f"p{index}#1",
                "chunk_type": "text_span",
                "text": f"Paper {index} reports 42.",
                "metadata": {"page": 1},
            }
            for index in range(1, candidate_count + 1)
        ],
    )
    store = _RUNNER.ChunkStore(chunks)
    reader = _RUNNER.PairwiseAOAIReader(store, _RUNNER.FakeLLM())
    query = Query(
        query_id="q1",
        question="What value is reported?",
        answer_types=["freeform"],
        table_schema=None,
    )
    candidates = tuple(
        CandidatePaper(f"p{index}", index)
        for index in range(1, candidate_count + 1)
    )
    handoff = _RUNNER.CandidateHandoff(query=query, candidate_papers=candidates)
    owner_resolution = _RUNNER.resolve_named_owner(query, candidates)
    judgments = []
    for candidate in candidates[:judgment_count]:
        records = store.load_paper(candidate.paper_id)
        chunk_id = str(records[0]["chunk_id"])
        relevant = candidate.paper_id == "p1"
        evidence_ids = [chunk_id] if relevant else []
        judgments.append(
            {
                "query_id": query.query_id,
                "paper_id": candidate.paper_id,
                "rank": candidate.rank,
                "status": "complete",
                "cache_key": reader.judgment_cache_key(
                    query,
                    candidate,
                    records,
                    owner_resolution=owner_resolution,
                ),
                "named_owner_resolution": owner_resolution,
                "is_relevant_to_answer": relevant,
                "has_usable_answer_evidence": relevant,
                "send_to_answer_agent": relevant,
                "evidence_chunk_ids": evidence_ids,
                "context_chunk_ids": [chunk_id],
                "evidence": (
                    [
                        {
                            "chunk_id": chunk_id,
                            "source_type": "text_span",
                            "locator": {"page": 1},
                            "purpose": "answer",
                            "quote_or_value": "",
                        }
                    ]
                    if relevant
                    else []
                ),
                "attached_image_count": 0,
                "attached_image_chunk_ids": [],
            }
        )

    prediction = Prediction(
        query_id=query.query_id,
        gold_papers=[{"paper_id": "p1"}],
        evidence=(
            [
                Evidence(
                    paper_id="p1",
                    source_type="text_span",
                    locator=EvidenceLocator(page=1),
                    evidence_text_or_value="42",
                )
            ]
            if include_evidence
            else []
        ),
        answer=Answer(freeform={"text": "42"}),
        candidate_papers=[candidate.paper_id for candidate in candidates],
    )
    answer_record = {
        "query_id": query.query_id,
        "status": "complete",
        "cache_key": reader.answer_cache_key(query, judgments),
        "accepted_paper_ids": ["p1"],
        "stage1_relevant_paper_ids": ["p1"],
        "prediction": prediction.to_dict(),
    }
    submission = _RUNNER.prediction_to_submission(
        query, prediction, require_evidence=include_evidence
    )
    run_dir = tmp_path / "run"
    query_dir = run_dir / query.query_id
    query_dir.mkdir(parents=True)
    _write_jsonl(query_dir / "candidate_judgments.jsonl", judgments)
    (query_dir / "answer.json").write_text(
        json.dumps(answer_record), encoding="utf-8"
    )
    (query_dir / "submission.json").write_text(
        json.dumps(submission), encoding="utf-8"
    )
    return run_dir, handoff, reader, submission


def test_materialize_rejects_submission_that_differs_from_answer_checkpoint(
    tmp_path,
):
    run_dir, handoff, reader, submission = _checkpointed_run(tmp_path)
    assert _RUNNER.materialize_run_outputs(run_dir, [handoff], reader) == (1, 1)

    submission["answer"]["freeform"]["text"] = "tampered"
    (run_dir / "q1" / "submission.json").write_text(
        json.dumps(submission), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match answer checkpoint"):
        _RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_materialize_support_only_papers_may_be_subset_of_stage2_handoff(tmp_path):
    run_dir, handoff, reader, submission = _checkpointed_run(
        tmp_path, candidate_count=2
    )
    answer_path = run_dir / "q1" / "answer.json"
    answer_record = json.loads(answer_path.read_text(encoding="utf-8"))
    answer_record["accepted_paper_ids"] = ["p1", "p2"]
    answer_record["submission_paper_ids"] = ["p1"]
    answer_path.write_text(json.dumps(answer_record), encoding="utf-8")

    assert _RUNNER.materialize_run_outputs(run_dir, [handoff], reader) == (1, 1)
    assert submission["gold_papers"] == [{"paper_id": "p1"}]


def test_materialize_accepts_test_extra_submission_without_evidence(tmp_path):
    run_dir, handoff, reader, submission = _checkpointed_run(
        tmp_path, include_evidence=False
    )

    assert set(submission) == {"query_id", "gold_papers", "answer"}
    assert _RUNNER.materialize_run_outputs(
        run_dir,
        [handoff],
        reader,
        require_evidence=False,
    ) == (1, 1)


def test_materialize_recreates_missing_submission_from_current_answer(tmp_path):
    run_dir, handoff, reader, expected_submission = _checkpointed_run(tmp_path)
    per_query_submission = run_dir / "q1" / "submission.json"
    per_query_submission.unlink()

    assert _RUNNER.materialize_run_outputs(
        run_dir,
        [handoff],
        reader,
    ) == (1, 1)

    assert json.loads(per_query_submission.read_text(encoding="utf-8")) == (
        expected_submission
    )
    assert json.loads((run_dir / "submission.jsonl").read_text(encoding="utf-8")) == (
        expected_submission
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_id", "q_other", "answer checkpoint query_id mismatch"),
        ("status", "running", "answer checkpoint status is not complete"),
    ],
)
def test_materialize_rejects_invalid_answer_checkpoint_identity(
    tmp_path, field, value, message
):
    run_dir, handoff, reader, _ = _checkpointed_run(tmp_path)
    answer_path = run_dir / "q1" / "answer.json"
    answer_record = json.loads(answer_path.read_text(encoding="utf-8"))
    answer_record[field] = value
    answer_path.write_text(json.dumps(answer_record), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_materialize_rejects_stage1_relevance_mismatch_in_answer_checkpoint(
    tmp_path,
):
    run_dir, handoff, reader, _ = _checkpointed_run(tmp_path)
    answer_path = run_dir / "q1" / "answer.json"
    answer_record = json.loads(answer_path.read_text(encoding="utf-8"))
    answer_record["stage1_relevant_paper_ids"] = ["p2"]
    answer_path.write_text(json.dumps(answer_record), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match prediction.gold_papers"):
        _RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_judgment_checkpoint_rejects_mismatched_expanded_evidence(tmp_path):
    run_dir, handoff, reader, _ = _checkpointed_run(tmp_path)
    judgment_path = run_dir / "q1" / "candidate_judgments.jsonl"
    rows = [
        json.loads(line)
        for line in judgment_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["evidence"] = []
    judgments = {row["paper_id"]: row for row in rows}

    status = _RUNNER.validate_judgment_checkpoint(handoff, judgments, reader)

    assert status.stale_paper_ids == ("p1",)


def test_unfiltered_run_requires_confirmation_after_printing_scope(capsys):
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query("q1", "question", ["freeform"]),
            (CandidatePaper("p1", 1), CandidatePaper("p2", 2)),
        ),
        _RUNNER.CandidateHandoff(
            Query("q2", "question", ["freeform"]),
            (CandidatePaper("p3", 1),),
        ),
    ]
    args = SimpleNamespace(
        stage="all",
        paper_id=None,
        query_id=[],
        confirm_full_run=False,
    )

    with pytest.raises(SystemExit, match="--confirm-full-run"):
        _RUNNER._print_and_confirm_run_plan(args, handoffs)

    assert (
        "queries=2, candidate_pairs=3, deterministic_owner_rejections=0, "
        "minimum_calls_without_cache=5, stage=all"
        in capsys.readouterr().out
    )


def test_q004_run_plan_keeps_all_pairs_but_counts_31_zero_call_rejections():
    candidates = (
        CandidatePaper(
            "dynapipe",
            1,
            "DynaPipe: Dynamic Layer Redistribution for Efficient Serving of LLMs",
        ),
        *(
            CandidatePaper(
                f"wrong-{rank}",
                rank,
                f"Unrelated Pipeline Paper {rank}",
            )
            for rank in range(2, 33)
        ),
    )
    handoff = _RUNNER.CandidateHandoff(
        Query(
            "q_004",
            "How many subfigures are there in Figure 4 of the DynaPipe paper?",
            ["multiple_choice"],
            options={"A": "2", "B": "4", "C": "8", "D": "16"},
        ),
        candidates,
    )

    audit = _RUNNER._named_owner_audit([handoff])

    assert audit["version"] == _RUNNER.NAMED_OWNER_RESOLVER_VERSION
    assert audit["deterministic_owner_rejections"] == 31
    assert audit["queries"][0]["paper_id"] == "dynapipe"
    assert audit["queries"][0]["hard_gate"] is True
    assert audit["queries"][0]["candidate_pairs"] == 32
    assert _RUNNER._planned_aoai_calls(
        [handoff], stage="judge", paper_id=None
    ) == (1, 32, 31, 1)
    assert _RUNNER._planned_aoai_calls(
        [handoff], stage="all", paper_id=None
    ) == (1, 32, 31, 2)


def test_global_runner_checkpoints_q004_wrong_31_with_one_provider_call(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidates = (
        CandidatePaper(
            "p1",
            1,
            "DynaPipe: Dynamic Layer Redistribution for Efficient Serving of LLMs",
        ),
        *(
            CandidatePaper(
                f"p{rank}",
                rank,
                f"Unrelated Pipeline Paper {rank}",
            )
            for rank in range(2, 33)
        ),
    )
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": candidate.paper_id,
                "chunk_id": f"{candidate.paper_id}#text",
                "chunk_type": "text_span",
                "text": f"Canonical content for {candidate.title}.",
                "metadata": {"page": 1},
            }
            for candidate in candidates
        ],
    )
    query = Query(
        "q_004",
        "How many subfigures are there in Figure 4 of the DynaPipe paper?",
        ["multiple_choice"],
        options={"A": "2", "B": "4", "C": "8", "D": "16"},
    )
    response = json.dumps(
        {
            "is_relevant_to_answer": True,
            "has_usable_answer_evidence": False,
            "evidence_chunk_ids": [],
        }
    )
    llm = _RUNNER.FakeLLM([response])
    reader = _RUNNER.PairwiseAOAIReader(_RUNNER.ChunkStore(chunks), llm)
    handoff = _RUNNER.CandidateHandoff(query, candidates)
    paths = _RUNNER.QueryRunPaths.under(run_dir, query.query_id)
    paths.directory.mkdir(parents=True)
    state = _RUNNER.QueryExecutionState(
        handoff=handoff,
        paths=paths,
        judgments={},
        target_candidates=list(candidates),
    )

    _RUNNER.run_candidate_judgments_globally(
        reader=reader,
        states=[state],
        run_dir=run_dir,
        workers=16,
        force=False,
    )

    assert len(llm.calls) == 1
    persisted = [
        json.loads(line)
        for line in paths.judgments.read_text(encoding="utf-8").splitlines()
    ]
    assert len(persisted) == 32
    assert persisted[0]["paper_id"] == "p1"
    assert persisted[0]["base_judgment_call_count"] == 1
    assert persisted[0]["is_relevant_to_answer"] is True
    assert persisted[0]["has_usable_answer_evidence"] is False
    assert persisted[0]["send_to_answer_agent"] is False
    assert persisted[0]["evidence_chunk_ids"] == []
    assert persisted[0]["requested_image_count"] == 0
    assert persisted[0]["requested_image_chunk_ids"] == []
    assert persisted[0]["attached_image_count"] == 0
    assert persisted[0]["attached_image_chunk_ids"] == []
    deterministic = persisted[1:]
    assert len(deterministic) == 31
    assert all(item["label"] == "irrelevant" for item in deterministic)
    assert all(item["is_relevant_to_answer"] is False for item in deterministic)
    assert all(
        item["has_usable_answer_evidence"] is False for item in deterministic
    )
    assert all(item["send_to_answer_agent"] is False for item in deterministic)
    assert all(item["evidence_chunk_ids"] == [] for item in deterministic)
    assert all(item["identity_conflict"] is True for item in deterministic)
    assert all(item["provider_invocation_count"] == 0 for item in deterministic)
    assert all(item["calls"] == [] for item in deterministic)
    assert _RUNNER.validate_judgment_checkpoint(
        handoff,
        state.judgments,
        reader,
    ).current


def test_run_plan_warns_that_more_than_main_test_scope_is_optional(capsys):
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query(f"q{index}", "question", ["freeform"]),
            (CandidatePaper(f"p{index}", 1),),
        )
        for index in range(72)
    ]
    args = SimpleNamespace(
        stage="all",
        paper_id=None,
        query_id=[],
        confirm_full_run=True,
        confirm_optional_test_extra=False,
    )

    with pytest.raises(SystemExit, match="--confirm-optional-test-extra"):
        _RUNNER._print_and_confirm_run_plan(args, handoffs)

    output = capsys.readouterr().out
    assert "queries=72, candidate_pairs=72" in output
    assert "required challenge test contains 71 questions" in output
    assert "4,901 optional diagnostic questions" in output

    args.confirm_optional_test_extra = True
    _RUNNER._print_and_confirm_run_plan(args, handoffs)


def test_parser_explains_main_and_optional_test_scopes():
    help_text = " ".join(_RUNNER.build_parser().format_help().split())

    assert "required challenge test has 71 questions" in help_text
    assert "test_extra's 4,901 questions are optional" in help_text
    assert "Second cost gate required whenever more than 71 queries" in help_text


@pytest.mark.parametrize(
    "config_name",
    ["aoai_pairwise_reader.yaml", "aoai_pairwise_reader_hybrid.yaml"],
)
def test_production_configs_use_one_paper_context_without_batch_settings(
    config_name,
):
    config = _RUNNER.load_config(ROOT / "configs" / "agent_style" / config_name)
    params = config["params"]

    assert params["max_paper_context_chars"] == 220_000
    assert params["max_judgment_prompt_chars"] == 240_000
    assert params["max_paper_images"] == 10
    assert "max_batch_chars" not in params
    assert "batch_overlap_chars" not in params
    assert "max_images_per_batch" not in params
    assert "judgment_image_mode" not in params
    assert "image_refine_labels" not in params


def test_unfiltered_run_is_rejected_before_provider_construction(
    monkeypatch, capsys
):
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query("q1", "question", ["freeform"]),
            (CandidatePaper("p1", 1),),
        )
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_aoai_pairwise_reader.py",
            "--queries",
            "queries.jsonl",
            "--candidates",
            "candidates.jsonl",
            "--chunks",
            "chunks.jsonl",
            "--run-dir",
            "run",
        ],
    )
    monkeypatch.setattr(
        _RUNNER,
        "load_config",
        lambda _path: {"name": "aoai_pairwise_reader"},
    )
    monkeypatch.setattr(
        _RUNNER,
        "load_candidate_handoffs",
        lambda *_args, **_kwargs: handoffs,
    )

    def fail_if_provider_is_built(_config):
        raise AssertionError("provider must not be built before confirmation")

    monkeypatch.setattr(_RUNNER, "build_llm", fail_if_provider_is_built)

    with pytest.raises(SystemExit, match="--confirm-full-run"):
        _RUNNER.main()

    assert "minimum_calls_without_cache=2" in capsys.readouterr().out


def test_explicit_query_selection_does_not_require_full_run_confirmation(capsys):
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query("q1", "question", ["freeform"]),
            (CandidatePaper("p1", 1),),
        ),
        _RUNNER.CandidateHandoff(
            Query("q2", "question", ["freeform"]),
            (CandidatePaper("p2", 1),),
        ),
    ]
    args = SimpleNamespace(
        stage="judge",
        paper_id=None,
        query_id=["q1", "q2"],
        confirm_full_run=False,
    )

    _RUNNER._print_and_confirm_run_plan(args, handoffs)

    assert "minimum_calls_without_cache=2" in capsys.readouterr().out


def test_answer_type_selection_preserves_input_order_and_intersects_query_ids():
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query("q1", "question", ["multiple_choice"]),
            (CandidatePaper("p1", 1),),
        ),
        _RUNNER.CandidateHandoff(
            Query("q2", "question", ["table"]),
            (CandidatePaper("p2", 1),),
        ),
        _RUNNER.CandidateHandoff(
            Query("q3", "question", ["table"]),
            (CandidatePaper("p3", 1),),
        ),
    ]

    selected = _RUNNER._select_handoffs(
        handoffs,
        ["q3", "q2"],
        answer_type="table",
    )

    assert [item.query.query_id for item in selected] == ["q2", "q3"]
    with pytest.raises(ValueError, match="selected no inputs"):
        _RUNNER._select_handoffs(
            handoffs,
            ["q1"],
            answer_type="table",
        )


def test_answer_type_selection_does_not_require_full_run_confirmation(capsys):
    handoffs = [
        _RUNNER.CandidateHandoff(
            Query("q1", "question", ["table"]),
            (CandidatePaper("p1", 1),),
        )
    ]
    args = SimpleNamespace(
        stage="answer",
        paper_id=None,
        query_id=[],
        answer_type="table",
        confirm_full_run=False,
    )

    _RUNNER._print_and_confirm_run_plan(args, handoffs)

    assert "queries=1" in capsys.readouterr().out


def test_parser_documents_answer_type_branch_filter():
    help_text = " ".join(_RUNNER.build_parser().format_help().split())

    assert "--answer-type" in help_text
    assert "isolates the table branch" in help_text


def test_materialize_ignores_stale_answer_payload_until_it_is_recomputed(tmp_path):
    run_dir, handoff, reader, _ = _checkpointed_run(tmp_path)
    answer_path = run_dir / "q1" / "answer.json"
    answer_record = json.loads(answer_path.read_text(encoding="utf-8"))
    answer_record["cache_key"] = "stale-prompt-version"
    answer_record["prediction"] = "old payload no longer parseable"
    answer_path.write_text(json.dumps(answer_record), encoding="utf-8")

    assert _RUNNER.materialize_run_outputs(run_dir, [handoff], reader) == (1, 0)
    trace = json.loads((run_dir / "reading_traces.jsonl").read_text(encoding="utf-8"))
    assert trace["answer_checkpoint_current"] is False


def test_materialize_rejects_answer_with_missing_candidate_judgment(tmp_path):
    run_dir, handoff, reader, _ = _checkpointed_run(
        tmp_path, candidate_count=2, judgment_count=1
    )

    with pytest.raises(ValueError, match="incomplete candidate judgments; missing 1"):
        _RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_runner_checkpoints_each_pair_and_recovers_answer_submission_gap(tmp_path):
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    config = tmp_path / "reader.yaml"
    run_dir = tmp_path / "run"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "What value is reported?",
                "answer_types": ["freeform"],
                "table_schema": None,
            }
        ],
    )
    _write_jsonl(
        candidates,
        [
            {
                "query_id": "q1",
                "candidate_papers": [
                    {
                        "rank": 1,
                        "paper_id": "p1",
                        "title": "Paper One",
                        "venue": "ACL",
                        "year": 2025,
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        metadata,
        [{"paper_id": "p1", "title": "Paper One", "venue": "ACL", "year": 2025}],
    )
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "The reported value is 42.",
                "metadata": {"page": 2},
            }
        ],
    )
    judgment = {
        "is_relevant_to_answer": True,
        "has_usable_answer_evidence": True,
        "evidence_chunk_ids": ["p1#1"],
    }
    answer = {
        "status": "ready",
        "papers": [{"paper_id": "p1", "evidence_chunk_ids": ["p1#1"]}],
        "paper_relevance": [
            {
                "paper_id": "p1",
                "role": "target_owner",
                "reason": "owns and reports the requested value",
            }
        ],
        "derivation": {
            "facts": [
                {
                    "id": "f_reported_value",
                    "name": "reported value",
                    "value": "42",
                    "value_kind": "reported",
                    "paper_id": "p1",
                    "chunk_ids": ["p1#1"],
                }
            ],
            "operations": [],
            "answer_bindings": [
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "f_reported_value",
                    "answer_fragment": "42",
                }
            ],
            "final_semantic_answer": "42",
        },
        "answer": {"freeform": {"text": "42"}},
        "support": [
            {
                "answer_path": "answer.freeform.text",
                "paper_id": "p1",
                "chunk_ids": ["p1#1"],
            }
        ],
        "completeness": {"answered_parts": ["value"], "missing": []},
    }
    config.write_text(
        yaml.safe_dump(
            {
                "name": "aoai_pairwise_reader",
                "llm": {
                    "name": "fake",
                    "params": {
                        "responses": [json.dumps(judgment), json.dumps(answer)]
                    },
                },
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/run_aoai_pairwise_reader.py",
        "--queries",
        str(queries),
        "--candidates",
        str(candidates),
        "--paper-metadata",
        str(metadata),
        "--chunks",
        str(chunks),
        "--reader",
        str(config),
        "--run-dir",
        str(run_dir),
        "--max-candidates",
        "1",
        "--confirm-full-run",
    ]

    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert (
        "image paths are disabled; configure --image-root to enable visual input"
        in result.stdout
    )
    judgments = [
        json.loads(line)
        for line in (run_dir / "q1" / "candidate_judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(judgments) == 1
    assert judgments[0]["paper_id"] == "p1"
    assert judgments[0]["is_relevant_to_answer"] is True
    assert judgments[0]["has_usable_answer_evidence"] is True
    assert judgments[0]["send_to_answer_agent"] is True
    assert judgments[0]["evidence_chunk_ids"] == ["p1#1"]
    trace = json.loads((run_dir / "reading_traces.jsonl").read_text())
    assert trace["relevance_judgments"][0]["is_relevant_to_answer"] is True
    assert trace["relevance_judgments"][0]["has_usable_answer_evidence"] is True
    assert trace["relevance_judgments"][0]["send_to_answer_agent"] is True
    assert trace["relevance_judgments"][0]["evidence_chunk_ids"] == ["p1#1"]
    assert trace["submission"]["gold_papers"] == [{"paper_id": "p1"}]
    assert trace["submission"]["answer"] == {"freeform": {"text": "42"}}
    assert json.loads((run_dir / "submission.jsonl").read_text())["query_id"] == "q1"
    answer_attempts = [
        json.loads(line)
        for line in (run_dir / "q1" / "answer_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(answer_attempts) == 1
    assert answer_attempts[0]["query_id"] == "q1"
    assert answer_attempts[0]["parse_error"] is None

    # Simulate a process dying after checkpoint_answer_update wrote answer.json
    # but before it wrote submission.json. The coordinator's pre-call bulk
    # invalidation also means no old aggregate row should be trusted on restart.
    (run_dir / "q1" / "submission.json").unlink()
    invalidate_aggregate_query(run_dir, "q1")

    resumed = subprocess.run(
        [*command, "--resume"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "[q1] answer cached" in resumed.stdout
    assert "[q1] answer complete" not in resumed.stdout
    assert (run_dir / "q1" / "submission.json").is_file()
    assert json.loads((run_dir / "submission.jsonl").read_text())["query_id"] == "q1"
