from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from littraceqa import pairwise_run_store as pairwise_run_store_module
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
invalidate_aggregate_query = _RUNNER.invalidate_aggregate_query

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
    assert "Read only the supplied candidate papers and evidence" in captured["system"]
    assert "do not search or use external knowledge" in captured["system"]


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

    assert strict_args.allow_missing_figure_images is False
    assert allowed_args.allow_missing_figure_images is True
    assert strict_manifest["reader"]["allow_missing_figure_images"] is False
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


def test_judgment_update_invalidates_aggregate_before_checkpoint(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        _RUNNER,
        "invalidate_aggregate_query",
        lambda run_dir, query_id: events.append(("invalidate", run_dir, query_id)),
    )
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

    assert [event[0] for event in events] == ["invalidate", "checkpoint"]


def test_answer_update_invalidates_aggregate_before_both_checkpoints(
    monkeypatch, tmp_path
):
    events = []
    monkeypatch.setattr(
        _RUNNER,
        "invalidate_aggregate_query",
        lambda run_dir, query_id: events.append(("invalidate", run_dir, query_id)),
    )
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

    assert [event[0] for event in events] == ["invalidate", "write", "write"]


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
    judgments = []
    for candidate in candidates[:judgment_count]:
        records = store.load_paper(candidate.paper_id)
        chunk_id = str(records[0]["chunk_id"])
        judgments.append(
            {
                "query_id": query.query_id,
                "paper_id": candidate.paper_id,
                "rank": candidate.rank,
                "status": "complete",
                "cache_key": reader.judgment_cache_key(query, candidate, records),
                "label": "direct_answer",
                "relevant": True,
                "satisfied_constraints": ["reported value"],
                "missing_constraints": [],
                "evidence": [
                    {
                        "chunk_id": chunk_id,
                        "source_type": "text_span",
                        "locator": {"page": 1},
                        "quote_or_value": "42",
                    }
                ],
                "candidate_answer": {"meaning": "42"},
                "reason": "direct statement",
                "visual_conflict": False,
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
        "queries=2, candidate_pairs=3, minimum_calls_without_cache=5, stage=all"
        in capsys.readouterr().out
    )


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
        "paper_role": "target_owner",
        "label": "direct_answer",
        "answerable_from_this_paper": True,
        "satisfied_constraints": ["reported value"],
        "missing_constraints": [],
        "blocking_mismatches": [],
        "visual": {"required": False, "status": "not_needed"},
        "evidence": [
            {
                "chunk_id": "p1#1",
                "purpose": "answer",
                "quote_or_value": "42",
            }
        ],
        "candidate_answer": {
            "units": [
                {
                    "name": "reported value",
                    "value": "42",
                    "value_kind": "reported",
                    "matched_option_labels": [],
                }
            ],
            "rows": [],
        },
        "confidence": 1.0,
        "reason": "direct statement",
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
    trace = json.loads((run_dir / "reading_traces.jsonl").read_text())
    assert trace["relevance_judgments"][0]["relevant"] is True
    assert trace["submission"]["answer"] == {"freeform": {"text": "42"}}
    assert json.loads((run_dir / "submission.jsonl").read_text())["query_id"] == "q1"

    # Simulate a process dying after checkpoint_answer_update wrote answer.json
    # but before it wrote submission.json. Its pre-write invalidation also means
    # no old aggregate row should be trusted on restart.
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
