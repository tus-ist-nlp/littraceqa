from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from littraceqa.di_pipeline.llm import azure_openai as azure_openai_module


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_aoai_pairwise_reader", ROOT / "scripts/run_aoai_pairwise_reader.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
_SAFE_QUERY_ID = _RUNNER._SAFE_QUERY_ID
invalidate_aggregate_query = _RUNNER.invalidate_aggregate_query


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
    assert "与えられた候補論文と根拠だけ" in captured["system"]
    assert "検索や外部知識を使わない" in captured["system"]


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
        [*base_args, "--allow-missing-figure-images"]
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


def test_manifest_fingerprints_all_pairwise_runtime_dependencies(
    tmp_path, monkeypatch
):
    expected_dependencies = {
        "src/littraceqa/di_pipeline/agent/evidence.py",
        "src/littraceqa/di_pipeline/agent/json_utils.py",
        "src/littraceqa/di_pipeline/contracts.py",
        "src/littraceqa/di_pipeline/llm/azure_openai.py",
        "src/littraceqa/di_pipeline/llm/base.py",
        "src/littraceqa/di_pipeline/llm/fake.py",
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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _checkpointed_run(
    tmp_path: Path, *, candidate_count: int = 1, judgment_count: int | None = None
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
    query = _RUNNER.Query(
        query_id="q1",
        question="What value is reported?",
        answer_types=["freeform"],
        table_schema=None,
    )
    candidates = tuple(
        _RUNNER.CandidatePaper(f"p{index}", index)
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

    prediction = _RUNNER.Prediction(
        query_id=query.query_id,
        gold_papers=[{"paper_id": "p1"}],
        evidence=[
            _RUNNER.Evidence(
                paper_id="p1",
                source_type="text_span",
                locator=_RUNNER.EvidenceLocator(page=1),
                evidence_text_or_value="42",
            )
        ],
        answer=_RUNNER.Answer(freeform={"text": "42"}),
        candidate_papers=[candidate.paper_id for candidate in candidates],
    )
    answer_record = {
        "query_id": query.query_id,
        "status": "complete",
        "cache_key": reader.answer_cache_key(query, judgments),
        "prediction": prediction.to_dict(),
    }
    submission = _RUNNER.prediction_to_submission(query, prediction)
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


def test_materialize_rejects_answer_with_missing_candidate_judgment(tmp_path):
    run_dir, handoff, reader, _ = _checkpointed_run(
        tmp_path, candidate_count=2, judgment_count=1
    )

    with pytest.raises(ValueError, match="incomplete candidate judgments; missing 1"):
        _RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_runner_checkpoints_each_pair_and_emits_analyzer_trace(tmp_path):
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
        "label": "direct_answer",
        "answerable_from_this_paper": True,
        "satisfied_constraints": ["reported value"],
        "missing_constraints": [],
        "evidence": [{"chunk_id": "p1#1", "quote_or_value": "42"}],
        "candidate_answer": {"meaning": "42"},
        "confidence": 1.0,
        "reason": "direct statement",
    }
    answer = {
        "papers": [{"paper_id": "p1", "evidence_chunk_ids": ["p1#1"]}],
        "answer": {"freeform": {"text": "42"}},
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
    ]

    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
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

    resumed = subprocess.run(
        [*command, "--resume"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "cached" in resumed.stdout
