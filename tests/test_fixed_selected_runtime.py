from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from littraceqa.aoai_pairwise_reader import (
    FIXED_SELECTED_CHECKPOINT_KIND,
    FIXED_SELECTED_PAPER_POLICY,
    ChunkStore,
    PairwiseAOAIReader,
)
from littraceqa.candidate_handoff import (
    CandidateHandoff,
    CandidatePaper,
    load_candidate_handoffs,
)
from littraceqa.di_pipeline.contracts import (
    Answer,
    Evidence,
    EvidenceLocator,
    Prediction,
    Query,
)
from littraceqa.di_pipeline.llm.fake import FakeLLM


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(
    "fixed_selected_runner", ROOT / "scripts/run_aoai_pairwise_reader.py"
)
RENDERER = _load_script(
    "fixed_selected_renderer", ROOT / "scripts/render_aoai_prompts.py"
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _minimal_manifest_inputs(tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    reader_config = tmp_path / "reader.yaml"
    _write_jsonl(
        chunks,
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "The value is 42.",
                "metadata": {"page": 1},
            },
            {
                "paper_id": "p2",
                "chunk_id": "p2#1",
                "chunk_type": "text_span",
                "text": "This paper provides a necessary condition.",
                "metadata": {"page": 2},
            },
        ],
    )
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "What value is reported?",
                "answer_types": ["freeform"],
            }
        ],
    )
    _write_jsonl(
        candidates,
        [{"query_id": "q1", "candidate_papers": ["p1", "p2"]}],
    )
    _write_jsonl(
        metadata,
        [
            {"paper_id": "p1", "title": "Paper One", "venue": "ACL", "year": 2025},
            {"paper_id": "p2", "title": "Paper Two", "venue": "ACL", "year": 2025},
        ],
    )
    reader_config.write_text("name: fixed-selected-test\n", encoding="utf-8")
    return chunks, queries, candidates, metadata, reader_config


def test_fixed_selected_manifest_records_policy_and_disables_owner_filter(tmp_path):
    chunks, queries, candidates, metadata, reader_config = _minimal_manifest_inputs(
        tmp_path
    )
    args = RUNNER.build_parser().parse_args(
        [
            "--queries",
            str(queries),
            "--candidates",
            str(candidates),
            "--paper-metadata",
            str(metadata),
            "--chunks",
            str(chunks),
            "--reader",
            str(reader_config),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    config = {
        "llm": {"name": "fake", "params": {}},
        "params": {"paper_set_policy": FIXED_SELECTED_PAPER_POLICY},
    }
    handoffs = load_candidate_handoffs(queries, candidates, metadata)

    manifest = RUNNER.build_manifest(
        args, config, ChunkStore(chunks), handoffs
    )

    assert manifest["schema_version"] == 3
    assert manifest["workflow"] == "fixed_selected_paper_evidence_reading"
    assert manifest["reader"]["paper_set_policy"] == FIXED_SELECTED_PAPER_POLICY
    assert manifest["reader"]["max_candidates"] is None
    assert manifest["reader"]["named_owner_resolution"] == {
        "version": RUNNER.NAMED_OWNER_RESOLVER_VERSION,
        "scope": "disabled_for_fixed_selected_papers",
        "deterministic_owner_rejections": 0,
        "queries": [],
    }
    assert RUNNER._planned_aoai_calls(
        handoffs,
        stage="all",
        paper_id=None,
        paper_set_policy=FIXED_SELECTED_PAPER_POLICY,
    ) == (1, 2, 0, 3)


def test_fixed_selected_production_config_can_cite_every_selected_paper() -> None:
    config = RUNNER.load_config(
        ROOT / "configs" / "agent_style" / "aoai_selected_paper_reader.yaml"
    )
    params = config["params"]

    assert params["max_evidence"] >= params["max_answer_papers"]


def test_fixed_selected_main_rejects_max_candidates_before_loading_or_provider(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_aoai_pairwise_reader.py",
            "--queries",
            "unused-queries.jsonl",
            "--candidates",
            "unused-candidates.jsonl",
            "--chunks",
            "unused-chunks.jsonl",
            "--reader",
            "unused-reader.yaml",
            "--run-dir",
            "unused-run",
            "--max-candidates",
            "1",
        ],
    )
    monkeypatch.setattr(
        RUNNER,
        "load_config",
        lambda _path: {
            "name": "fixed-selected-test",
            "params": {"paper_set_policy": FIXED_SELECTED_PAPER_POLICY},
        },
    )
    monkeypatch.setattr(
        RUNNER,
        "load_candidate_handoffs",
        lambda *_args, **_kwargs: pytest.fail("inputs must not be loaded"),
    )
    monkeypatch.setattr(
        RUNNER,
        "build_llm",
        lambda _config: pytest.fail("provider must not be constructed"),
    )

    with pytest.raises(SystemExit, match="forbidden in fixed-selected mode"):
        RUNNER.main()


def _fixed_selected_checkpoint(tmp_path: Path):
    chunks, queries, candidates_path, metadata, _ = _minimal_manifest_inputs(tmp_path)
    handoff = load_candidate_handoffs(queries, candidates_path, metadata)[0]
    reader = PairwiseAOAIReader(
        ChunkStore(chunks), FakeLLM(), paper_set_policy=FIXED_SELECTED_PAPER_POLICY
    )
    judgments: list[dict] = []
    for candidate in handoff.candidate_papers:
        records = reader.chunk_store.load_paper(candidate.paper_id)
        chunk_id = str(records[0]["chunk_id"])
        has_evidence = candidate.paper_id == "p1"
        fact = {
            "chunk_id": chunk_id,
            "purpose": "answer_value",
            "fact": "The reported value is 42.",
            "source_excerpt": "The value is 42.",
        }
        judgments.append(
            {
                "query_id": handoff.query.query_id,
                "paper_id": candidate.paper_id,
                "rank": candidate.rank,
                "status": "complete",
                "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
                "cache_key": reader.judgment_cache_key(
                    handoff.query, candidate, records
                ),
                "is_relevant_to_answer": True,
                "has_usable_answer_evidence": has_evidence,
                "send_to_answer_agent": has_evidence,
                "evidence_chunk_ids": [chunk_id] if has_evidence else [],
                "context_chunk_ids": [chunk_id],
                "evidence": (
                    [
                        {
                            "chunk_id": chunk_id,
                            "source_type": "text_span",
                            "locator": {"page": 1},
                            "purpose": "answer_value",
                            "quote_or_value": "The value is 42.",
                        }
                    ]
                    if has_evidence
                    else []
                ),
                "extracted_facts": [fact] if has_evidence else [],
                "attached_image_count": 0,
                "attached_image_chunk_ids": [],
            }
        )

    prediction = Prediction(
        query_id=handoff.query.query_id,
        gold_papers=[{"paper_id": "p1"}, {"paper_id": "p2"}],
        evidence=[
            Evidence(
                paper_id="p1",
                source_type="text_span",
                locator=EvidenceLocator(page=1),
                evidence_text_or_value="42",
            )
        ],
        answer=Answer(freeform={"text": "42"}),
        candidate_papers=["p1", "p2"],
    )
    answer_record = {
        "query_id": handoff.query.query_id,
        "status": "complete",
        "paper_set_policy": FIXED_SELECTED_PAPER_POLICY,
        "cache_key": reader.answer_cache_key(handoff.query, judgments),
        "accepted_paper_ids": ["p1"],
        "stage1_relevant_paper_ids": None,
        "submission_paper_ids": ["p1", "p2"],
        "prediction": prediction.to_dict(),
    }
    expected_submission = RUNNER.prediction_to_submission(handoff.query, prediction)
    run_dir = tmp_path / "run"
    query_dir = run_dir / handoff.query.query_id
    query_dir.mkdir(parents=True)
    _write_jsonl(query_dir / "candidate_judgments.jsonl", judgments)
    (query_dir / "answer.json").write_text(
        json.dumps(answer_record), encoding="utf-8"
    )
    (query_dir / "submission.json").write_text(
        json.dumps(expected_submission), encoding="utf-8"
    )
    return run_dir, handoff, reader, expected_submission


def test_fixed_selected_checkpoint_restores_all_selected_papers(tmp_path):
    run_dir, handoff, reader, expected_submission = _fixed_selected_checkpoint(
        tmp_path
    )
    paths = RUNNER.QueryRunPaths.under(run_dir, handoff.query.query_id)
    paths.submission.unlink()

    assert RUNNER.materialize_run_outputs(run_dir, [handoff], reader) == (1, 1)

    restored = json.loads(paths.submission.read_text(encoding="utf-8"))
    assert restored == expected_submission
    assert restored["gold_papers"] == [
        {"paper_id": "p1"},
        {"paper_id": "p2"},
    ]
    assert json.loads((run_dir / "submission.jsonl").read_text(encoding="utf-8")) == (
        expected_submission
    )


def test_fixed_selected_checkpoint_cannot_shrink_authoritative_papers_together(
    tmp_path,
):
    run_dir, handoff, reader, _ = _fixed_selected_checkpoint(tmp_path)
    answer_path = run_dir / "q1" / "answer.json"
    answer_record = json.loads(answer_path.read_text(encoding="utf-8"))
    answer_record["submission_paper_ids"] = ["p1"]
    answer_record["prediction"]["gold_papers"] = [{"paper_id": "p1"}]
    answer_path.write_text(json.dumps(answer_record), encoding="utf-8")
    (run_dir / "q1" / "submission.json").unlink()

    with pytest.raises(ValueError, match="externally selected candidate papers"):
        RUNNER.materialize_run_outputs(run_dir, [handoff], reader)


def test_fixed_selected_checkpoint_requires_fixed_extraction_shape(tmp_path):
    run_dir, handoff, reader, _ = _fixed_selected_checkpoint(tmp_path)
    judgment_path = run_dir / "q1" / "candidate_judgments.jsonl"
    judgments = [
        json.loads(line)
        for line in judgment_path.read_text(encoding="utf-8").splitlines()
    ]
    judgments[0].pop("checkpoint_kind")
    _write_jsonl(judgment_path, judgments)

    status = RUNNER.validate_judgment_checkpoint(
        handoff,
        {item["paper_id"]: item for item in judgments},
        reader,
    )

    assert status.stale_paper_ids == ("p1",)


def test_fixed_selected_preview_matches_production_prompts_and_metadata(tmp_path):
    chunks, queries, candidates, metadata, _ = _minimal_manifest_inputs(tmp_path)
    paper_text = tmp_path / "paper.txt"
    evidence_text = tmp_path / "evidence.txt"
    accepted_summary = tmp_path / "accepted.json"
    output = tmp_path / "preview.json"
    paper_text.write_text(
        '[chunk {"paper_id":"p1","chunk_id":"p1#1","source_type":"text_span",'
        '"locator":{"page":1}}]\nThe value is 42.',
        encoding="utf-8",
    )
    evidence_text.write_text(paper_text.read_text(encoding="utf-8"), encoding="utf-8")
    accepted = [
        {
            "paper_id": "p1",
            "title": "Paper One",
            "rank": 1,
            "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
            "is_relevant_to_answer": True,
            "has_usable_answer_evidence": True,
            "send_to_answer_agent": True,
            "evidence": [
                {
                    "chunk_id": "p1#1",
                    "source_type": "text_span",
                    "locator": {"page": 1},
                    "purpose": "answer_value",
                }
            ],
            "extracted_facts": [
                {
                    "chunk_id": "p1#1",
                    "purpose": "answer_value",
                    "fact": "The reported value is 42.",
                    "source_excerpt": "The value is 42.",
                }
            ],
        }
    ]
    accepted_summary.write_text(json.dumps(accepted), encoding="utf-8")

    assert (
        RENDERER.main(
            [
                "--queries",
                str(queries),
                "--candidates",
                str(candidates),
                "--paper-metadata",
                str(metadata),
                "--paper-id",
                "p1",
                "--paper-set-policy",
                FIXED_SELECTED_PAPER_POLICY,
                "--paper-text-file",
                str(paper_text),
                "--accepted-summary-file",
                str(accepted_summary),
                "--evidence-file",
                str(evidence_text),
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    preview = json.loads(output.read_text(encoding="utf-8"))
    handoff: CandidateHandoff = load_candidate_handoffs(
        queries, candidates, metadata
    )[0]
    candidate: CandidatePaper = handoff.candidate_papers[0]
    reader = PairwiseAOAIReader(
        ChunkStore(chunks), FakeLLM(), paper_set_policy=FIXED_SELECTED_PAPER_POLICY
    )
    record = reader.chunk_store.load_paper("p1")[0]
    production_judgment = reader._judgment_prompt(
        query=handoff.query,
        candidate=candidate,
        context={
            "text": paper_text.read_text(encoding="utf-8"),
            "records_by_id": {"p1#1": record},
            "image_paths": [],
            "compacted": True,
            "total_chunk_count": 1,
            "selected_chunk_ids": ["p1#1"],
            "omitted_chunk_ids": [],
        },
    )
    production_answer = reader._answer_prompt(
        handoff.query,
        accepted,
        {
            "text": evidence_text.read_text(encoding="utf-8"),
            "records_by_id": {"p1#1": record},
            "image_paths": [],
        },
    )

    assert preview["candidate_payload"] == {
        "paper_id": "p1",
        "rank": 1,
        "title": "Paper One",
        "venue": "ACL",
        "year": 2025,
    }
    assert preview["paper_set_policy"] == FIXED_SELECTED_PAPER_POLICY
    assert [item["prompt_version"] for item in preview["prompts"]] == [
        reader.judgment_prompt_version,
        reader.answer_prompt_version,
    ]
    assert [item["text"] for item in preview["prompts"]] == [
        production_judgment,
        production_answer,
    ]
    assert preview["few_shot_examples"]["judgment"] == (
        reader.judgment_example_ids(handoff.query)
    )
