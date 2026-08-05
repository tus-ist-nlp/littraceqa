from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_runner_writes_submission_and_separate_analysis(tmp_path):
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    config = tmp_path / "agent.yaml"
    output = tmp_path / "submission.jsonl"

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
                "metadata": {"page": 3},
            }
        ],
    )
    config.write_text(
        yaml.safe_dump(
            {
                "name": "corpus_qa",
                "llm": {
                    "name": "fake",
                    "params": {
                        "responses": [
                            '{"targets":[{"name":"reported value","search_terms":[]}]'
                            ',"venues":[],"years":[],"modalities":["text_span"],'
                            '"requires_multiple_papers":false}',
                            '{"paper_ids":["p1"],"unresolved_targets":[]}',
                            '{"papers":[{"paper_id":"p1","evidence_chunk_ids":["p1#1"]}],'
                            '"answer":{"freeform":{"text":"42"}}}',
                        ]
                    },
                },
                "params": {},
            }
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "scripts/run_corpus_qa.py",
        "--queries",
        str(queries),
        "--candidates",
        str(candidates),
        "--paper-metadata",
        str(metadata),
        "--chunks",
        str(chunks),
        "--agent",
        str(config),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    submission = json.loads(output.read_text(encoding="utf-8"))
    assert set(submission) == {"query_id", "gold_papers", "evidence", "answer"}
    assert submission["answer"] == {"freeform": {"text": "42"}}
    analysis_path = output.with_suffix(output.suffix + ".analysis.jsonl")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["trace"]
    assert analysis["candidate_papers"] == ["p1"]
    checkpoint_path = output.with_suffix(output.suffix + ".checkpoint.jsonl")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert checkpoint["query_id"] == "q1"
    assert manifest["inputs"]["chunks"]["sha256"]
    assert set(manifest["git"]) == {"sha"}
    assert manifest["preflight"]["canonical_papers_missing_from_corpus"] == []
    assert manifest["preflight"]["errors"] == []

    resumed = subprocess.run(
        [*command, "--resume"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert output.read_text(encoding="utf-8").count("\n") == 1
