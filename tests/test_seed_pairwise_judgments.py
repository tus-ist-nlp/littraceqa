from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_seed_function():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/seed_pairwise_judgments.py"
    spec = importlib.util.spec_from_file_location("seed_pairwise_judgments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.seed


seed = _load_seed_function()


def _write_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"run":"old"}\n', encoding="utf-8")
    query = source / "q_001"
    query.mkdir()
    (query / "candidate_judgments.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q_001",
                "paper_id": "paper_1",
                "status": "complete",
                "cache_key": "abc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (query / "answer.json").write_text('{"must":"not copy"}\n', encoding="utf-8")
    return source


def test_seeds_only_stage1_and_records_provenance(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    destination = tmp_path / "destination"
    result = seed(source, destination)
    assert result["queries"] == 1
    assert result["candidate_judgments"] == 1
    assert (destination / "q_001/candidate_judgments.jsonl").is_file()
    assert not (destination / "q_001/answer.json").exists()
    assert not (destination / "manifest.json").exists()
    assert json.loads((destination / "seeded_judgments.json").read_text())[
        "source_manifest_sha256"
    ]


def test_refuses_destination_with_manifest(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already has a manifest"):
        seed(source, destination)


def test_rejects_incomplete_source_checkpoint(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    path = source / "q_001/candidate_judgments.jsonl"
    record = json.loads(path.read_text())
    record["status"] = "error"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid complete"):
        seed(source, tmp_path / "destination")
