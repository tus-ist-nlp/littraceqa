from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/sync_official_release.py"
    spec = importlib.util.spec_from_file_location("sync_official_release", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SYNC = _load_module()


def test_checked_in_manifest_pins_current_official_splits() -> None:
    manifest = _SYNC.load_manifest(_SYNC.DEFAULT_MANIFEST)

    assert manifest["revision"] == "bd35dc14cf0483e0ffa51fa2a54d2689c13f9845"
    assert manifest["files"]["data/sample_submission.jsonl"] == (
        "d64dbf95b5ddf1e68bb714f77e5a185e3cc3407469160e0c4e0d34041de7e251"
    )
    assert "data/test.jsonl" in manifest["files"]
    assert "data/test_extra.jsonl" in manifest["files"]
    assert "scripts/evaluate.py" in manifest["files"]
    assert "scripts/validate_submission.py" in manifest["files"]
    assert all(len(value) == 64 for value in manifest["files"].values())


def test_manifest_rejects_path_traversal(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repository": "LitTraceQA/LitTraceQA",
                "revision": "a" * 40,
                "files": {"../escape": "b" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe release path"):
        _SYNC.load_manifest(manifest_path)
