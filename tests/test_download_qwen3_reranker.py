from __future__ import annotations

from pathlib import Path

import pytest

from scripts.download_qwen3_reranker import download, reranker_spec


def _write_search_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_download_uses_model_revision_and_cache_from_config(tmp_path):
    search = _write_search_config(
        tmp_path / "search.yaml",
        """
reranker:
  name: qwen3
  params:
    model: Qwen/example
    revision: abc123
""",
    )
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    snapshot = download(
        search,
        Path("~/model-cache"),
        snapshot_download=fake_download,
    )

    assert snapshot == tmp_path / "snapshot"
    assert calls == [
        {
            "repo_id": "Qwen/example",
            "revision": "abc123",
            "cache_dir": str(Path("~/model-cache").expanduser()),
        }
    ]


@pytest.mark.parametrize(
    "body, message",
    [
        ("reranker: {name: none, params: {}}\n", "does not select qwen3"),
        ("reranker: {name: qwen3, params: {revision: abc}}\n", "model is missing"),
        ("reranker: {name: qwen3, params: {model: Qwen/x}}\n", "must be pinned"),
    ],
)
def test_reranker_spec_rejects_incomplete_config(tmp_path, body, message):
    search = _write_search_config(tmp_path / "search.yaml", body)

    with pytest.raises(ValueError, match=message):
        reranker_spec(search)
