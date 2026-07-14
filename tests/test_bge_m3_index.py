"""Focused tests for the exact BGE-M3 NumPy index."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from litqa.contracts import Chunk
from litqa.index.bge_m3_index import BGEM3NumpyIndex


def _chunk(chunk_id: str, paper_id: str, text: str, chunk_type: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        chunk_type=chunk_type,
        metadata={"observed": True},
    )


def _embedding(text: str) -> np.ndarray:
    lowered = text.lower()
    if "alpha" in lowered:
        return np.array([1.0, 0.0], dtype=np.float32)
    if "beta" in lowered:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([2 ** -0.5, 2 ** -0.5], dtype=np.float32)


def _fake_embed(texts: list[str]) -> np.ndarray:
    return np.stack([_embedding(text) for text in texts])


def test_build_filters_chunk_types_and_exact_search_is_deterministic(
    tmp_path: Path, monkeypatch
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    index = BGEM3NumpyIndex(
        index_dir=str(tmp_path / "index"),
        model_path=str(model_path),
        include_chunk_types=["title_abstract"],
    )
    monkeypatch.setattr(index, "_embed", _fake_embed)
    index.build(
        [
            _chunk("a", "p1", "alpha", "title_abstract"),
            _chunk("ignored", "p1", "beta", "figure"),
            _chunk("b", "p2", "beta", "title_abstract"),
        ]
    )

    first = index.search("alpha query", 20)
    second = index.search("alpha query", 20)

    assert [result.paper_id for result in first] == ["p1", "p2"]
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]
    assert first[0].source == "bge_m3_numpy"
    assert np.load(tmp_path / "index" / "embeddings.npy").shape == (2, 2)


def test_load_validates_config_and_restores_saved_chunks(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    index_dir = tmp_path / "index"
    built = BGEM3NumpyIndex(
        index_dir=str(index_dir),
        model_path=str(model_path),
        max_length=512,
    )
    monkeypatch.setattr(built, "_embed", _fake_embed)
    built.build([_chunk("a", "p1", "alpha", "title_abstract")])

    loaded = BGEM3NumpyIndex(
        index_dir=str(index_dir),
        model_path=str(model_path),
        max_length=512,
    )
    loaded.load()
    monkeypatch.setattr(loaded, "_embed", _fake_embed)

    assert loaded.search("alpha", 1)[0].chunk_id == "a"
    config = json.loads((index_dir / "index_config.json").read_text())
    assert config["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert config["pooling"] == "sentence_transformers_model_config"
    assert config["normalize_embeddings"] is True
    assert config["row_count"] == 1
    assert len(config["embeddings_sha256"]) == 64
    assert len(config["chunks_sha256"]) == 64

    mismatched = BGEM3NumpyIndex(
        index_dir=str(index_dir),
        model_path=str(model_path),
        max_length=256,
    )
    with pytest.raises(ValueError, match="does not match"):
        mismatched.load()


def test_build_rejects_empty_selection_and_duplicate_chunk_ids(
    tmp_path: Path, monkeypatch
):
    index = BGEM3NumpyIndex(
        index_dir=str(tmp_path / "index"),
        include_chunk_types=["title_abstract"],
    )
    monkeypatch.setattr(index, "_embed", _fake_embed)

    with pytest.raises(ValueError, match="no selected"):
        index.build([_chunk("figure", "p1", "alpha", "figure")])
    with pytest.raises(ValueError, match="unique chunk_id"):
        index.build(
            [
                _chunk("same", "p1", "alpha", "title_abstract"),
                _chunk("same", "p2", "beta", "title_abstract"),
            ]
        )


@pytest.mark.parametrize("max_length", [0, 8193, True])
def test_constructor_rejects_unsafe_max_length(tmp_path: Path, max_length):
    with pytest.raises(ValueError, match="max_length"):
        BGEM3NumpyIndex(index_dir=str(tmp_path / "index"), max_length=max_length)


def test_load_rejects_artifact_changed_after_config_commit(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    index_dir = tmp_path / "index"
    built = BGEM3NumpyIndex(index_dir=str(index_dir), model_path=str(model_path))
    monkeypatch.setattr(built, "_embed", _fake_embed)
    built.build([_chunk("a", "p1", "alpha", "title_abstract")])
    with (index_dir / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    loaded = BGEM3NumpyIndex(index_dir=str(index_dir), model_path=str(model_path))
    with pytest.raises(ValueError, match="Chunk checksum"):
        loaded.load()


def test_build_rejects_non_normalized_embeddings(tmp_path: Path, monkeypatch):
    index = BGEM3NumpyIndex(index_dir=str(tmp_path / "index"))
    monkeypatch.setattr(
        index,
        "_embed",
        lambda texts: np.full((len(texts), 2), 2.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="L2-normalized"):
        index.build([_chunk("a", "p1", "alpha", "title_abstract")])


def test_lazy_model_load_is_pinned_offline_and_prefix_free(tmp_path: Path, monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, target, **kwargs):
            calls["target"] = target
            calls["constructor"] = kwargs
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["encode"] = kwargs
            calls["max_seq_length"] = self.max_seq_length
            return np.array([[1.0, 0.0]], dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    index = BGEM3NumpyIndex(
        index_dir=str(tmp_path / "index"),
        model="BAAI/bge-m3",
        revision="fixed-revision",
        batch_size=1,
        device="cpu",
        max_length=512,
        local_files_only=True,
    )

    assert index._embed(["observed text"]).shape == (1, 2)  # noqa: SLF001
    assert calls == {
        "target": "BAAI/bge-m3",
        "constructor": {
            "device": "cpu",
            "local_files_only": True,
            "revision": "fixed-revision",
        },
        "texts": ["observed text"],
        "encode": {
            "batch_size": 1,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "prompt_name": None,
            "prompt": "",
        },
        "max_seq_length": 512,
    }


@pytest.mark.parametrize("schema_version", [True, 3, [], "2"])
def test_load_rejects_invalid_schema_version(
    tmp_path: Path, schema_version
):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_config.json").write_text(
        json.dumps({"schema_version": schema_version}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema version"):
        BGEM3NumpyIndex(index_dir=str(index_dir)).load()


def test_equal_scores_keep_input_order_before_and_after_reload(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    index_dir = tmp_path / "index"
    built = BGEM3NumpyIndex(index_dir=str(index_dir), model_path=str(model_path))
    monkeypatch.setattr(built, "_embed", _fake_embed)
    built.build(
        [
            _chunk("first", "p-first", "gamma", "title_abstract"),
            _chunk("second", "p-second", "delta", "title_abstract"),
        ]
    )
    assert [item.paper_id for item in built.search("gamma", 2)] == [
        "p-first",
        "p-second",
    ]

    loaded = BGEM3NumpyIndex(index_dir=str(index_dir), model_path=str(model_path))
    loaded.load()
    monkeypatch.setattr(loaded, "_embed", _fake_embed)
    assert [item.paper_id for item in loaded.search("gamma", 2)] == [
        "p-first",
        "p-second",
    ]
