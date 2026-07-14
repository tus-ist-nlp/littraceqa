"""compose_config() のパス自動導出・衝突回避ロジックのテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import litqa.config as config_module
from litqa.config import _build_component, compose_config


def _paths() -> dict:
    return {
        "pdf_dir": "/data/pdfs",
        "chunks_dir": "/data/chunks",
        "index_dir": "/data/index",
        "docint_chunks": "/data/docint_chunks",
        "backup_dir": "/data/backup",
        "paper_metadata": "data/paper_metadata.jsonl",
    }


def _search() -> dict:
    return {
        "per_index_k": 100,
        "indexers": [{"name": "bm25s", "params": {}}],
        "fuser": {"name": "rrf", "params": {"k": 60, "weights": {"bm25s": 1.0}}},
        "reranker": {"name": "none", "params": {}},
    }


def _agent() -> dict:
    return {"name": "simple", "params": {"top_k": 20}}


def test_agent_configs_keep_twenty_retrieval_candidates_without_gold_count_hints():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "agent_style"
    simple = yaml.safe_load((config_root / "simple.yaml").read_text(encoding="utf-8"))
    verifying = yaml.safe_load(
        (config_root / "verifying.yaml").read_text(encoding="utf-8")
    )
    iterative = yaml.safe_load(
        (config_root / "iterative.yaml").read_text(encoding="utf-8")
    )

    assert simple["params"]["top_k"] == simple["params"]["max_papers"] == 20
    assert verifying["params"]["top_k"] == verifying["params"]["max_papers"] == 20
    assert iterative["params"]["top_k"] == iterative["params"]["max_papers"] == 20
    assert iterative["params"]["sufficient_papers"] is None


def test_optional_component_module_is_imported_only_when_built(monkeypatch):
    imported: list[str] = []
    sentinel = object()

    monkeypatch.setattr(config_module.registry, "list_registered", lambda: [])
    monkeypatch.setattr(
        config_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    monkeypatch.setattr(
        config_module.registry,
        "build",
        lambda kind, name, **kwargs: sentinel,
    )

    built = _build_component("preprocessor", "pypdf", pdf_dir="/tmp/pdfs")

    assert built is sentinel
    assert imported == ["litqa.preprocess.pypdf_chunker"]


def test_optional_component_reports_the_missing_dependency(monkeypatch):
    def fail_import(name: str):
        raise ModuleNotFoundError(
            "No module named 'pypdf'", name="pypdf"
        )

    monkeypatch.setattr(config_module.registry, "list_registered", lambda: [])
    monkeypatch.setattr(config_module.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="missing dependency 'pypdf'"):
        _build_component("preprocessor", "pypdf", pdf_dir="/tmp/pdfs")


def test_compose_config_shape():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "pypdf", "params": {"max_chars_per_chunk": 2000}},
        search=_search(),
        agent=_agent(),
    )

    assert cfg["preprocessor"] == {
        "name": "pypdf",
        "params": {"max_chars_per_chunk": 2000, "pdf_dir": "/data/pdfs"},
    }
    assert cfg["retriever"]["per_index_k"] == 100
    assert cfg["retriever"]["fuser"] == _search()["fuser"]
    assert cfg["retriever"]["reranker"] == _search()["reranker"]
    assert cfg["agent"] == _agent()
    assert cfg["paths"]["chunks"] == "/data/chunks/pypdf_chunks.jsonl"


def test_index_dir_is_namespaced_by_process_name():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "pypdf", "params": {}},
        search=_search(),
        agent=_agent(),
    )

    index_dir = cfg["retriever"]["indexers"][0]["params"]["index_dir"]
    assert index_dir == "/data/index/pypdf/bm25s"


def test_index_id_separates_variants_of_the_same_indexer():
    search = _search()
    search["indexers"] = [
        {
            "name": "bm25s",
            "index_id": "bm25_first",
            "params": {},
        },
        {
            "name": "bm25s",
            "index_id": "bm25_second",
            "params": {},
        },
    ]

    cfg = compose_config(
        paths=_paths(),
        process={"name": "pypdf", "params": {}},
        search=search,
        agent=_agent(),
    )

    index_dirs = [
        item["params"]["index_dir"] for item in cfg["retriever"]["indexers"]
    ]
    assert index_dirs == [
        "/data/index/pypdf/bm25_first",
        "/data/index/pypdf/bm25_second",
    ]


@pytest.mark.parametrize("index_id", ["", ".", "..", "nested/path"])
def test_index_id_must_be_path_safe(index_id: str):
    search = _search()
    search["indexers"][0]["index_id"] = index_id

    with pytest.raises(ValueError, match="index_id"):
        compose_config(
            paths=_paths(),
            process={"name": "pypdf", "params": {}},
            search=search,
            agent=_agent(),
        )


def test_same_search_style_with_different_process_style_does_not_collide():
    pypdf_cfg = compose_config(
        paths=_paths(),
        process={"name": "pypdf", "params": {}},
        search=_search(),
        agent=_agent(),
    )
    figure_cfg = compose_config(
        paths=_paths(),
        process={"name": "figure_vlm", "params": {}},
        search=_search(),
        agent=_agent(),
    )

    pypdf_index_dir = pypdf_cfg["retriever"]["indexers"][0]["params"]["index_dir"]
    figure_index_dir = figure_cfg["retriever"]["indexers"][0]["params"]["index_dir"]

    assert pypdf_index_dir != figure_index_dir
    assert pypdf_cfg["paths"]["chunks"] != figure_cfg["paths"]["chunks"]


def test_explicit_pdf_dir_and_index_dir_override_auto_derivation():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "pypdf", "params": {"pdf_dir": "/custom/pdfs"}},
        search={
            "per_index_k": 100,
            "indexers": [{"name": "bm25s", "params": {"index_dir": "/custom/index"}}],
            "fuser": {"name": "rrf", "params": {}},
            "reranker": {"name": "none", "params": {}},
        },
        agent=_agent(),
    )

    assert cfg["preprocessor"]["params"]["pdf_dir"] == "/custom/pdfs"
    assert cfg["retriever"]["indexers"][0]["params"]["index_dir"] == "/custom/index"


def test_original_dicts_are_not_mutated():
    search = _search()
    process = {"name": "pypdf", "params": {}}

    compose_config(paths=_paths(), process=process, search=search, agent=_agent())

    assert search["indexers"][0]["params"] == {}
    assert process["params"] == {}


def test_mineru_uses_its_configured_source_path():
    paths = _paths()
    paths["mineru_root"] = "/shared/mineru"

    cfg = compose_config(
        paths=paths,
        process={
            "name": "mineru",
            "path_key": "mineru_root",
            "params": {"max_chars_per_chunk": 2000},
        },
        search=_search(),
        agent=_agent(),
    )

    assert cfg["preprocessor"]["params"] == {
        "max_chars_per_chunk": 2000,
        "mineru_root": "/shared/mineru",
    }
    assert cfg["paths"]["chunks"] == "/data/chunks/mineru_chunks.jsonl"
    assert cfg["retriever"]["indexers"][0]["params"]["index_dir"] == (
        "/data/index/mineru/bm25s"
    )


def test_mineru_requires_an_explicit_root():
    with pytest.raises(KeyError, match="mineru_root"):
        compose_config(
            paths=_paths(),
            process={"name": "mineru", "path_key": "mineru_root", "params": {}},
            search=_search(),
            agent=_agent(),
        )


def test_mineru_v2_uses_a_separate_chunk_and_index_namespace():
    paths = _paths()
    paths["mineru_root"] = "/shared/mineru"

    cfg = compose_config(
        paths=paths,
        process={
            "name": "mineru_v2",
            "source": "mineru",
            "path_key": "mineru_root",
            "params": {
                "max_chars_per_chunk": 2000,
                "content_version": "v2",
            },
        },
        search=_search(),
        agent=_agent(),
    )

    assert cfg["preprocessor"]["params"]["content_version"] == "v2"
    assert cfg["paths"]["chunks"] == "/data/chunks/mineru_v2_chunks.jsonl"
    assert cfg["retriever"]["indexers"][0]["params"]["index_dir"] == (
        "/data/index/mineru_v2/bm25s"
    )
