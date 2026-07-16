"""compose_config() のパス自動導出・衝突回避ロジックのテスト。"""

from __future__ import annotations

from litqa.config import compose_config


def _paths() -> dict:
    return {
        "pdf_dir": "/data/pdfs",
        "chunks_dir": "/data/chunks",
        "index_dir": "/data/index",
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


def test_compose_config_shape():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {"max_chars_per_chunk": 2000}},
        search=_search(),
        agent=_agent(),
    )

    assert cfg["preprocessor"] == {
        "name": "marker",
        "params": {"max_chars_per_chunk": 2000, "pdf_dir": "/data/pdfs"},
    }
    assert cfg["retriever"]["per_index_k"] == 100
    assert cfg["retriever"]["fuser"] == _search()["fuser"]
    assert cfg["retriever"]["reranker"] == _search()["reranker"]
    assert cfg["agent"] == _agent()
    assert cfg["paths"]["chunks"] == "/data/chunks/marker_chunks.jsonl"


def test_index_dir_is_namespaced_by_process_name():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {}},
        search=_search(),
        agent=_agent(),
    )

    index_dir = cfg["retriever"]["indexers"][0]["params"]["index_dir"]
    assert index_dir == "/data/index/marker/bm25s"


def test_same_search_style_with_different_process_style_does_not_collide():
    marker_cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {}},
        search=_search(),
        agent=_agent(),
    )
    figure_cfg = compose_config(
        paths=_paths(),
        process={"name": "figure_vlm", "params": {}},
        search=_search(),
        agent=_agent(),
    )

    marker_index_dir = marker_cfg["retriever"]["indexers"][0]["params"]["index_dir"]
    figure_index_dir = figure_cfg["retriever"]["indexers"][0]["params"]["index_dir"]

    assert marker_index_dir != figure_index_dir
    assert marker_cfg["paths"]["chunks"] != figure_cfg["paths"]["chunks"]


def test_explicit_pdf_dir_and_index_dir_override_auto_derivation():
    cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {"pdf_dir": "/custom/pdfs"}},
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
    process = {"name": "marker", "params": {}}

    compose_config(paths=_paths(), process=process, search=search, agent=_agent())

    assert search["indexers"][0]["params"] == {}
    assert process["params"] == {}
