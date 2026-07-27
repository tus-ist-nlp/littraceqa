"""compose_config() のパス自動導出・衝突回避ロジックのテスト。"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.config import compose_config, override_rerank_pool


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
    assert "retriever_wrapper" not in cfg["retriever"]
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


def test_reranker_candidate_pool_is_optional_and_preserved():
    search = _search()
    search["pool_k"] = 50
    search["attribute_weight"] = 0.2

    cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {}},
        search=search,
        agent=_agent(),
    )

    assert cfg["retriever"]["pool_k"] == 50
    assert cfg["retriever"]["attribute_weight"] == 0.2


def test_retriever_wrapper_is_opt_in_and_copied():
    search = _search()
    search["retriever_wrapper"] = {
        "name": "seed_expansion",
        "params": {
            "candidate_k": 50,
            "seed_text_chars": 512,
            "rrf_k": 60,
            "max_results": 10,
        },
    }

    cfg = compose_config(
        paths=_paths(),
        process={"name": "marker", "params": {}},
        search=search,
        agent=_agent(),
    )

    assert cfg["retriever"]["retriever_wrapper"] == search["retriever_wrapper"]
    assert cfg["retriever"]["retriever_wrapper"] is not search["retriever_wrapper"]
    assert (
        cfg["retriever"]["retriever_wrapper"]["params"]
        is not search["retriever_wrapper"]["params"]
    )


def test_wrapper_paper_embedding_index_name_is_resolved_under_process_index():
    search = _search()
    search["retriever_wrapper"] = {
        "name": "seed_expansion",
        "params": {
            "paper_embedding_index_name": "specter2_paper_embeddings",
            "method_dense_tail_weight": 0.35,
        },
    }

    cfg = compose_config(
        paths=_paths(),
        process={"name": "mineru", "params": {}},
        search=search,
        agent=_agent(),
    )

    params = cfg["retriever"]["retriever_wrapper"]["params"]
    assert "paper_embedding_index_name" not in params
    assert (
        params["paper_embedding_index_dir"]
        == "/data/index/mineru/specter2_paper_embeddings"
    )
    assert search["retriever_wrapper"]["params"][
        "paper_embedding_index_name"
    ] == "specter2_paper_embeddings"


@pytest.mark.parametrize(
    "index_name",
    ["", "../outside", "/absolute", 42],
)
def test_wrapper_paper_embedding_index_name_rejects_unsafe_values(index_name):
    search = _search()
    search["retriever_wrapper"] = {
        "name": "seed_expansion",
        "params": {"paper_embedding_index_name": index_name},
    }

    with pytest.raises(ValueError, match="paper_embedding_index_name"):
        compose_config(
            paths=_paths(),
            process={"name": "mineru", "params": {}},
            search=search,
            agent=_agent(),
        )


def test_rerank_pool_override_is_bounded_and_does_not_mutate_source():
    search = _search()
    search["reranker"] = {"name": "qwen3", "params": {}}

    updated = override_rerank_pool(search, 50)

    assert "pool_k" not in search
    assert updated["pool_k"] == 50
    with pytest.raises(ValueError, match="between 1 and 1000"):
        override_rerank_pool(search, 0)
    with pytest.raises(ValueError, match="enabled reranker"):
        override_rerank_pool(_search(), 20)


def test_rerank_pool_override_routes_to_retriever_wrapper():
    search = _search()
    search["reranker"] = {"name": "qwen3", "params": {}}
    search["retriever_wrapper"] = {
        "name": "seed_expansion",
        "params": {"candidate_k": 50, "rerank_pool_k": 20},
    }

    updated = override_rerank_pool(search, 50)

    assert updated["pool_k"] == 50
    assert updated["retriever_wrapper"]["params"]["rerank_pool_k"] == 50
    assert search["retriever_wrapper"]["params"]["rerank_pool_k"] == 20
