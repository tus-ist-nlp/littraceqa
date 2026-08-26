"""compose_config() のパス自動導出・衝突回避ロジックのテスト。"""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.config import build_pipeline, compose_config
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.attribute_filter import (
    AttributeExtractor,
    LLMAttributeExtractor,
)


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
    # reranker は省略可（`_search()` は書いていない）。無ければ None が入る。
    assert cfg["retriever"]["reranker"] is None
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
            "fuser": {"name": "paper_rrf", "params": {}},
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


# ---- build_pipeline の属性フィルタ配線 -------------------------------------


def _pipeline_cfg(tmp_path, attribute_filter: dict) -> dict:
    """build_pipeline に渡せる最小の cfg。索引はスタブに差し替える。"""
    metadata = tmp_path / "paper_metadata.jsonl"
    metadata.write_text(
        json.dumps({"paper_id": "naacl2025_000", "venue": "NAACL", "year": 2025}) + "\n",
        encoding="utf-8",
    )
    return {
        "paths": {"paper_metadata": str(metadata)},
        "retriever": {
            "per_index_k": 10,
            "pool_k": None,
            "indexers": [{"name": "_stub_for_test", "params": {}}],
            "fuser": {"name": "paper_rrf", "params": {}},
            "reranker": None,
            "attribute_filter": attribute_filter,
        },
        "agent": {
            "name": "reading",
            "llm": {"name": "fake", "params": {}},
            "params": {"retrieve_top_k": 5},
        },
    }


@pytest.fixture(autouse=True, scope="module")
def _stub_indexer():
    @register("indexer", "_stub_for_test")
    class _StubIndexer:
        name = "_stub_for_test"

        def search(self, query: str, top_k: int) -> list:
            return []

    return _StubIndexer


def test_llm_extract_wraps_the_regex_extractor(tmp_path):
    """llm_extract: true で LLM 抽出器が組み立てられること。"""
    _, retriever, _ = build_pipeline(
        _pipeline_cfg(tmp_path, {"enabled": True, "llm_extract": True})
    )
    assert isinstance(retriever.attribute_extractor, LLMAttributeExtractor)


def test_llm_extract_defaults_to_regex_only(tmp_path):
    """既定（llm_extract を書かない）では従来どおり正規表現のままであること。"""
    _, retriever, _ = build_pipeline(_pipeline_cfg(tmp_path, {"enabled": True}))
    assert isinstance(retriever.attribute_extractor, AttributeExtractor)
    assert not isinstance(retriever.attribute_extractor, LLMAttributeExtractor)


def test_attribute_filter_disabled_leaves_no_extractor(tmp_path):
    _, retriever, _ = build_pipeline(_pipeline_cfg(tmp_path, {"enabled": False}))
    assert retriever.attribute_extractor is None


def test_config_label_prefixes_subfolder_configs() -> None:
    """agent_style のサブフォルダに置いた config は、フォルダ名込みのラベルになる。

    stem だけだと reading_loop/rrf.yaml と reading_expand_rrf/rrf.yaml が
    どちらも "rrf" になり、report/*.md の名前と実験セレクタで区別できなくなる。
    """
    from littraceqa.common import config_label

    # configs/{kind}/ 直下は従来どおり stem のまま
    assert config_label("configs/agent_style/reading.yaml") == "reading"
    assert config_label("configs/paths/default.yaml") == "default"
    assert config_label("configs/search_style/bm25.yaml") == "bm25"

    # サブフォルダはフォルダ名を前に付ける
    assert config_label("configs/agent_style/reading_loop/rrf.yaml") == "reading_loop_rrf"
    assert config_label("configs/agent_style/reading_normal/fat.yaml") == "reading_normal_fat"
    assert (
        config_label("configs/agent_style/reading_expand_insert/fused.yaml")
        == "reading_expand_insert_fused"
    )

    # フォルダ名の末尾の語と同じ stem は重ねない
    # （フォルダ分けする前のファイル名と同じラベルになる）
    assert config_label("configs/agent_style/reading_expand_rrf/rrf.yaml") == "reading_expand_rrf"
    assert (
        config_label("configs/agent_style/reading_expand_insert/insert.yaml")
        == "reading_expand_insert"
    )
    assert (
        config_label("configs/agent_style/reading_expand_rrf/cand50.yaml")
        == "reading_expand_rrf_cand50"
    )

    # search_style も同じ規則。フォルダ名は畳む前のファイル名の接頭辞そのままなので、
    # ラベルは畳む前と一致する（過去の report/*.md と並べて読める）。
    assert (
        config_label("configs/search_style/bm25_colbert/colbert.yaml") == "bm25_colbert"
    )
    assert (
        config_label("configs/search_style/bm25_colbert/gte_modern.yaml")
        == "bm25_colbert_gte_modern"
    )
    assert (
        config_label("configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/8b.yaml")
        == "bm25_qwen3_8b_rerank_qwen3_8b"
    )
    assert (
        config_label(
            "configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml"
        )
        == "bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100"
    )
    assert (
        config_label("configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml")
        == "bm25_specter2_body_qwen3"
    )


@pytest.mark.parametrize("kind", ["agent_style", "search_style"])
def test_config_label_covers_every_config_file(kind: str) -> None:
    """実在する全ファイルでラベルが一意になる（衝突すると実験どうしを比較できない）。"""
    from pathlib import Path

    from littraceqa.common import config_label

    paths = sorted(Path("configs", kind).rglob("*.yaml"))
    assert paths, f"{kind} に yaml が1枚も無い"
    labels = [config_label(p) for p in paths]
    assert len(set(labels)) == len(labels), sorted(labels)
