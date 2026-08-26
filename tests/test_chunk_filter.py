"""chunk_types フィルタと index_name による索引パス分離のテスト。

この2つが無いと、モデルを設計どおりの粒度で使う構成が組めない:
* chunk_types: SPECTER2 は title+abstract で学習された論文単位モデルなので、
  本文チャンクを個別に埋め込むのは設計外。abstract だけに絞れるようにする。
* index_name: 同じ indexer の別バリアント（全チャンク版 / abstract版）を並べたとき、
  索引ディレクトリを分けないと互いを上書きしてしまう。
"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.config import compose_config
from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.chunk_filter import ALL_CHUNK_TYPES, filter_chunk_types


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"p0#c{i}", paper_id="p0", text=f"t{i}", chunk_type=t, metadata={})
        for i, t in enumerate(
            ["title_abstract", "text_span", "text_span", "table", "figure"]
        )
    ]


def test_none_keeps_every_chunk():
    assert len(filter_chunk_types(_chunks(), None)) == 5
    assert len(filter_chunk_types(_chunks(), [])) == 5


def test_filters_to_the_requested_types():
    kept = filter_chunk_types(_chunks(), ["title_abstract"])
    assert [c.chunk_type for c in kept] == ["title_abstract"]

    body = filter_chunk_types(_chunks(), ["text_span", "table", "figure"])
    assert [c.chunk_type for c in body] == ["text_span", "text_span", "table", "figure"]


def test_typo_in_chunk_types_is_rejected_loudly():
    """静かに0件の索引を作るより、その場で落ちた方がよい。"""
    with pytest.raises(ValueError, match="unknown chunk_types"):
        filter_chunk_types(_chunks(), ["titel_abstract"])


def test_all_chunk_types_matches_what_the_chunkers_emit():
    assert "title_abstract" in ALL_CHUNK_TYPES
    assert "text_span" in ALL_CHUNK_TYPES
    assert "table" in ALL_CHUNK_TYPES


def _compose(indexers: list[dict]) -> dict:
    return compose_config(
        paths={
            "pdf_dir": "/d/pdfs",
            "chunks_dir": "/d/chunks",
            "index_dir": "/d/index",
            "paper_metadata": "m.jsonl",
        },
        process={"name": "mineru", "params": {}},
        search={
            "per_index_k": 100,
            "indexers": indexers,
            "fuser": {"name": "paper_rrf", "params": {}},
        },
        agent={"name": "simple", "params": {}},
    )


def test_index_name_defaults_to_the_indexer_name():
    cfg = _compose([{"name": "faiss_specter2", "params": {}}])
    assert cfg["retriever"]["indexers"][0]["params"]["index_dir"] == (
        "/d/index/mineru/faiss_specter2"
    )


def test_index_name_separates_variants_of_the_same_indexer():
    """abstract版が全チャンク版の索引を上書きしないこと。"""
    cfg = _compose(
        [
            {"name": "faiss_specter2", "params": {}},
            {
                "name": "faiss_specter2",
                "index_name": "faiss_specter2_abstract",
                "params": {"chunk_types": ["title_abstract"]},
            },
        ]
    )
    dirs = [ix["params"]["index_dir"] for ix in cfg["retriever"]["indexers"]]
    assert dirs == [
        "/d/index/mineru/faiss_specter2",
        "/d/index/mineru/faiss_specter2_abstract",
    ]
    assert len(set(dirs)) == 2, "索引パスが衝突している"
