"""構成そのもの（`di_pipeline.pipeline`）のテスト。

**設定を差し替える仕組みは持たないので、テストするのは「今の構成が意図どおりか」**。
索引パスの導出（取り違えると数時間のビルドを上書きする）と、各段に本当に意図した
モデル・パラメータが渡っているかを固定する。
"""

from __future__ import annotations

import yaml

from littraceqa.di_pipeline.index.faiss_qwen3 import INDEX_NAME, PRODUCTION_PARAMS
from littraceqa.di_pipeline.pipeline import (
    PROCESS,
    Paths,
    build_expander,
    build_indexers,
    build_preprocessor,
    build_retriever,
)


def _paths(tmp_path) -> Paths:
    (tmp_path / "chunks").mkdir()
    (tmp_path / f"chunks/{PROCESS}_chunks.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "meta.jsonl").write_text("", encoding="utf-8")
    config = tmp_path / "paths.yaml"
    config.write_text(
        yaml.safe_dump({
            "pdf_dir": str(tmp_path / "pdfs"),
            "chunks_dir": str(tmp_path / "chunks"),
            "index_dir": str(tmp_path / "index"),
            "paper_metadata": str(tmp_path / "meta.jsonl"),
        }),
        encoding="utf-8",
    )
    return Paths.load(config)


def test_index_paths_are_namespaced_by_the_preprocessor(tmp_path):
    """索引は前処理名の下に置く。前処理を変えて作り直しても既存索引と衝突しない。"""
    paths = _paths(tmp_path)
    assert paths.index("bm25s") == tmp_path / "index" / PROCESS / "bm25s"
    assert paths.chunks == tmp_path / "chunks" / f"{PROCESS}_chunks.jsonl"


def test_every_index_has_its_own_directory(tmp_path):
    """索引パスが重なっていないこと。**重なると先に作った索引を上書きする。**"""
    dirs = [str(ix.index_dir) for ix in build_indexers(_paths(tmp_path))]
    assert len(dirs) == len(set(dirs)), dirs


def test_embedding_index_shares_its_settings_with_the_shard_builder(tmp_path):
    """埋め込みの設定は index/faiss_qwen3.py の定数から取る。

    分散ビルド（scripts/build_faiss_qwen3_shard.py）が同じ定数を使うので、
    **構築時と検索時でモデルや前置詞がズレる事故が起きない。**
    """
    embedder = build_indexers(_paths(tmp_path))[2]
    assert embedder.model_name == PRODUCTION_PARAMS["model"]
    assert embedder.doc_prefix == PRODUCTION_PARAMS["doc_prefix"]
    assert embedder.index_dir.name == INDEX_NAME
    # 検索時は devices[0] しか使わないので1枚だけ（残りを reranker に空ける）。
    assert embedder.devices == ["cuda:0"]


def test_retriever_wiring(tmp_path):
    """検索の各段が意図した構成で組まれていること。"""
    retriever = build_retriever(_paths(tmp_path))
    assert [type(ix).__name__ for ix in retriever.indexers] == [
        "BM25Index", "BM25PaperIndex", "Qwen3FAISSIndex",
    ]
    assert type(retriever.fuser).__name__ == "PaperRRFFuser"  # 1論文1票
    assert retriever.reranker.model_name == "Qwen/Qwen3-Reranker-8B"
    # reranker を2枚に分けるのは、pool_k 件をクエリのたびに推論するため。
    assert retriever.reranker.devices == ["cuda:1", "cuda:2"]
    assert (retriever.per_index_k, retriever.pool_k) == (100, 200)
    assert retriever.seed_expansion.query_chars == 512
    assert retriever.rerank_blend.protect_top == 20
    # **上げてはいけない**（NAACL で faiss search が61倍に膨らんだ）。
    assert retriever.max_fetch_k == 3000


def test_expander_uses_three_independent_sources(tmp_path):
    """3ソースは違う gold を拾うので併用する（MLT だけが拾える gold が2本ある）。"""
    expander = build_expander(_paths(tmp_path))
    assert [type(s).__name__ for s in expander.sources] == [
        "Specter2PaperExpander", "BibCouplingExpander", "BM25MLTExpander",
    ]
    assert all(s.neighbors == 100 for s in expander.sources)


def test_preprocessor_reads_mineru_output(tmp_path):
    assert type(build_preprocessor(_paths(tmp_path))).__name__ == "MinerUChunker"
