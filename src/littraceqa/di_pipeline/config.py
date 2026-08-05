"""configs/{paths,process_style,search_style,agent_style}/*.yaml を読み込み、
registry.build() でパイプラインの各段を組み立てるモジュール。

前処理・検索手法・エージェント・共有パスは、それぞれ独立したyamlファイルとして
以下の4フォルダに分けて置く。実行時にこの4つから1ファイルずつ選んで
compose_config() で組み合わせる（詳細は CLAUDE.md 参照）。

    configs/paths/default.yaml:
      pdf_dir: /data2/littraceqa/pdfs
      chunks_dir: /data2/littraceqa/chunks
      index_dir: /data2/littraceqa/index
      paper_metadata: data/paper_metadata.jsonl

    configs/process_style/marker.yaml:
      name: marker
      params: { force_ocr: true, use_llm: false }

    configs/search_style/bm25_qwen3.yaml:
      per_index_k: 100
      indexers:
        - { name: bm25s, params: {} }
        - { name: faiss_qwen3, params: { model: ... } }
      fuser:
        name: rrf
        params: { k: 60, weights: { bm25s: 1.0, faiss_qwen3: 1.0 } }
      reranker:
        name: none
        params: {}

    configs/agent_style/reading.yaml:
      name: reading
      llm: { name: azure_openai, params: { reasoning_effort: medium } }
      params: { top_k: 20, max_steps: 3 }

    # agent.llm で構築するクライアントを指定する（LLM を使わないエージェントなら省略可）。
    # yaml のファイル名は任意のラベルで、実際に組み立てるクラスは name フィールド
    # （registry に @register("agent", name) されたキー）で決まる。両者が一致する
    # 必要はない。

process_style / search_style の各ファイルには pdf_dir / index_dir を書かない。
compose_config() が paths から自動導出する（同じ search_style を別の
process_style と組み合わせても索引パスが衝突しないようにするため）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from littraceqa.di_pipeline import registry

# APIキー等はリポジトリ直下の .env から読む（コードにも yaml にも書かない）。
# 既に export されている環境変数は上書きしない。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from littraceqa.di_pipeline.agent.iterative import IterativeAgent  # noqa: F401
from littraceqa.di_pipeline.agent.reading import ReadingAgent  # noqa: F401
from littraceqa.di_pipeline.agent.simple import SimpleAgent  # noqa: F401
from littraceqa.di_pipeline.agent.verifying import VerifyingAgent  # noqa: F401
from littraceqa.di_pipeline.index.bm25_index import BM25Index  # noqa: F401
from littraceqa.di_pipeline.index.colbert_index import ColBERTIndex  # noqa: F401
from littraceqa.di_pipeline.index.faiss_qwen3 import Qwen3FAISSIndex  # noqa: F401
from littraceqa.di_pipeline.index.faiss_specter2 import Specter2FAISSIndex  # noqa: F401
from littraceqa.di_pipeline.index.siglip_image import SiglipImageIndex  # noqa: F401
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM  # noqa: F401
from littraceqa.di_pipeline.llm.fake import FakeLLM  # noqa: F401
from littraceqa.di_pipeline.llm.openai_compatible import (  # noqa: F401
    OpenAICompatibleLLM,
)
from littraceqa.di_pipeline.preprocess.figure_vlm import FigureVLMChunker  # noqa: F401
from littraceqa.di_pipeline.preprocess.marker_chunker import MarkerChunker  # noqa: F401
from littraceqa.di_pipeline.preprocess.mineru_chunker import MinerUChunker  # noqa: F401
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever
from littraceqa.di_pipeline.retrieve.reranker import NoneReranker  # noqa: F401
from littraceqa.di_pipeline.retrieve.rrf import RRFFuser  # noqa: F401


def load_config(path: str | Path) -> dict:
    """yaml ファイルを読み込み、dict をそのまま返す。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compose_config(paths: dict, process: dict, search: dict, agent: dict) -> dict:
    """paths/process_style/search_style/agent_style の4dictから、

    build_pipeline() がそのまま扱える {paths, preprocessor, retriever, agent}
    形のcfgを組み立てる。pdf_dir / index_dir / chunks は process の名前を
    キーにして paths から自動導出し、同じ search_style を別の process_style
    と組み合わせても索引パスが衝突しないようにする（明示指定があれば優先する）。
    """
    process_name = process["name"]

    preprocessor_params = dict(process.get("params", {}))
    preprocessor_params.setdefault("pdf_dir", paths["pdf_dir"])

    indexers = []
    for indexer in search["indexers"]:
        indexer_params = dict(indexer.get("params", {}))
        # 同じ indexer の別バリアント（モデル違い・chunk_types 違い）を共存させるための
        # 索引ディレクトリ名。省略すると indexer 名がそのまま使われる。
        # 例: faiss_specter2 を「全チャンク版」と「abstractのみ版」で並べたいとき、
        # index_name を分けないと同じパスを奪い合って上書きしてしまう。
        # search_style に絶対パスを書かない方針は保ったまま、末尾だけを変える。
        index_name = indexer.get("index_name", indexer["name"])
        indexer_params.setdefault(
            "index_dir", f"{paths['index_dir']}/{process_name}/{index_name}"
        )
        indexers.append({"name": indexer["name"], "params": indexer_params})

    resolved_paths = dict(paths)
    resolved_paths["chunks"] = f"{paths['chunks_dir']}/{process_name}_chunks.jsonl"

    return {
        "paths": resolved_paths,
        "preprocessor": {"name": process_name, "params": preprocessor_params},
        "retriever": {
            "per_index_k": search["per_index_k"],
            "indexers": indexers,
            "fuser": search["fuser"],
            "reranker": search["reranker"],
        },
        "agent": agent,
    }


def build_pipeline(cfg: dict) -> tuple[Any, HybridRetriever, Any]:
    """cfg から preprocessor, retriever, agent のインスタンスを組み立てる。"""
    indexers = [
        registry.build("indexer", ix["name"], **ix.get("params", {}))
        for ix in cfg["retriever"]["indexers"]
    ]
    fuser = registry.build(
        "fuser",
        cfg["retriever"]["fuser"]["name"],
        **cfg["retriever"]["fuser"].get("params", {}),
    )

    reranker_cfg = cfg["retriever"].get("reranker")
    reranker = None
    if reranker_cfg:
        reranker = registry.build(
            "reranker", reranker_cfg["name"], **reranker_cfg.get("params", {})
        )

    retriever = HybridRetriever(
        indexers=indexers,
        fuser=fuser,
        reranker=reranker,
        per_index_k=cfg["retriever"]["per_index_k"],
    )

    agent_cfg = cfg["agent"]
    llm_kwargs: dict[str, Any] = {}
    llm_cfg = agent_cfg.get("llm")
    if llm_cfg:
        llm_kwargs["llm"] = registry.build("llm", llm_cfg["name"], **llm_cfg.get("params", {}))

    agent = registry.build(
        "agent",
        agent_cfg["name"],
        retriever=retriever,
        **llm_kwargs,
        **agent_cfg.get("params", {}),
    )

    preprocessor = None
    preprocessor_cfg = cfg.get("preprocessor")
    if preprocessor_cfg:
        preprocessor = registry.build(
            "preprocessor", preprocessor_cfg["name"], **preprocessor_cfg.get("params", {})
        )

    return preprocessor, retriever, agent
