"""configs/{paths,process_style,search_style,agent_style}/*.yaml を読み込み、
registry.build() でパイプラインの各段を組み立てるモジュール。

前処理・検索手法・エージェント・共有パスは、それぞれ独立したyamlファイルとして
以下の4フォルダに分けて置く。実行時にこの4つから1ファイルずつ選んで
compose_config() で組み合わせる（詳細は CLAUDE.md 参照）。

    configs/paths/default.yaml:
      pdf_dir: /data2/littraceqa/pdfs
      docint_chunks: /data2/littraceqa/docint_chunks
      chunks_dir: /data2/littraceqa/chunks
      index_dir: /data2/littraceqa/index
      backup_dir: /data2/littraceqa/backup
      paper_metadata: data/paper_metadata.jsonl

    configs/process_style/pypdf.yaml:
      name: pypdf
      params: { max_chars_per_chunk: 2000 }

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

    configs/agent_style/simple.yaml:
      name: simple
      params: { top_k: 20 }

    # LLM を使うエージェント（例: iterative）は agent.llm で構築するクライアントを指定する:
    #   name: iterative
    #   llm: { name: fake, params: {} }
    #   params: { top_k: 20, max_steps: 3 }

process_style / search_style の各ファイルには pdf_dir / index_dir を書かない。
compose_config() が paths から自動導出する（同じ search_style を別の
process_style と組み合わせても索引パスが衝突しないようにするため）。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from litqa import registry
from litqa.agent.iterative import IterativeAgent  # noqa: F401
from litqa.agent.simple import SimpleAgent  # noqa: F401
from litqa.agent.verifying import VerifyingAgent  # noqa: F401
from litqa.index.bm25_index import BM25Index  # noqa: F401
from litqa.index.bge_m3_index import BGEM3NumpyIndex  # noqa: F401
from litqa.index.paper_bm25 import PaperBM25Index  # noqa: F401
from litqa.llm.fake import FakeLLM  # noqa: F401
from litqa.preprocess.figure_vlm import FigureVLMChunker  # noqa: F401
from litqa.preprocess.mineru_chunker import MinerUChunker  # noqa: F401
from litqa.retrieve.hybrid import HybridRetriever
from litqa.retrieve.paper_rank_rrf import PaperRankRRFFuser  # noqa: F401
from litqa.retrieve.reranker import NoneReranker  # noqa: F401
from litqa.retrieve.rrf import RRFFuser  # noqa: F401


_LAZY_COMPONENT_MODULES = {
    ("indexer", "colbert"): "litqa.index.colbert_index",
    ("indexer", "faiss_qwen3"): "litqa.index.faiss_qwen3",
    ("indexer", "faiss_specter2"): "litqa.index.faiss_specter2",
    ("indexer", "siglip_image"): "litqa.index.siglip_image",
    ("preprocessor", "marker"): "litqa.preprocess.marker_chunker",
    ("preprocessor", "pypdf"): "litqa.preprocess.pypdf_chunker",
}


def _build_component(kind: str, name: str, **kwargs: Any) -> Any:
    """Build one component, importing optional backends only when selected."""
    key = (kind, name)
    module_name = _LAZY_COMPONENT_MODULES.get(key)
    if module_name is not None and key not in set(registry.list_registered()):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            dependency = exc.name or "an optional dependency"
            raise RuntimeError(
                f"cannot build {kind}:{name}; missing dependency {dependency!r}"
            ) from exc
    return registry.build(kind, name, **kwargs)


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
    path_key = process.get("path_key", "pdf_dir")
    if path_key not in paths and path_key not in preprocessor_params:
        raise KeyError(
            f"process '{process_name}' requires path '{path_key}' in paths or params"
        )
    if path_key in paths:
        preprocessor_params.setdefault(path_key, paths[path_key])

    indexers = []
    for indexer in search["indexers"]:
        indexer_params = dict(indexer.get("params", {}))
        index_id = indexer.get("index_id", indexer["name"])
        if (
            not isinstance(index_id, str)
            or not index_id
            or Path(index_id).name != index_id
            or index_id in {".", ".."}
        ):
            raise ValueError("index_id must be a non-empty path-safe name")
        indexer_params.setdefault(
            "index_dir", f"{paths['index_dir']}/{process_name}/{index_id}"
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
        _build_component("indexer", ix["name"], **ix.get("params", {}))
        for ix in cfg["retriever"]["indexers"]
    ]
    fuser = _build_component(
        "fuser",
        cfg["retriever"]["fuser"]["name"],
        **cfg["retriever"]["fuser"].get("params", {}),
    )

    reranker_cfg = cfg["retriever"].get("reranker")
    reranker = None
    if reranker_cfg:
        reranker = _build_component(
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
        llm_kwargs["llm"] = _build_component(
            "llm", llm_cfg["name"], **llm_cfg.get("params", {})
        )

    agent = _build_component(
        "agent",
        agent_cfg["name"],
        retriever=retriever,
        **llm_kwargs,
        **agent_cfg.get("params", {}),
    )

    preprocessor = None
    preprocessor_cfg = cfg.get("preprocessor")
    if preprocessor_cfg:
        preprocessor = _build_component(
            "preprocessor", preprocessor_cfg["name"], **preprocessor_cfg.get("params", {})
        )

    return preprocessor, retriever, agent
