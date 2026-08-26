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

from littraceqa.chunk_store import ChunkStore
from littraceqa.common import ROOT
from littraceqa.di_pipeline import registry

# APIキー等はリポジトリ直下の .env から読む（コードにも yaml にも書かない）。
# 既に export されている環境変数は上書きしない。
load_dotenv(ROOT / ".env")
from littraceqa.di_pipeline.agent.reading import CombineConfig, ReadingAgent  # noqa: F401
from littraceqa.di_pipeline.index.bm25_index import BM25Index  # noqa: F401
from littraceqa.di_pipeline.index.bm25_paper_index import BM25PaperIndex  # noqa: F401
from littraceqa.di_pipeline.index.faiss_qwen3 import Qwen3FAISSIndex  # noqa: F401
from littraceqa.di_pipeline.index.faiss_specter2 import Specter2FAISSIndex  # noqa: F401
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM  # noqa: F401
from littraceqa.di_pipeline.llm.fake import FakeLLM  # noqa: F401
from littraceqa.di_pipeline.preprocess.mineru_chunker import MinerUChunker  # noqa: F401
from littraceqa.di_pipeline.retrieve.attribute_filter import (
    AttributeExtractor,
    LLMAttributeExtractor,
)
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever
from littraceqa.di_pipeline.retrieve.paper_expander import (  # noqa: F401
    BibCouplingExpander,
    BM25MLTExpander,
    FusedPaperExpander,
    Specter2PaperExpander,
)
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser  # noqa: F401
from littraceqa.di_pipeline.retrieve.reranker import NoneReranker  # noqa: F401
from littraceqa.di_pipeline.retrieve.reranker import Qwen3Reranker  # noqa: F401
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

    # agent の expansion（論文→論文展開）は索引の**名前**だけを yaml に書き、
    # 実際のパスは indexers と同じ規則で paths から導出する
    # （agent_style に絶対パスを書かない方針を保つ）。
    #
    # 単一ソース（旧形式）と複数ソース（sources: [...]、RRF融合）の両方を受ける。
    agent_cfg = dict(agent)
    expansion = agent_cfg.get("expansion")
    if expansion:
        expansion = dict(expansion)
        sources = expansion.pop("sources", None)
        if sources is None:
            sources = [
                {"name": "specter2", "index_name": expansion.pop("index_name", None)}
            ]
        resolved_sources = []
        for source in sources:
            source = dict(source)
            name = source.pop("name", "specter2")
            params = dict(source.pop("params", {}))
            params.update(source)  # name 以外はそのまま params 扱い
            if name == "specter2":
                index_name = params.pop("index_name", None) or "faiss_specter2_abstract"
                params.setdefault(
                    "index_dir", f"{paths['index_dir']}/{process_name}/{index_name}"
                )
            elif name == "bib_coupling":
                index_name = params.pop("index_name", None) or "bib_coupling"
                params.setdefault("chunks", resolved_paths["chunks"])
                params.setdefault(
                    "cache_path",
                    f"{paths['index_dir']}/{process_name}/{index_name}/refs.pkl",
                )
            elif name == "bm25_mlt":
                # 構築済みの bm25s_paper 索引をそのまま読む（追加構築なし）。
                # 行番号 -> paper_id と anchor 用 title+abstract のキャッシュだけ
                # 別ディレクトリに置く（索引ディレクトリを汚さないため）。
                index_name = params.pop("index_name", None) or "bm25s_paper"
                params.setdefault(
                    "index_dir", f"{paths['index_dir']}/{process_name}/{index_name}"
                )
                params.setdefault(
                    "cache_path",
                    f"{paths['index_dir']}/{process_name}/bm25_mlt/anchor_text.pkl",
                )
            resolved_sources.append({"name": name, "params": params})
        expansion["sources"] = resolved_sources
        agent_cfg["expansion"] = expansion

    return {
        "paths": resolved_paths,
        "preprocessor": {"name": process_name, "params": preprocessor_params},
        "retriever": {
            "per_index_k": search["per_index_k"],
            # reranker に渡す候補プール件数。search_style 側で省略可 (既定は top_k*3)。
            "pool_k": search.get("pool_k"),
            "indexers": indexers,
            "fuser": search["fuser"],
            "reranker": search["reranker"],
            # 質問が明示した会議名・年で絞り込む設定。search_style 側で省略可
            # （省略すると無効で、従来どおりのコードパスを通る）。
            "attribute_filter": search.get("attribute_filter"),
            # reranker の順位を融合前の順位と RRF で混ぜる設定。省略すると
            # 従来どおり reranker が順位を完全に置き換える。
            "rerank_blend": search.get("rerank_blend"),
            # 1位論文の語彙を質問に足して引き直す設定（pseudo relevance feedback）。
            # 省略 / enabled: false なら検索は従来どおり1回だけ。
            "seed_expansion": search.get("seed_expansion"),
        },
        "agent": agent_cfg,
    }


# expansion ブロックのうち、A/B 統合（ReadingAgent 側）が使うキー。
# 残りは expander 自身のパラメータ（neighbors / rrf_k / ソース固有）。
_COMBINE_KEYS = {
    "anchors": "anchors",
    "combine_rrf_k": "rrf_k",
    "related_weight": "related_weight",
    "related_offset": "related_offset",
    "anchor_from": "anchor_from",
}


def build_paper_expander(agent_cfg: dict) -> tuple[Any, CombineConfig]:
    """agent の expansion ブロックを、展開器と A/B 統合の設定に振り分ける。

    **どちらが何を使うかをここで1回だけ決める。** 展開器は「渡された起点の近傍を
    返す」だけ、統合の重みや起点の決め方は ReadingAgent の持ち物なので、
    yaml の1ブロックを2つの宛先に配る場所が要る。

    ソースが1つならそのまま、複数なら RRF 融合する FusedPaperExpander で包む。
    """
    expansion_cfg = dict(agent_cfg.get("expansion") or {})
    if not expansion_cfg:
        return None, CombineConfig()

    source_cfgs = expansion_cfg.pop("sources", [])
    combine = CombineConfig(
        **{
            field: expansion_cfg.pop(key)
            for key, field in _COMBINE_KEYS.items()
            if key in expansion_cfg
        }
    )
    # 残りは展開器のもの。未知のキーは黙って捨てず名前を挙げる
    # （綴り間違いが既定値のまま走ると、効いていない設定で実験してしまう）。
    unknown = sorted(set(expansion_cfg) - {"neighbors", "rrf_k"})
    if unknown:
        raise ValueError(
            f"unknown expansion keys: {unknown}. "
            f"valid: {sorted(set(_COMBINE_KEYS) | {'sources', 'neighbors', 'rrf_k'})}"
        )
    neighbors = expansion_cfg.get("neighbors")
    shared = {} if neighbors is None else {"neighbors": neighbors}

    sources = [
        registry.build("expander", s["name"], **{**shared, **s.get("params", {})})
        for s in source_cfgs
    ]
    if len(sources) == 1:
        return sources[0], combine
    fused = FusedPaperExpander(
        sources=sources,
        **shared,
        **{k: v for k, v in expansion_cfg.items() if k == "rrf_k"},
    )
    return fused, combine


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

    agent_cfg = cfg["agent"]
    llm_kwargs: dict[str, Any] = {}
    llm_cfg = agent_cfg.get("llm")
    if llm_cfg:
        llm_kwargs["llm"] = registry.build("llm", llm_cfg["name"], **llm_cfg.get("params", {}))

    # 論文→論文展開（agent yaml の expansion ブロック）。無ければ候補列は検索の順位のまま。
    expander, combine = build_paper_expander(agent_cfg)
    if expander is not None:
        llm_kwargs["paper_expander"] = expander
        llm_kwargs["combine"] = combine

    # 属性フィルタ（会議名・年）。設定が無い / enabled: false なら無効のまま。
    attribute_cfg = cfg["retriever"].get("attribute_filter") or {}
    attribute_kwargs: dict[str, Any] = {}
    if attribute_cfg.get("enabled"):
        extractor: Any = AttributeExtractor(cfg["paths"]["paper_metadata"])
        # llm_extract: true のとき、正規表現が取れなかった質問だけ LLM に判定させる
        # （エージェントと同じ LLM を使い回す）。LLM が無い構成では黙って正規表現のまま。
        if attribute_cfg.get("llm_extract") and "llm" in llm_kwargs:
            extractor = LLMAttributeExtractor(extractor, llm_kwargs["llm"])
        attribute_kwargs = {
            "attribute_extractor": extractor,
            "fetch_safety": attribute_cfg.get("safety", 1.5),
            "max_fetch_k": attribute_cfg.get("max_fetch_k", 5000),
            "min_filtered_results": attribute_cfg.get("min_results", 10),
        }

    # Seed Expansion（1位論文の語彙を質問に足して引き直す）。書かなければ無効で、
    # anchor の本文を引く ChunkStore も作らない（起動のたびに 3.8GB の索引を
    # 触らせないため）。
    seed_kwargs: dict[str, Any] = {}
    seed_cfg = cfg["retriever"].get("seed_expansion") or {}
    if seed_cfg.get("enabled"):
        seed_kwargs = {
            "seed_expansion": seed_cfg,
            "anchor_store": ChunkStore(cfg["paths"]["chunks"]),
        }

    retriever = HybridRetriever(
        indexers=indexers,
        fuser=fuser,
        reranker=reranker,
        per_index_k=cfg["retriever"]["per_index_k"],
        pool_k=cfg["retriever"].get("pool_k"),
        rerank_blend=cfg["retriever"].get("rerank_blend"),
        **seed_kwargs,
        **attribute_kwargs,
    )

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
