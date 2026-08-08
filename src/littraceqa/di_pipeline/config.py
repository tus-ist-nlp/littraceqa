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

import importlib
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.retrieve.base import Retriever
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever


# Each built-in implementation registers itself when its module is imported.
# Lazy placeholders retain the registry's public keys while avoiding optional
# dependencies for components that are not selected by the composed config.
_BUILTIN_COMPONENTS: dict[tuple[str, str], tuple[str, str]] = {
    ("agent", "iterative"): (
        "littraceqa.di_pipeline.agent.iterative",
        "IterativeAgent",
    ),
    ("agent", "reading"): ("littraceqa.di_pipeline.agent.reading", "ReadingAgent"),
    ("agent", "simple"): ("littraceqa.di_pipeline.agent.simple", "SimpleAgent"),
    ("agent", "verifying"): (
        "littraceqa.di_pipeline.agent.verifying",
        "VerifyingAgent",
    ),
    ("indexer", "bm25s"): ("littraceqa.di_pipeline.index.bm25_index", "BM25Index"),
    ("indexer", "paper_bm25"): (
        "littraceqa.di_pipeline.index.paper_bm25",
        "PaperBM25Index",
    ),
    ("indexer", "colbert"): (
        "littraceqa.di_pipeline.index.colbert_index",
        "ColBERTIndex",
    ),
    ("indexer", "faiss_qwen3"): (
        "littraceqa.di_pipeline.index.faiss_qwen3",
        "Qwen3FAISSIndex",
    ),
    ("indexer", "faiss_specter2"): (
        "littraceqa.di_pipeline.index.faiss_specter2",
        "Specter2FAISSIndex",
    ),
    ("indexer", "siglip_image"): (
        "littraceqa.di_pipeline.index.siglip_image",
        "SiglipImageIndex",
    ),
    ("llm", "azure_openai"): (
        "littraceqa.di_pipeline.llm.azure_openai",
        "AzureOpenAILLM",
    ),
    ("llm", "fake"): ("littraceqa.di_pipeline.llm.fake", "FakeLLM"),
    ("preprocessor", "figure_vlm"): (
        "littraceqa.di_pipeline.preprocess.figure_vlm",
        "FigureVLMChunker",
    ),
    ("preprocessor", "marker"): (
        "littraceqa.di_pipeline.preprocess.marker_chunker",
        "MarkerChunker",
    ),
    ("preprocessor", "mineru"): (
        "littraceqa.di_pipeline.preprocess.mineru_chunker",
        "MinerUChunker",
    ),
    ("fuser", "rrf"): ("littraceqa.di_pipeline.retrieve.rrf", "RRFFuser"),
    ("fuser", "paper_rank_rrf"): (
        "littraceqa.di_pipeline.retrieve.paper_rank_rrf",
        "PaperRankRRFFuser",
    ),
    ("reranker", "none"): (
        "littraceqa.di_pipeline.retrieve.reranker",
        "NoneReranker",
    ),
    ("reranker", "qwen3"): (
        "littraceqa.di_pipeline.retrieve.qwen3_reranker",
        "Qwen3Reranker",
    ),
    ("retriever_wrapper", "seed_expansion"): (
        "littraceqa.di_pipeline.retrieve.seed_expansion",
        "SeedExpansionRetriever",
    ),
}


def _load_project_dotenv() -> None:
    """Load the project dotenv file when LLM construction actually needs it."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        if exc.name != "dotenv":
            raise
        return

    # Keep exported environment variables authoritative over local dotenv values.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _lazy_component_class(
    kind: str,
    module_name: str,
    class_name: str,
) -> type[Any]:
    """Create a registry-compatible placeholder that imports on construction."""

    class LazyComponent:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            if kind == "llm":
                _load_project_dotenv()
            module = importlib.import_module(module_name)
            implementation = getattr(module, class_name)
            return implementation(*args, **kwargs)

    LazyComponent.__name__ = class_name
    LazyComponent.__qualname__ = class_name
    LazyComponent.__module__ = module_name
    return LazyComponent


def _register_lazy_builtins() -> None:
    """Expose every built-in registry key without importing its implementation."""
    registered = set(registry.list_registered())
    for (kind, name), (module_name, class_name) in _BUILTIN_COMPONENTS.items():
        if (kind, name) in registered:
            continue
        placeholder = _lazy_component_class(kind, module_name, class_name)
        registry.register(kind, name)(placeholder)


_register_lazy_builtins()


def load_config(path: str | Path) -> dict:
    """Load a YAML file and return its mapping unchanged."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_rerank_pool(search: dict, pool_k: int | None) -> dict:
    """Return a copied search config with a bounded reranker candidate pool."""
    if pool_k is None:
        return search
    if not 1 <= pool_k <= 1000:
        raise ValueError("reranker pool size must be between 1 and 1000")
    if (search.get("reranker") or {}).get("name", "none") == "none":
        raise ValueError("reranker pool size requires an enabled reranker")
    updated = dict(search)
    updated["pool_k"] = pool_k
    wrapper = search.get("retriever_wrapper")
    if wrapper:
        wrapper_copy = dict(wrapper)
        wrapper_params = dict(wrapper_copy.get("params", {}))
        wrapper_params["rerank_pool_k"] = pool_k
        wrapper_copy["params"] = wrapper_params
        updated["retriever_wrapper"] = wrapper_copy
    return updated


def compose_config(
    paths: dict,
    process: dict,
    search: dict,
    agent: dict,
    select: dict | None = None,
) -> dict:
    """paths/process_style/search_style/agent_style/select_style から、

    build_pipeline() がそのまま扱える {paths, preprocessor, retriever, agent}
    形のcfgを組み立てる。pdf_dir / index_dir / chunks は process の名前を
    キーにして paths から自動導出し、同じ search_style を別の process_style
    と組み合わせても索引パスが衝突しないようにする（明示指定があれば優先する）。

    select_style は提出する論文集合の決め方で、省略できる。省略すると agent が
    自前の打ち切り（paper_cutoff）をそのまま使う。指定すると agent の params へ
    paper_selector として畳み込まれるので、agent_style を3種類に複製せずに
    提出方法だけを差し替えられる。
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

    retriever = {
        "per_index_k": search["per_index_k"],
        "indexers": indexers,
        "fuser": search["fuser"],
        "reranker": search["reranker"],
    }
    if "pool_k" in search:
        retriever["pool_k"] = search["pool_k"]
    if "attribute_weight" in search:
        retriever["attribute_weight"] = search["attribute_weight"]
    if "retriever_wrapper" in search:
        wrapper = search["retriever_wrapper"]
        wrapper_params = dict(wrapper.get("params", {}))
        paper_embedding_index_name = wrapper_params.pop(
            "paper_embedding_index_name",
            None,
        )
        if paper_embedding_index_name is not None:
            if (
                not isinstance(paper_embedding_index_name, str)
                or not paper_embedding_index_name.strip()
                or Path(paper_embedding_index_name).name
                != paper_embedding_index_name
            ):
                raise ValueError(
                    "paper_embedding_index_name must be a non-empty directory name"
                )
            wrapper_params.setdefault(
                "paper_embedding_index_dir",
                (
                    f"{paths['index_dir']}/{process_name}/"
                    f"{paper_embedding_index_name}"
                ),
            )
        if wrapper_params.get("structured_filter") or wrapper_params.get(
            "exact_method_search"
        ):
            # The alias index is built from the same corpus metadata the rest
            # of the pipeline uses, so it is derived rather than configured.
            wrapper_params.setdefault(
                "paper_metadata_path",
                str(resolved_paths["paper_metadata"]),
            )
        retriever["retriever_wrapper"] = {
            "name": wrapper["name"],
            "params": wrapper_params,
        }

    composed_agent = dict(agent)
    if select is not None:
        if not isinstance(select, dict) or not select.get("name"):
            raise ValueError("select_style must be a mapping with a name")
        agent_params = dict(composed_agent.get("params", {}))
        agent_params["paper_selector"] = {
            "name": select["name"],
            "params": dict(select.get("params", {})),
        }
        composed_agent["params"] = agent_params

    return {
        "paths": resolved_paths,
        "preprocessor": {"name": process_name, "params": preprocessor_params},
        "retriever": retriever,
        "agent": composed_agent,
    }


def build_pipeline(
    cfg: dict,
    *,
    build_agent: bool = True,
    build_preprocessor: bool = True,
) -> tuple[Any, Retriever, Any | None]:
    """Build pipeline components with optional agent and preprocessor stages.

    Retrieval-only jobs do not need an LLM client or PDF parser. Skipping those
    stages keeps existing-index evaluation independent from Azure and
    preprocessing dependencies.
    """
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
    if reranker_cfg and reranker_cfg.get("name", "none") != "none":
        reranker = registry.build(
            "reranker", reranker_cfg["name"], **reranker_cfg.get("params", {})
        )

    wrapper_cfg = cfg["retriever"].get("retriever_wrapper")
    core_reranker = None if wrapper_cfg else reranker
    core_retriever = HybridRetriever(
        indexers=indexers,
        fuser=fuser,
        reranker=core_reranker,
        per_index_k=cfg["retriever"]["per_index_k"],
        pool_k=cfg["retriever"].get("pool_k"),
        attribute_weight=cfg["retriever"].get("attribute_weight", 0.25),
    )

    retriever = core_retriever
    if wrapper_cfg:
        retriever = registry.build(
            "retriever_wrapper",
            wrapper_cfg["name"],
            retriever=core_retriever,
            reranker=reranker,
            **wrapper_cfg.get("params", {}),
        )

    agent = None
    if build_agent:
        agent_cfg = cfg["agent"]
        llm_kwargs: dict[str, Any] = {}
        llm_cfg = agent_cfg.get("llm")
        if llm_cfg:
            llm_kwargs["llm"] = registry.build(
                "llm", llm_cfg["name"], **llm_cfg.get("params", {})
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
    if build_preprocessor and preprocessor_cfg:
        preprocessor = registry.build(
            "preprocessor", preprocessor_cfg["name"], **preprocessor_cfg.get("params", {})
        )

    return preprocessor, retriever, agent
