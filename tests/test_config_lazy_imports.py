"""Focused tests for config-driven lazy component imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an import check in a fresh interpreter with the local source tree."""
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    source_path = str(_PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_config_import_and_yaml_loading_do_not_import_optional_components(tmp_path):
    config_path = tmp_path / "paths.yaml"
    config_path.write_text("pdf_dir: /data/pdfs\n", encoding="utf-8")
    script = r"""
import sys

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.config import compose_config, load_config

assert load_config(sys.argv[1]) == {"pdf_dir": "/data/pdfs"}
assert compose_config
expected_keys = {
    ("agent", "reading"),
    ("indexer", "bm25s"),
    ("indexer", "faiss_qwen3"),
    ("llm", "azure_openai"),
    ("preprocessor", "marker"),
    ("preprocessor", "mineru"),
    ("reranker", "qwen3"),
    ("retriever_wrapper", "seed_expansion"),
}
assert expected_keys <= set(registry.list_registered())
fake_llm = registry.build("llm", "fake", responses=["ok"])
assert fake_llm("prompt") == "ok"
assert "littraceqa.di_pipeline.llm.fake" in sys.modules

for module_name in (
    "dotenv",
    "openai",
    "torch",
    "faiss",
    "littraceqa.di_pipeline.agent.reading",
    "littraceqa.di_pipeline.index.faiss_qwen3",
    "littraceqa.di_pipeline.llm.azure_openai",
    "littraceqa.di_pipeline.preprocess.marker_chunker",
    "littraceqa.di_pipeline.preprocess.mineru_chunker",
    "littraceqa.di_pipeline.retrieve.qwen3_reranker",
    "littraceqa.di_pipeline.retrieve.seed_expansion",
):
    assert module_name not in sys.modules, module_name
"""

    result = _run_isolated(script, str(config_path))

    assert result.returncode == 0, result.stderr


def test_agentless_mineru_bm25_build_imports_only_selected_components(tmp_path):
    script = r"""
import sys
from pathlib import Path

from littraceqa.di_pipeline.config import build_pipeline, compose_config

root = Path(sys.argv[1])
cfg = compose_config(
    paths={
        "pdf_dir": str(root / "pdfs"),
        "chunks_dir": str(root / "chunks"),
        "index_dir": str(root / "index"),
        "paper_metadata": str(root / "papers.jsonl"),
    },
    process={"name": "mineru", "params": {"mineru_dir": str(root / "mineru")}},
    search={
        "per_index_k": 20,
        "indexers": [{"name": "bm25s", "params": {}}],
        "fuser": {"name": "rrf", "params": {}},
        "reranker": {"name": "none", "params": {}},
    },
    agent={
        "name": "reading",
        "llm": {"name": "azure_openai", "params": {}},
        "params": {"top_k": 20},
    },
)
preprocessor, retriever, agent = build_pipeline(cfg, build_agent=False)

assert type(preprocessor).__name__ == "MinerUChunker"
assert type(retriever.indexers[0]).__name__ == "BM25Index"
assert type(retriever.fuser).__name__ == "RRFFuser"
assert retriever.reranker is None
assert agent is None

required_modules = {
    "littraceqa.di_pipeline.index.bm25_index",
    "littraceqa.di_pipeline.preprocess.mineru_chunker",
    "littraceqa.di_pipeline.retrieve.rrf",
}
assert required_modules <= set(sys.modules)
for module_name in (
    "dotenv",
    "openai",
    "torch",
    "faiss",
    "littraceqa.di_pipeline.agent.reading",
    "littraceqa.di_pipeline.index.faiss_qwen3",
    "littraceqa.di_pipeline.index.faiss_specter2",
    "littraceqa.di_pipeline.llm.azure_openai",
    "littraceqa.di_pipeline.preprocess.marker_chunker",
    "littraceqa.di_pipeline.retrieve.reranker",
    "littraceqa.di_pipeline.retrieve.seed_expansion",
):
    assert module_name not in sys.modules, module_name
"""

    result = _run_isolated(script, str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_retrieval_only_build_skips_preprocessor_and_agent_imports(tmp_path):
    script = r"""
import sys
from pathlib import Path

from littraceqa.di_pipeline.config import build_pipeline, compose_config

root = Path(sys.argv[1])
cfg = compose_config(
    paths={
        "pdf_dir": str(root / "pdfs"),
        "chunks_dir": str(root / "chunks"),
        "index_dir": str(root / "index"),
        "paper_metadata": str(root / "papers.jsonl"),
    },
    process={"name": "mineru", "params": {"mineru_dir": str(root / "mineru")}},
    search={
        "per_index_k": 20,
        "indexers": [{"name": "bm25s", "params": {}}],
        "fuser": {"name": "rrf", "params": {}},
        "reranker": {"name": "none", "params": {}},
    },
    agent={
        "name": "reading",
        "llm": {"name": "azure_openai", "params": {}},
        "params": {"top_k": 20},
    },
)
preprocessor, retriever, agent = build_pipeline(
    cfg,
    build_agent=False,
    build_preprocessor=False,
)

assert preprocessor is None
assert agent is None
assert type(retriever.indexers[0]).__name__ == "BM25Index"
assert type(retriever.fuser).__name__ == "RRFFuser"
assert "littraceqa.di_pipeline.index.bm25_index" in sys.modules
assert "littraceqa.di_pipeline.retrieve.rrf" in sys.modules
for module_name in (
    "dotenv",
    "openai",
    "littraceqa.di_pipeline.agent.reading",
    "littraceqa.di_pipeline.llm.azure_openai",
    "littraceqa.di_pipeline.preprocess.mineru_chunker",
    "littraceqa.di_pipeline.retrieve.seed_expansion",
):
    assert module_name not in sys.modules, module_name
"""

    result = _run_isolated(script, str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_qwen3_reranker_construction_does_not_load_the_model():
    script = r"""
import sys

from littraceqa.di_pipeline import registry
import littraceqa.di_pipeline.config

reranker = registry.build("reranker", "qwen3", local_files_only=True)
assert type(reranker).__name__ == "Qwen3Reranker"
assert reranker.local_files_only is True
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""

    result = _run_isolated(script)

    assert result.returncode == 0, result.stderr


def test_seed_expansion_qwen_reranks_the_fixed_final_paper_set(tmp_path):
    script = r"""
import sys
from pathlib import Path

from littraceqa.di_pipeline.config import build_pipeline, compose_config, load_config

root = Path(sys.argv[1])
cfg = compose_config(
    paths={
        "pdf_dir": str(root / "pdfs"),
        "chunks_dir": str(root / "chunks"),
        "index_dir": str(root / "index"),
        "paper_metadata": str(root / "papers.jsonl"),
    },
    process={"name": "mineru", "params": {"mineru_dir": str(root / "mineru")}},
    search=load_config(
        "configs/search_style/"
        "bm25_paper_rank_seed_expansion_qwen3_reranker.yaml"
    ),
    agent={"name": "simple", "params": {"top_k": 10}},
)
preprocessor, retriever, agent = build_pipeline(
    cfg,
    build_agent=False,
    build_preprocessor=False,
)

assert preprocessor is None
assert agent is None
assert type(retriever).__name__ == "SeedExpansionRetriever"
assert retriever.retriever.reranker is None
assert type(retriever.reranker).__name__ == "Qwen3Reranker"
assert retriever.rerank_pool_k == 50
assert retriever.max_results == 50
assert retriever.final_rerank_protected_top_k == 20
assert retriever.stable_prefix_k == 10
assert retriever.rerank_final_candidates is True
assert retriever.final_rerank_document_chars == 2000
assert retriever.paper_dense_consensus_seed_k == 3
assert retriever.paper_dense_reciprocal_seed_k == 8
assert retriever.paper_dense_reciprocal_forward_k == 20
assert retriever.paper_dense_reciprocal_reverse_k == 10
assert retriever.paper_dense_reciprocal_min_support == 6
assert retriever.paper_dense_reciprocal_max_candidates == 32
assert retriever.open_set_seed_k == 5
assert retriever.open_set_min_support == 2
assert retriever.open_set_max_seed_rank == 2
assert retriever.open_set_slot_k == 20
assert retriever.reranker.device == "cuda:0"
assert retriever.reranker.dtype == "bfloat16"
assert retriever.reranker.batch_size == 4
assert retriever.reranker.base_rank_weight == 0.59
assert retriever.reranker._model is None
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""

    result = _run_isolated(script, str(tmp_path))

    assert result.returncode == 0, result.stderr
