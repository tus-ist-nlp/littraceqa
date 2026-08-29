"""The configuration itself (`di_pipeline.pipeline`).

**There is no mechanism for swapping methods, so what is tested is whether the one
configuration is what it is meant to be.** This pins down how the index paths are
derived — get one wrong and a build of several hours is overwritten — and that each
stage really receives the intended model and parameters.
"""

from __future__ import annotations

import yaml

from littraceqa.di_pipeline.index.faiss_qwen3 import INDEX_NAME, PRODUCTION_PARAMS
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.di_pipeline.pipeline import (
    PROCESS,
    Paths,
    build_agent,
    build_expander,
    build_expander_index,
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
    """Indexes sit under the preprocessor's name, so rebuilding with a different
    preprocessor cannot collide with the existing ones."""
    paths = _paths(tmp_path)
    assert paths.index("bm25s") == tmp_path / "index" / PROCESS / "bm25s"
    assert paths.chunks == tmp_path / "chunks" / f"{PROCESS}_chunks.jsonl"


def test_every_index_has_its_own_directory(tmp_path):
    """No two index paths coincide. **A collision overwrites the index built first.**"""
    dirs = [str(ix.index_dir) for ix in build_indexers(_paths(tmp_path))]
    assert len(dirs) == len(set(dirs)), dirs


def test_embedding_index_shares_its_settings_with_the_shard_builder(tmp_path):
    """The embedding settings come from the constant in index/faiss_qwen3.py.

    The distributed build (scripts/build_faiss_qwen3_shard.py) reads the same
    constant, which is what makes **a model or prefix mismatch between build time
    and search time impossible.**
    """
    embedder = build_indexers(_paths(tmp_path))[2]
    assert embedder.model_name == PRODUCTION_PARAMS["model"]
    assert embedder.doc_prefix == PRODUCTION_PARAMS["doc_prefix"]
    assert embedder.index_dir.name == INDEX_NAME
    # Search uses devices[0] only, so one GPU here; the rest are left to the reranker.
    assert embedder.devices == ["cuda:0"]


def test_retriever_wiring(tmp_path):
    """Each stage of retrieval is assembled as intended."""
    retriever = build_retriever(_paths(tmp_path))
    assert [type(ix).__name__ for ix in retriever.indexers] == [
        "BM25Index", "BM25PaperIndex", "Qwen3FAISSIndex",
    ]
    assert type(retriever.fuser).__name__ == "PaperRRFFuser"  # one vote per paper
    assert retriever.reranker.model_name == "Qwen/Qwen3-Reranker-8B"
    # The reranker gets two GPUs because it scores pool_k chunks on every query.
    assert retriever.reranker.devices == ["cuda:1", "cuda:2"]
    assert (retriever.per_index_k, retriever.pool_k) == (100, 200)
    assert retriever.seed_expansion.query_chars == 512
    assert retriever.rerank_blend.protect_top == 20
    # **Never raise this**: on NAACL it blew faiss search up 61x.
    assert retriever.max_fetch_k == 3000


def test_expander_uses_three_independent_sources(tmp_path):
    """All three sources are used because they find different gold — two gold papers
    are reachable only through MLT."""
    expander = build_expander(_paths(tmp_path))
    assert [type(s).__name__ for s in expander.sources] == [
        "Specter2PaperExpander", "BibCouplingExpander", "BM25MLTExpander",
    ]
    assert all(s.neighbors == 100 for s in expander.sources)


def test_expander_index_can_be_rebuilt(tmp_path):
    """**There is a way to rebuild ranking B's index.**

    The SPECTER2 index has a writer (Specter2FAISSIndex) and a reader
    (Specter2PaperExpander) in different classes, and the reader opens faiss
    directly. Search keeps working even if nothing ever calls the writer, so
    "it cannot be rebuilt" would go unnoticed until the index was gone. This pins
    the two to the same location.
    """
    paths = _paths(tmp_path)
    builder = build_expander_index(paths)
    reader = build_expander(paths).sources[0]
    assert str(builder.index_dir) == str(reader.index_dir)
    # A whole-paper model, so only title+abstract is indexed.
    assert builder.chunk_types == ["title_abstract"]


def test_expander_index_is_not_a_search_index(tmp_path):
    """SPECTER2 never reaches the fuser; it exists only to look up ranking B."""
    names = [type(ix).__name__ for ix in build_indexers(_paths(tmp_path))]
    assert "Specter2FAISSIndex" not in names


def test_agent_is_assembled_with_the_measured_settings(tmp_path):
    """**Actually run the production assembly.**

    Exercising build_retriever and build_expander individually says nothing about
    build_agent, which ties them together — and that did break once: a param removed
    from ReadingConfig was still being passed, and run_search.py died on startup.
    """
    agent = build_agent(_paths(tmp_path), llm=FakeLLM())

    assert agent.config.max_steps == 3
    assert agent.config.max_candidates == 20
    assert agent.config.max_papers == 10
    assert agent.config.subquery_count == 4
    # Table chunks are not used for a paper's representative score; the reader still
    # sees them as before.
    assert agent.config.paper_score_skip_chunk_types == ("table",)
    # The A/B fusion. At k=60 a deep rank still beats A's top hit, hence 10.
    assert agent.combine.rrf_k == 10
    assert agent.combine.anchors == 1
    assert agent.combine.anchor_from == "verdict"
    assert agent.combine.related_weight == 1.0 and agent.combine.related_offset == 0
    # Both retrieval and expansion are attached.
    assert agent.retriever.reranker.model_name == "Qwen/Qwen3-Reranker-8B"
    assert len(agent.paper_expander.sources) == 3


def test_preprocessor_reads_mineru_output(tmp_path):
    assert type(build_preprocessor(_paths(tmp_path))).__name__ == "MinerUChunker"
