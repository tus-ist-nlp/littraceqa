"""The retrieval system itself. **Read this file to understand the whole pipeline.**

How one question becomes 50 candidate papers:

    question
     ├─ extract the venue/year constraint once (AttributeExtractor)
     ├─ an LLM splits the question into 4 subqueries          ┐
     │   for each subquery:                                    │
     │     query 3 indexes (chunk BM25 / paper BM25 / Qwen3-8B)│
     │     fuse per paper (one vote per paper)                 │ iterate,
     │     seed expansion (re-query with the top paper's terms)│ at most
     │     score with Qwen3-Reranker-8B, blend the ranks       │ 3 rounds
     ├─ show the top 20 papers to a reading LLM, which picks    │
     │   the evidence and says whether it is sufficient         │
     └─ if not, re-split from what is missing and loop back    ┘
        ↓
    fuse ranking A (question→paper) with ranking B (paper→paper) → 50 candidates

Measurements and the reasoning behind each value live in CLAUDE.md; the
implementations live in each module's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from littraceqa.chunk_store import ChunkStore
from littraceqa.common import ROOT
from littraceqa.di_pipeline.agent import CombineConfig, ReadingAgent, ReadingConfig
from littraceqa.di_pipeline.expander import (
    BibCouplingExpander,
    BM25MLTExpander,
    FusedPaperExpander,
    Specter2PaperExpander,
)
from littraceqa.di_pipeline.faiss_qwen3 import INDEX_NAME as QWEN3_INDEX_NAME
from littraceqa.di_pipeline.faiss_qwen3 import PRODUCTION_PARAMS as QWEN3_PARAMS
from littraceqa.di_pipeline.faiss_qwen3 import Qwen3FAISSIndex
from littraceqa.di_pipeline.indexes import BM25Index, BM25PaperIndex, Specter2FAISSIndex
from littraceqa.di_pipeline.llm import AzureOpenAILLM, LLMClient
from littraceqa.di_pipeline.preprocess import MinerUChunker
from littraceqa.di_pipeline.reranker import Qwen3Reranker
from littraceqa.di_pipeline.retrieve import (
    AttributeExtractor,
    HybridRetriever,
    PaperRRFFuser,
    RerankBlend,
    SeedExpansion,
)

# API keys and the like come from .env at the repo root (never from code or yaml).
# Variables already exported in the environment win.
load_dotenv(ROOT / ".env")

# Name of the preprocessor. It sits in the middle of every index path
# (`{index_dir}/mineru/bm25s`), so rebuilding with a different preprocessor
# cannot clobber the existing indexes.
PROCESS = "mineru"

# Name of the SPECTER2 index. **The model works on whole papers, so only
# title+abstract is indexed** — the proximity adapter was trained on
# title+abstract, and embedding body fragments, tables or equations separately
# takes the input off that distribution. The "abstract" suffix is a leftover
# from when a whole-chunk variant existed alongside it.
SPECTER2_INDEX_NAME = "faiss_specter2_abstract"


@dataclass(frozen=True)
class Paths:
    """Only the locations that differ per machine. **No method settings here.**

    Loaded from `configs/paths/*.yaml`, which exists because machines keep their
    data in different places.
    """

    pdf_dir: Path
    chunks_dir: Path
    index_dir: Path
    paper_metadata: Path

    @classmethod
    def load(cls, path: str | Path) -> Paths:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            pdf_dir=Path(raw["pdf_dir"]),
            chunks_dir=Path(raw["chunks_dir"]),
            index_dir=Path(raw["index_dir"]),
            paper_metadata=Path(raw["paper_metadata"]),
        )

    @property
    def chunks(self) -> Path:
        """Chunks written by the preprocessor (both indexing and reading use them)."""
        return self.chunks_dir / f"{PROCESS}_chunks.jsonl"

    def index(self, name: str) -> Path:
        """Where one index lives. Keep `name` unique per index — a collision
        silently overwrites an index that took hours to build."""
        return self.index_dir / PROCESS / name


def build_preprocessor(paths: Paths) -> MinerUChunker:
    """PDF → chunks. Only reads the content_list.json that scripts/run_mineru.py wrote."""
    return MinerUChunker(pdf_dir=str(paths.pdf_dir), max_chars_per_chunk=2000)


def build_indexers(paths: Paths) -> list:
    """The three search indexes. `--build` needs only these plus the preprocessor
    (no reranker, no ChunkStore)."""
    return [
        # Strong when the question's terms land inside a single chunk.
        BM25Index(index_dir=str(paths.index("bm25s"))),
        # Treats a whole paper as one document. When the question's terms are
        # scattered across a paper, no single chunk holds them all and the chunk
        # index goes weak, so the two are used together.
        BM25PaperIndex(index_dir=str(paths.index("bm25s_paper"))),
        # Catches paraphrases that share no vocabulary. Model settings live in
        # faiss_qwen3.py so the distributed builder can share them. Only one
        # GPU here: search uses devices[0] only, leaving the rest for the reranker.
        Qwen3FAISSIndex(
            index_dir=str(paths.index(QWEN3_INDEX_NAME)),
            devices="cuda:0",
            **QWEN3_PARAMS,
        ),
    ]


def build_retriever(paths: Paths) -> HybridRetriever:
    """One question → a ranking of chunks. Fuses 3 indexes per paper, then blends
    the reranker's ranking into it.

    **Assumes the chunks already exist** (`anchor_store` reads them). Index builds
    use `build_indexers()` directly.
    """
    return HybridRetriever(
        indexers=build_indexers(paths),
        # **One vote per paper.** Fusing per chunk lets long papers and
        # table-heavy papers occupy the top purely by chunk count, and the metric
        # is per paper, so that distortion lands straight on the score.
        fuser=PaperRRFFuser(k=60, chunks_per_paper=3),
        # Multi-GPU disables torch.compile automatically (dynamo breaks when a
        # compiled model is called from several threads). Weights load lazily.
        reranker=Qwen3Reranker(
            model="Qwen/Qwen3-Reranker-8B",
            devices="cuda:1,cuda:2",
            fp16=True,
            max_batch_tokens=2048,
            batch_size=4,
            max_tokens=2048,
        ),
        per_index_k=100,
        pool_k=200,
        # Append the top paper's vocabulary to the question and query again.
        # **Runs before the reranker**, so inference cost does not go up.
        seed_expansion=SeedExpansion(query_chars=512),
        anchor_store=ChunkStore(str(paths.chunks)),
        # Blend the reranker's ranking instead of letting it replace the order.
        rerank_blend=RerankBlend(
            original_weight=0.6, rerank_weight=0.4, rrf_k=60, protect_top=20
        ),
        # Only fires when the question names a venue; otherwise the original path.
        attribute_extractor=AttributeExtractor(paths.paper_metadata),
        fetch_safety=1.5,
        # **Do not raise this.** Matching it to per_index_k (40000) blew up faiss
        # search from 1.5s to 91.1s on NAACL (4.3% selectivity).
        max_fetch_k=3000,
        min_filtered_results=10,
    )


def build_expander_index(paths: Paths) -> Specter2FAISSIndex:
    """The SPECTER2 index that ranking B reads. **Not a search index, so it is not
    in `build_indexers`** — it never reaches the fuser and only serves
    `build_expander()` looking up neighbours.

    `--build` still has to create it. The writer and the reader are different
    classes, so without this there would be no way to rebuild the index at all.
    """
    return Specter2FAISSIndex(
        index_dir=str(paths.index(SPECTER2_INDEX_NAME)),
        model="allenai/specter2_base",
        chunk_types=["title_abstract"],
        batch_size=128,
        fp16=True,
    )


def build_expander(paths: Paths) -> FusedPaperExpander:
    """Paper-to-paper proximity (ranking B). **The three sources find different
    gold papers, which is why all three are used.**

    Of 37 gold papers that fell outside the candidate list, SPECTER2 recovered 15,
    bibliographic coupling 11 and full-text MLT 16; 2 were reachable only via MLT.
    """
    return FusedPaperExpander(
        sources=[
            # Neighbours in SPECTER2(proximity) space. Reads a prebuilt index.
            Specter2PaperExpander(
                index_dir=str(paths.index(SPECTER2_INDEX_NAME)), neighbors=100
            ),
            # Jaccard over the arXiv IDs each paper cites. **Not a citation graph** —
            # this corpus only covers 2024-2025, so contemporaries cannot cite each
            # other; the link comes from older work they share. min_shared=2 drops
            # generic citations (Adam, ResNet) that would connect everything.
            BibCouplingExpander(
                chunks=str(paths.chunks),
                cache_path=str(paths.index("bib_coupling") / "refs.pkl"),
                neighbors=100,
                min_shared=2,
            ),
            # More-like-this over full paper text, querying the prebuilt
            # bm25s_paper index.
            BM25MLTExpander(
                index_dir=str(paths.index("bm25s_paper")),
                cache_path=str(paths.index("bm25_mlt") / "anchor_text.pkl"),
                neighbors=100,
                query_chars=1200,
            ),
        ],
        neighbors=100,
        rrf_k=60,
    )


def build_agent(paths: Paths, llm: LLMClient | None = None) -> ReadingAgent:
    """Split → read → re-search for what is missing, then fuse A and B into 50."""
    return ReadingAgent(
        build_retriever(paths),
        llm=llm or AzureOpenAILLM(reasoning_effort="medium"),
        paper_expander=build_expander(paths),
        # **k=10.** At k=60, with lists of 50, a paper present in both rankings
        # beats A's top hit no matter how deep it sits (2/(61+r) > 1/61 for r < 61).
        combine=CombineConfig(
            rrf_k=10,
            related_weight=1.0,
            related_offset=0,
            anchors=1,
            anchor_from="verdict",
        ),
        config=ReadingConfig(
            max_steps=3,
            retrieve_top_k=20,
            max_candidates=20,
            chunks_per_paper=2,
            snippet_chars=1800,
            max_papers=10,
            # Table chunks are dense with numbers and short labels, so a single
            # table can spike a paper's representative score even when the paper
            # is off-topic. **They still reach the reading LLM as usual.**
            paper_score_skip_chunk_types=("table",),
        ),
    )
