"""The indexes that are cheap enough to build in one pass: BM25 and SPECTER2.

Three classes, one per index in `pipeline.build_indexers()` plus the SPECTER2 index
that ranking B reads. Qwen3-Embedding lives in faiss_qwen3.py instead, because the
distributed builder imports that module on machines where bm25s and the SPECTER2
adapters are not installed.

## BM25 over chunks


Builds a bm25s (https://github.com/xhluca/bm25s) index from the Chunks in
chunks.jsonl and answers a query with RetrievalResults.

The index itself persists to `index_dir` through `bm25s.BM25.save/load`, but bm25s
stores only the terms — none of a Chunk's metadata (paper_id, chunk_type, ...) —
so a copy of chunks.jsonl is written beside it. That copy is what turns a hit back
into a Chunk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import bm25s
import faiss
import numpy as np
import torch
from adapters import AutoAdapterModel
from tqdm import tqdm
from transformers import AutoTokenizer

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, filter_chunk_types

_CHUNKS_FILENAME = "chunks.jsonl"


class BM25Index:
    name = "bm25s"

    def __init__(
        self,
        index_dir: str,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        stopwords: str | None = "en",
        stemmer: str | None = None,
    ):
        """BM25's scoring and tokenisation parameters. The values actually used are
        in `pipeline.build_indexers()`, which leaves every one of them at its default.

        - k1 / b / method: bm25s.BM25's scoring parameters. **Left at their
          defaults** — sweeping them against validation recall never beat the
          default. They are baked into the score matrix at build time, so changing
          one means rebuilding the index.
        - stopwords / stemmer: bm25s.tokenize's side. **Build and search must
          tokenise identically**, so these two are held on the instance and used by
          both (construction and search go through the same `build_indexers()`).
        - stemmer takes a language name (e.g. "english") and stems with PyStemmer;
          omitted, nothing is stemmed. Naming one where PyStemmer is not installed
          raises ImportError at build or load.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.method = method
        self.stopwords = stopwords
        self._stemmer = self._build_stemmer(stemmer)
        self._chunks: list[Chunk] = []
        self._retriever: bm25s.BM25 | None = None

    @staticmethod
    def _build_stemmer(stemmer: str | None):
        if not stemmer:
            return None
        import Stemmer  # PyStemmer; imported here so it is only needed when asked for.

        return Stemmer.Stemmer(stemmer)

    def _tokenize(self, texts: list[str]):
        return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self._stemmer)

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus_tokens = self._tokenize([chunk.text for chunk in self._chunks])
        retriever = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        retriever.index(corpus_tokens)
        retriever.save(str(self.index_dir))
        self._save_chunks()
        self._retriever = retriever

    def load(self) -> None:
        self._retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        self._chunks = self._load_chunks()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._retriever is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        query_tokens = self._tokenize([query])
        doc_indices, scores = self._retriever.retrieve(query_tokens, k=k)
        results: list[RetrievalResult] = []
        for doc_index, score in zip(doc_indices[0], scores[0]):
            chunk = self._chunks[int(doc_index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(score),
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata,
                    source=self.name,
                )
            )
        return results

    def _save_chunks(self) -> None:
        path = self.index_dir / _CHUNKS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _load_chunks(self) -> list[Chunk]:
        path = self.index_dir / _CHUNKS_FILENAME
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                chunks.append(Chunk(**record))
        return chunks


# ============================================================================
# BM25 over whole papers
# ============================================================================
#
# Sparse retrieval over whole papers, with BM25.
#
# Where bm25_index.py makes one document per chunk, this groups the Chunks by
# paper_id and makes **one document per paper**. Both are used together: when a
# question's terms are scattered across a paper, no single chunk holds them all and
# the chunk index goes weak.
#
# Every Chunk's text carries the same prefix, `"[{venue} {year}] {title}\n{body}"`.
# Concatenating them as they are would repeat the venue and title words once per
# chunk and skew the BM25 score, so the prefix is kept once per paper and only the
# bodies are joined.
#
# **A hit here has no real chunk_id** — it gets the pseudo id `"{paper_id}#paper"` —
# so it is never handed to ReadingAgent as evidence. It ranks papers and nothing
# else; `PAPER_LEVEL_SOURCES` in retrieve.py keeps these pseudo chunks from
# being chosen to represent a paper whenever a real chunk exists.

_PAPERS_FILENAME = "papers.jsonl"


def _paper_prefix(metadata: dict) -> str:
    return f"[{metadata['venue']} {metadata['year']}] {metadata['title']}\n"


def _build_paper_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Collapse the Chunks of each paper into one, keeping the prefix only once."""
    bodies: dict[str, list[str]] = {}
    metadata: dict[str, dict] = {}
    for chunk in chunks:
        prefix = _paper_prefix(chunk.metadata)
        body = chunk.text.removeprefix(prefix) if chunk.text.startswith(prefix) else chunk.text
        bodies.setdefault(chunk.paper_id, []).append(body)
        metadata.setdefault(chunk.paper_id, chunk.metadata)

    return [
        Chunk(
            chunk_id=f"{paper_id}#paper",
            paper_id=paper_id,
            text=_paper_prefix(metadata[paper_id]) + "\n".join(bodies[paper_id]),
            chunk_type="paper",
            metadata=metadata[paper_id],
        )
        for paper_id in bodies
    ]


class BM25PaperIndex:
    name = "bm25s_paper"

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._papers: list[Chunk] = []
        self._retriever: bm25s.BM25 | None = None

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._papers = _build_paper_chunks(chunks)
        corpus_tokens = bm25s.tokenize(
            [paper.text for paper in self._papers], stopwords="en"
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str(self.index_dir))
        self._save_papers()
        self._retriever = retriever

    def load(self) -> None:
        self._retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        self._papers = self._load_papers()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._retriever is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        k = min(top_k, len(self._papers))
        if k <= 0:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="en")
        doc_indices, scores = self._retriever.retrieve(query_tokens, k=k)
        results: list[RetrievalResult] = []
        for doc_index, score in zip(doc_indices[0], scores[0]):
            paper = self._papers[int(doc_index)]
            results.append(
                RetrievalResult(
                    chunk_id=paper.chunk_id,
                    paper_id=paper.paper_id,
                    score=float(score),
                    text=paper.text,
                    chunk_type=paper.chunk_type,
                    metadata=paper.metadata,
                    source=self.name,
                )
            )
        return results

    def _save_papers(self) -> None:
        path = self.index_dir / _PAPERS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for paper in self._papers:
                f.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")

    def _load_papers(self) -> list[Chunk]:
        path = self.index_dir / _PAPERS_FILENAME
        papers: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                papers.append(Chunk(**record))
        return papers


# ============================================================================
# SPECTER2, over FAISS
# ============================================================================
#
# Dense retrieval with SPECTER2, over FAISS.
#
# Embeds a Chunk's text with SPECTER2 into a FAISS inner-product index
# (`IndexFlatIP`). The embeddings are L2-normalised, so the inner product *is* cosine
# similarity.
#
# SPECTER2 tells the document side from the query side by swapping adapters
# ("proximity" for documents, "adhoc_query" for queries). Pooling is the CLS token —
# SPECTER2 is a BERT-family encoder.
#
# **The adapters ship in the `adapters` library's format, not peft's**
# (allenai/specter2's adapter_config.json has no `peft_type`), so loading them
# through peft dies with `KeyError: 'peft_type'`. `AutoAdapterModel` is what reads
# them.
#
# **Do not move `_MAX_TOKENS` off 512.** SPECTER2 is a BERT with
# max_position_embeddings=512, but its tokenizer leaves model_max_length unset, so a
# larger value is not truncated even with `truncation=True` — the input runs past the
# position embeddings and forward crashes. About 8% of MinerU's chunks are longer
# than 512 tokens, so this is hit routinely, not rarely.

# The sidecar written beside a faiss index, holding the chunks in row order. Same
# name and same role as BM25Index's, hence the shared constant at the top.
_INDEX_FILENAME = "index.faiss"

# SPECTER2 (BERT)'s max_position_embeddings. Never raise it; see the module docstring.
_MAX_TOKENS = 512


class Specter2FAISSIndex:
    name = "faiss_specter2"

    def __init__(
        self,
        index_dir: str,
        model: str = "allenai/specter2_base",
        batch_size: int = 128,
        device: str = "cuda",
        fp16: bool = True,
        chunk_types: list[str] | None = None,
        doc_adapter: str = "proximity",
        query_adapter: str = "adhoc_query",
        doc_adapter_id: str = "allenai/specter2",
        query_adapter_id: str = "allenai/specter2_adhoc_query",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        # Which chunk_types go into the index; None takes them all.
        # **SPECTER2's proximity adapter is a whole-paper model trained on
        # title+abstract**, so `["title_abstract"]` is the designed use. Embedding
        # body chunks separately takes the input off that training distribution.
        self.chunk_types = chunk_types
        # The embeddings get L2-normalised anyway, so fp16 costs no measurable
        # accuracy. Measured on an RTX 3090: 134 chunks/s at fp32, 476 at fp16 (3.6x).
        self.fp16 = fp16 and device.startswith("cuda")
        self.doc_adapter = doc_adapter
        self.query_adapter = query_adapter
        self.doc_adapter_id = doc_adapter_id
        self.query_adapter_id = query_adapter_id

        self._tokenizer = None
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = filter_chunk_types(chunks, self.chunk_types)
        if not self._chunks:
            raise ValueError(
                f"no chunk matches chunk_types={self.chunk_types}"
            )
        print(f"  {self.name}: embedding {len(self._chunks):,} chunks (chunk_types={self.chunk_types or 'all'})")
        embeddings = self._embed(
            [chunk.text for chunk in self._chunks], adapter=self.doc_adapter
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_dir / _INDEX_FILENAME))
        self._save_chunks()
        self._index = index

    def load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / _INDEX_FILENAME))
        self._chunks = self._load_chunks()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        query_embedding = self._embed([query], adapter=self.query_adapter)
        scores, indices = self._index.search(query_embedding, k)
        results: list[RetrievalResult] = []
        for score, doc_index in zip(scores[0], indices[0]):
            if doc_index < 0:
                continue
            chunk = self._chunks[int(doc_index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(score),
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata,
                    source=self.name,
                )
            )
        return results

    def _embed(self, texts: list[str], adapter: str) -> np.ndarray:
        self._ensure_model()
        self._model.set_active_adapters(adapter)

        # A build runs through millions of chunks; without progress there is no way
        # to tell it apart from a hang. Not printed when embedding a single query.
        starts = range(0, len(texts), self.batch_size)
        if len(texts) > self.batch_size:
            starts = tqdm(starts, desc=f"{self.name} embedding", unit="batch")

        all_embeddings: list[np.ndarray] = []
        for start in starts:
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=_MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            pooled = outputs.last_hidden_state[:, 0]  # the CLS token
            all_embeddings.append(pooled.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from adapters import AutoAdapterModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoAdapterModel.from_pretrained(self.model_name)
        model.load_adapter(
            self.doc_adapter_id, load_as=self.doc_adapter, set_active=True
        )
        model.load_adapter(self.query_adapter_id, load_as=self.query_adapter)
        if self.fp16:
            model = model.half()
        model = model.to(self.device)
        model.eval()
        self._model = model

    def _save_chunks(self) -> None:
        path = self.index_dir / _CHUNKS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _load_chunks(self) -> list[Chunk]:
        path = self.index_dir / _CHUNKS_FILENAME
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                chunks.append(Chunk(**record))
        return chunks
