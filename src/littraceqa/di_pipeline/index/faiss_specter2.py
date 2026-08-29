"""Dense retrieval with SPECTER2, over FAISS.

Embeds a Chunk's text with SPECTER2 into a FAISS inner-product index
(`IndexFlatIP`). The embeddings are L2-normalised, so the inner product *is* cosine
similarity.

SPECTER2 tells the document side from the query side by swapping adapters
("proximity" for documents, "adhoc_query" for queries). Pooling is the CLS token —
SPECTER2 is a BERT-family encoder.

**The adapters ship in the `adapters` library's format, not peft's**
(allenai/specter2's adapter_config.json has no `peft_type`), so loading them
through peft dies with `KeyError: 'peft_type'`. `AutoAdapterModel` is what reads
them.

**Do not move `_MAX_TOKENS` off 512.** SPECTER2 is a BERT with
max_position_embeddings=512, but its tokenizer leaves model_max_length unset, so a
larger value is not truncated even with `truncation=True` — the input runs past the
position embeddings and forward crashes. About 8% of MinerU's chunks are longer
than 512 tokens, so this is hit routinely, not rarely.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.index.chunk_filter import filter_chunk_types

_CHUNKS_FILENAME = "chunks.jsonl"
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
