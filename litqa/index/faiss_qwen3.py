"""FAISS + Qwen3-Embedding による dense 検索インデックス。

Chunk のテキストを Qwen3-Embedding でベクトル化し、FAISS の内積 (IndexFlatIP)
インデックスを構築する。埋め込みは L2 正規化してあるため、内積がそのまま
コサイン類似度になる。

Qwen3-Embedding は文書側/クエリ側を prefix (「passage: 」/「query: 」) で
区別し、プーリングは最後の非パディングトークンを使う。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"


@register("indexer", "faiss_qwen3")
class Qwen3FAISSIndex:
    name = "faiss_qwen3"

    def __init__(
        self,
        index_dir: str,
        model: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 32,
        device: str = "cuda",
        doc_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

        self._tokenizer = None
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        embeddings = self._embed(
            [chunk.text for chunk in self._chunks], prefix=self.doc_prefix
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
        query_embedding = self._embed([query], prefix=self.query_prefix)
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

    def _embed(self, texts: list[str], prefix: str = "") -> np.ndarray:
        self._ensure_model()
        texts = [prefix + text for text in texts]

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
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


def _last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """パディング方向(left/right)によらず最後の非パディングトークンを取り出す。

    Qwen3-Embedding が公式に採用しているプーリング方法。
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, sequence_lengths]
