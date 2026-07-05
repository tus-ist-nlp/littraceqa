"""BM25 による疎検索インデックス。

chunks.jsonl 由来の Chunk 列から bm25s (https://github.com/xhluca/bm25s) の
BM25 インデックスを構築し、クエリに対して RetrievalResult を返す。
索引本体は bm25s.BM25.save/load で index_dir に永続化するが、bm25s は
Chunk のメタデータ（paper_id, chunk_type, metadata など）までは持たないため、
検索結果を Chunk に戻すための chunks.jsonl も同じディレクトリに保存する。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import bm25s

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"


@register("indexer", "bm25s")
class BM25Index:
    name = "bm25s"

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._retriever: bm25s.BM25 | None = None

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in self._chunks], stopwords="en"
        )
        retriever = bm25s.BM25()
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
        query_tokens = bm25s.tokenize([query], stopwords="en")
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
