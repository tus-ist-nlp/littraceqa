"""PyLate + ColBERT による late interaction 検索インデックス。

Chunk のテキストを ColBERT でトークン単位のベクトル列として埋め込み、
PLAID インデックスを構築する。検索時はクエリの各トークンが文書の最も近い
トークンとのスコアを取る max-sim により late interaction で順位付けする。

dense 検索（1文書=1ベクトル）より精密だが、メモリ・計算コストは重い。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pylate import indexes, models, retrieve

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_PLAID_INDEX_NAME = "colbert"


@register("indexer", "colbert")
class ColBERTIndex:
    name = "colbert"

    def __init__(
        self,
        index_dir: str,
        model: str = "colbert-ir/colbertv2.0",
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device

        self._model = None
        self._index: indexes.PLAID | None = None
        self._retriever: retrieve.ColBERT | None = None
        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[str, Chunk] = {}

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._ensure_model()
        doc_embeddings = self._model.encode(
            sentences=[chunk.text for chunk in self._chunks],
            batch_size=self.batch_size,
            is_query=False,
            show_progress_bar=False,
        )
        self._index = indexes.PLAID(
            index_folder=str(self.index_dir),
            index_name=_PLAID_INDEX_NAME,
            override=True,
            show_progress=False,
        )
        self._index.add_documents(
            documents_ids=[chunk.chunk_id for chunk in self._chunks],
            documents_embeddings=doc_embeddings,
        )
        self._retriever = retrieve.ColBERT(index=self._index)
        self._save_chunks()

    def load(self) -> None:
        self._chunks = self._load_chunks()
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._index = indexes.PLAID(
            index_folder=str(self.index_dir),
            index_name=_PLAID_INDEX_NAME,
            override=False,
            show_progress=False,
        )
        self._retriever = retrieve.ColBERT(index=self._index)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._retriever is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        self._ensure_model()
        query_embedding = self._model.encode(
            sentences=[query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )
        results_nested = self._retriever.retrieve(
            queries_embeddings=query_embedding, k=k
        )
        if results_nested:
            results_raw = results_nested[0]
        else:
            results_raw = []
        results: list[RetrievalResult] = []
        for result_dict in results_raw:
            chunk_id = result_dict["id"]
            score = result_dict["score"]
            if chunk_id not in self._chunk_by_id:
                continue
            chunk = self._chunk_by_id[chunk_id]
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

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model = models.ColBERT(
            model_name_or_path=self.model_name,
            device=self.device,
        )

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
