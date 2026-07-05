"""FAISS + SigLIP による画像そのものの dense 検索インデックス。

figure_vlm の Chunk（metadata["image_path"] を持つ figure/table チャンク）が
対象。VLM でテキスト化する faiss_qwen3 経路とは異なり、切り出した図・表の
画像をそのまま vision encoder（SigLIP）でベクトル化する。クエリは SigLIP の
text encoder でベクトル化するため、bm25s/faiss_qwen3 とは別の埋め込み空間になる
（RRFFuser はスコアではなく順位で統合するため、空間が異なっても問題なく併用できる）。

image_path を持たないチャンク（本文など）は索引対象外としてスキップする。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"


@register("indexer", "siglip_image")
class SiglipImageIndex:
    name = "siglip_image"

    def __init__(
        self,
        index_dir: str,
        model: str = "google/siglip-base-patch16-224",
        batch_size: int = 16,
        device: str = "cuda",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device

        self._processor = None
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("image_path")
            and Path(chunk.metadata["image_path"]).exists()
        ]
        self._save_chunks()
        if not self._chunks:
            self._index = None
            return

        embeddings = self._embed_images(
            [chunk.metadata["image_path"] for chunk in self._chunks]
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_dir / _INDEX_FILENAME))
        self._index = index

    def load(self) -> None:
        self._chunks = self._load_chunks()
        index_path = self.index_dir / _INDEX_FILENAME
        self._index = faiss.read_index(str(index_path)) if index_path.exists() else None

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self._chunks or self._index is None:
            return []
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        query_embedding = self._embed_text([query])
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

    def _embed_images(self, image_paths: list[str]) -> np.ndarray:
        self._ensure_model()
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            all_embeddings.append(features.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        inputs = self._processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        embeddings = features.float().cpu().numpy().astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._processor = AutoProcessor.from_pretrained(self.model_name)
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
