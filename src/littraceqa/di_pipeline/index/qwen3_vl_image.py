"""FAISS + Qwen3-VL-Embedding による図表画像の dense 検索インデックス。

MinerU が切り出した図表画像（Chunk の metadata["image_path"]）をそのまま
ベクトル化する。siglip_image.py と同じ位置づけ（画像を画像のまま埋め込む）だが、
モデルが違う:

* SigLIP は画像用と短いテキスト用のエンコーダが分かれた CLIP 系で、長い質問文に弱い。
* Qwen3-VL-Embedding は画像と長文を**同一モデル・同一空間**で埋め込む。
  「Figure 4 のサブ図は何個か」のような具体的な質問と図表の対応付けに向く。

figure_vlm（VLM に図をテキスト説明させてから text 埋め込みする経路）とも違い、
**テキストへの変換というボトルネックを挟まない**。しかも MinerU 変換の時点で
画像は既に切り出されており（{mineru_dir}/{paper_id}/auto/images/*.jpg）、
MinerUChunker が metadata["image_path"] に入れているので、追加の前処理は要らない。

image_path を持たないチャンク（本文など）は索引対象外としてスキップする。
実測(RTX3090, fp16): 5.2 枚/秒, VRAM ピーク19.5GB, 4096次元。
figure+table は全チャンク256万件のうち約52.8万件なので、4GPU で約7時間・索引8.7GB。

**隔離 venv (.venv-vl) が必要**:
本体の .venv は pylate(ColBERT) が sentence-transformers==5.3.0 を固定しているが、
Qwen3-VL-Embedding は 5.4.0 以降を要求するため共存できない。sentence_transformers の
import をモジュール先頭ではなく _ensure_model() 内に置いてあるのは、本体 venv で
config.py からこのモジュールを import しても壊れないようにするため（索引を実際に
使うときだけ .venv-vl で動かす）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"


@register("indexer", "qwen3_vl_image")
class Qwen3VLImageIndex:
    name = "qwen3_vl_image"

    def __init__(
        self,
        index_dir: str,
        model: str = "Qwen/Qwen3-VL-Embedding-8B",
        # 実測でVRAMピークは bs=4 で18.0GB / bs=8 で19.5GB。RTX3090(24GB)には載るが
        # スループットは 4.9 -> 5.2 枚/秒とほぼ頭打ちなので、上げても得が少ない。
        batch_size: int = 8,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16 and str(device).startswith("cuda")

        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        # 画像が実在するものだけを対象にする。MinerU が失敗した論文で image_path だけ
        # 残っていることがあるため、存在チェックまでして黙って落とす。
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

        print(
            f"  {self.name}: {len(self._chunks):,} 件の図表画像を "
            f"{self.model_name} で埋め込み (batch_size={self.batch_size})"
        )
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
        query_embedding = self._embed_texts([query])
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

    # ---- 埋め込み ---------------------------------------------------------

    def _embed_images(self, image_paths: list[str]) -> np.ndarray:
        """図表画像を埋め込む。バッチごとに float32 化して溜める。

        52.8万件を一度に list[list[float]] で持つとメモリが持たないので、
        バッチ単位で ndarray にしてから concatenate する
        （faiss_azure_openai.py で同じ理由の OOM を踏んだ）。
        """
        from PIL import Image  # 隔離 venv 側にしか無いので遅延 import

        self._ensure_model()
        arrays: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            vectors = self._model.encode(
                images, batch_size=self.batch_size, show_progress_bar=False
            )
            arrays.append(np.asarray(vectors, dtype="float32"))
            if (start // self.batch_size) % 500 == 0:
                done = min(start + self.batch_size, len(image_paths))
                print(f"    {done:,}/{len(image_paths):,} 枚")

        embeddings = np.concatenate(arrays, axis=0)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """クエリ側。画像と同じモデル・同じ空間に落ちる（ここが SigLIP との違い）。"""
        self._ensure_model()
        vectors = self._model.encode(
            texts, batch_size=len(texts), show_progress_bar=False
        )
        embeddings = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # 本体 venv (sentence-transformers 5.3.0) では読めないので、モジュール先頭では
        # なくここで import する。実行は .venv-vl（5.4.0 以降）で行う。
        import torch
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"dtype": torch.float16} if self.fp16 else {}
        self._model = SentenceTransformer(
            self.model_name, device=self.device, model_kwargs=model_kwargs
        )

    # ---- チャンクの保存・読み込み ------------------------------------------

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
