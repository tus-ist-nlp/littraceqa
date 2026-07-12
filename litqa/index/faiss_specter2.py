"""FAISS + SPECTER2 による dense 検索インデックス。

Chunk のテキストを SPECTER2 でベクトル化し、FAISS の内積 (IndexFlatIP)
インデックスを構築する。埋め込みは L2 正規化してあるため、内積がそのまま
コサイン類似度になる。

SPECTER2 は文書側/クエリ側をアダプタ切り替えで区別する
（doc側 "proximity" / query側 "adhoc_query"）。プーリングは CLS トークン
（SPECTER2 は BERT 系エンコーダ）。

アダプタは peft ではなく adapters ライブラリ形式で配布されている
（allenai/specter2 の adapter_config.json は peft_type を持たない）。peft で読むと
KeyError: 'peft_type' で落ちるため、AutoAdapterModel を使う。

_MAX_TOKENS は 512 から動かさないこと。SPECTER2 は max_position_embeddings=512 の
BERT だが、tokenizer の model_max_length が未設定なので、これより大きい値を渡すと
truncation=True でも切り詰められず、位置埋め込みの範囲外になって forward が落ちる
（MinerU のチャンクは約8%が512トークンを超える）。
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

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"

# SPECTER2 (BERT) の max_position_embeddings。上記の通り増やしてはいけない。
_MAX_TOKENS = 512


@register("indexer", "faiss_specter2")
class Specter2FAISSIndex:
    name = "faiss_specter2"

    def __init__(
        self,
        index_dir: str,
        model: str = "allenai/specter2_base",
        batch_size: int = 128,
        device: str = "cuda",
        fp16: bool = True,
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
        # 埋め込みはどうせ L2 正規化するので fp16 でも精度への影響は無視できる。
        # RTX 3090 実測で fp32 134 chunks/s に対し fp16 476 chunks/s（約3.6倍）。
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
        self._chunks = list(chunks)
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

        # 索引構築では 250万件規模を回すので、進捗が見えないと生きているのか
        # 判断できない。クエリ1件の埋め込みでは出さない。
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
            pooled = outputs.last_hidden_state[:, 0]  # CLS トークン
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
