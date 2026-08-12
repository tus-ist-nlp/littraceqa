"""FAISS + Azure OpenAI Embeddings による dense 検索インデックス。

Chunk のテキストを Azure OpenAI の embedding デプロイ（.env の
AZURE_OPENAI_EMBEDDING_DEPLOYMENT、既定 text-embedding-3-large）でベクトル化し、
FAISS の内積 (IndexFlatIP) インデックスを構築する。埋め込みは L2 正規化してあるため、
内積がそのままコサイン類似度になる。

ローカルGPUモデル（faiss_qwen3 / faiss_specter2）と違い埋め込みはAPI呼び出しなので
GPU/デバイス管理は不要な代わりに、レート制限・一時エラーへのリトライが要る。
`src/littraceqa/azure/build_azure_search_index.py` が Azure AI Search パイプライン用に
持っている `embed_texts_with_retry`（context-length時の適応的再切り詰め、429/5xx時の
指数バックオフ）をそのまま再利用する。

課金は入力トークン数で決まり、`AZURE_OPENAI_EMBEDDING_REQUEST_DIMENSIONS` で次元削減を
リクエストしても安くならない。全chunk（約256万件、平均1,020文字/chunk）を埋め込むと
概算 $75〜100程度（2026-07時点の見積もり、text-embedding-3-large換算）。

`embed_texts_with_retry` の再切り詰めは文字数ベース（`MAX_EMBED_CHARS`、英語散文の
~4文字/トークンを想定）なので、パイプ区切りの表や文字化けOCRのような~2文字/トークン
以下の密なチャンクだと半分に切ってもまだ8,192トークン上限を超え、バッチが
恒久的に失敗しうる（chunkとembeddingの対応がずれるので静かにスキップせず例外で
止める設計）。これを回避するため、送信前に tiktoken で実トークン数を測って
`_EMBED_TOKEN_LIMIT` に収まるよう切り詰める。文字数の目算と違い実測なので、
内容に関わらず1回の送信で確実に上限内に収まる。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm

from littraceqa.azure.azure_config import OpenAISettings, build_openai_client, load_environment
from littraceqa.azure.build_azure_search_index import embed_texts_with_retry
from littraceqa.common import batched
from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.index.chunk_filter import filter_chunk_types
from littraceqa.di_pipeline.registry import register

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"
# モデルの実上限は8,192トークン。安全マージンを引いて8,000で切り詰める。
_EMBED_TOKEN_LIMIT = 8000


@register("indexer", "faiss_azure_openai")
class AzureOpenAIFAISSIndex:
    name = "faiss_azure_openai"

    def __init__(
        self,
        index_dir: str,
        batch_size: int = 256,
        workers: int = 4,
        chunk_types: list[str] | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.workers = workers
        # 索引に入れる chunk_type を絞る。None なら全部。
        self.chunk_types = chunk_types

        self._settings: OpenAISettings | None = None
        self._client = None
        self._encoding: tiktoken.Encoding | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = filter_chunk_types(chunks, self.chunk_types)
        if not self._chunks:
            raise ValueError(
                f"chunk_types={self.chunk_types} に一致するチャンクが1件もありません"
            )
        print(
            f"  {self.name}: {len(self._chunks):,} 件を埋め込み "
            f"(chunk_types={self.chunk_types or '全部'})"
        )
        embeddings = self._embed_texts([chunk.text for chunk in self._chunks])
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

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        load_environment()
        self._settings = OpenAISettings.from_env()
        self._client = build_openai_client(self._settings)
        try:
            self._encoding = tiktoken.encoding_for_model(self._settings.embedding_deployment)
        except KeyError:
            # デプロイ名がtiktokenの既知モデル名と一致しない場合のフォールバック。
            # text-embedding-3-*/ada-002は全てcl100k_base。
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate_to_token_limit(self, text: str) -> str:
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= _EMBED_TOKEN_LIMIT:
            return text
        return self._encoding.decode(tokens[:_EMBED_TOKEN_LIMIT])

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        batch_vectors = embed_texts_with_retry(self._client, self._settings, batch)
        if batch_vectors is None:
            raise RuntimeError(
                f"{self.name}: 埋め込みに失敗したバッチがあります"
                "（context-length以外の恒久的な4xxエラー、あるいはリトライ上限到達）。"
                "chunkとembeddingの対応がずれるため、索引を静かに欠損させず失敗させます。"
            )
        # ここで即座にfloat32配列へ変換する。Pythonのlist[list[float]]のまま
        # 256万件ぶん溜め込むと（floatオブジェクトのオーバーヘッドで）numpy配列の
        # 何倍ものメモリを食い、大規模ビルドがメモリ枯渇でホストごと固まる
        # （実際に82%進行時点でnlp01がハングし手動再起動が必要になった）。
        return np.asarray(batch_vectors, dtype="float32")

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_client()
        truncated = [self._truncate_to_token_limit(text) or " " for text in texts]
        batches = list(batched(truncated, self.batch_size))

        # 索引構築では256万件規模を回すので、進捗が見えないと生きているのか
        # 判断できない。クエリ1件の埋め込みでは出さない。
        if len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                arrays = list(
                    tqdm(
                        pool.map(self._embed_batch, batches),
                        total=len(batches),
                        desc=f"{self.name} embedding",
                        unit="batch",
                    )
                )
        else:
            arrays = [self._embed_batch(batch) for batch in batches]

        embeddings = np.concatenate(arrays, axis=0)
        faiss.normalize_L2(embeddings)
        return embeddings

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
