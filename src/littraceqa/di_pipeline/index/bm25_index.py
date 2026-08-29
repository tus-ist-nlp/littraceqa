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

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult

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
        """BM25 のスコアリング・トークナイズ系パラメータ。実効値は
        `pipeline.build_indexers()` にあり、そこでは全部既定のまま使っている。

        - k1 / b / method: bm25s.BM25 のスコアリングパラメータ。**既定値のまま使っている**
          （validation の recall で振って既定を超えなかった）。build 時にスコア行列へ
          焼き込まれるので、変えたら索引を作り直すこと。
        - stopwords / stemmer: bm25s.tokenize のトークナイズ系。build と search で
          同じ設定を使う必要があるため、この2つはインスタンスに保持して両方で使う
          （構築時と検索時で同じインスタンス生成コード＝`build_indexers()` を通る）。
        - stemmer は言語名（例: "english"）を渡すと PyStemmer で語幹化する。
          省略時は語幹化しない。PyStemmer 未導入の環境で stemmer を指定すると
          build/load 時に ImportError になる。
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
        import Stemmer  # PyStemmer。指定時のみ必要にして通常構成の依存を増やさない。

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
