"""BM25 による疎検索インデックス（論文単位版）。

通常の bm25_index.py は chunk 単位でドキュメントを作るが、こちらは Chunk 列を
paper_id ごとにまとめ、論文1本を1ドキュメントとして bm25s の BM25 インデックスを
構築する。chunk単位版との比較用 ablation indexer。

各 Chunk の text は "[{venue} {year}] {title}\n{body}" という形式で、全チャンクに
同じ prefix が付いている。そのまま連結すると venue/title の語が chunk 数だけ水増し
されて BM25 スコアが歪むため、prefix は論文ごとに1回だけ残し、body だけを連結する。

論文単位の検索結果は chunk_id を持たない（chunk_id は "{paper_id}#paper" という
擬似IDを充てる）ため、ReadingAgent の根拠チャンク特定には使わない想定。
gold paper（正解論文ID）の特定精度だけを見る用途に限定する。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import bm25s

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.registry import register

_PAPERS_FILENAME = "papers.jsonl"


def _paper_prefix(metadata: dict) -> str:
    return f"[{metadata['venue']} {metadata['year']}] {metadata['title']}\n"


def _build_paper_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """paper_id ごとに Chunk をまとめ、prefix を1回だけ残した「論文1本 = 1 Chunk」に潰す。"""
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


@register("indexer", "bm25s_paper")
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
