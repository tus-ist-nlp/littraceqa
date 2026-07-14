"""Paper-level BM25 index built from deterministic aggregates of common chunks."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator

from litqa.contracts import Chunk, RetrievalResult
from litqa.index.bm25_index import BM25Index
from litqa.registry import register


_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)|[A-Z])?[.):\-]?\s*"
    r"(?:references?|bibliography)\s*$",
    re.IGNORECASE,
)


def _body_text(chunk: Chunk) -> str:
    """Remove the repeated paper prefix used by the common Chunk contract."""
    _, separator, body = chunk.text.partition("\n")
    return (body if separator else chunk.text).strip()


def _paper_chunk(
    paper_id: str,
    chunks: list[Chunk],
    exclude_references: bool,
) -> Chunk:
    first = chunks[0]
    metadata = dict(first.metadata)
    title = str(metadata.get("title") or "").strip()
    parts: list[str] = []
    heading = ""
    inside_references = False

    for chunk in chunks:
        body = _body_text(chunk)
        if not body:
            continue
        if chunk.chunk_type == "title_abstract":
            parts.append(body)
            continue

        if chunk.metadata.get("text_level") is not None:
            heading = body
            inside_references = bool(_REFERENCE_HEADING_RE.fullmatch(heading))
            if not (exclude_references and inside_references):
                parts.append(body)
            continue

        if exclude_references and inside_references:
            continue
        parts.append(f"{heading}\n{body}" if heading else body)

    text = "\n".join(part for part in (title, *parts) if part)
    metadata.update(
        {
            "paper_aggregation": "whole_paper",
            "source_chunk_count": len(chunks),
            "indexed_char_count": len(text),
        }
    )
    return Chunk(
        chunk_id=f"{paper_id}#paper",
        paper_id=paper_id,
        text=text,
        chunk_type="paper",
        metadata=metadata,
    )


def aggregate_papers(
    chunks: Iterable[Chunk],
    exclude_references: bool = True,
) -> Iterator[Chunk]:
    """Yield one paper document at a time from paper-contiguous input chunks."""
    current_paper_id: str | None = None
    current_chunks: list[Chunk] = []
    completed: set[str] = set()

    for chunk in chunks:
        if chunk.paper_id != current_paper_id:
            if current_paper_id is not None:
                yield _paper_chunk(
                    current_paper_id,
                    current_chunks,
                    exclude_references=exclude_references,
                )
                completed.add(current_paper_id)
            if chunk.paper_id in completed:
                raise ValueError(
                    "paper_bm25 requires chunks for each paper to be contiguous: "
                    f"{chunk.paper_id} appeared more than once"
                )
            current_paper_id = chunk.paper_id
            current_chunks = []
        current_chunks.append(chunk)

    if current_paper_id is not None:
        yield _paper_chunk(
            current_paper_id,
            current_chunks,
            exclude_references=exclude_references,
        )


@register("indexer", "paper_bm25")
class PaperBM25Index:
    """Search one aggregated BM25 document per paper as a coarse retrieval stage."""

    name = "paper_bm25"

    def __init__(
        self,
        index_dir: str,
        exclude_references: bool = True,
        result_text_chars: int = 2000,
    ):
        if result_text_chars <= 0:
            raise ValueError("result_text_chars must be positive")
        self.exclude_references = exclude_references
        self.result_text_chars = result_text_chars
        self._delegate = BM25Index(index_dir=index_dir)

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._delegate.build(
            aggregate_papers(chunks, exclude_references=self.exclude_references)
        )

    def load(self) -> None:
        self._delegate.load()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        results = self._delegate.search(query, top_k)
        return [
            dataclasses.replace(
                result,
                text=result.text[: self.result_text_chars],
                source=self.name,
            )
            for result in results
        ]
