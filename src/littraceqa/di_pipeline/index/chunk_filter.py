"""Which chunk_types an index takes in.

The same set of chunks is handed to every index; this lets each one keep the
granularity its model was built for. **A model taken off its design granularity
does not perform:**

* SPECTER2's proximity adapter was trained on title+abstract — a **whole-paper**
  model. Embedding body fragments, tables or equations separately takes the input
  off that distribution, so it is given `title_abstract` only.
* BM25 matches on vocabulary, so granularity does not matter to it. LitTraceQA's
  questions are full of method, dataset and metric names (PointLoRA, ModelNet40,
  ECM-XL), which is exactly where exact rare-term matching wins — every chunk.
* Passage-level dense vectors (Qwen3 and the like) go on body chunks.
"""

from __future__ import annotations

from collections.abc import Iterable

from littraceqa.di_pipeline.contracts import Chunk

# The chunk_type values MinerUChunker actually emits.
ALL_CHUNK_TYPES = (
    "title_abstract",
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
)


def filter_chunk_types(
    chunks: Iterable[Chunk], chunk_types: list[str] | None
) -> list[Chunk]:
    """Keep only the chunks whose chunk_type is listed; None keeps everything."""
    if not chunk_types:
        return list(chunks)

    unknown = set(chunk_types) - set(ALL_CHUNK_TYPES)
    if unknown:
        raise ValueError(
            f"unknown chunk_types: {sorted(unknown)} "
            f"(expected a subset of {list(ALL_CHUNK_TYPES)})"
        )

    wanted = set(chunk_types)
    return [chunk for chunk in chunks if chunk.chunk_type in wanted]
