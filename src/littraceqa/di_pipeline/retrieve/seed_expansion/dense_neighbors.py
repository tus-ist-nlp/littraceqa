"""Shared guards for reading paper-embedding neighbour results.

The three dense lanes consume the same neighbour payloads, so the checks that
reject unusable scores and empty documents live here instead of being repeated
per lane.
"""

from __future__ import annotations

import math
from numbers import Real

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult


def finite_similarity(result) -> float | None:
    """Return the neighbour score as a float, or ``None`` when unusable."""

    similarity = getattr(result, "score", None)
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, Real)
        or not math.isfinite(similarity)
    ):
        return None
    return float(similarity)


def usable_document(paper_index, paper_id: str) -> Chunk | None:
    """Return a non-empty paper document, or ``None`` when unusable."""

    document = paper_index.get_document(paper_id)
    if (
        not isinstance(document, Chunk)
        or document.paper_id != paper_id
        or not isinstance(document.text, str)
        or not document.text.strip()
    ):
        return None
    return document


def leading_seed_ids(results: list[RetrievalResult], seed_k: int) -> list[str]:
    """Collect the first distinct paper IDs to use as dense seeds."""

    seed_ids: list[str] = []
    for result in results[:seed_k]:
        paper_id = result.paper_id
        if not isinstance(paper_id, str) or not paper_id or paper_id in seed_ids:
            continue
        seed_ids.append(paper_id)
    return seed_ids
