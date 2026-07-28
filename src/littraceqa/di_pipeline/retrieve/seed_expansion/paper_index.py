"""Shared lookup for the paper-level index the expansion stages depend on."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


PAPER_INDEX_NAME = "paper_bm25"


def find_paper_index(
    indexers: Iterable[Any],
    *required_methods: str,
) -> Any | None:
    """Return the first paper index exposing every required method."""

    return next(
        (
            indexer
            for indexer in indexers
            if getattr(indexer, "name", None) == PAPER_INDEX_NAME
            and all(
                callable(getattr(indexer, method, None))
                for method in required_methods
            )
        ),
        None,
    )
