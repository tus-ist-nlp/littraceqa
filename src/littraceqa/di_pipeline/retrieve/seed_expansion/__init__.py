"""Seed expansion retrieval, split into the stages a search runs through.

Importing this package registers the ``seed_expansion`` retriever wrapper.
"""

from __future__ import annotations

from littraceqa.di_pipeline.retrieve.seed_expansion.retriever import (
    SeedExpansionRetriever,
)


__all__ = ["SeedExpansionRetriever"]
