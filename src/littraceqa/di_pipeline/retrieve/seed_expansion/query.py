"""Query and hint extraction.

Turns the incoming question into the query strings and the ``SearchHints`` that
each retrieval lane runs with.  Method hints are deliberately withheld from the
retrieval lanes and only applied during final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.attributes import (
    extract_literal_search_hints,
)


def paper_context(seed: RetrievalResult) -> str | None:
    """Return non-empty paper-level expansion text when available."""

    metadata = seed.metadata if isinstance(seed.metadata, dict) else {}
    context = metadata.get("paper_rank_expansion_text")
    if isinstance(context, str) and context.strip():
        return context
    return None


def without_method_hints(hints: SearchHints | None) -> SearchHints | None:
    """Keep stable paper attributes while deferring methods to final ranking."""

    if hints is None:
        return None
    return SearchHints(
        venues=hints.venues,
        years=hints.years,
    )


def _normalized_title(seed: RetrievalResult) -> str:
    metadata = seed.metadata if isinstance(seed.metadata, dict) else {}
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return " ".join(title.split())
    return ""


def _starts_with_title(context: str, title: str) -> bool:
    """Return whether normalized paper context begins with its title."""

    folded_context = context.casefold()
    folded_title = title.casefold()
    return folded_context == folded_title or folded_context.startswith(
        f"{folded_title} "
    )


@dataclass(frozen=True)
class QueryPreparation:
    """Builds the expansion queries and resolves the hints for one search."""

    seed_text_chars: int
    literal_attribute_hints: bool
    literal_method_hints: bool

    def effective_hints(
        self,
        query: str,
        hints: SearchHints | None,
    ) -> SearchHints | None:
        """Prefer caller hints and fall back to literal extraction when opted in."""

        if hints is not None or not self.literal_attribute_hints:
            return hints
        return extract_literal_search_hints(
            query,
            include_methods=self.literal_method_hints,
        )

    def expanded_query(
        self,
        query: str,
        seed: RetrievalResult,
    ) -> str | None:
        """Expand with paper-level context, falling back to the legacy form."""

        context = paper_context(seed)
        if context is None:
            return self.legacy_expanded_query(query, seed)

        normalized_title = _normalized_title(seed)
        representative_text = " ".join(context.split())[
            : self.seed_text_chars
        ].strip()
        include_title = bool(
            normalized_title
            and not _starts_with_title(representative_text, normalized_title)
        )

        return " ".join(
            part
            for part in (
                " ".join(query.split()),
                normalized_title if include_title else "",
                representative_text,
            )
            if part
        )

    def legacy_expanded_query(
        self,
        query: str,
        seed: RetrievalResult,
    ) -> str | None:
        """Build the original title-plus-query-matched-chunk expansion."""

        normalized_title = _normalized_title(seed)
        if not normalized_title or not isinstance(seed.text, str):
            return None

        representative_text = " ".join(seed.text.split())[
            : self.seed_text_chars
        ].strip()
        if not representative_text:
            return None
        return " ".join(
            (
                " ".join(query.split()),
                normalized_title,
                representative_text,
            )
        )
