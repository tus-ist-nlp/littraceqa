"""Candidate protection.

Reserves final slots that later stages must not evict: the papers whose title
the query names outright.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.method_aliases import GENERIC_TITLE_ALIASES


_ALIAS_WORD_RE = re.compile(r"[A-Za-z0-9]+")
MAX_PROTECTED_TITLES = 4


def title_alias(result: RetrievalResult) -> str | None:
    """Return a conservative identifier-like prefix from a candidate title."""

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    title = metadata.get("title")
    if not isinstance(title, str):
        return None

    normalized_title = unicodedata.normalize("NFKC", title)
    prefix, separator, _ = normalized_title.partition(":")
    if not separator:
        return None
    alias = " ".join(prefix.split())
    words = _ALIAS_WORD_RE.findall(alias)
    if not (3 <= len(alias) <= 40) or not (1 <= len(words) <= 4):
        return None

    generic_key = re.sub(r"[^A-Z0-9]+", "", alias.upper())
    if generic_key in GENERIC_TITLE_ALIASES:
        return None

    letters = [character for character in alias if character.isalpha()]
    is_camel_case = bool(re.search(r"[a-z0-9][A-Z]", alias))
    is_all_caps = bool(letters) and all(character.isupper() for character in letters)
    is_identifier_like = (
        is_camel_case
        or is_all_caps
        or any(character.isdigit() for character in alias)
        or "-" in alias
    )
    return alias if is_identifier_like else None


def standalone_alias_position(query: str, alias: str) -> int | None:
    """Return the first standalone alias position after NFKC normalization."""

    normalized_query = unicodedata.normalize("NFKC", query)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        + re.escape(alias)
        + r"(?![A-Za-z0-9_])"
    )
    match = pattern.search(normalized_query)
    return match.start() if match is not None else None


@dataclass(frozen=True)
class ExplicitTitleGuard:
    """Reserves bounded final slots for unambiguous titles named in the query."""

    enabled: bool
    max_protected_titles: int

    def restore(
        self,
        query: str,
        candidate_pool: list[RetrievalResult],
        selected: list[RetrievalResult],
        output_k: int,
    ) -> list[RetrievalResult]:
        """Put back query-named papers that ranking dropped from the output."""

        if not self.enabled or output_k <= 0:
            return selected

        protected = self._protected_matches(query, candidate_pool)
        if not protected:
            return selected

        protected_ids = {candidate.paper_id for _, _, candidate, _ in protected}
        selected_ids = {candidate.paper_id for candidate in selected}
        missing = [
            item for item in protected if item[2].paper_id not in selected_ids
        ]
        if not missing:
            return selected

        restored = list(selected)
        for _, rank, candidate, alias in missing:
            if len(restored) >= output_k:
                replacement_index = next(
                    (
                        index
                        for index in range(len(restored) - 1, -1, -1)
                        if restored[index].paper_id not in protected_ids
                    ),
                    None,
                )
                if replacement_index is None:
                    break
                del restored[replacement_index]

            metadata = dict(candidate.metadata)
            metadata["explicit_title_guard_alias"] = alias
            metadata["pre_title_guard_rank"] = rank
            restored.append(replace(candidate, metadata=metadata))

        return restored[:output_k]

    def _protected_matches(
        self,
        query: str,
        candidate_pool: list[RetrievalResult],
    ) -> list[tuple[int, int, RetrievalResult, str]]:
        """Find unambiguous aliases that the query mentions on their own."""

        aliases: dict[str, list[tuple[int, RetrievalResult, str]]] = {}
        for rank, candidate in enumerate(candidate_pool, start=1):
            alias = title_alias(candidate)
            if alias is None:
                continue
            aliases.setdefault(alias.casefold(), []).append(
                (rank, candidate, alias)
            )

        protected: list[tuple[int, int, RetrievalResult, str]] = []
        for matches in aliases.values():
            if len({candidate.paper_id for _, candidate, _ in matches}) != 1:
                continue
            rank, candidate, alias = matches[0]
            position = standalone_alias_position(query, alias)
            if position is not None:
                protected.append((position, rank, candidate, alias))
        protected.sort(key=lambda item: (item[0], item[1], item[2].paper_id))
        return protected[: self.max_protected_titles]
