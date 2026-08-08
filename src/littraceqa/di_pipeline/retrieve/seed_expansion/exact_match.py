"""Admit the paper whose title names a method the question spells out.

Lexical retrieval scores a name like ``AD-GS`` as two short tokens, so the
paper that owns it can fall outside the candidate set entirely while the
question names it outright. This lane builds an alias index from paper titles
once and adds the owning paper directly.

An alias only counts when it identifies exactly one paper. Allowing two owners
already admits shared terms such as "RAG" and the metric "mAP", which on the
validation questions produced matches that were never gold; requiring a single
owner gave 21 gold papers with no false match at all.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.question_entities import (
    fold_alias,
    is_generic_alias,
    looks_like_method_name,
    question_aliases,
)

MAX_EXACT_MATCH_PAPERS = 20

_TITLE_PREFIX_SEPARATORS = (":", " - ", " — ")
_PARENTHETICAL_RE = re.compile(r"\(([^)]{2,40})\)")


def title_aliases(title: object) -> tuple[str, ...]:
    """Extract the method names a title advertises.

    Two shapes cover almost every paper: ``NAME: long description`` and a
    parenthetical abbreviation inside the description.
    """

    if not isinstance(title, str) or not title.strip():
        return ()
    normalized = " ".join(unicodedata.normalize("NFKC", title).split())

    aliases: list[str] = []
    for separator in _TITLE_PREFIX_SEPARATORS:
        prefix, found, _ = normalized.partition(separator)
        if found and prefix.strip():
            aliases.append(prefix.strip())
            break
    aliases.extend(
        match.group(1).strip() for match in _PARENTHETICAL_RE.finditer(normalized)
    )

    accepted: list[str] = []
    for alias in aliases:
        # MinerU splits capitalised names ("V ideo LL a MB"), so the despaced
        # form has to be indexed alongside the written one.
        for candidate in (alias, alias.replace(" ", "")):
            if (
                looks_like_method_name(candidate)
                and not is_generic_alias(candidate)
                and fold_alias(candidate)
                and candidate not in accepted
            ):
                accepted.append(candidate)
    return tuple(accepted)


@dataclass(frozen=True)
class MethodAliasIndex:
    """Folded method alias -> the single paper whose title advertises it."""

    papers_by_key: dict[str, str]

    @classmethod
    def from_metadata(cls, metadata_path: str | Path) -> MethodAliasIndex:
        """Build the index from the corpus metadata used for retrieval."""

        collected: dict[str, list[str]] = {}
        with Path(metadata_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper_id = record.get("paper_id")
                if not isinstance(paper_id, str) or not paper_id:
                    continue
                for alias in title_aliases(record.get("title")):
                    owners = collected.setdefault(fold_alias(alias), [])
                    if paper_id not in owners:
                        owners.append(paper_id)
        return cls(
            papers_by_key={
                key: owners[0] for key, owners in collected.items() if len(owners) == 1
            }
        )

    def lookup(self, question: object) -> tuple[tuple[str, str], ...]:
        """Return (alias, paper_id) pairs the question names, in reading order."""

        matches: list[tuple[str, str]] = []
        seen: set[str] = set()
        for alias in question_aliases(question):
            paper_id = self.papers_by_key.get(fold_alias(alias))
            if paper_id is not None and paper_id not in seen:
                seen.add(paper_id)
                matches.append((alias, paper_id))
        return tuple(matches)


@dataclass
class ExactMethodMatch:
    """Add papers the question names outright but the search never retrieved."""

    enabled: bool
    metadata_path: str | None
    max_papers: int
    seed_text_chars: int
    _index: MethodAliasIndex | None = field(default=None, init=False, repr=False)
    _unavailable: bool = field(default=False, init=False, repr=False)

    def candidates(
        self,
        question: str,
        indexers,
        *,
        exclude_paper_ids: set[str],
    ) -> list[RetrievalResult]:
        """Return named papers that are not already in the candidate set."""

        if not self.enabled or self.max_papers <= 0:
            return []
        index = self._load_index()
        if index is None:
            return []
        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return []

        results: list[RetrievalResult] = []
        for alias, paper_id in index.lookup(question):
            if len(results) >= self.max_papers:
                break
            if paper_id in exclude_paper_ids:
                continue
            try:
                document = paper_index.get_document(paper_id)
            except Exception:
                continue
            if (
                document is None
                or getattr(document, "paper_id", None) != paper_id
                or not isinstance(getattr(document, "text", None), str)
                or not document.text.strip()
            ):
                continue
            metadata = (
                dict(document.metadata) if isinstance(document.metadata, dict) else {}
            )
            metadata["exact_method_alias"] = alias
            results.append(
                RetrievalResult(
                    chunk_id=document.chunk_id,
                    paper_id=paper_id,
                    score=0.0,
                    text=document.text[: self.seed_text_chars],
                    chunk_type=document.chunk_type,
                    metadata=metadata,
                    source="exact_method_match",
                )
            )
        return results

    def _load_index(self) -> MethodAliasIndex | None:
        """Build the alias index once, and never retry a failed build."""

        if self._unavailable or not self.metadata_path:
            return None
        if self._index is not None:
            return self._index
        try:
            self._index = MethodAliasIndex.from_metadata(self.metadata_path)
        except OSError:
            self._unavailable = True
            return None
        return self._index
