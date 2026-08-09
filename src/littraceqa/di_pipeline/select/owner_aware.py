"""Prefer papers that explicitly claim the methods named in a question."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path

from littraceqa.common import read_json
from littraceqa.di_pipeline.index.method_sidecar import (
    METHOD_GRAPH_SCHEMA_VERSION,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.question_entities import (
    fold_alias,
    question_aliases,
)
from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
    ordered_paper_ids,
    select_papers,
)

_PAPER_SET_RE = re.compile(
    r"\b(?:papers|works|studies|articles|publications|submissions)\b",
    re.IGNORECASE,
)
_SINGULAR_PAPER_RE = re.compile(
    r"\b(?:paper|work|study|article|publication)\b",
    re.IGNORECASE,
)
_EXPLICIT_PAPER_SUFFIX_RE = re.compile(
    r"(?:['’]s)?(?:\s+|-)(?:(?:original|method|model|framework)\s+)?paper\b",
    re.IGNORECASE,
)


def explicitly_names_paper(question: object, alias: str) -> bool:
    """Return whether ``alias`` is followed closely by the word ``paper``."""

    if not isinstance(question, str):
        return False
    text = unicodedata.normalize("NFKC", question)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        suffix = text[match.end() : match.end() + 32]
        if _EXPLICIT_PAPER_SUFFIX_RE.match(suffix):
            return True
    return False


def names_a_paper_set(question: object) -> bool:
    """Return whether the wording explicitly refers to multiple papers."""

    return isinstance(question, str) and bool(_PAPER_SET_RE.search(question))


def names_one_paper(question: object) -> bool:
    """Return whether the wording contains one singular paper reference."""

    if not isinstance(question, str) or names_a_paper_set(question):
        return False
    text = unicodedata.normalize("NFKC", question)
    return len(_SINGULAR_PAPER_RE.findall(text)) == 1


class MethodOwnerIndex:
    """Resolve question aliases to unique paper owners from a method sidecar."""

    def __init__(self, owners: Mapping[str, str]) -> None:
        self._exact = dict(owners)
        folded: dict[str, set[str]] = {}
        for alias, paper_id in owners.items():
            key = fold_alias(alias)
            if key:
                folded.setdefault(key, set()).add(paper_id)
        self._folded = {
            key: next(iter(paper_ids))
            for key, paper_ids in folded.items()
            if len(paper_ids) == 1
        }

    @classmethod
    def load(cls, path: str | Path) -> MethodOwnerIndex:
        path = Path(path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        if payload.get("schema_version") != METHOD_GRAPH_SCHEMA_VERSION:
            raise ValueError(f"{path} has an incompatible method graph schema")
        owners = payload.get("owners")
        if not isinstance(owners, dict) or any(
            not isinstance(alias, str)
            or not alias
            or not isinstance(paper_id, str)
            or not paper_id
            for alias, paper_id in owners.items()
        ):
            raise ValueError(f"{path} has an invalid owners mapping")
        return cls(owners)

    def owner_matches(
        self,
        question: object,
        candidates: Collection[str],
    ) -> tuple[tuple[str, str], ...]:
        """Return ``(alias, paper_id)`` pairs in question order.

        Exact spelling wins over folded matching. A folded spelling is used
        only when all equivalent aliases have the same owner.
        """

        candidate_ids = set(candidates)
        matches: list[tuple[str, str]] = []
        seen_papers: set[str] = set()
        for alias in question_aliases(question):
            paper_id = self._exact.get(alias)
            if paper_id is None:
                paper_id = self._folded.get(fold_alias(alias))
            if paper_id not in candidate_ids or paper_id in seen_papers:
                continue
            matches.append((alias, paper_id))
            seen_papers.add(paper_id)
        return tuple(matches)


class OwnerAwarePaperSelector:
    """Use method ownership to fill a cardinality-based paper selection.

    The selector only reorders its input candidates. Candidate generation and
    adding papers outside the retrieval pool remain retrieval responsibilities.
    """

    def __init__(
        self,
        method_owner_index_path: str | Path | None = None,
        default_count: int = 1,
        open_set_count: int = 1,
        stated_count_margin: int = 0,
        max_papers: int = 10,
        require_evidence: bool = False,
    ) -> None:
        if not method_owner_index_path:
            raise ValueError("method_owner_index_path is required")
        self.method_owner_index_path = Path(method_owner_index_path)
        self.cardinality = CardinalityPaperSelector(
            default_count=default_count,
            open_set_count=open_set_count,
            stated_count_margin=stated_count_margin,
            max_papers=max_papers,
            require_evidence=require_evidence,
        )
        self._method_owners: MethodOwnerIndex | None = None

    def _owner_index(self) -> MethodOwnerIndex:
        if self._method_owners is None:
            self._method_owners = MethodOwnerIndex.load(
                self.method_owner_index_path
            )
        return self._method_owners

    def expected_count(self, question: object) -> tuple[int, str]:
        """Return the count before candidate-specific owner matching."""

        return self.cardinality.expected_count(question)

    def select(
        self,
        question: object,
        candidates: Sequence[str] | Iterable[str],
        evidence_paper_ids: Collection[str] | None = None,
    ) -> PaperSelection:
        ordered = ordered_paper_ids(candidates)
        base_count, reason = self.cardinality.expected_count(question)
        matches = self._owner_index().owner_matches(question, ordered)
        owner_ids = [paper_id for _, paper_id in matches]
        count = base_count
        if (
            reason in {"default", "open_set_enumeration"}
            and names_a_paper_set(question)
        ):
            count = min(
                max(base_count, len(owner_ids)),
                self.cardinality.max_papers,
            )

        explicit_owners = [
            paper_id
            for alias, paper_id in matches
            if explicitly_names_paper(question, alias)
        ]
        if base_count == 1 and explicit_owners:
            preferred = explicit_owners + owner_ids + ordered
        else:
            preferred = ordered[:1] + owner_ids + ordered[1:]
        reranked = ordered_paper_ids(preferred)

        if count > 1 and names_one_paper(question) and ordered:
            confirmed = ordered_paper_ids(ordered[:1] + owner_ids)
            count = min(count, len(confirmed))

        if reranked != ordered or count > base_count:
            reason = f"{reason}+method_owner"
        if count < base_count:
            reason = f"{reason}+single_paper_guard"
        return select_papers(
            reranked,
            count=count,
            reason=reason,
            require_evidence=self.cardinality.require_evidence,
            evidence_paper_ids=evidence_paper_ids,
        )
