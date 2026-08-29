"""Narrow results by the venue and year the question states.

Some LitTraceQA questions name their search scope outright ("Which NAACL 2025
papers ...", "Among ICML 2025 papers ..."). Five of 55 validation queries do, and
in those the gold papers satisfied the constraint 18 times out of 18 — when the
constraint is stated, filtering on it is always right.

No index changes are needed. ``RetrievalResult.metadata`` already carries venue
and year (see metadata_base in ``preprocess/mineru_chunker.py``), so dropping
results afterwards works identically for every index.

**It only fires when exactly one venue can be extracted.** Why:

* Filtering on the year alone buys little: the corpus is 91.3% 2025 and 8.7% 2024.
* Some questions target every venue explicitly ("Across all venues, among
  2025 ..."), and their gold spanned iccv / neurips / icml.
* A cited paper's venue can leak in ("Which CVPR 2025 papers cite UniAD (...,
  CVPR2023)"), so finding two or more venues means giving up.

When nothing is extracted an empty AttributeFilter is returned and the caller
takes its normal path, so questions that name no venue behave exactly as before.

**Regular expressions are enough.** Across the 55 validation queries nothing was
missed: 5 fired and 2 `all venues` cases were correctly declined. An LLM fallback
for aliases such as "NIPS" was built and removed after measuring zero gain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


# The venues present in the corpus (all 9 values of venue in
# data/paper_metadata.jsonl). Looked up lower-cased to absorb spelling variants.
_VENUES = ("NeurIPS", "ICLR", "EMNLP", "ACL", "ICML", "CVPR", "ICCV", "ECCV", "NAACL")

# Questions that explicitly target every venue; no venue must be extracted.
_ALL_VENUES_RE = re.compile(r"\ball\s+venues\b", re.I)

# Only a year adjacent to the venue (separated by space, comma or apostrophe),
# so "CVPR 2025 papers cite UniAD (..., CVPR2023)" is not dragged to the distant year.
_YEAR_RE = r"(20\d{2})"


@dataclass(frozen=True)
class AttributeFilter:
    """The attribute constraint applied to results. Empty means no constraint."""

    venue: str | None = None
    year: int | None = None

    def is_empty(self) -> bool:
        return self.venue is None and self.year is None

    def matches(self, metadata: dict | None) -> bool:
        """Does this chunk's metadata satisfy the constraint?"""
        metadata = metadata or {}
        if self.venue is not None and metadata.get("venue") != self.venue:
            return False
        if self.year is not None and metadata.get("year") != self.year:
            return False
        return True


class AttributeExtractor:
    """Builds an AttributeFilter from a question and reports its selectivity.

    Selectivity is computed from paper counts in paper_metadata.jsonl rather than
    chunk counts. Chunks per paper do not vary much by venue, so the ratio is close
    enough for working back to a fetch size.
    """

    def __init__(self, paper_metadata: str | Path):
        self._venue_by_lower = {v.lower(): v for v in _VENUES}
        self._total = 0
        self._counts: dict[tuple[str | None, int | None], int] = {}
        self._load(Path(paper_metadata))
        # Match venues on word boundaries so ACL does not match inside NAACL.
        self._venue_re = re.compile(
            r"\b(" + "|".join(re.escape(v) for v in _VENUES) + r")\b", re.I
        )

    def _load(self, path: Path) -> None:
        papers: list[tuple[str, int]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                venue = record.get("venue")
                year = record.get("year")
                papers.append((venue, int(year) if year is not None else None))

        self._total = len(papers)
        counts: dict[tuple[str | None, int | None], int] = {}
        for venue, year in papers:
            # Count (venue, None), (venue, year) and (None, year) so selectivity
            # can be looked up for a constraint missing either part.
            for key in ((venue, None), (venue, year), (None, year)):
                counts[key] = counts.get(key, 0) + 1
        self._counts = counts

    def exists(self, venue: str | None, year: int | None) -> bool:
        """Does the corpus contain at least one paper with this (venue, year)?"""
        if venue is None and year is None:
            return False
        return self._counts.get((venue, year), 0) > 0

    def extract(self, question: str) -> AttributeFilter:
        """Extract the constraint from a question; an empty AttributeFilter if there is none."""
        if not question or _ALL_VENUES_RE.search(question):
            return AttributeFilter()

        found = {self._venue_by_lower[m.group(1).lower()] for m in self._venue_re.finditer(question)}
        if len(found) != 1:
            # Neither zero venues nor two or more (a cited venue leaked in) is
            # usable, so both decline.
            return AttributeFilter()
        venue = next(iter(found))

        return AttributeFilter(venue=venue, year=self._adjacent_year(question, venue))

    def _adjacent_year(self, question: str, venue: str) -> int | None:
        """Return only a year adjacent to the venue, ignoring years elsewhere.

        In "Which CVPR 2025 papers cite UniAD (Planning-oriented ..., CVPR2023)" the
        CVPR2023 also reads as adjacent, so finding more than one means declining.
        """
        pattern = re.compile(r"\b" + re.escape(venue) + r"\b[\s,'’]*" + _YEAR_RE, re.I)
        years = {int(m.group(1)) for m in pattern.finditer(question)}
        if len(years) != 1:
            return None
        year = next(iter(years))
        # A year absent from the corpus would always filter to nothing, so the
        # year is dropped in that case.
        if self._counts.get((venue, year), 0) == 0:
            return None
        return year

    def selectivity(self, attribute_filter: AttributeFilter) -> float:
        """Fraction of papers satisfying the constraint, floored to avoid dividing by zero."""
        if attribute_filter.is_empty() or self._total == 0:
            return 1.0
        matched = self._counts.get((attribute_filter.venue, attribute_filter.year), 0)
        if matched <= 0:
            return 1.0
        return matched / self._total


def filter_results(results: list, attribute_filter: AttributeFilter) -> list:
    """Keep only the results that satisfy the constraint."""
    if attribute_filter.is_empty():
        return list(results)
    return [r for r in results if attribute_filter.matches(r.metadata)]
