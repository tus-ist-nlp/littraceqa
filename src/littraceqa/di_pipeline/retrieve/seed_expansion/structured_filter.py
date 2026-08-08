"""Search one modality of one venue when the question names both.

Enumeration questions such as "which CVPR 2025 papers use UniAD as a baseline
in their main comparison table" state two hard constraints that ordinary
lexical scoring cannot enforce: the venue and the part of the paper the
evidence lives in. Ranking the whole corpus by term overlap buries the answer,
because papers from every venue mention the term somewhere.

Applying both constraints turns the corpus into a shortlist. On the validation
questions "UniAD x table x CVPR 2025" leaves 19 papers and contains all nine
gold papers within the top 16, where the unrestricted lexical ranking placed
them between 7th and 68th.

The lane is deliberately narrow. Besides a venue, a year and a modality, the
question must actually be asking for an unknown set of papers. Those three
constraints alone are not enough: "For the two ICCV 2025 papers, compare their
optimization iterations" states all of them while already fixing its answer to
two named systems, and promoting a venue-wide shortlist over that question's
ranking pushed the papers it names out of the top twenty.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    is_open_set_enumeration,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.question_entities import (
    question_aliases,
)

MAX_STRUCTURED_FILTER_PAPERS = 20

# Naming a modality is not enough on its own. "figure" is also the English
# word for a number, and questions use it that way constantly ("the speedup
# figure DASH reports"). A modality only counts when the question also says
# something structural about it, or points at a numbered one.
_MODALITY_RULES: tuple[tuple[str, str], ...] = (
    (
        "figure",
        r"(?:"
        r"\bfigures?\s*\d"                       # "Figure 4"
        r"|\b(?:framework|pipeline|architecture|overview|method|teaser"
        r"|introductory|title|schematic|qualitative|main)[\s-]\w*\s*figures?\b"
        r"|\bfigures?\b[^.?!]{0,40}?\b(?:illustrat\w*|depict\w*|shown"
        r"|visuali[sz]\w*|plots?|panel|caption|diagram)\b"
        r"|\b(?:illustrat\w*|depict\w*|shown|visuali[sz]\w*|plots?)\b"
        r"[^.?!]{0,40}?\bfigures?\b"
        r"|\bdiagrams?\b|\bschematics?\b"
        r")",
    ),
    (
        "table",
        r"(?:"
        r"\btables?\s*\d"                        # "Table 2"
        r"|\b(?:comparison|main|results?|baseline|ablation|summary)"
        r"[\s-]\w*\s*tables?\b"
        r"|\btables?\b[^.?!]{0,40}?\b(?:report\w*|list\w*|compar\w*|row|column)\b"
        r")",
    ),
    (
        "equation_algorithm",
        r"(?:\bequations?\s*\d|\bequations?\b|\balgorithms?\s*\d"
        r"|\bformulat\w*\b|\bloss function\b)",
    ),
)

# Venue strings as they appear in paper metadata, with the spellings a question
# may use for them.
_VENUE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NeurIPS", ("neurips", "nips")),
    ("NAACL", ("naacl",)),
    ("EMNLP", ("emnlp",)),
    ("ACL", ("acl",)),
    ("ICLR", ("iclr",)),
    ("ICML", ("icml",)),
    ("CVPR", ("cvpr",)),
    ("ICCV", ("iccv",)),
    ("ECCV", ("eccv",)),
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class StructuredConstraint:
    """The venue, year and modality a question restricts its answer to."""

    venue: str
    year: int
    chunk_type: str


def detect_constraint(question: object) -> StructuredConstraint | None:
    """Return the constraint a question states, or ``None`` when incomplete.

    Every part is required. A modality alone is far too weak, because papers
    from every venue put the same term in a table. Enumeration is required for
    the opposite reason: when a question already names the papers it compares,
    its own ranking is the better one and must not be rebuilt.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    if not is_open_set_enumeration(question):
        return None
    normalized = unicodedata.normalize("NFKC", question)
    lowered = normalized.casefold()

    venue = next(
        (
            name
            for name, cues in _VENUE_CUES
            if any(re.search(rf"\b{cue}\b", lowered) for cue in cues)
        ),
        None,
    )
    if venue is None:
        return None

    years = [int(match.group(0)) for match in _YEAR_RE.finditer(normalized)]
    if not years:
        return None

    chunk_type = next(
        (
            name
            for name, pattern in _MODALITY_RULES
            if re.search(pattern, lowered, re.IGNORECASE)
        ),
        None,
    )
    if chunk_type is None:
        return None
    # A question naming several years is about a comparison across them, so the
    # first one is the venue's year.
    return StructuredConstraint(venue=venue, year=years[0], chunk_type=chunk_type)


@dataclass
class StructuredFilterSearch:
    """Admit papers whose stated modality and venue both match the question."""

    enabled: bool
    metadata_path: str | None
    max_papers: int
    search_depth: int
    seed_text_chars: int
    _venues: dict[str, tuple[str, int]] | None = field(
        default=None, init=False, repr=False
    )
    _unavailable: bool = field(default=False, init=False, repr=False)

    def candidates(
        self,
        question: str,
        indexers,
        *,
        exclude_paper_ids: set[str],
    ) -> list[RetrievalResult]:
        """Return matching papers ordered by their best chunk hit."""

        if not self.enabled or self.max_papers <= 0:
            return []
        constraint = detect_constraint(question)
        if constraint is None:
            return []
        venues = self._load_venues()
        if venues is None:
            return []
        index = _chunk_index(indexers)
        if index is None:
            return []

        try:
            hits = index.search(_search_terms(question), self.search_depth)
        except Exception:
            return []

        # The stated modality comes first because it is the precise signal, but
        # a paper can satisfy the question while only the body text is
        # searchable: one gold paper draws "MCTS Procedure" inside its figure,
        # which MinerU never extracts, yet its prose says MCTS. Scanning body
        # text after the modality recovers that paper without disturbing the
        # papers the modality already ranked.
        results: list[RetrievalResult] = []
        seen: set[str] = set(exclude_paper_ids)
        for wanted_type in (constraint.chunk_type, "text_span"):
            for hit in hits:
                if len(results) >= self.max_papers:
                    break
                paper_id = getattr(hit, "paper_id", None)
                if not isinstance(paper_id, str) or paper_id in seen:
                    continue
                if getattr(hit, "chunk_type", None) != wanted_type:
                    continue
                if venues.get(paper_id) != (constraint.venue, constraint.year):
                    continue
                seen.add(paper_id)
                metadata = (
                    dict(hit.metadata) if isinstance(hit.metadata, dict) else {}
                )
                metadata.update(
                    {
                        "structured_filter_venue": constraint.venue,
                        "structured_filter_year": constraint.year,
                        "structured_filter_chunk_type": constraint.chunk_type,
                        "structured_filter_matched_type": wanted_type,
                    }
                )
                results.append(
                    RetrievalResult(
                        chunk_id=hit.chunk_id,
                        paper_id=paper_id,
                        score=0.0,
                        text=hit.text[: self.seed_text_chars],
                        chunk_type=hit.chunk_type,
                        metadata=metadata,
                        source="structured_filter",
                    )
                )
        return results

    def _load_venues(self) -> dict[str, tuple[str, int]] | None:
        """Read paper_id -> (venue, year) once, and never retry a failed read."""

        if self._unavailable or not self.metadata_path:
            return None
        if self._venues is not None:
            return self._venues
        venues: dict[str, tuple[str, int]] = {}
        try:
            with Path(self.metadata_path).open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    paper_id = record.get("paper_id")
                    venue = record.get("venue")
                    year = record.get("year")
                    if (
                        isinstance(paper_id, str)
                        and isinstance(venue, str)
                        and isinstance(year, int)
                    ):
                        venues[paper_id] = (venue, year)
        except OSError:
            self._unavailable = True
            return None
        self._venues = venues
        return venues


def _search_terms(question: str) -> str:
    """Search for the entity the question is about, not its whole sentence.

    The constraint already fixes the venue and modality, so what remains is
    finding the term inside them. Scoring the full sentence dilutes that term
    across the prose around it: on the validation questions, searching the
    sentence put four of nine gold papers in the top ten, while searching the
    named entities alone put all nine there.
    """

    # The venue is already a hard filter, so repeating it only matches the
    # boilerplate every paper of that venue contains.
    venue_words = {cue for _, cues in _VENUE_CUES for cue in cues}
    aliases = [
        alias
        for alias in question_aliases(question)
        if not any(word in alias.casefold() for word in venue_words)
    ]
    return " ".join(aliases) if aliases else question


def _chunk_index(indexers):
    """Return the chunk-level searchable index, not the paper-level one."""

    return next(
        (
            indexer
            for indexer in indexers
            if callable(getattr(indexer, "search", None))
            and not callable(getattr(indexer, "get_document", None))
        ),
        None,
    )
