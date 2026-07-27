"""Deterministic, non-LLM soft reranking with explicit paper attributes."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import replace
from typing import Iterable

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.method_aliases import (
    GENERIC_METHOD_ALIASES,
    extract_self_owned_method_aliases,
    method_aliases_equal,
)


_VENUE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("naacl", ("naacl", "north american chapter of the association for computational linguistics")),
    ("emnlp", ("emnlp", "empirical methods in natural language processing")),
    ("acl", ("acl", "annual meeting of the association for computational linguistics")),
    ("neurips", ("neurips", "nips", "neural information processing systems")),
    ("iclr", ("iclr", "international conference on learning representations")),
    ("icml", ("icml", "international conference on machine learning")),
    ("cvpr", ("cvpr", "computer vision and pattern recognition")),
    ("iccv", ("iccv", "international conference on computer vision")),
    ("eccv", ("eccv", "european conference on computer vision")),
    ("aaai", ("aaai", "aaai conference on artificial intelligence")),
    ("ijcai", ("ijcai", "international joint conference on artificial intelligence")),
)
_VENUE_DISPLAY_NAMES = {
    canonical: "NeurIPS" if canonical == "neurips" else canonical.upper()
    for canonical, _ in _VENUE_PATTERNS
}
_YEAR_PATTERN = r"(?:19|20)\d{2}"
_PAIR_SEPARATOR_PATTERN = r"(?:\s*[-/]\s*|\s+)"
_CANDIDATE_SCOPE_YEAR_RE = re.compile(
    rf"\bamong\s+(?:the\s+)?(?P<year>{_YEAR_PATTERN})\b"
    r"(?=[^.!?()\n]{0,80}\b(?:methods?|papers?|works?|approaches?)\b)",
    re.IGNORECASE,
)
_METHOD_WORD = r"[A-Za-z][A-Za-z0-9]*(?:[.+-][A-Za-z0-9]+)*\.?"
_METHOD_PHRASE = rf"{_METHOD_WORD}(?:\s+{_METHOD_WORD}){{0,2}}"
_TARGET_METHOD_PATTERNS = (
    re.compile(
        rf"\b(?:in|from)\s+the\s+(?P<method>{_METHOD_PHRASE})\s+paper\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bdoes\s+(?P<method>{_METHOD_PHRASE})\s+"
        r"(?:with|on|using)\s+[^?!,\n]{1,80}?\s+"
        r"(?:use|achieve|report|obtain|employ|require)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bdoes\s+(?P<method>{_METHOD_PHRASE})\s+"
        r"(?:use|achieve|report|obtain|employ|require)\b",
        re.IGNORECASE,
    ),
)
_METHOD_LIST_RE = re.compile(
    r"\bfor\s+(?P<methods>[^?!.\n]{1,200})"
    r"(?=(?:\s+as\s+reported\s+in\s+their\s+respective\s+papers)?\s*[?!.])",
    re.IGNORECASE,
)
_GENERIC_METHOD_TERMS = GENERIC_METHOD_ALIASES | frozenset(
    {
        "AI",
        "BERT",
        "BLEU",
        "CIFAR10",
        "CNN",
        "COCO",
        "CVPR",
        "DNN",
        "EMNLP",
        "FID",
        "GAN",
        "GPT",
        "ICCV",
        "ICLR",
        "ICML",
        "LLM",
        "LORA",
        "LSTM",
        "MLP",
        "NAACL",
        "NEURIPS",
        "NLP",
        "OCR",
        "PDF",
        "RAG",
        "SOTA",
        "VAE",
        "VIT",
        "VLM",
        "VQA",
    }
)


def _outside_parentheses(value: str) -> str:
    """Return NFKC-normalized text with parenthetical content removed."""

    text = unicodedata.normalize("NFKC", value)
    output: list[str] = []
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
            if depth == 1:
                output.append(" ")
        elif character == ")":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    output.append(" ")
        elif depth == 0:
            output.append(character)
    return "".join(output)


def _venue_alias_pattern(alias: str) -> str:
    """Build a standalone, whitespace-tolerant regex for one venue alias."""

    words = alias.split()
    body = r"\s+".join(re.escape(word) for word in words)
    return rf"(?<![A-Za-z0-9_])(?:{body})(?![A-Za-z0-9_])"


def _deduplicate(values: Iterable[object]) -> tuple:
    """Deduplicate ordered values without changing their first occurrence."""

    return tuple(dict.fromkeys(values))


def _is_identifier_like_method(value: str) -> bool:
    """Return whether a short literal looks like a named method identifier."""

    candidate = " ".join(value.split()).strip(" ,;:")
    if not re.fullmatch(_METHOD_PHRASE, candidate):
        return False
    words = candidate.split()
    if not (1 <= len(words) <= 3) or not (2 <= len(candidate) <= 40):
        return False
    generic_key = re.sub(r"[^A-Z0-9]+", "", candidate.upper())
    if generic_key in _GENERIC_METHOD_TERMS:
        return False
    identifier_words = [
        bool(
            re.search(r"[a-z0-9][A-Z]", word) is not None
            or (
                sum(character.isalpha() for character in word) >= 2
                and all(
                    not character.isalpha() or character.isupper()
                    for character in word
                )
            )
            or any(character.isdigit() for character in word)
            or any(character in ".+-" for character in word)
        )
        for word in words
    ]
    if len(words) == 1:
        return identifier_words[0]
    if all(identifier_words):
        return True
    return bool(
        re.fullmatch(r"[A-Z][a-z]{1,4}\.", words[0])
        and all(identifier_words[1:])
    )


def extract_target_method_hints(query: str) -> tuple[str, ...]:
    """Extract method names only from narrow target-paper question forms.

    This deliberately ignores general mentions such as papers that merely cite
    a baseline. It is intended as an opt-in heuristic until a search agent
    supplies explicit ``SearchHints.methods``.
    """

    if not isinstance(query, str) or not query.strip():
        return ()

    text = _outside_parentheses(query)
    positioned: list[tuple[int, str]] = []
    for pattern in _TARGET_METHOD_PATTERNS:
        for match in pattern.finditer(text):
            method = " ".join(match.group("method").split())
            if _is_identifier_like_method(method):
                positioned.append((match.start("method"), method))

    for match in _METHOD_LIST_RE.finditer(text):
        value = re.sub(
            r"\s+as\s+reported\s+in\s+their\s+respective\s+papers\s*$",
            "",
            match.group("methods"),
            flags=re.IGNORECASE,
        )
        offset = match.start("methods")
        for part_match in re.finditer(r"[^,]+", value):
            part = re.sub(
                r"^\s*(?:and|or)\s+",
                "",
                part_match.group(),
                flags=re.IGNORECASE,
            ).strip()
            for conjunction_part in re.split(r"\s+(?:and|or)\s+", part):
                method = " ".join(conjunction_part.split()).strip(" ,;:")
                if _is_identifier_like_method(method):
                    positioned.append(
                        (offset + part_match.start(), method)
                    )

    positioned.sort(key=lambda item: (item[0], item[1]))
    return _deduplicate(method for _, method in positioned)


def extract_literal_search_hints(
    query: str,
    *,
    include_methods: bool = False,
) -> SearchHints:
    """Extract literal venue/year and optional target-method constraints.

    Parenthetical text is ignored because it commonly describes a cited
    baseline rather than the papers requested by the question. Venue-year
    pairs take precedence over unpaired venue mentions. A year without a venue
    is accepted only in the narrow ``among YEAR ... papers/methods`` form.
    Method extraction is opt-in and limited to target-paper question patterns.
    """

    if not isinstance(query, str) or not query.strip():
        return SearchHints()

    text = _outside_parentheses(query)
    methods = extract_target_method_hints(query) if include_methods else ()
    paired: list[tuple[int, str, int]] = []
    standalone: list[tuple[int, str]] = []
    for canonical, aliases in _VENUE_PATTERNS:
        display_name = _VENUE_DISPLAY_NAMES[canonical]
        for alias in aliases:
            venue_pattern = _venue_alias_pattern(alias)
            for match in re.finditer(venue_pattern, text, re.IGNORECASE):
                standalone.append((match.start(), display_name))

            venue_year_re = re.compile(
                rf"{venue_pattern}{_PAIR_SEPARATOR_PATTERN}"
                rf"(?P<year>{_YEAR_PATTERN})(?!\d)",
                re.IGNORECASE,
            )
            year_venue_re = re.compile(
                rf"(?<!\d)(?P<year>{_YEAR_PATTERN})"
                rf"{_PAIR_SEPARATOR_PATTERN}{venue_pattern}",
                re.IGNORECASE,
            )
            for pattern in (venue_year_re, year_venue_re):
                for match in pattern.finditer(text):
                    paired.append(
                        (match.start(), display_name, int(match.group("year")))
                    )

    if paired:
        paired.sort(key=lambda item: (item[0], item[1], item[2]))
        return SearchHints(
            venues=_deduplicate(venue for _, venue, _ in paired),
            years=_deduplicate(year for _, _, year in paired),
            methods=methods,
        )

    standalone.sort(key=lambda item: (item[0], item[1]))
    scope_years = (
        int(match.group("year"))
        for match in _CANDIDATE_SCOPE_YEAR_RE.finditer(text)
    )
    return SearchHints(
        venues=_deduplicate(venue for _, venue in standalone),
        years=_deduplicate(scope_years),
        methods=methods,
    )


def _as_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _canonical_venue(value: object) -> str:
    normalized = _normalize_text(value)
    # Years and ordinal edition numbers are not part of a venue identity.
    normalized = re.sub(r"\b(?:19|20)\d{2}\b", "", normalized)
    normalized = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", normalized)
    normalized = " ".join(normalized.split())
    padded = f" {normalized} "
    for canonical, aliases in _VENUE_PATTERNS:
        if any(f" {alias} " in padded for alias in aliases):
            return canonical
    return normalized


def _normalized_set(values: Iterable[object], *, venue: bool = False) -> set[str]:
    normalize = _canonical_venue if venue else _normalize_text
    return {normalized for value in values if (normalized := normalize(value))}


def _method_values(values: Iterable[object]) -> tuple[str, ...]:
    """Return non-empty method strings without discarding identifier case."""

    return tuple(
        value
        for value in values
        if isinstance(value, str) and value.strip()
    )


def _method_sets_match(
    requested: tuple[str, ...],
    available: tuple[str, ...],
) -> bool:
    """Match complete aliases conservatively after separator normalization."""

    return any(
        method_aliases_equal(wanted, actual)
        for wanted in requested
        for actual in available
    )


def _year_set(values: Iterable[object]) -> set[int]:
    years: set[int] = set()
    for value in values:
        try:
            years.add(int(value))
        except (TypeError, ValueError):
            continue
    return years


def _attribute_signal(result: RetrievalResult, hints: SearchHints) -> tuple[float, tuple[str, ...]]:
    """Return a signal in [-1, 1] and the names of explicitly matched fields."""

    metadata = result.metadata or {}
    signals: list[float] = []
    matches: list[str] = []

    requested_venues = _normalized_set(hints.venues, venue=True)
    if requested_venues:
        available = _normalized_set(_as_values(metadata.get("venue")), venue=True)
        if not available:
            signals.append(0.0)
        elif requested_venues & available:
            signals.append(1.0)
            matches.append("venue")
        else:
            signals.append(-1.0)

    requested_years = _year_set(hints.years)
    if requested_years:
        available_years = _year_set(_as_values(metadata.get("year")))
        if not available_years:
            signals.append(0.0)
        elif requested_years & available_years:
            signals.append(1.0)
            matches.append("year")
        else:
            signals.append(-1.0)

    requested_methods = _method_values(hints.methods)
    if requested_methods:
        method_values: list[object] = []
        for key in ("methods", "method_names", "method"):
            method_values.extend(_as_values(metadata.get(key)))
        title = metadata.get("title")
        if isinstance(title, str):
            # Only a distinctive title prefix is treated as a method alias.
            # The entire title is retained solely for an exact title match.
            method_values.append(title)
            method_values.extend(
                evidence.alias
                for evidence in extract_self_owned_method_aliases(title, "")
            )
        available_methods = _method_values(method_values)
        if available_methods and _method_sets_match(
            requested_methods,
            available_methods,
        ):
            signals.append(1.0)
            matches.append("method")

    if not signals:
        return 0.0, ()
    return sum(signals) / len(signals), tuple(matches)


def rerank_by_attributes(
    candidates: list[RetrievalResult],
    hints: SearchHints | None,
    *,
    attribute_weight: float = 0.25,
) -> list[RetrievalResult]:
    """Softly rerank candidates while retaining missing and mismatched metadata.

    The original rank is normalized to [0, 1], then a bounded attribute signal
    is added.  Missing metadata has a neutral signal, explicit matches receive a
    boost, and explicit venue/year mismatches receive a penalty. Method hints
    are positive-only because related gold papers need not own the method named
    in the question. If there are no hints or no candidate explicitly matches
    any hint, the original results are returned unchanged as a safe fallback.
    """

    if not math.isfinite(attribute_weight) or attribute_weight < 0:
        raise ValueError("attribute_weight must be a finite non-negative number")
    if not candidates or hints is None or hints.is_empty:
        return list(candidates)

    signals = [_attribute_signal(candidate, hints) for candidate in candidates]
    if not any(matches for _, matches in signals):
        return list(candidates)

    denominator = max(len(candidates) - 1, 1)
    ranked: list[tuple[float, int, RetrievalResult]] = []
    for original_index, (candidate, (signal, matches)) in enumerate(zip(candidates, signals)):
        base_rank_score = 1.0 - original_index / denominator
        combined_score = base_rank_score + attribute_weight * signal
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "pre_attribute_score": candidate.score,
                "pre_attribute_rank": original_index + 1,
                "attribute_signal": signal,
                "attribute_matches": list(matches),
            }
        )
        ranked.append(
            (
                combined_score,
                original_index,
                replace(candidate, score=combined_score, metadata=metadata),
            )
        )

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in ranked]
