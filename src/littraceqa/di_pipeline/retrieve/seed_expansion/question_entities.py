"""Pull the identifier-like names a question mentions out of its prose.

A method name is what a structured search should look for: the sentence around
it is boilerplate that every paper of the venue shares. Folding case, hyphens
and spaces lets ``AD-GS``, ``ad-gs`` and ``adgs`` agree, and the identifier
shape test is what stops ordinary words from folding into the same space.
"""

from __future__ import annotations

import re
import unicodedata

from littraceqa.di_pipeline.retrieve.method_aliases import (
    GENERIC_METHOD_ALIASES,
    GENERIC_TITLE_ALIASES,
)


_CANDIDATE_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z0-9]+)*"
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def fold_alias(value: object) -> str:
    """Reduce an alias to the key used for matching, or '' when unusable.

    Case, hyphens, underscores, dots and spaces all disappear, because papers
    and questions disagree about them constantly (``AD-GS`` vs ``AD GS``).
    """

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def looks_like_method_name(alias: str) -> bool:
    """Return whether an alias is distinctive enough to match on.

    Plain lowercase words are rejected: folding removes the very features that
    separate a method name from prose, so the surface form has to carry them.
    """

    words = _WORD_RE.findall(alias)
    if not words or len(words) > 4:
        return False
    if not 2 <= len(alias) <= 40:
        return False
    if sum(len(word) for word in words) < 3:
        return False
    if not any(character.isalpha() for character in alias):
        return False
    # At least one word must carry identifier shape on its own. Requiring it
    # per word is what separates "GSPlat" and "AD-GS" from ordinary Title Case
    # such as "Anomaly Detection", whose capitals are just English.
    return any(_word_is_identifier(word) for word in words)


def _word_is_identifier(word: str) -> bool:
    return (
        bool(re.search(r"[A-Z]", word[1:]))
        or (len(word) >= 2 and word.isupper())
        or any(character.isdigit() for character in word)
    )


def is_generic_alias(alias: str) -> bool:
    key = " ".join(alias.split()).casefold()
    return key in GENERIC_METHOD_ALIASES or key in GENERIC_TITLE_ALIASES


def _is_lowercased_identifier(token: str) -> bool:
    """Return whether a token is a method name someone wrote in lower case.

    Questions occasionally spell ``AD-GS`` as ``ad-gs``. Case alone cannot be
    required then, so the internal punctuation or digit has to carry the
    identifier shape instead; a plain word such as "accuracy" still fails.
    """

    if len(fold_alias(token)) < 4:
        return False
    return any(character in "-_+." for character in token) or any(
        character.isdigit() for character in token
    )


def question_aliases(question: object) -> tuple[str, ...]:
    """Extract identifier-like names a question mentions, in reading order."""

    if not isinstance(question, str):
        return ()
    normalized = unicodedata.normalize("NFKC", question)
    seen: set[str] = set()
    found: list[str] = []
    for match in _CANDIDATE_TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("-_+.")
        if is_generic_alias(token):
            continue
        if not (
            looks_like_method_name(token) or _is_lowercased_identifier(token)
        ):
            continue
        key = fold_alias(token)
        if not key or key in seen:
            continue
        seen.add(key)
        found.append(token)
    return tuple(found)
