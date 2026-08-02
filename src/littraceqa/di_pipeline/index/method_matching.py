"""Decide when a text mention really refers to a named method.

A bare alias match is not enough: short or generic aliases collide with
ordinary words, so a mention only counts when the surrounding text carries
method-like context. These rules are shared by owner lookup and by the
method-graph sidecar that persists the result.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence

from littraceqa.di_pipeline.contracts import Chunk


METHOD_FIRST_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
METHOD_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9_])"
)
METHOD_CONTEXT_CHARS = 200
METHOD_CONTEXT_WORD_RE = re.compile(r"[a-z][a-z0-9]*")
GENERIC_METHOD_CONTEXT_WORDS = frozenset(
    {
        "algorithm",
        "algorithms",
        "approach",
        "approaches",
        "based",
        "effective",
        "efficient",
        "framework",
        "frameworks",
        "general",
        "learning",
        "loss",
        "losses",
        "method",
        "methods",
        "model",
        "models",
        "module",
        "modules",
        "new",
        "novel",
        "objective",
        "objectives",
        "paper",
        "proposed",
        "simple",
        "strategy",
        "strategies",
        "system",
        "systems",
        "task",
        "tasks",
        "technique",
        "techniques",
        "training",
        "unified",
        "using",
        "with",
        "without",
        "work",
    }
)


def require_positive_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def generic_alias_key(alias: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", alias.upper())


def first_method_token(alias: str) -> str | None:
    match = METHOD_FIRST_TOKEN_RE.search(alias)
    return match.group(0) if match is not None else None


def standalone_alias_pattern(alias: str) -> re.Pattern[str]:
    body = r"\s+".join(
        re.escape(part) for part in alias.split()
    )
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])")


def owner_literal_alias_pattern(alias: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in alias.split())
    return re.compile(
        rf"(?<![A-Za-z0-9_+./-]){body}(?![A-Za-z0-9_])"
    )


def normalized_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def case_preserving_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(re.findall(r"[A-Za-z0-9]+", normalized))


def is_mixed_case_alias(alias: str) -> bool:
    letters = [character for character in alias if character.isalpha()]
    return (
        any(character.islower() for character in letters)
        and any(character.isupper() for character in letters)
    )


def alias_alnum_length(alias: str) -> int:
    return sum(character.isalnum() for character in alias)


def distinctive_context_words(value: object) -> frozenset[str]:
    if not isinstance(value, str):
        return frozenset()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return frozenset(
        word
        for word in METHOD_CONTEXT_WORD_RE.findall(normalized)
        if len(word) >= 4 and word not in GENERIC_METHOD_CONTEXT_WORDS
    )


def mention_has_method_context(
    text: str,
    start: int,
    end: int,
    context_pattern: re.Pattern[str] | None,
    required_word_count: int,
) -> bool:
    if context_pattern is None:
        return True
    observed: set[str] = set()
    for match in context_pattern.finditer(
        text,
        max(0, start - METHOD_CONTEXT_CHARS),
        min(len(text), end + METHOD_CONTEXT_CHARS),
    ):
        observed.add(match.group(0).casefold())
        if len(observed) >= required_word_count:
            return True
    return False


def method_context_pattern(
    context_words: frozenset[str],
) -> re.Pattern[str] | None:
    if not context_words:
        return None
    alternatives = "|".join(
        re.escape(word)
        for word in sorted(context_words, key=lambda word: (-len(word), word))
    )
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def method_corpus_signature(documents: Sequence[Chunk]) -> str:
    """Return a deterministic signature for validating the graph sidecar."""
    digest = hashlib.sha256()
    for document in documents:
        metadata = document.metadata
        record = {
            "paper_id": document.paper_id,
            "chunk_id": document.chunk_id,
            "method_names": metadata.get("method_names") or [],
            "method_alias_evidence": (
                metadata.get("method_alias_evidence") or []
            ),
        }
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
