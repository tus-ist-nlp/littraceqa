"""Conservative extraction and matching of explicitly owned method aliases.

The extractor intentionally does not infer an acronym from a title.  It keeps
only names that a paper explicitly presents as its own method, model,
framework, or component.  This makes the output suitable for weak retrieval
signals without relying on gold labels, validation-only fields, or an LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache


MAX_METHOD_ALIASES = 6

# These names are common concepts, datasets, metrics, models, venues, or file
# formats.  Even when one occurs near a self-description, it is too ambiguous
# to identify a paper-level method safely.
GENERIC_METHOD_ALIASES = frozenset(
    {
        "AAAI",
        "ACC",
        "ACL",
        "ADAM",
        "AI",
        "API",
        "AUC",
        "BERT",
        "BLEU",
        "CIFAR",
        "CM",
        "CMS",
        "CNN",
        "COCO",
        "CPU",
        "CV",
        "CVPR",
        "DETR",
        "DNN",
        "DPO",
        "ECCV",
        "ELBO",
        "EM",
        "EMNLP",
        "FID",
        "FLOP",
        "FLOPS",
        "GAN",
        "GPT",
        "GPU",
        "GRU",
        "HTML",
        "ICCV",
        "ICLR",
        "ICML",
        "ID",
        "IJCAI",
        "IMAGENET",
        "IOU",
        "JSON",
        "KL",
        "LLAMA",
        "LLAVA",
        "LLM",
        "LLMS",
        "LORA",
        "LSTM",
        "LVLM",
        "LVLMS",
        "MAE",
        "MAP",
        "MCQ",
        "ML",
        "MLLM",
        "MLP",
        "MNIST",
        "MSE",
        "NAACL",
        "NAS",
        "NEURIPS",
        "NIPS",
        "NLP",
        "OCR",
        "ODE",
        "OOD",
        "PDF",
        "PDE",
        "POPE",
        "PSNR",
        "QA",
        "QWEN",
        "RAG",
        "RAM",
        "RELU",
        "RL",
        "RLHF",
        "RMSE",
        "RNN",
        "ROUGE",
        "SDE",
        "SGD",
        "SOTA",
        "SSIM",
        "SVM",
        "TPU",
        "URL",
        "VAE",
        "VICUNA",
        "VIT",
        "VLM",
        "VLMS",
        "VQA",
        "XML",
    }
)

_ALIAS_TOKEN = r"[A-Za-z][A-Za-z0-9]*(?:[-+.][A-Za-z0-9]+)*"
_ALIAS_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_REFERENCE_HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?"
    r"(?:(?:\d+(?:\.\d+)*)|[A-Z])?[.):\-]?[ \t]*"
    r"(?:references?|bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_PARENTHETICAL_ALIAS_RE = re.compile(
    r"\(\s*"
    r"(?:(?P<qualifier>(?:hereafter\s+)?"
    r"(?:referred|refered)\s+to\s+as)\s+)?"
    rf"(?P<alias>{_ALIAS_TOKEN})\s*\)"
)
_STRONG_SELF_CUE_RE = re.compile(
    r"\b(?:we|this\s+(?:paper|work))\s+"
    r"(?:propose|introduce|present|develop)\b",
    re.IGNORECASE,
)
_OWNED_NOUN_RE = re.compile(
    r"\b(?:our|the\s+proposed)\s+(?:new\s+|proposed\s+|resulting\s+)*"
    r"(?:method|model|framework|approach|algorithm|technique|strategy|"
    r"objective|loss|module)\b",
    re.IGNORECASE,
)
_NAMING_CUE_RE = re.compile(
    r"\b(?:term(?:ed)?|name[ds]?|call(?:ed)?|dubbed|"
    r"(?:refer(?:red)?|refered)\s+to)\b",
    re.IGNORECASE,
)
_EMPLOY_CUE_RE = re.compile(r"\bwe\s+(?:employ|use)\b", re.IGNORECASE)
_LONG_NAME_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:termed|named|called|dubbed|"
    r"(?:referred|refered)\s+to\s+as)\s+"
    r"|"
    r"\b(?:we|this\s+(?:paper|work))\s+"
    r"(?:propose|introduce|present|develop|employ|use)\s+"
    r")",
    re.IGNORECASE,
)
_DIRECT_ALIAS_PATTERNS = (
    re.compile(
        r"\b(?:our|the)\s+(?:new\s+|proposed\s+|resulting\s+)*"
        r"(?:method|model|framework|approach|algorithm|technique|strategy|"
        r"objective|loss|module)\s*"
        r"(?:,\s*|\bis\s+|(?:named|called|termed|dubbed)\s+)"
        rf"(?P<alias>{_ALIAS_TOKEN})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:term|name|call)\s+"
        r"(?:(?:this|our)\s+(?:method|model|framework|approach|algorithm|"
        r"technique|strategy|objective|loss|module)\s+|it\s+)?"
        r"(?:as\s+)?"
        rf"(?P<alias>{_ALIAS_TOKEN})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+refer\s+to\b[^.!?\n]{0,100}?\bas\s+"
        rf"(?P<alias>{_ALIAS_TOKEN})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:propose|introduce|present|develop)\s+"
        r"(?:(?:a|an|the)\s+)?"
        r"(?:(?:new|novel|simple|effective|lightweight)\s+){0,3}"
        r"(?:(?:method|model|framework|approach|algorithm|technique|strategy|"
        r"objective|loss|module)\s+(?:named|called|termed|dubbed)\s+)?"
        rf"(?P<alias>{_ALIAS_TOKEN})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
)
_DERIVED_ALIAS_RE = re.compile(
    rf"(?P<owned>{_ALIAS_TOKEN})\s*"
    r"\(\s*(?:hereafter\s+)?denoted\s+as\s+"
    rf"(?P<alias>{_ALIAS_TOKEN})\s*\)",
    re.IGNORECASE,
)
_OURS_ALIAS_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<alias>{_ALIAS_TOKEN})"
    r"\s*\(\s*ours\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MethodAliasEvidence:
    """One explicit statement that a paper owns a method alias.

    ``start`` and ``end`` are offsets in NFKC-normalized title or body text,
    as identified by ``source``.  ``long_name`` is retained when a
    parenthetical definition provides one; it is never synthesized.
    """

    alias: str
    source: str
    start: int
    end: int
    long_name: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable evidence record."""

        return {
            "alias": self.alias,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "long_name": self.long_name,
        }


def text_before_references(text: object) -> str:
    """Return normalized paper text before an exact references heading.

    A prose sentence containing the word "references" is not treated as a
    heading.  Missing or non-string text yields an empty string so one malformed
    paper cannot stop a batch.
    """

    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    match = _REFERENCE_HEADING_RE.search(normalized)
    return normalized[: match.start()] if match is not None else normalized


def _generic_key(alias: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", alias.upper())


def _normalize_alias(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    alias = " ".join(unicodedata.normalize("NFKC", value).split())
    alias = alias.strip(" \t\r\n,;:$")
    words = _ALIAS_WORD_RE.findall(alias)
    alnum_count = sum(len(word) for word in words)
    if not (3 <= alnum_count and 2 <= len(alias) <= 40):
        return None
    if not 1 <= len(words) <= 4:
        return None
    if _generic_key(alias) in GENERIC_METHOD_ALIASES:
        return None

    letters = [character for character in alias if character.isalpha()]
    has_mixed_identifier_case = bool(re.search(r"[a-z0-9][A-Z]", alias))
    is_all_caps = bool(letters) and all(character.isupper() for character in letters)
    contains_distinctive_word = any(
        len(word) >= 3 and word.isupper() for word in words
    )
    is_distinctive = (
        has_mixed_identifier_case
        or is_all_caps
        or contains_distinctive_word
        or any(character.isdigit() for character in alias)
    )
    return alias if is_distinctive else None


def _sentence_window(text: str, position: int, max_chars: int = 400) -> str:
    start = max(0, position - max_chars)
    for separator in (".", "?", "!", "\n"):
        boundary = text.rfind(separator, start, position)
        if boundary >= start:
            start = max(start, boundary + 1)
    return text[start:position]


def _has_self_ownership_cue(
    text: str,
    position: int,
    *,
    qualified_reference: bool,
) -> bool:
    window = _sentence_window(text, position)
    if (
        _STRONG_SELF_CUE_RE.search(window)
        or _OWNED_NOUN_RE.search(window)
        or _NAMING_CUE_RE.search(window)
    ):
        return True
    # "We use X (ABC)" often names a baseline.  It is accepted only when the
    # parenthesis itself explicitly says that ABC is the name used in this paper.
    return qualified_reference and _EMPLOY_CUE_RE.search(window) is not None


def _clean_long_name(value: str) -> str | None:
    value = re.sub(
        r"\\(?:textit|emph|mathrm|mathbf)\s*\{([^{}]*)\}",
        r"\1",
        value,
    )
    value = value.replace("$", "").replace("{", "").replace("}", "")
    value = " ".join(value.split()).strip(" \t\r\n,;:-")
    value = re.sub(
        r"^(?:(?:a|an|the|new|novel|simple|effective|lightweight|both)\s+)+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^(?:training\s+)?(?:method|model|framework|approach|algorithm|"
        r"technique|strategy|objective|loss|module)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = value.strip(" \t\r\n,;:-")
    if not value or len(value) > 120:
        return None
    return value


def _long_name_before_parenthesis(text: str, position: int) -> str | None:
    window = _sentence_window(text, position)
    cues = list(_LONG_NAME_CUE_RE.finditer(window))
    if cues:
        candidate = window[cues[-1].end() :]
    else:
        candidate = window
    # A sentence may define several aliases in a coordinated list.  Only the
    # phrase after the preceding definition belongs to the current alias.
    if ")" in candidate:
        candidate = candidate.rsplit(")", 1)[-1]
        candidate = re.sub(r"^\s*(?:,?\s*and|,)\s+", "", candidate)
    return _clean_long_name(candidate)


def _title_prefix_evidence(title: str) -> MethodAliasEvidence | None:
    prefix, separator, _ = title.partition(":")
    if not separator:
        return None
    alias = _normalize_alias(prefix)
    if alias is None:
        return None
    start = title.find(prefix)
    return MethodAliasEvidence(
        alias=alias,
        source="title_prefix",
        start=start,
        end=start + len(prefix),
    )


def extract_self_owned_method_aliases(
    title: object,
    text: object,
    *,
    max_aliases: int = MAX_METHOD_ALIASES,
) -> tuple[MethodAliasEvidence, ...]:
    """Extract at most six deterministic, explicitly self-owned method names.

    Sources are considered in this order: a distinctive title prefix before a
    colon, parenthetical self-definitions, and direct naming statements.  Body
    scanning continues through appendices but stops at the first exact
    references or bibliography heading.
    """

    if isinstance(max_aliases, bool) or not isinstance(max_aliases, int):
        raise TypeError("max_aliases must be an integer")
    if not 1 <= max_aliases <= MAX_METHOD_ALIASES:
        raise ValueError(
            f"max_aliases must be between 1 and {MAX_METHOD_ALIASES}"
        )

    normalized_title = (
        unicodedata.normalize("NFKC", title) if isinstance(title, str) else ""
    )
    body = text_before_references(text)
    candidates: list[MethodAliasEvidence] = []

    title_evidence = _title_prefix_evidence(normalized_title)
    if title_evidence is not None:
        candidates.append(title_evidence)

    body_candidates: list[tuple[int, int, MethodAliasEvidence]] = []
    for match in _PARENTHETICAL_ALIAS_RE.finditer(body):
        qualifier = match.group("qualifier")
        if not _has_self_ownership_cue(
            body,
            match.start(),
            qualified_reference=qualifier is not None,
        ):
            continue
        alias = _normalize_alias(match.group("alias"))
        if alias is None:
            continue
        body_candidates.append(
            (
                match.start("alias"),
                0,
                MethodAliasEvidence(
                    alias=alias,
                    source="parenthetical_definition",
                    start=match.start("alias"),
                    end=match.end("alias"),
                    long_name=_long_name_before_parenthesis(
                        body,
                        match.start(),
                    ),
                ),
            )
        )

    for pattern_index, pattern in enumerate(_DIRECT_ALIAS_PATTERNS, start=1):
        for match in pattern.finditer(body):
            alias = _normalize_alias(match.group("alias"))
            if alias is None:
                continue
            # In "we propose LongName (ABC)", ABC is the explicit alias and
            # LongName is its retained expansion, not a second alias.
            next_non_space = match.end()
            while next_non_space < len(body) and body[next_non_space].isspace():
                next_non_space += 1
            parenthetical = _PARENTHETICAL_ALIAS_RE.match(
                body,
                next_non_space,
            )
            if (
                parenthetical is not None
                and _normalize_alias(parenthetical.group("alias")) is not None
            ):
                continue
            body_candidates.append(
                (
                    match.start("alias"),
                    pattern_index,
                    MethodAliasEvidence(
                        alias=alias,
                        source="naming_statement",
                        start=match.start("alias"),
                        end=match.end("alias"),
                    ),
                )
            )

    for match in _OURS_ALIAS_RE.finditer(body):
        alias = _normalize_alias(match.group("alias"))
        if alias is None:
            continue
        body_candidates.append(
            (
                match.start("alias"),
                len(_DIRECT_ALIAS_PATTERNS) + 1,
                MethodAliasEvidence(
                    alias=alias,
                    source="ours_label",
                    start=match.start("alias"),
                    end=match.end("alias"),
                ),
            )
        )

    owned_aliases = {
        evidence.alias for evidence in candidates
    } | {
        evidence.alias for _, _, evidence in body_candidates
    }
    for match in _DERIVED_ALIAS_RE.finditer(body):
        # A secondary name is accepted only when the same paper has already
        # made an explicit ownership claim for the name immediately preceding
        # the parenthesis.  This avoids claiming aliases from baseline prose.
        if match.group("owned") not in owned_aliases:
            continue
        alias = _normalize_alias(match.group("alias"))
        if alias is None:
            continue
        body_candidates.append(
                (
                    match.start("alias"),
                    len(_DIRECT_ALIAS_PATTERNS) + 2,
                    MethodAliasEvidence(
                        alias=alias,
                        source="derived_naming_statement",
                    start=match.start("alias"),
                    end=match.end("alias"),
                ),
            )
        )

    body_candidates.sort(key=lambda item: (item[0], item[1], item[2].alias))
    candidates.extend(evidence for _, _, evidence in body_candidates)

    selected: list[MethodAliasEvidence] = []
    seen: set[str] = set()
    for evidence in candidates:
        if evidence.alias in seen:
            continue
        seen.add(evidence.alias)
        selected.append(evidence)
        if len(selected) >= max_aliases:
            break
    return tuple(selected)


@lru_cache(maxsize=4096)
def _standalone_pattern(alias: str) -> re.Pattern[str]:
    parts = alias.split()
    body = r"\s+".join(re.escape(part) for part in parts)
    return re.compile(
        rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])"
    )


def standalone_exact_alias_positions(
    text: object,
    alias: object,
    *,
    exclude_references: bool = True,
) -> tuple[int, ...]:
    """Return standalone, case-sensitive alias offsets in normalized text."""

    if not isinstance(exclude_references, bool):
        raise TypeError("exclude_references must be a boolean")
    if not isinstance(text, str) or not isinstance(alias, str):
        return ()

    normalized_alias = " ".join(
        unicodedata.normalize("NFKC", alias).split()
    )
    if not normalized_alias:
        return ()
    normalized_text = (
        text_before_references(text)
        if exclude_references
        else unicodedata.normalize("NFKC", text)
    )
    pattern = _standalone_pattern(normalized_alias)
    return tuple(match.start() for match in pattern.finditer(normalized_text))


def has_standalone_exact_alias(
    text: object,
    alias: object,
    *,
    exclude_references: bool = True,
) -> bool:
    """Return whether an exact-case standalone alias occurs in paper text."""

    return bool(
        standalone_exact_alias_positions(
            text,
            alias,
            exclude_references=exclude_references,
        )
    )


def method_aliases_equal(left: object, right: object) -> bool:
    """Compare method aliases after normalizing separators, never substrings.

    NFKC normalization and alphanumeric tokenization make punctuation-only
    variants such as ``D-FINE`` and ``D FINE`` equivalent. Mixed-case
    identifiers remain case-sensitive so distinct names such as ``sCT`` and
    ``SCT`` cannot collapse into one another. Other aliases are compared
    case-insensitively.
    """

    if not isinstance(left, str) or not isinstance(right, str):
        return False

    def normalized(value: str, *, preserve_case: bool) -> str:
        value = unicodedata.normalize("NFKC", value)
        if not preserve_case:
            value = value.casefold()
        pattern = r"[A-Za-z0-9]+" if preserve_case else r"[a-z0-9]+"
        return " ".join(re.findall(pattern, value))

    left_case = normalized(left, preserve_case=True)
    right_case = normalized(right, preserve_case=True)
    if not left_case or not right_case:
        return False

    def is_mixed_case(value: str) -> bool:
        letters = [character for character in value if character.isalpha()]
        return (
            any(character.islower() for character in letters)
            and any(character.isupper() for character in letters)
        )

    if is_mixed_case(left_case) or is_mixed_case(right_case):
        return left_case == right_case
    return normalized(left, preserve_case=False) == normalized(
        right,
        preserve_case=False,
    )


__all__ = [
    "GENERIC_METHOD_ALIASES",
    "MAX_METHOD_ALIASES",
    "MethodAliasEvidence",
    "extract_self_owned_method_aliases",
    "has_standalone_exact_alias",
    "method_aliases_equal",
    "standalone_exact_alias_positions",
    "text_before_references",
]
