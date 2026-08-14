"""Reading-only LitTraceQA agent over the fixed paper ranking from PR #7.

The retrieval result is already fixed for each query.  This module never
searches, reranks, decomposes a query for retrieval, or reads development-only
labels.  It performs exactly two semantic stages:

1. Pair one observable query with one candidate paper hydrated from the MinerU
   corpus and ask an LLM whether the paper is answer-relevant and whether the
   supplied context contains usable answer evidence, citing exact chunk IDs.
2. Keep every answer-relevant paper for paper scoring, but give Stage 2 only the
   original chunks whose paper passed both judgments and whose IDs validated.

Stage 1 first applies a narrow, candidate-set-unique canonical-owner gate for
paper-local objects. A decisive wrong owner is checkpointed without AOAI; every
other query-paper pair sends exactly one selected paper context. Long papers and
image-heavy papers are compacted deterministically before that single request;
they are never partitioned into additional semantic calls. A failed API call,
invalid JSON, or invented chunk ID raises an error; it is never converted into
an ``irrelevant`` decision.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import math
import re
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Required, TypedDict, cast

from littraceqa.answer_derivation import (
    ANSWER_DERIVATION_VERSION,
    DerivationValidationError,
    citation_author_filter,
    citation_identity_key,
    explicit_singleton_eligibility_clause,
    has_explicit_singleton_eligibility_filter,
    is_aggregate_citation_count_query,
    is_axis_extent_lookup_query,
    requires_extremum_operation,
    validate_answer_semantics,
    validate_citation_count_items,
)
from littraceqa.candidate_handoff import (
    CandidatePaper,
    require_production_query,
)
from littraceqa.chunk_store import ChunkStore, Record
from littraceqa.citation_locator import (
    CITATION_LOCATOR_VERSION,
    bibliography_entry_supports_value,
    infer_citation_locator_overrides,
    requested_ordinal_bibliography_entries,
    requested_reference_ordinal,
)
from littraceqa.corpus_preflight import requires_visual_image
from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.mineru_record import (
    MAX_AOAI_IMAGES_PER_REQUEST,
    QUERY_AWARE_OBJECT_LOCATOR_VERSION,
    SPLIT_CAPTION_TABLE_LOCATOR_VERSION,
    coarse_locator,
    query_aware_prediction_locator,
    readable_image_path,
    record_source_type,
    recover_split_caption_table_records,
    submission_evidence_eligible,
)
from littraceqa.pairwise_prompts import (
    ANSWER_PROMPT_VERSION,
    FIXED_SELECTED_ANSWER_PROMPT_VERSION,
    JUDGMENT_PROMPT_VERSION,
    JUDGMENT_QUESTION_TYPE_VERSION,
    SELECTED_EVIDENCE_PROMPT_VERSION,
    answer_response_shape,
    example_manifest,
    judgment_question_type,
    render_answer_prompt,
    render_judgment_prompt,
    render_selected_evidence_prompt,
    requires_coordinated_metric_table_context,
    requires_explicit_visual_mention,
    requires_scaling_eligibility_output,
    selected_evidence_example_manifest,
)
from littraceqa.query_requirements import (
    QUERY_REQUIREMENTS_VERSION,
    explicit_table_row_items,
    table_row_key_texts,
    unaccounted_explicit_table_items,
)

PAPER_CONTEXT_SELECTOR_VERSION = "query-lexical-v5-scoped-metric-tables"
NAMED_OWNER_RESOLVER_VERSION = "named-owner-v2-grammatical-local-object-only"
ANSWER_IMAGE_HANDOFF_VERSION = "answer-purpose-image-first-v2-mixed-visual-text"
SUBMISSION_PAPER_SELECTION_VERSION = "answer-support-only-v1"
MAX_ANSWER_REPAIR_ATTEMPTS = 5
PAIRWISE_CANDIDATE_POLICY = "pairwise_candidates"
FIXED_SELECTED_PAPER_POLICY = "fixed_selected"
PAPER_SET_POLICIES = frozenset(
    {PAIRWISE_CANDIDATE_POLICY, FIXED_SELECTED_PAPER_POLICY}
)
FIXED_SELECTED_CHECKPOINT_KIND = "fixed_selected_evidence"
FIXED_SELECTED_ANSWER_FALLBACK_VERSION = (
    "query-ranked-eligible-v2-multi-paper-table-supplements"
)
FIXED_SELECTED_TABLE_SUPPLEMENTS_PER_PAPER = 2
MIXED_VISUAL_TEXT_SUPPLEMENTS_PER_PAPER = 4
_SELECTED_EVIDENCE_PURPOSES = frozenset(
    {
        "answer_value",
        "comparison_operand",
        "eligibility_condition",
        "table_row",
        "visual_fact",
        "citation_fact",
    }
)

JUDGMENT_LABELS = (
    "direct_answer",
    "partial_answer",
    "supporting_only",
    "mention_only",
    "irrelevant",
    "unreadable",
)
RELEVANT_LABELS = frozenset(
    {"direct_answer", "partial_answer", "supporting_only"}
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+%/^-]*")
_OBJECT_REFERENCE_RE = re.compile(
    r"(?i)\b(figure|fig\.?|table|equation|eq\.?|algorithm|reference|ref\.?|citation)"
    r"\s*(?:no\.?\s*)?([A-Za-z]?\d+[A-Za-z]?)\b"
)
_LOCAL_NUMBERED_OBJECT_RE = re.compile(
    r"(?i)\b(?:figure|fig\.?|table|equation|eq\.?|reference|ref\.?)\s*"
    r"(?:no\.?\s*)?\d+[A-Za-z]?(?:\s*\([A-Za-z0-9]+\))?\b"
)
_LOCAL_ORDINAL_REFERENCE_RE = re.compile(
    r"(?i)\b(?:\d+(?:st|nd|rd|th)\s+reference|last\s+reference)\b"
)
_LOCAL_REFERENCE_INVENTORY_RE = re.compile(
    r"(?i)\b(?:how\s+many\s+references|how\s+many\s+papers\s+"
    r"(?:were\s+)?cited)\b"
)
_DIRECT_REPORTED_OPTIMUM_QUERY_RE = re.compile(
    r"\bwhat\s+(?:(?:is|was)\s+)?(?:the\s+)?(?:reported\s+)?"
    r"(?:optimal|best)\b[^?\n]{0,100}?\b"
    r"(?:factor|coefficient|hyperparameter|parameter|setting|threshold|"
    r"temperature|rate|ratio|weight|gamma|lambda|alpha|beta|rho)\b",
    re.IGNORECASE,
)
_EXPLICIT_MULTI_SOURCE_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:compare|comparison|versus|vs\.?|also|while|whereas)\b"
    r"|\b(?:baseline|other|another|second)\s+paper\b"
    r"|[,;]\s*and\s+(?:what|which|how|where|when|who|does|do|is|are|"
    r"report|give|the)\b"
    r")"
)
_LOCAL_OBJECT_GRAMMAR = (
    r"(?:"
    r"(?:figure|fig\.?|table|equation|eq\.?|reference|ref\.?)\s*"
    r"(?:no\.?\s*)?\d+[A-Za-z]?(?:\s*\([A-Za-z0-9]+\))?"
    r"|\d+(?:st|nd|rd|th)\s+reference"
    r"|last\s+reference"
    r"|how\s+many\s+references"
    r"|how\s+many\s+papers\s+(?:were\s+)?cited"
    r")"
)
_PAPER_AFTER_RE = re.compile(r"(?i)\bpaper\s+([^?]+?)\s*\??$")
_OWNER_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)


class NamedOwnerResolution(TypedDict):
    """Candidate-set-level resolution of one explicit named paper owner."""

    version: str
    status: str
    owner_phrase: str
    paper_id: str
    title: str
    match_kind: str
    hard_gate: bool
    reason: str


def _normalize_owner_text(value: str) -> str:
    """Normalize title text without inventing acronym expansions."""

    decoded = html.unescape(unicodedata.normalize("NFKC", str(value or "")))
    return " ".join(_OWNER_TOKEN_RE.findall(decoded.lower()))


def _normalized_source_text(value: str) -> str:
    """Normalize harmless PDF/OCR typography while preserving source content."""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(value or "")))
    normalized = (
        normalized.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2212", "-")
    )
    return " ".join(normalized.split())


def _source_excerpt_is_visible(excerpt: str, record: Record) -> bool:
    """Return whether a copied excerpt occurs in the cited original chunk."""

    needle = _normalized_source_text(excerpt)
    haystack = _normalized_source_text(str(record.get("text") or ""))
    return bool(needle and needle in haystack)


def _record_with_prompt_visible_text(record: Record, rendered: str) -> Record:
    """Copy a record with text restricted to what the model actually received."""

    _, separator, visible_text = rendered.partition("\n")
    visible_record = dict(record)
    visible_record["text"] = visible_text if separator else ""
    return visible_record  # type: ignore[return-value]


def _candidate_title_aliases(title: str) -> tuple[str, ...]:
    """Return only literal, distinctive aliases supplied by canonical metadata."""

    raw_title = html.unescape(unicodedata.normalize("NFKC", str(title or "")))
    raw_aliases: list[tuple[str, bool]] = [(raw_title, False)]
    if ":" in raw_title:
        raw_aliases.append((raw_title.split(":", 1)[0], True))
    aliases: list[str] = []
    for raw_alias, is_prefix in raw_aliases:
        alias = _normalize_owner_text(raw_alias)
        compact = alias.replace(" ", "")
        if len(compact) < 5 or alias in aliases:
            continue
        if is_prefix and len(alias.split()) == 1:
            letters = "".join(
                character for character in raw_alias if character.isalpha()
            )
            distinctive_single_token = (
                any(character.isdigit() for character in raw_alias)
                or (len(letters) >= 3 and letters.isupper())
                or bool(re.search(r"[a-z0-9][A-Z]", raw_alias))
            )
            if not distinctive_single_token:
                continue
        aliases.append(alias)
    return tuple(sorted(aliases, key=lambda value: (-len(value), value)))


_COMPOUND_NAME_JOINERS = frozenset("-‐‑‒–—−")


def _candidate_is_only_a_compound_name_component(
    question: str,
    candidate: CandidatePaper,
    records: Iterable[Record],
) -> bool:
    """Reject a base-method title matched only inside a longer method name.

    For example, ``D-FINE`` in ``DEIM-D-FINE-X`` is not a standalone request
    for the D-FINE paper.  The guard is intentionally fail-open: it does not
    apply when the alias also occurs standalone or when the selected evidence
    itself contains the complete compound name from the question.
    """

    raw_question = html.unescape(unicodedata.normalize("NFKC", question))
    compound_names: list[str] = []
    found_alias = False
    found_standalone_alias = False
    for alias in _candidate_title_aliases(candidate.title):
        for match in re.finditer(
            _literal_alias_pattern(alias), raw_question, re.IGNORECASE
        ):
            found_alias = True
            left_attached = (
                match.start() >= 2
                and raw_question[match.start() - 1] in _COMPOUND_NAME_JOINERS
                and raw_question[match.start() - 2].isalnum()
            )
            right_attached = (
                match.end() + 1 < len(raw_question)
                and raw_question[match.end()] in _COMPOUND_NAME_JOINERS
                and raw_question[match.end() + 1].isalnum()
            )
            if not (left_attached or right_attached):
                found_standalone_alias = True
                continue
            start = match.start()
            end = match.end()
            while start > 0 and (
                raw_question[start - 1].isalnum()
                or raw_question[start - 1] in _COMPOUND_NAME_JOINERS
            ):
                start -= 1
            while end < len(raw_question) and (
                raw_question[end].isalnum()
                or raw_question[end] in _COMPOUND_NAME_JOINERS
            ):
                end += 1
            compound_names.append(raw_question[start:end])
    if not found_alias or found_standalone_alias or not compound_names:
        return False

    normalized_context = " ".join(
        _normalize_owner_text(str(record.get("text") or "")) for record in records
    )
    return not any(
        _normalize_owner_text(compound_name) in normalized_context
        for compound_name in compound_names
    )


def _literal_alias_pattern(alias: str) -> str:
    """Render a normalized canonical alias as a punctuation-tolerant regex."""

    tokens = alias.split()
    return r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(
        re.escape(token) for token in tokens
    ) + r"(?![A-Za-z0-9])"


def _alias_grammatically_owns_local_object(question: str, alias: str) -> bool:
    """Check explicit grammatical ownership, not mere title co-occurrence.

    The destructive gate is limited to constructions such as ``Figure 4 of
    DynaPipe``, ``references in the SecEmb paper``, ``in DynaPipe, Figure 4``,
    or ``DynaPipe's Figure 4``.  A title that is merely cited or compared in the
    same question must remain an AOAI judgment.
    """

    raw_question = html.unescape(unicodedata.normalize("NFKC", question or ""))
    alias_pattern = _literal_alias_pattern(alias)
    paper_owner = (
        rf"(?:the\s+)?(?:paper\s+)?{alias_pattern}(?:\s+paper)?"
    )
    object_then_owner = re.compile(
        rf"\b{_LOCAL_OBJECT_GRAMMAR}(?![A-Za-z0-9])[^?.;:]{{0,100}}?"
        rf"\b(?:of|in)\b\s+{paper_owner}",
        re.IGNORECASE,
    )
    scoped_owner_then_object = re.compile(
        rf"\b(?:in|according\s+to)\s+{paper_owner}\s*[,;:]"
        rf"[^?.;:]{{0,120}}?\b{_LOCAL_OBJECT_GRAMMAR}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    possessive_or_compound_owner = re.compile(
        rf"{alias_pattern}(?:\s+paper)?(?:['’]s|\s+)"
        rf"(?:the\s+)?\b{_LOCAL_OBJECT_GRAMMAR}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return any(
        pattern.search(raw_question)
        for pattern in (
            object_then_owner,
            scoped_owner_then_object,
            possessive_or_compound_owner,
        )
    )


def _identity_only_blocking_mismatch(value: str) -> bool:
    """Recognize only explicit paper/title identity denials.

    A resolved canonical owner makes these denials false.  Broader scientific
    mismatches (dataset, setting, metric, population, and so on) are deliberately
    retained so Stage 2 cannot rescue a genuinely incompatible source.
    """

    normalized = " ".join(str(value or "").lower().split())
    return any(
        re.search(pattern, normalized)
        for pattern in (
            (
                r"\b(?:wrong|different|mismatched|unrelated)\s+"
                r"(?:candidate\s+)?(?:paper|title|owner)\b"
            ),
            (
                r"\b(?:candidate\s+)?(?:paper|title|owner)"
                r"(?:\s+identity)?\s+(?:does not match|doesn't match|is wrong|"
                r"is not\s+(?:the\s+)?(?:requested|target|named)"
                r"(?:\s+paper)?)\b"
            ),
            (
                r"\bnot\s+(?:the\s+)?(?:requested|target|named)\s+"
                r"(?:paper|title|owner)\b"
            ),
        )
    )


def _query_is_explicit_multi_source(question: str) -> bool:
    """Keep comparisons and multiple paper-local objects out of hard gating."""

    local_objects = re.findall(_LOCAL_OBJECT_GRAMMAR, question, re.IGNORECASE)
    return bool(_EXPLICIT_MULTI_SOURCE_RE.search(question)) or len(local_objects) > 1


def _single_full_title_typo_match(
    question: str,
    candidates: Iterable[CandidatePaper],
) -> tuple[CandidatePaper, str] | None:
    """Resolve one long post-``paper`` title with exactly one benign word typo.

    This deliberately does not fuzz short aliases or acronyms.  It exists for
    query-authored full-title slips such as ``Leaner`` versus ``Linear`` and is
    candidate-set unique before it can influence any metadata or cache key.
    """

    match = _PAPER_AFTER_RE.search(question)
    if match is None:
        return None
    owner_phrase = _normalize_owner_text(match.group(1))
    phrase_tokens = owner_phrase.split()
    if len(phrase_tokens) < 4 or len(owner_phrase.replace(" ", "")) < 20:
        return None

    matches: list[CandidatePaper] = []
    for candidate in candidates:
        title = _normalize_owner_text(candidate.title)
        title_tokens = title.split()
        if len(title_tokens) != len(phrase_tokens):
            continue
        differing = [
            (left, right)
            for left, right in zip(phrase_tokens, title_tokens, strict=True)
            if left != right
        ]
        if len(differing) != 1:
            continue
        left, right = differing[0]
        if SequenceMatcher(None, left, right).ratio() < (2 / 3):
            continue
        if SequenceMatcher(None, owner_phrase, title).ratio() < 0.90:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        return None
    return matches[0], owner_phrase


def resolve_named_owner(
    query: Query,
    candidates: Iterable[CandidatePaper],
) -> NamedOwnerResolution:
    """Resolve one high-confidence literal named owner across the whole ranking.

    The hard gate is intentionally narrower than resolution: it activates only
    for a numbered figure/table/equation, a numbered/last reference, or an
    explicit section/bibliography citation inventory.  General single-paper and
    all multi-source questions remain LLM judgments so comparison/option papers
    are not lost merely because another title is named in the question.
    """

    ordered_candidates = tuple(candidates)
    normalized_question = f" {_normalize_owner_text(query.question)} "
    exact_matches: list[tuple[CandidatePaper, tuple[str, ...]]] = []
    for candidate in ordered_candidates:
        matched_aliases = [
            alias
            for alias in _candidate_title_aliases(candidate.title)
            if f" {alias} " in normalized_question
        ]
        if matched_aliases:
            exact_matches.append((candidate, tuple(matched_aliases)))

    match_kind = ""
    owner_phrase = ""
    literal_aliases: tuple[str, ...] = ()
    resolved: CandidatePaper | None = None
    if len(exact_matches) == 1:
        resolved, literal_aliases = exact_matches[0]
        owner_phrase = literal_aliases[0]
        match_kind = "literal_title_or_prefix"
    elif not exact_matches:
        typo_match = _single_full_title_typo_match(
            query.question, ordered_candidates
        )
        if typo_match is not None:
            resolved, owner_phrase = typo_match
            match_kind = "single_word_full_title_typo"

    if resolved is None:
        status = "ambiguous" if len(exact_matches) > 1 else "unresolved"
        reason = (
            "multiple candidate titles are literally named in the query"
            if exact_matches
            else "no unique high-confidence canonical-title match"
        )
        return {
            "version": NAMED_OWNER_RESOLVER_VERSION,
            "status": status,
            "owner_phrase": "",
            "paper_id": "",
            "title": "",
            "match_kind": "",
            "hard_gate": False,
            "reason": reason,
        }

    grammatical_owner = (
        match_kind == "literal_title_or_prefix"
        and not _query_is_explicit_multi_source(query.question)
        and any(
            _alias_grammatically_owns_local_object(query.question, alias)
            for alias in literal_aliases
        )
    )
    return {
        "version": NAMED_OWNER_RESOLVER_VERSION,
        "status": "resolved",
        "owner_phrase": owner_phrase,
        "paper_id": resolved.paper_id,
        "title": resolved.title,
        "match_kind": match_kind,
        "hard_gate": grammatical_owner,
        "reason": (
            "unique literal title grammatically owns the paper-local object"
            if grammatical_owner
            else (
                "unique fuzzy title resolved for metadata only; fuzzy matches never gate"
                if match_kind == "single_word_full_title_typo"
                else (
                    "unique named title resolved; comparison or multi-source "
                    "questions never hard gate"
                    if _query_is_explicit_multi_source(query.question)
                    else (
                        "unique named title resolved without explicit "
                        "local-object ownership"
                    )
                )
            )
        ),
    }


class PaperContext(TypedDict):
    """The only selected paper context sent for one Stage-1 judgment."""

    text: str
    records_by_id: dict[str, Record]
    image_paths: list[str]
    compacted: bool
    total_chunk_count: int
    selected_chunk_ids: list[str]
    omitted_chunk_ids: list[str]
    full_text_characters: int
    selected_text_characters: int
    character_limit: int
    total_readable_image_count: int
    selected_image_chunk_ids: list[str]
    omitted_image_chunk_ids: list[str]
    selection: list[dict[str, Any]]


class AnswerContext(TypedDict):
    """Validated handoff plus separately tracked deterministic supplements."""

    text: str
    records_by_id: dict[str, Record]
    image_paths: list[str]
    stage1_handoff_chunk_ids: list[str]
    python_supplemental_chunk_ids: list[str]
    fixed_selected_table_supplement_chunk_ids: list[str]
    answer_evidence_image_chunk_ids: list[str]
    attached_answer_evidence_image_chunk_ids: list[str]


class CompletionResult(TypedDict, total=False):
    """Serializable model response plus optional provider metadata."""

    text: Required[str]
    request_id: str | None
    model: str | None
    deployment: str
    usage: Any
    latency_seconds: float
    max_completion_tokens: int
    finish_reason: str
    rate_limit: dict[str, str]
    requested_image_count: int
    attached_image_count: int
    image_fallback_reason: str
    provider_invocation_count: int
    provider_attempt_id: str
    provider_semantic_phase: str
    provider_invocation_index: int
    prompt_content_filter_fallback_reason: str
    prompt_content_filter_blocked_categories: list[str]
    prompt_content_filter_blocked_attempts: list[dict[str, Any]]
    blocked_prompt_sha256: str
    blocked_prompt_characters: int
    blocked_context_chunk_ids: list[str]


ProviderAttemptCallback = Callable[[dict[str, Any]], None]


def _finalize_provider_response_attempt(
    completion: CompletionResult,
    *,
    callback: ProviderAttemptCallback | None,
    semantic_phase: str,
    logical_attempt_index: int,
    parse_error: str | None,
) -> None:
    """Finalize one prepared response without storing its response body."""

    if callback is None:
        return
    attempt_id = str(completion.get("provider_attempt_id") or "")
    if not attempt_id:
        raise RuntimeError("provider response is missing its prepared attempt_id")
    safe_metadata_keys = (
        "request_id",
        "model",
        "deployment",
        "usage",
        "latency_seconds",
        "max_completion_tokens",
        "finish_reason",
        "rate_limit",
        "estimated_reserved_tokens",
        "launch_interval_seconds",
        "target_tpm",
        "requested_image_count",
        "attached_image_count",
        "image_fallback_reason",
        "prompt_sha256",
        "prompt_characters",
    )
    callback(
        {
            "attempt_id": attempt_id,
            "event_kind": "finalize",
            "outcome": "response",
            "semantic_phase": semantic_phase,
            "logical_attempt_index": logical_attempt_index,
            "provider_invocation_index": int(
                completion.get("provider_invocation_index") or 1
            ),
            "provider_invocation_count": 1,
            "parse_error": parse_error,
            **{
                key: completion.get(key)
                for key in safe_metadata_keys
                if key in completion
            },
        }
    )


class ReadingResponseError(RuntimeError):
    """The model returned a response that cannot safely drive the next stage."""


class JudgmentEvidenceChunkError(ReadingResponseError):
    """Stage 1 cited a chunk outside its selected candidate-paper context."""


class JudgmentResponseExhaustedError(ReadingResponseError):
    """Both the initial Stage-1 response and its repair failed validation.

    This narrow exception lets the global coordinator isolate one malformed
    model response without mistaking authentication, rate-limit, corpus, or
    adapter failures for a harmless candidate-level problem.  The paid calls
    remain attached for durable error auditing.
    """

    def __init__(self, message: str, *, calls: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.calls = calls


class AnswerEvidenceLocatorError(ReadingResponseError):
    """Stage 2 cited evidence that cannot be serialized with an official locator."""


class NoRelevantCandidatesError(RuntimeError):
    """No candidate was accepted, so an evidence-grounded answer is impossible."""


_NUMBERED_CITATION_MARKER_RE = re.compile(
    r"(?<!\w)(?:\[\s*(?P<bracket>\d+)\s*\]|"
    r"(?:reference|ref\.?)\s*:?\s*(?P<label>\d+)\b)",
    re.IGNORECASE,
)


def _numbered_citation_entries(text: str) -> list[tuple[int, str]]:
    """Split numbered bibliography text at each explicit citation marker.

    MinerU can place several bibliography entries in one chunk (and sometimes
    even on one physical line).  Returning the text only up to the next marker
    prevents an author in ``[2]`` from being used to validate ``[1]``.
    """

    matches = list(_NUMBERED_CITATION_MARKER_RE.finditer(text))
    entries: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        raw_number = match.group("bracket") or match.group("label")
        if raw_number is None:
            continue
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        entries.append((int(raw_number), text[match.start() : end]))
    return entries


def _metadata_citation_number(value: Any) -> int | None:
    """Read a citation locator only when its complete value names one number."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[0]) if len(numbers) == 1 else None


def _citation_identity_supported_by_records(
    item: str,
    records: Iterable[Record],
    *,
    required_author: str | None = None,
) -> bool:
    """Check one normalized identity against the cited answer-chunk text."""

    identity_key = citation_identity_key(item)
    if identity_key is None:
        return False
    if identity_key.startswith("number:"):
        number = int(identity_key.removeprefix("number:"))
        required_author_pattern = (
            re.compile(
                rf"(?<!\w){re.escape(required_author)}(?!\w)", re.IGNORECASE
            )
            if required_author
            else None
        )
        for record in records:
            metadata = record.get("metadata") or {}
            text = str(record.get("text") or "")
            numbered_entries = _numbered_citation_entries(text)
            for entry_number, entry_text in numbered_entries:
                if entry_number != number:
                    continue
                if (
                    required_author_pattern is None
                    or required_author_pattern.search(entry_text)
                ):
                    return True

            # A matching citation_id is an entry-level locator when the text has
            # no explicit numbered boundaries.  If boundaries are present, do
            # not widen the author check back to the whole multi-entry chunk.
            if (
                _metadata_citation_number(metadata.get("citation_id")) == number
                and not numbered_entries
                and (
                    required_author_pattern is None
                    or required_author_pattern.search(text)
                )
            ):
                return True
        return False

    _, author, year = identity_key.split(":", maxsplit=2)
    inline_pattern = re.compile(
        rf"(?<!\w){re.escape(author)}(?!\w)"
        rf"(?:\s+et\s+al\.?)?\s*,?\s*\(?\s*{re.escape(year)}\s*\)?"
    )
    bibliography_start = re.compile(
        r"^[A-ZÀ-ÖØ-Þ][^,\n]{0,60},\s+"
        r"(?:[A-ZÀ-ÖØ-Þ]\.?(?:\s|,)|[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)",
    )
    target_entry_start = re.compile(
        rf"^{re.escape(author)},\s+", re.IGNORECASE
    )
    required_author_pattern = (
        re.compile(rf"(?<!\w){re.escape(required_author)}(?!\w)")
        if required_author
        else None
    )
    for record in records:
        raw_text = str(record.get("text") or "")
        if inline_pattern.search(raw_text.casefold()) and (
            required_author is None or required_author == author
        ):
            return True

        # Bibliography entries in MinerU text start on their own line.  Include
        # wrapped continuation lines, but stop before the next surname/initial
        # entry so an adjacent paper's year can never support this identity.
        lines = raw_text.splitlines()
        for index, line in enumerate(lines):
            if not target_entry_start.search(line.strip()):
                continue
            entry_lines = [line]
            for following in lines[index + 1 :]:
                if bibliography_start.search(following.strip()):
                    break
                entry_lines.append(following)
            entry = " ".join(entry_lines).casefold()
            if re.search(
                rf"(?<!\d){re.escape(year)}(?!\w)", entry
            ) and (
                required_author_pattern is None
                or required_author_pattern.search(entry)
            ):
                return True
    return False


def _validate_stage1_citation_count(
    *,
    query: Query,
    candidate_answer: dict[str, Any],
    evidence: list[dict[str, Any]],
    allowed_records: dict[str, Record],
) -> None:
    """Validate the explicit Stage-1 inventory for a scalar citation count."""

    units = candidate_answer.get("units")
    if not isinstance(units, list) or len(units) != 1 or not isinstance(units[0], dict):
        raise ReadingResponseError(
            "aggregate citation count requires exactly one candidate_answer unit "
            "with integer value and counted_items"
        )
    unit = units[0]
    value = unit.get("value")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReadingResponseError(
            "aggregate citation count candidate_answer unit.value must be an integer"
        )
    try:
        counted_items = validate_citation_count_items(
            unit.get("counted_items"),
            path="candidate_answer.units[0].counted_items",
        )
    except DerivationValidationError as exc:
        raise ReadingResponseError(str(exc)) from exc
    if value != len(counted_items):
        raise ReadingResponseError(
            f"aggregate citation count unit.value={value} but counted_items "
            f"contain {len(counted_items)} identities"
        )

    answer_records = [
        allowed_records[str(item["chunk_id"])]
        for item in evidence
        if item.get("purpose") == "answer"
    ]
    unsupported = [
        item
        for item in counted_items
        if not _citation_identity_supported_by_records(
            item,
            answer_records,
            required_author=citation_author_filter(query),
        )
    ]
    if unsupported:
        raise ReadingResponseError(
            "aggregate citation count identities are not supported by the cited "
            f"answer chunks: {unsupported}"
        )

    raw_labels = unit.get("matched_option_labels") or []
    if "multiple_choice" in query.answer_types and raw_labels:
        for raw_label in raw_labels:
            option_text = (query.options or {}).get(str(raw_label))
            if (
                not isinstance(option_text, str)
                or not re.fullmatch(r"\s*\d+\s*", option_text)
                or int(option_text) != value
            ):
                raise ReadingResponseError(
                    "aggregate citation count matched option text must be a bare "
                    f"integer equal to validated count {value}"
                )
    unit["counted_items"] = counted_items


def _validate_stage2_citation_count_support(
    *,
    query: Query,
    derivation: dict[str, Any],
    context_records: dict[str, Record],
) -> None:
    """Ground every counted identity in the count facts' submitted chunks."""

    facts_by_id = {
        str(fact["id"]): fact
        for fact in derivation.get("facts") or []
        if isinstance(fact, dict) and fact.get("id")
    }
    for operation in derivation.get("operations") or []:
        if not isinstance(operation, dict) or operation.get("kind") != "count":
            continue
        records: list[Record] = []
        seen_chunk_ids: set[str] = set()
        for fact_id in operation.get("fact_ids") or []:
            fact = facts_by_id.get(str(fact_id))
            if fact is None:
                continue
            for raw_chunk_id in fact.get("chunk_ids") or []:
                chunk_id = str(raw_chunk_id)
                if chunk_id in seen_chunk_ids:
                    continue
                record = context_records.get(chunk_id)
                if record is not None:
                    seen_chunk_ids.add(chunk_id)
                    records.append(record)
        unsupported = [
            str(item)
            for item in operation.get("items") or []
            if not _citation_identity_supported_by_records(
                str(item),
                records,
                required_author=citation_author_filter(query),
            )
        ]
        if unsupported:
            raise ReadingResponseError(
                "aggregate citation count identities are not supported by the "
                f"referenced fact chunks: {unsupported}"
            )


def _validate_ordinal_reference_support(
    *,
    query: Query,
    derivation: dict[str, Any],
    context_records: dict[str, Record],
) -> None:
    """Ground final facts in bibliography label ``[N]``, never body order.

    An ordinal phrase such as ``third reference`` is a bibliography lookup.
    In-text citation order is neither stable nor an official evidence locator,
    so a fact used by the final answer must be visible inside the exact ``[N]``
    entry of one of its own cited chunks.
    """

    ordinal = requested_reference_ordinal(query.question)
    if ordinal is None:
        return
    target_entries = requested_ordinal_bibliography_entries(
        query.question, context_records.values()
    )
    if not target_entries:
        raise ReadingResponseError(
            f"ordinal bibliography lookup requests label [{ordinal}], but no "
            "explicit matching bibliography entry is available in answer context"
        )

    facts_by_id = {
        str(fact["id"]): fact
        for fact in derivation.get("facts") or []
        if isinstance(fact, dict) and fact.get("id")
    }
    operations_by_id = {
        str(operation["id"]): operation
        for operation in derivation.get("operations") or []
        if isinstance(operation, dict) and operation.get("id")
    }
    bound_fact_ids: set[str] = set()
    for binding in derivation.get("answer_bindings") or []:
        if not isinstance(binding, dict):
            continue
        source_id = str(binding.get("source_id") or "")
        if binding.get("source_type") == "fact":
            bound_fact_ids.add(source_id)
            continue
        if binding.get("source_type") != "operation":
            continue
        operation = operations_by_id.get(source_id)
        if operation is not None:
            bound_fact_ids.update(
                str(fact_id) for fact_id in operation.get("fact_ids") or []
            )

    unsupported: list[str] = []
    for fact_id in sorted(bound_fact_ids):
        fact = facts_by_id.get(fact_id)
        if fact is None:
            continue
        paper_id = str(fact.get("paper_id") or "")
        chunk_ids = {str(value) for value in fact.get("chunk_ids") or []}
        matching_entries = [
            entry
            for entry in target_entries
            if entry.paper_id == paper_id and entry.chunk_id in chunk_ids
        ]
        if not any(
            bibliography_entry_supports_value(entry, fact.get("value"))
            for entry in matching_entries
        ):
            unsupported.append(fact_id)
    if unsupported:
        raise ReadingResponseError(
            f"ordinal bibliography lookup means bibliography label [{ordinal}], "
            "not the ordinal in-text citation occurrence; final answer fact(s) "
            f"are not supported inside entry [{ordinal}]: {unsupported}"
        )


_VISUAL_SUBFIGURE_COUNT_RE = re.compile(
    r"(?i)\bhow\s+many\s+(?:subfigures?|subplots?)\b"
)
_SPATIAL_AXES_ID_RE = re.compile(
    r"(?i)(?:\b(?:top|bottom|upper|lower|left|right|middle|center|centre)\b|"
    r"\b(?:row|col(?:umn)?)\s*[-:#]?\s*\d+\b)"
)


def _integer_count(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise ReadingResponseError(f"visual subfigure count {path} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    raise ReadingResponseError(f"visual subfigure count {path} must be an integer")


def _validate_stage1_visual_subfigure_count(
    *,
    query: Query,
    candidate_answer: dict[str, Any],
) -> None:
    """Require an auditable spatial inventory for an image panel count."""

    if _VISUAL_SUBFIGURE_COUNT_RE.search(query.question) is None:
        return
    units = candidate_answer.get("units")
    if (
        not isinstance(units, list)
        or len(units) != 1
        or not isinstance(units[0], dict)
    ):
        raise ReadingResponseError(
            "visual subfigure count requires exactly one candidate_answer unit"
        )
    unit = units[0]
    value = _integer_count(unit.get("value"), path="unit.value")
    items = unit.get("counted_items")
    if not isinstance(items, list) or not items:
        raise ReadingResponseError(
            "visual subfigure count requires non-empty counted_items with one "
            "distinct spatial identifier per independently bounded axes region"
        )
    normalized_items: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ReadingResponseError(
                f"visual subfigure count counted_items[{index}] must be non-empty text"
            )
        normalized = _normalize_owner_text(item)
        if not normalized or _SPATIAL_AXES_ID_RE.search(item) is None:
            raise ReadingResponseError(
                "visual subfigure count counted_items must use a distinct spatial "
                "identifier for each axes region; a bare group label, row name, "
                "or model family is not a subfigure"
            )
        normalized_items.append(normalized)
    if len(normalized_items) != len(set(normalized_items)):
        raise ReadingResponseError(
            "visual subfigure count counted_items must be distinct"
        )
    if value != len(items):
        raise ReadingResponseError(
            f"visual subfigure count unit.value={value} but counted_items contain "
            f"{len(items)} spatial axes"
        )
    raw_labels = unit.get("matched_option_labels") or []
    if not isinstance(raw_labels, list):
        raise ReadingResponseError(
            "visual subfigure count matched_option_labels must be a list"
        )
    for label in raw_labels:
        option_value = _integer_count(
            (query.options or {}).get(str(label)),
            path=f"matched option {label!r}",
        )
        if option_value != value:
            raise ReadingResponseError(
                f"visual subfigure count value={value} does not equal matched "
                f"option {label!r} value={option_value}"
            )
    unit["value"] = value
    unit["counted_items"] = list(items)


def _answer_review_pool(
    judgments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the validated Stage-2 handoff pool in candidate-rank order.

    The current three-field schema requires A, B, ``send_to_answer_agent``, and
    non-empty validated IDs. The narrow target-owner rescue below applies only
    to legacy multi-field judgments kept for checkpoint compatibility.
    """

    ordered = sorted(judgments, key=lambda value: int(value.get("rank") or 0))
    selected: list[dict[str, Any]] = []
    seen_papers: set[str] = set()
    for item in ordered:
        paper_id = str(item.get("paper_id") or "")
        simple_schema = "send_to_answer_agent" in item
        if simple_schema:
            accepted = bool(
                item.get("is_relevant_to_answer") is True
                and item.get("has_usable_answer_evidence") is True
                and item.get("send_to_answer_agent") is True
                and (item.get("evidence_chunk_ids") or [])
            )
            owner_recheck = False
        else:
            accepted = (
                item.get("relevant") is True
                and item.get("label") in RELEVANT_LABELS
            )
            owner_recheck = (
                not accepted
                and item.get("paper_role") == "target_owner"
                and item.get("identity_conflict") is not True
                and not (item.get("blocking_mismatches") or [])
                and bool(item.get("evidence") or [])
            )
        if not paper_id or paper_id in seen_papers or not (accepted or owner_recheck):
            continue
        seen_papers.add(paper_id)
        if accepted:
            selected.append(item)
            continue
        rescued = dict(item)
        rescued["stage1_label"] = str(item.get("label") or "")
        rescued["label"] = "supporting_only"
        rescued["answer_pool_reason"] = "target_owner_recheck"
        selected.append(rescued)
    return selected


def _is_image_content_policy_violation(exc: Exception) -> bool:
    """Return true only for an Azure/OpenAI image content-policy rejection."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code") == "content_policy_violation":
            return True
    return (
        getattr(exc, "status_code", None) in (None, 400)
        and "content_policy_violation" in str(exc).lower()
    )


def _prompt_content_filter_categories(exc: Exception) -> tuple[str, ...]:
    """Return categories Azure structurally reports as filtering the prompt."""

    if getattr(exc, "status_code", None) != 400:
        return ()
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return ()
    error = body.get("error", body)
    if not isinstance(error, dict):
        return ()
    if error.get("code") != "content_filter" or error.get("param") != "prompt":
        return ()
    inner_error = error.get("innererror")
    if not isinstance(inner_error, dict):
        return ()
    filter_result = inner_error.get("content_filter_result", inner_error)
    if not isinstance(filter_result, dict):
        return ()

    filtered_categories: list[str] = []
    for category, raw_result in filter_result.items():
        if not isinstance(category, str) or not isinstance(raw_result, dict):
            continue
        if category == "jailbreak":
            is_filtered = (
                raw_result.get("detected") is True
                and raw_result.get("filtered") is True
            )
        else:
            is_filtered = raw_result.get("filtered") is True
        if is_filtered:
            filtered_categories.append(category)
    return tuple(sorted(set(filtered_categories)))


def _prompt_filter_provider_invocation_count(exc: Exception) -> int:
    """Read the application-level provider calls made before a prompt rejection."""

    raw_count = getattr(exc, "_littraceqa_provider_invocation_count", 1)
    return raw_count if isinstance(raw_count, int) and raw_count > 0 else 1


def _record_prompt_filter_provider_invocation_count(
    exc: Exception, count: int, prompt: str
) -> None:
    """Attach private audit metadata without changing the raised exception type."""

    try:
        setattr(exc, "_littraceqa_provider_invocation_count", count)
        setattr(
            exc,
            "_littraceqa_prompt_sha256",
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
        setattr(exc, "_littraceqa_prompt_characters", len(prompt))
    except Exception:
        # Some third-party exception implementations may prohibit attributes.
        # The known Azure/OpenAI errors support them; retain the conservative
        # one-call default for an incompatible exception object.
        pass


def _judgment_call_record(
    completion: CompletionResult,
    *,
    phase: str,
    attempt: str,
    parse_error: str | None,
    context: PaperContext,
) -> dict[str, Any]:
    """Build the durable audit record for one judgment model call."""
    return {
        **{key: value for key, value in completion.items() if key != "text"},
        "usage": completion.get("usage"),
        "phase": phase,
        "attempt": attempt,
        "context_chunk_ids": list(context["records_by_id"]),
        "image_paths": list(context["image_paths"]),
        "raw_response": completion["text"],
        "parse_error": parse_error,
    }


class PairwiseAOAIReader:
    """Judge fixed candidates, then answer from validated Stage-1 handoff chunks."""

    supports_named_owner_resolution = True
    supports_provider_attempt_ledger = True

    def __init__(
        self,
        chunk_store: ChunkStore,
        llm: LLMClient,
        max_paper_context_chars: int = 220_000,
        max_judgment_prompt_chars: int = 240_000,
        max_paper_images: int = MAX_AOAI_IMAGES_PER_REQUEST,
        answer_context_chars: int = 220_000,
        answer_neighbor_chunks: int = 1,
        max_answer_images: int = MAX_AOAI_IMAGES_PER_REQUEST,
        max_answer_papers: int = 50,
        max_evidence: int = 32,
        max_evidence_per_paper: int | None = None,
        judgment_max_completion_tokens: int | None = None,
        answer_max_completion_tokens: int | None = None,
        paper_set_policy: str = PAIRWISE_CANDIDATE_POLICY,
    ) -> None:
        if paper_set_policy not in PAPER_SET_POLICIES:
            raise ValueError(
                f"paper_set_policy must be one of {sorted(PAPER_SET_POLICIES)}"
            )
        if max_paper_context_chars < 8_000:
            raise ValueError("max_paper_context_chars must be at least 8000")
        if max_judgment_prompt_chars < max_paper_context_chars + 8_000:
            raise ValueError(
                "max_judgment_prompt_chars must reserve at least 8000 characters "
                "above max_paper_context_chars"
            )
        if (
            max_paper_images < 0
            or max_answer_images < 0
            or max_paper_images > MAX_AOAI_IMAGES_PER_REQUEST
            or max_answer_images > MAX_AOAI_IMAGES_PER_REQUEST
        ):
            raise ValueError(
                "image limits must be between 0 and "
                f"{MAX_AOAI_IMAGES_PER_REQUEST} per AOAI request"
            )
        if answer_context_chars < 8_000:
            raise ValueError("answer_context_chars must be at least 8000")
        for name, value in (
            ("judgment_max_completion_tokens", judgment_max_completion_tokens),
            ("answer_max_completion_tokens", answer_max_completion_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")
        resolved_per_paper = (
            min(4, max_evidence)
            if max_evidence_per_paper is None
            else max_evidence_per_paper
        )
        if (
            answer_neighbor_chunks < 0
            or max_answer_papers < 1
            or max_evidence < 1
            or resolved_per_paper < 1
            or resolved_per_paper > max_evidence
        ):
            raise ValueError("invalid answer context limits")
        self.chunk_store = chunk_store
        self.llm = llm
        self.max_paper_context_chars = max_paper_context_chars
        self.max_judgment_prompt_chars = max_judgment_prompt_chars
        self.max_paper_images = max_paper_images
        self.answer_context_chars = answer_context_chars
        self.answer_neighbor_chunks = answer_neighbor_chunks
        self.max_answer_images = max_answer_images
        self.max_answer_papers = max_answer_papers
        self.max_evidence = max_evidence
        self.max_evidence_per_paper = resolved_per_paper
        self.judgment_max_completion_tokens = judgment_max_completion_tokens
        self.answer_max_completion_tokens = answer_max_completion_tokens
        self.paper_set_policy = paper_set_policy
        # In fixed-selected mode every supplied paper is authoritative. A
        # named-owner heuristic may guide extraction but must never remove one.
        self.supports_named_owner_resolution = (
            paper_set_policy != FIXED_SELECTED_PAPER_POLICY
        )

    @property
    def judgment_prompt_version(self) -> str:
        return (
            SELECTED_EVIDENCE_PROMPT_VERSION
            if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            else JUDGMENT_PROMPT_VERSION
        )

    @property
    def answer_prompt_version(self) -> str:
        return (
            FIXED_SELECTED_ANSWER_PROMPT_VERSION
            if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            else ANSWER_PROMPT_VERSION
        )

    def judgment_example_ids(self, query: Query) -> list[str]:
        if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY:
            return selected_evidence_example_manifest(query)
        return example_manifest(query)["judgment"]

    # ---- Stage 1: one query x one candidate paper -------------------------

    def judgment_cache_key(
        self,
        query: Query,
        candidate: CandidatePaper,
        records: list[Record],
        *,
        owner_resolution: NamedOwnerResolution | None = None,
    ) -> str:
        require_production_query(query)
        payload = {
            "prompt_version": self.judgment_prompt_version,
            "paper_set_policy": self.paper_set_policy,
            "question_type_version": JUDGMENT_QUESTION_TYPE_VERSION,
            "question_type": judgment_question_type(query),
            "named_owner_resolution": owner_resolution
            or {
                "version": NAMED_OWNER_RESOLVER_VERSION,
                "status": "not_computed",
                "hard_gate": False,
            },
            "few_shot_examples": self.judgment_example_ids(query),
            "query": _production_query_payload(query),
            "candidate": _candidate_payload(candidate),
            "paper_content_sha256": _records_sha256(records),
            "paper_image_content_sha256": _image_content_sha256(records),
            "limits": {
                "paper_context_selector_version": PAPER_CONTEXT_SELECTOR_VERSION,
                "max_paper_context_chars": self.max_paper_context_chars,
                "max_judgment_prompt_chars": self.max_judgment_prompt_chars,
                "max_paper_images": self.max_paper_images,
                "judgment_max_completion_tokens": (
                    self.judgment_max_completion_tokens
                ),
            },
        }
        return _json_sha256(payload)

    def judge_candidate(
        self,
        query: Query,
        candidate: CandidatePaper,
        *,
        owner_resolution: NamedOwnerResolution | None = None,
        provider_attempt_callback: ProviderAttemptCallback | None = None,
    ) -> dict[str, Any]:
        """Run an independent relevance/evidence judgment for one candidate paper."""

        require_production_query(query)
        records = self.chunk_store.load_paper(candidate.paper_id)
        if not records:
            raise FileNotFoundError(
                f"{query.query_id}: candidate paper is absent from MinerU corpus: "
                f"{candidate.paper_id}"
            )
        if (
            owner_resolution is not None
            and owner_resolution["status"] == "resolved"
            and owner_resolution["hard_gate"]
            and owner_resolution["paper_id"] != candidate.paper_id
        ):
            return self._deterministic_wrong_owner_judgment(
                query=query,
                candidate=candidate,
                records=records,
                owner_resolution=owner_resolution,
            )
        resolved_target_owner = bool(
            owner_resolution is not None
            and owner_resolution["status"] == "resolved"
            and owner_resolution["hard_gate"]
            and owner_resolution["paper_id"] == candidate.paper_id
        )
        context = self._paper_context(query, candidate, records)
        prompt = self._judgment_prompt(
            query=query,
            candidate=candidate,
            context=context,
        )
        if len(prompt) > self.max_judgment_prompt_chars:
            reduced_limit = max(
                8_000,
                self.max_paper_context_chars
                - (len(prompt) - self.max_judgment_prompt_chars)
                - 1_000,
            )
            context = self._paper_context(
                query, candidate, records, max_chars=reduced_limit
            )
            prompt = self._judgment_prompt(
                query=query,
                candidate=candidate,
                context=context,
            )
        if len(prompt) > self.max_judgment_prompt_chars:
            raise ValueError(
                f"{query.query_id}/{candidate.paper_id}: rendered judgment prompt "
                f"has {len(prompt)} characters, exceeding "
                f"max_judgment_prompt_chars={self.max_judgment_prompt_chars}"
            )
        blocked_filter_attempts: list[dict[str, Any]] = []

        def record_blocked_filter_attempt(
            *,
            blocked_prompt: str,
            blocked_context: PaperContext,
            phase: str,
            categories: tuple[str, ...],
            error: Exception,
        ) -> None:
            blocked_filter_attempts.append(
                {
                    "phase": phase,
                    "categories": list(categories),
                    "prompt_sha256": getattr(
                        error,
                        "_littraceqa_prompt_sha256",
                        hashlib.sha256(
                            blocked_prompt.encode("utf-8")
                        ).hexdigest(),
                    ),
                    "prompt_characters": getattr(
                        error,
                        "_littraceqa_prompt_characters",
                        len(blocked_prompt),
                    ),
                    "context_chunk_ids": list(
                        blocked_context["selected_chunk_ids"]
                    ),
                    "provider_invocation_count": (
                        _prompt_filter_provider_invocation_count(error)
                    ),
                }
            )

        try:
            completion = self._complete(
                prompt,
                context["image_paths"],
                max_completion_tokens=self.judgment_max_completion_tokens,
                provider_attempt_callback=provider_attempt_callback,
                semantic_phase="judgment_initial_full_context",
            )
        except Exception as exc:
            categories = _prompt_content_filter_categories(exc)
            if not categories:
                raise
            record_blocked_filter_attempt(
                blocked_prompt=prompt,
                blocked_context=context,
                phase="full_context",
                categories=categories,
                error=exc,
            )

            fallback_context = self._prompt_content_filter_fallback_context(
                query=query,
                records=records,
                original_context=context,
            )
            fallback_phase = (
                "title_abstract"
                if fallback_context["selected_chunk_ids"]
                else "metadata_only"
            )
            fallback_prompt = self._judgment_prompt(
                query=query,
                candidate=candidate,
                context=fallback_context,
            )
            if len(fallback_prompt) > self.max_judgment_prompt_chars:
                fallback_context = self._prompt_content_filter_fallback_context(
                    query=query,
                    records=records,
                    original_context=context,
                    include_title_abstract=False,
                )
                fallback_phase = "metadata_only"
                fallback_prompt = self._judgment_prompt(
                    query=query,
                    candidate=candidate,
                    context=fallback_context,
                )
            if len(fallback_prompt) > self.max_judgment_prompt_chars:
                raise ValueError(
                    f"{query.query_id}/{candidate.paper_id}: metadata-only "
                    "judgment fallback exceeds max_judgment_prompt_chars="
                    f"{self.max_judgment_prompt_chars}"
                )

            # Build a fresh request instead of appending instructions to the
            # rejected prompt. This guarantees that body-level attack examples
            # from the blocked context cannot survive into the retry.
            try:
                completion = self._complete(
                    fallback_prompt,
                    fallback_context["image_paths"],
                    max_completion_tokens=self.judgment_max_completion_tokens,
                    provider_attempt_callback=provider_attempt_callback,
                    semantic_phase=f"judgment_initial_{fallback_phase}",
                )
            except Exception as fallback_exc:
                fallback_categories = _prompt_content_filter_categories(
                    fallback_exc
                )
                if not fallback_categories or fallback_phase == "metadata_only":
                    raise
                record_blocked_filter_attempt(
                    blocked_prompt=fallback_prompt,
                    blocked_context=fallback_context,
                    phase=fallback_phase,
                    categories=fallback_categories,
                    error=fallback_exc,
                )
                fallback_context = self._prompt_content_filter_fallback_context(
                    query=query,
                    records=records,
                    original_context=context,
                    include_title_abstract=False,
                )
                fallback_phase = "metadata_only"
                fallback_prompt = self._judgment_prompt(
                    query=query,
                    candidate=candidate,
                    context=fallback_context,
                )
                if len(fallback_prompt) > self.max_judgment_prompt_chars:
                    raise ValueError(
                        f"{query.query_id}/{candidate.paper_id}: metadata-only "
                        "judgment fallback exceeds max_judgment_prompt_chars="
                        f"{self.max_judgment_prompt_chars}"
                    )
                completion = self._complete(
                    fallback_prompt,
                    fallback_context["image_paths"],
                    max_completion_tokens=self.judgment_max_completion_tokens,
                    provider_attempt_callback=provider_attempt_callback,
                    semantic_phase=f"judgment_initial_{fallback_phase}",
                )

            first_blocked = blocked_filter_attempts[0]
            completion["prompt_content_filter_fallback_reason"] = (
                f"azure_prompt_content_filter_{fallback_phase}"
            )
            completion["prompt_content_filter_blocked_categories"] = sorted(
                {
                    category
                    for attempt in blocked_filter_attempts
                    for category in attempt["categories"]
                }
            )
            completion["prompt_content_filter_blocked_attempts"] = (
                blocked_filter_attempts
            )
            # Retain the original single-block fields for checkpoint consumers
            # written before multi-step fallback was introduced.
            completion["blocked_prompt_sha256"] = first_blocked["prompt_sha256"]
            completion["blocked_prompt_characters"] = first_blocked[
                "prompt_characters"
            ]
            completion["blocked_context_chunk_ids"] = first_blocked[
                "context_chunk_ids"
            ]
            completion["provider_invocation_count"] = int(
                completion.get("provider_invocation_count") or 1
            ) + sum(
                int(attempt["provider_invocation_count"])
                for attempt in blocked_filter_attempts
            )
            context = fallback_context
            prompt = fallback_prompt
        calls: list[dict[str, Any]] = []
        validated_completion = completion
        try:
            parsed = self._parse_judgment(
                query=query,
                candidate=candidate,
                payload_text=completion["text"],
                allowed_records=context["records_by_id"],
                attached_image_paths=_completion_attached_paths(
                    completion, context["image_paths"]
                ),
                resolved_target_owner=resolved_target_owner,
                allow_legacy_schema=False,
            )
        except ReadingResponseError as exc:
            _finalize_provider_response_attempt(
                completion,
                callback=provider_attempt_callback,
                semantic_phase=str(
                    completion.get("provider_semantic_phase")
                    or "judgment_initial"
                ),
                logical_attempt_index=1,
                parse_error=str(exc),
            )
            calls.append(
                _judgment_call_record(
                    completion,
                    phase="single_context",
                    attempt="initial",
                    parse_error=str(exc),
                    context=context,
                )
            )
            repair_prompt = self._judgment_evidence_repair_prompt(
                original_prompt=prompt,
                rejected_response=completion["text"],
                error=exc,
                allowed_chunk_ids=list(context["records_by_id"]),
                eligible_chunk_ids=[
                    chunk_id
                    for chunk_id, record in context["records_by_id"].items()
                    if submission_evidence_eligible(record)
                ],
            )
            repair_completion = self._complete(
                repair_prompt,
                context["image_paths"],
                max_completion_tokens=self.judgment_max_completion_tokens,
                provider_attempt_callback=provider_attempt_callback,
                semantic_phase="judgment_evidence_repair",
            )
            try:
                parsed = self._parse_judgment(
                    query=query,
                    candidate=candidate,
                    payload_text=repair_completion["text"],
                    allowed_records=context["records_by_id"],
                    attached_image_paths=_completion_attached_paths(
                        repair_completion, context["image_paths"]
                    ),
                    resolved_target_owner=resolved_target_owner,
                    allow_legacy_schema=False,
                )
                validated_completion = repair_completion
            except ReadingResponseError as repair_exc:
                _finalize_provider_response_attempt(
                    repair_completion,
                    callback=provider_attempt_callback,
                    semantic_phase="judgment_evidence_repair",
                    logical_attempt_index=2,
                    parse_error=str(repair_exc),
                )
                calls.append(
                    _judgment_call_record(
                        repair_completion,
                        phase="single_context",
                        attempt="evidence_repair",
                        parse_error=str(repair_exc),
                        context=context,
                    )
                )
                raise JudgmentResponseExhaustedError(
                    f"{query.query_id}/{candidate.paper_id}: stage-1 response "
                    "remained invalid after one evidence repair: "
                    f"{repair_exc}",
                    calls=calls,
                ) from repair_exc
            else:
                _finalize_provider_response_attempt(
                    repair_completion,
                    callback=provider_attempt_callback,
                    semantic_phase="judgment_evidence_repair",
                    logical_attempt_index=2,
                    parse_error=None,
                )
                calls.append(
                    _judgment_call_record(
                        repair_completion,
                        phase="single_context",
                        attempt="evidence_repair",
                        parse_error=None,
                        context=context,
                    )
                )
        else:
            _finalize_provider_response_attempt(
                completion,
                callback=provider_attempt_callback,
                semantic_phase=str(
                    completion.get("provider_semantic_phase")
                    or "judgment_initial"
                ),
                logical_attempt_index=1,
                parse_error=None,
            )
            calls.append(
                _judgment_call_record(
                    completion,
                    phase="single_context",
                    attempt="initial",
                    parse_error=None,
                    context=context,
                )
            )

        parsed["relevant"] = bool(
            parsed.get("is_relevant_to_answer")
            if "is_relevant_to_answer" in parsed
            else parsed["label"] in RELEVANT_LABELS
        )
        parsed["identity_conflict"] = parsed["paper_role"] == "distractor"
        parsed["visual_conflict"] = False

        actual_judgment_image_paths = _completion_attached_paths(
            validated_completion, context["image_paths"]
        )
        actual_judgment_image_path_set = set(actual_judgment_image_paths)
        actual_judgment_image_chunk_ids = [
            chunk_id
            for chunk_id in context["selected_image_chunk_ids"]
            if (
                (record := context["records_by_id"].get(chunk_id)) is not None
                and readable_image_path(record) in actual_judgment_image_path_set
            )
        ]

        result = {
            "query_id": query.query_id,
            **_candidate_payload(candidate),
            "status": "complete",
            "prompt_version": self.judgment_prompt_version,
            "paper_set_policy": self.paper_set_policy,
            "question_type_version": JUDGMENT_QUESTION_TYPE_VERSION,
            "question_type": judgment_question_type(query),
            "few_shot_example_ids": self.judgment_example_ids(query),
            "cache_key": self.judgment_cache_key(
                query,
                candidate,
                records,
                owner_resolution=owner_resolution,
            ),
            "named_owner_resolution": owner_resolution,
            "paper_content_sha256": _records_sha256(records),
            "paper_image_content_sha256": _image_content_sha256(records),
            "paper_chunk_count": len(records),
            "paper_context_selector_version": PAPER_CONTEXT_SELECTOR_VERSION,
            "paper_context_compacted": context["compacted"],
            "context_chunk_count": len(context["selected_chunk_ids"]),
            "omitted_chunk_count": len(context["omitted_chunk_ids"]),
            "context_chunk_ids": context["selected_chunk_ids"],
            "omitted_chunk_ids": context["omitted_chunk_ids"],
            "paper_full_text_characters": context["full_text_characters"],
            "paper_context_characters": context["selected_text_characters"],
            "paper_context_character_limit": context["character_limit"],
            "paper_image_compacted": (
                context["total_readable_image_count"] > len(context["image_paths"])
            ),
            "paper_readable_image_count": context["total_readable_image_count"],
            "requested_image_count": len(context["image_paths"]),
            "requested_image_chunk_ids": context["selected_image_chunk_ids"],
            "attached_image_count": len(actual_judgment_image_paths),
            "attached_image_chunk_ids": actual_judgment_image_chunk_ids,
            "omitted_image_chunk_ids": context["omitted_image_chunk_ids"],
            "paper_context_selection": context["selection"],
            "base_judgment_call_count": 1,
            "logical_judgment_attempt_count": len(calls),
            "judgment_call_count": sum(
                int(call.get("provider_invocation_count") or 1) for call in calls
            ),
            "provider_invocation_count": sum(
                int(call.get("provider_invocation_count") or 1) for call in calls
            ),
            **parsed,
            "judgment": parsed,
            "calls": calls,
        }
        return result

    def _deterministic_wrong_owner_judgment(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        records: list[Record],
        owner_resolution: NamedOwnerResolution,
    ) -> dict[str, Any]:
        """Checkpoint a decisive canonical-title mismatch without an AOAI call."""

        resolved_title = owner_resolution["title"]
        mismatch = (
            f"candidate canonical title {candidate.title!r} is not the uniquely "
            f"resolved owner {resolved_title!r}"
        )
        parsed = {
            "is_relevant_to_answer": False,
            "has_usable_answer_evidence": False,
            "send_to_answer_agent": False,
            "model_is_relevant_to_answer": False,
            "model_has_usable_answer_evidence": False,
            "model_evidence_chunk_ids": [],
            "paper_role": "distractor",
            "label": "irrelevant",
            "answerable_from_this_paper": False,
            "satisfied_constraints": [],
            "missing_constraints": [
                f"paper-local object owned by {resolved_title}"
            ],
            "blocking_mismatches": [mismatch],
            "visual": {"required": False, "status": "not_needed"},
            "evidence": [],
            "evidence_chunk_ids": [],
            "candidate_answer": {"units": [], "rows": []},
            "confidence": 1.0,
            "reason": (
                "Authoritative candidate metadata conflicts with the uniquely "
                "resolved owner of the requested paper-local object."
            ),
            "relevant": False,
            "identity_conflict": True,
            "visual_conflict": False,
        }
        readable_image_chunk_ids = [
            str(record.get("chunk_id") or "")
            for record in records
            if readable_image_path(record)
        ]
        selected_ids: list[str] = []
        omitted_ids = [str(record.get("chunk_id") or "") for record in records]
        return {
            "query_id": query.query_id,
            **_candidate_payload(candidate),
            "status": "complete",
            "prompt_version": JUDGMENT_PROMPT_VERSION,
            "question_type_version": JUDGMENT_QUESTION_TYPE_VERSION,
            "question_type": judgment_question_type(query),
            "few_shot_example_ids": example_manifest(query)["judgment"],
            "cache_key": self.judgment_cache_key(
                query,
                candidate,
                records,
                owner_resolution=owner_resolution,
            ),
            "named_owner_resolution": owner_resolution,
            "deterministic_filter": NAMED_OWNER_RESOLVER_VERSION,
            "paper_content_sha256": _records_sha256(records),
            "paper_image_content_sha256": _image_content_sha256(records),
            "paper_chunk_count": len(records),
            "paper_context_selector_version": PAPER_CONTEXT_SELECTOR_VERSION,
            "paper_context_compacted": bool(records),
            "context_chunk_count": 0,
            "omitted_chunk_count": len(omitted_ids),
            "context_chunk_ids": selected_ids,
            "omitted_chunk_ids": omitted_ids,
            "paper_full_text_characters": sum(
                len(str(record.get("text") or "")) for record in records
            ),
            "paper_context_characters": 0,
            "paper_context_character_limit": self.max_paper_context_chars,
            "paper_image_compacted": bool(readable_image_chunk_ids),
            "paper_readable_image_count": len(readable_image_chunk_ids),
            "attached_image_count": 0,
            "attached_image_chunk_ids": [],
            "omitted_image_chunk_ids": readable_image_chunk_ids,
            "paper_context_selection": [],
            "base_judgment_call_count": 0,
            "logical_judgment_attempt_count": 0,
            "judgment_call_count": 0,
            "provider_invocation_count": 0,
            **parsed,
            "judgment": {
                key: value
                for key, value in parsed.items()
                if key not in {"relevant", "identity_conflict", "visual_conflict"}
            },
            "calls": [],
        }

    def _paper_context(
        self,
        query: Query,
        candidate: CandidatePaper,
        records: list[Record],
        *,
        max_chars: int | None = None,
    ) -> PaperContext:
        """Select one deterministic, query-focused context for a whole paper."""

        require_production_query(query)
        context_char_limit = self.max_paper_context_chars if max_chars is None else max_chars
        if context_char_limit < 8_000:
            raise ValueError("paper context character limit must be at least 8000")
        if not records:
            raise ValueError("paper has no readable MinerU chunks")
        chunk_ids: list[str] = []
        for record in records:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id:
                raise ValueError(
                    f"MinerU record has no chunk_id: paper_id={record.get('paper_id')!r}"
                )
            if chunk_id in chunk_ids:
                raise ValueError(
                    f"duplicate MinerU chunk_id in {candidate.paper_id}: {chunk_id}"
                )
            chunk_ids.append(chunk_id)

        focus = _query_selection_focus(query)
        scores, reasons = _paper_record_scores(
            records, focus, source_question=query.question
        )
        full_parts = [
            self._format_record(record, text=str(record.get("text") or ""))
            for record in records
        ]
        full_text = "\n\n".join(full_parts)

        explicit_indices = [
            index
            for index, item_reasons in enumerate(reasons)
            if "explicit_object_reference" in item_reasons
        ]
        title_indices = [
            index
            for index, record in enumerate(records)
            if index == 0 or record_source_type(record) == "title_abstract"
        ]
        scoped_figure_mention = requires_explicit_visual_mention(query.question)
        scoped_metric_tables = (
            requires_coordinated_metric_table_context(query.question)
            and any(record_source_type(record) == "table" for record in records)
        )
        scoped_image_source_type = (
            "figure" if scoped_figure_mention else "table" if scoped_metric_tables else None
        )
        image_indices = _ranked_image_record_indices(
            records,
            scores=scores,
            reasons=reasons,
            limit=self.max_paper_images,
            question=query.question,
            scoped_source_type=scoped_image_source_type,
        )
        if scoped_figure_mention:
            # The query explicitly scopes a literal term to a primary/main
            # Figure. Keep the judgment evidence boundary equally scoped:
            # candidate metadata already carries the title, while full body and
            # table text can leak the term from the wrong location.
            selected_indices = [
                index
                for index, record in enumerate(records)
                if record_source_type(record) == "figure"
            ]
            if not selected_indices:
                selected_indices = title_indices
            rendered_by_index = {
                index: self._format_record(
                    records[index],
                    text=_without_corpus_title_prefix(
                        str(records[index].get("text") or "")
                    ),
                )
                for index in selected_indices
            }
            scoped_text_length = sum(
                len(rendered_by_index[index]) for index in selected_indices
            ) + max(0, len(selected_indices) - 1) * 2
            if scoped_text_length > context_char_limit:
                priority = sorted(
                    selected_indices, key=lambda index: (-scores[index], index)
                )
                kept: dict[int, str] = {}
                used_chars = 0
                for position, index in enumerate(priority):
                    separator_chars = 2 if kept else 0
                    available = context_char_limit - used_chars - separator_chars
                    if available <= 256:
                        break
                    remaining = len(priority) - position
                    target = min(
                        available,
                        max(512, available // max(1, remaining)),
                    )
                    rendered = _bounded_formatted_record(
                        self,
                        records[index],
                        focus=focus,
                        max_chars=target,
                        source_text=_without_corpus_title_prefix(
                            str(records[index].get("text") or "")
                        ),
                    )
                    if len(rendered) > available:
                        continue
                    kept[index] = rendered
                    used_chars += separator_chars + len(rendered)
                if not kept:
                    raise ValueError(
                        f"{query.query_id}/{candidate.paper_id}: figure context "
                        "selection could not fit any chunk"
                    )
                selected_indices = sorted(kept)
                rendered_by_index = kept
            compacted = (
                scoped_text_length > context_char_limit
                or len(selected_indices) != len(records)
            )
        elif scoped_metric_tables:
            # Coordinated benchmark-value questions are especially vulnerable
            # to long-paper dilution and near-name values in prose. Candidate
            # metadata already establishes identity, so expose the complete
            # MinerU table set and its table images. If a paper has no tables,
            # this branch is not taken and the normal full-text path remains.
            selected_indices = [
                index
                for index, record in enumerate(records)
                if record_source_type(record) == "table"
            ]
            rendered_by_index = {
                index: self._format_record(
                    records[index],
                    text=_without_corpus_title_prefix(
                        str(records[index].get("text") or "")
                    ),
                )
                for index in selected_indices
            }
            scoped_text_length = sum(
                len(rendered_by_index[index]) for index in selected_indices
            ) + max(0, len(selected_indices) - 1) * 2
            if scoped_text_length > context_char_limit:
                priority = sorted(
                    selected_indices, key=lambda index: (-scores[index], index)
                )
                kept: dict[int, str] = {}
                used_chars = 0
                for index in priority:
                    separator_chars = 2 if kept else 0
                    available = context_char_limit - used_chars - separator_chars
                    if available <= 512:
                        break
                    rendered = rendered_by_index[index]
                    if len(rendered) > available:
                        rendered = _bounded_formatted_record(
                            self,
                            records[index],
                            focus=focus,
                            max_chars=available,
                            source_text=_without_corpus_title_prefix(
                                str(records[index].get("text") or "")
                            ),
                        )
                    if len(rendered) > available:
                        continue
                    kept[index] = rendered
                    used_chars += separator_chars + len(rendered)
                if not kept:
                    raise ValueError(
                        f"{query.query_id}/{candidate.paper_id}: table context "
                        "selection could not fit any chunk"
                    )
                selected_indices = sorted(kept)
                rendered_by_index = kept
            compacted = len(selected_indices) != len(records)
        elif len(full_text) <= context_char_limit:
            selected_indices = list(range(len(records)))
            rendered_by_index = dict(enumerate(full_parts))
            compacted = False
        else:
            compacted = True
            # Exact requested objects are non-negotiable.  The title/abstract
            # anchors paper identity, and each selected image needs its source
            # chunk in the same request.  All remaining chunks compete by a
            # deterministic query-only score; immediate neighbours are mildly
            # promoted to preserve local reading context.
            forced = _ordered_unique_ints(
                [*explicit_indices, *title_indices, *image_indices]
            )
            ranked = sorted(
                range(len(records)), key=lambda index: (-scores[index], index)
            )
            neighbour_indices: list[int] = []
            for index in [*explicit_indices, *ranked[:8]]:
                if index > 0:
                    neighbour_indices.append(index - 1)
                if index + 1 < len(records):
                    neighbour_indices.append(index + 1)
            priority = _ordered_unique_ints(
                [*forced, *neighbour_indices, *ranked]
            )
            selected: set[int] = set()
            rendered_by_index: dict[int, str] = {}
            used_chars = 0
            forced_set = set(forced)
            forced_remaining = len(forced)
            for index in priority:
                record = records[index]
                separator_chars = 2 if selected else 0
                available = context_char_limit - used_chars - separator_chars
                if available <= 256:
                    break
                full_rendered = full_parts[index]
                is_forced = index in forced_set
                if is_forced:
                    # Do not let one giant essential chunk consume the space
                    # needed by later explicit objects or selected images.
                    fair_share = max(512, available // max(1, forced_remaining))
                    target = min(available, max(512, fair_share))
                    rendered = _bounded_formatted_record(
                        self,
                        record,
                        focus=focus,
                        max_chars=target,
                    )
                    forced_remaining -= 1
                elif len(full_rendered) <= available:
                    rendered = full_rendered
                else:
                    # Continue looking: a later, shorter high-scoring record may
                    # still fit.  Non-essential chunks are never split into a
                    # second request.
                    continue
                if len(rendered) > available:
                    continue
                selected.add(index)
                rendered_by_index[index] = rendered
                used_chars += separator_chars + len(rendered)

            if not selected:
                raise ValueError(
                    f"{query.query_id}/{candidate.paper_id}: paper context selection "
                    "could not fit any chunk"
                )
            selected_indices = sorted(selected)

        selected_set = set(selected_indices)
        selected_parts = [rendered_by_index[index] for index in selected_indices]
        text = "\n\n".join(selected_parts)
        if len(text) > context_char_limit:
            raise AssertionError("paper context exceeds configured character limit")
        selected_image_indices = [
            index for index in image_indices if index in selected_set
        ]
        image_paths: list[str] = []
        for index in selected_image_indices:
            image_path = readable_image_path(records[index])
            if image_path and image_path not in image_paths:
                image_paths.append(image_path)

        readable_image_paths = {
            path
            for record in records
            if (path := readable_image_path(record))
        }
        attached_image_paths = set(image_paths)
        selected_image_chunk_ids = [
            chunk_ids[index]
            for index in selected_indices
            if (path := readable_image_path(records[index]))
            and path in attached_image_paths
        ]
        omitted_image_chunk_ids = [
            chunk_ids[index]
            for index, record in enumerate(records)
            if (path := readable_image_path(record))
            and path not in attached_image_paths
        ]

        records_by_id = {
            chunk_ids[index]: _record_with_prompt_visible_text(
                records[index], rendered_by_index[index]
            )
            for index in selected_indices
        }
        omitted_ids = [
            chunk_id
            for index, chunk_id in enumerate(chunk_ids)
            if index not in selected_set
        ]
        selection = [
            {
                "chunk_id": chunk_ids[index],
                "score": round(scores[index], 6),
                "reasons": reasons[index] or ["original_order_fallback"],
                "text_characters": len(rendered_by_index[index]),
                "image_attached": index in selected_image_indices,
            }
            for index in selected_indices
        ]
        return {
            "text": text,
            "records_by_id": records_by_id,
            "image_paths": image_paths,
            "compacted": compacted,
            "total_chunk_count": len(records),
            "selected_chunk_ids": list(records_by_id),
            "omitted_chunk_ids": omitted_ids,
            "full_text_characters": len(full_text),
            "selected_text_characters": len(text),
            "character_limit": context_char_limit,
            "total_readable_image_count": len(readable_image_paths),
            "selected_image_chunk_ids": selected_image_chunk_ids,
            "omitted_image_chunk_ids": omitted_image_chunk_ids,
            "selection": selection,
        }

    def _prompt_content_filter_fallback_context(
        self,
        *,
        query: Query,
        records: list[Record],
        original_context: PaperContext,
        include_title_abstract: bool = True,
    ) -> PaperContext:
        """Build a fresh title/abstract-only context after a prompt rejection."""

        chunk_ids = [str(record["chunk_id"]) for record in records]
        title_indices = (
            [
                index
                for index, record in enumerate(records)
                if str(record.get("chunk_type") or "") == "title_abstract"
            ]
            if include_title_abstract
            else []
        )
        selected_indices: list[int] = []
        rendered_by_index: dict[int, str] = {}
        used_characters = 0
        for position, index in enumerate(title_indices):
            separator_characters = 2 if selected_indices else 0
            remaining_characters = (
                self.max_paper_context_chars
                - used_characters
                - separator_characters
            )
            remaining_records = len(title_indices) - position
            if remaining_characters <= 0:
                break
            fair_share = remaining_characters // max(1, remaining_records)
            full_rendered = self._format_record(
                records[index], text=str(records[index].get("text") or "")
            )
            rendered = (
                full_rendered
                if len(full_rendered) <= fair_share
                else _bounded_formatted_record(
                    self,
                    records[index],
                    focus=query.question,
                    max_chars=fair_share,
                )
            )
            if not rendered:
                continue
            selected_indices.append(index)
            rendered_by_index[index] = rendered
            used_characters += separator_characters + len(rendered)

        if selected_indices:
            text = "\n\n".join(
                rendered_by_index[index] for index in selected_indices
            )
        else:
            text = (
                "[No title_abstract chunk is available. Judge only from the "
                "candidate metadata above and cite no paper chunk.]"
            )
        selected_set = set(selected_indices)
        records_by_id = {
            chunk_ids[index]: _record_with_prompt_visible_text(
                records[index], rendered_by_index[index]
            )
            for index in selected_indices
        }
        omitted_ids = [
            chunk_id
            for index, chunk_id in enumerate(chunk_ids)
            if index not in selected_set
        ]
        omitted_image_chunk_ids = [
            chunk_ids[index]
            for index, record in enumerate(records)
            if readable_image_path(record)
        ]
        selection = [
            {
                "chunk_id": chunk_ids[index],
                "score": 0.0,
                "reasons": ["prompt_content_filter_title_abstract_fallback"],
                "text_characters": len(rendered_by_index[index]),
                "image_attached": False,
            }
            for index in selected_indices
        ]
        return {
            "text": text,
            "records_by_id": records_by_id,
            "image_paths": [],
            "compacted": True,
            "total_chunk_count": len(records),
            "selected_chunk_ids": list(records_by_id),
            "omitted_chunk_ids": omitted_ids,
            "full_text_characters": original_context["full_text_characters"],
            "selected_text_characters": len(text),
            "character_limit": self.max_paper_context_chars,
            "total_readable_image_count": original_context[
                "total_readable_image_count"
            ],
            "selected_image_chunk_ids": [],
            "omitted_image_chunk_ids": omitted_image_chunk_ids,
            "selection": selection,
        }

    def _judgment_prompt(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        context: PaperContext,
    ) -> str:
        image_legend = self._image_legend(
            context["records_by_id"], context["image_paths"]
        )
        selected_chunk_count = len(context["selected_chunk_ids"])
        omitted_chunk_count = len(context["omitted_chunk_ids"])
        context_coverage = {
            "paper_context_complete": (
                not context["compacted"]
                and selected_chunk_count == context["total_chunk_count"]
                and omitted_chunk_count == 0
            ),
            "selected_chunk_count": selected_chunk_count,
            "total_chunk_count": context["total_chunk_count"],
            "omitted_chunk_count": omitted_chunk_count,
        }
        renderer = (
            render_selected_evidence_prompt
            if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            else render_judgment_prompt
        )
        return renderer(
            query=query,
            query_payload=_production_query_payload(query),
            candidate_payload=_candidate_payload(candidate),
            context_coverage=context_coverage,
            paper_text=context["text"],
            image_legend=image_legend,
        )

    def _parse_selected_evidence(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        payload: dict[str, Any],
        allowed_records: dict[str, Record],
        attached_image_paths: Iterable[str] | None,
    ) -> dict[str, Any]:
        """Validate atomic source facts for one externally selected paper."""

        if set(payload) != {"evidence_facts"}:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: fixed-selected extraction "
                "must return exactly the evidence_facts field"
            )
        raw_facts = payload["evidence_facts"]
        if not isinstance(raw_facts, list):
            raise ReadingResponseError("evidence_facts must be a list")
        if len(raw_facts) > 64:
            raise ReadingResponseError("evidence_facts must contain at most 64 items")

        attached_images = {
            str(Path(path).resolve()) for path in (attached_image_paths or [])
        }
        extracted_facts: list[dict[str, str]] = []
        dropped_evidence_facts: list[dict[str, Any]] = []
        seen_facts: set[tuple[str, str, str, str]] = set()
        evidence_chunk_ids: list[str] = []
        has_attached_visual_fact = False
        for index, raw_fact in enumerate(raw_facts):
            if not isinstance(raw_fact, dict) or set(raw_fact) != {
                "chunk_id",
                "purpose",
                "fact",
                "source_excerpt",
            }:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] must contain exactly chunk_id, "
                    "purpose, fact, and source_excerpt"
                )
            chunk_id = str(raw_fact.get("chunk_id") or "").strip()
            purpose = str(raw_fact.get("purpose") or "").strip()
            fact = str(raw_fact.get("fact") or "").strip()
            source_excerpt = str(raw_fact.get("source_excerpt") or "").strip()
            if not chunk_id or not fact:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] requires non-empty chunk_id and fact"
                )
            if purpose not in _SELECTED_EVIDENCE_PURPOSES:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] has invalid purpose {purpose!r}"
                )
            if len(fact) > 1_000 or len(source_excerpt) > 2_000:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] exceeds the fact/excerpt length limit"
                )
            record = allowed_records.get(chunk_id)
            if record is None:
                raise JudgmentEvidenceChunkError(
                    f"{query.query_id}/{candidate.paper_id}: model invented or "
                    f"cross-cited chunk_id {chunk_id!r}"
                )
            if str(record.get("paper_id") or "") != candidate.paper_id:
                raise JudgmentEvidenceChunkError(
                    f"{query.query_id}: chunk {chunk_id!r} belongs to another paper"
                )
            image_path = readable_image_path(record)
            actually_attached_image = bool(
                image_path
                and str(Path(image_path).resolve()) in attached_images
            )
            if source_excerpt:
                if not _source_excerpt_is_visible(source_excerpt, record):
                    # Fail closed at item granularity. Table OCR is especially
                    # prone to harmless spacing/markup paraphrases even after a
                    # repair call. An unverified excerpt must never reach Stage
                    # 2, but one bad item must not abort the other selected
                    # papers or the whole 55-query run. If every item is
                    # dropped, fixed-selected answer fallback re-reads original
                    # query-ranked chunks from this same paper.
                    dropped_evidence_facts.append(
                        {
                            "index": index,
                            "chunk_id": chunk_id,
                            "reason": "source_excerpt_not_visible_verbatim",
                        }
                    )
                    continue
            elif not actually_attached_image:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] may omit source_excerpt only for "
                    "an actually attached image"
                )
            elif purpose != "visual_fact":
                raise ReadingResponseError(
                    f"evidence_facts[{index}] may omit source_excerpt only when "
                    "purpose is visual_fact"
                )
            if purpose == "visual_fact" and not actually_attached_image:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] visual_fact requires an actually "
                    "attached source image"
                )
            if purpose == "visual_fact" and actually_attached_image:
                has_attached_visual_fact = True
            normalized = {
                "chunk_id": chunk_id,
                "purpose": purpose,
                "fact": fact,
                "source_excerpt": source_excerpt,
            }
            fingerprint = (chunk_id, purpose, fact, source_excerpt)
            if fingerprint in seen_facts:
                raise ReadingResponseError(
                    f"evidence_facts[{index}] duplicates an earlier atomic fact"
                )
            seen_facts.add(fingerprint)
            extracted_facts.append(normalized)
            if chunk_id not in evidence_chunk_ids:
                evidence_chunk_ids.append(chunk_id)

        evidence_contains_figure = any(
            record_source_type(allowed_records[chunk_id]) == "figure"
            for chunk_id in evidence_chunk_ids
        )
        if (
            requires_visual_image(query.question)
            and evidence_contains_figure
            and not has_attached_visual_fact
        ):
            raise ReadingResponseError(
                "an explicit visual query requires at least one visual_fact from "
                "an actually attached image"
            )

        has_evidence = bool(evidence_chunk_ids)
        evidence = []
        for chunk_id in evidence_chunk_ids:
            chunk_facts = [
                item for item in extracted_facts if item["chunk_id"] == chunk_id
            ]
            record = allowed_records[chunk_id]
            copied_excerpts = _ordered_unique(
                item["source_excerpt"]
                for item in chunk_facts
                if item["source_excerpt"]
            )
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "source_type": record_source_type(record),
                    "locator": coarse_locator(record),
                    "purpose": "answer",
                    # A free-form model fact is a routing hypothesis, not source
                    # evidence. Use only excerpts whose exact occurrence in the
                    # cited chunk was validated. Image-only facts have no OCR
                    # excerpt, so their concise fact remains a focus hint while
                    # the actually attached pixels stay authoritative.
                    "quote_or_value": " | ".join(copied_excerpts)
                    or " | ".join(item["fact"] for item in chunk_facts),
                }
            )
        # A compound query can pair a visual Figure/axis lookup from one paper
        # with an ordinary reported Table cell from another. Do not turn the
        # table paper into a visual source merely because the query has one
        # visual clause. The final query-level validator still requires at
        # least one visual fact, and an extracted visual_fact remains
        # paper-specific here.
        visual_required = has_attached_visual_fact
        return {
            "checkpoint_kind": FIXED_SELECTED_CHECKPOINT_KIND,
            "is_relevant_to_answer": True,
            "has_usable_answer_evidence": has_evidence,
            "send_to_answer_agent": has_evidence,
            "model_is_relevant_to_answer": None,
            "model_has_usable_answer_evidence": has_evidence,
            "model_evidence_chunk_ids": list(evidence_chunk_ids),
            "routing_adjustments": [
                "paper relevance is fixed by the external selected-paper input",
                *(
                    [
                        f"dropped {len(dropped_evidence_facts)} evidence fact(s) "
                        "whose source_excerpt was not visible verbatim"
                    ]
                    if dropped_evidence_facts
                    else []
                ),
            ],
            "question_type": judgment_question_type(query),
            "question_type_version": JUDGMENT_QUESTION_TYPE_VERSION,
            "paper_role": "answer_source" if has_evidence else "constraint_source",
            "label": "partial_answer" if has_evidence else "supporting_only",
            "answerable_from_this_paper": has_evidence,
            "satisfied_constraints": [],
            "missing_constraints": [],
            "blocking_mismatches": [],
            "visual": {
                "required": visual_required,
                "status": (
                    "inspected"
                    if visual_required and has_attached_visual_fact
                    else "missing" if visual_required else "not_needed"
                ),
            },
            "evidence": evidence,
            "evidence_chunk_ids": evidence_chunk_ids,
            "extracted_facts": extracted_facts,
            "dropped_evidence_facts": dropped_evidence_facts,
            "candidate_answer": {"units": [], "rows": []},
            "confidence": 0.0,
            "reason": (
                "Atomic facts were extracted from an externally selected paper."
                if has_evidence
                else "No requested source fact was found in the supplied stored context."
            ),
        }

    def _parse_simple_judgment(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        payload: dict[str, Any],
        allowed_records: dict[str, Record],
        attached_image_paths: Iterable[str] | None,
        resolved_target_owner: bool,
    ) -> dict[str, Any]:
        """Validate the three-field Stage-1 contract and derive safe routing."""

        required_keys = {
            "is_relevant_to_answer",
            "has_usable_answer_evidence",
            "evidence_chunk_ids",
        }
        if set(payload) != required_keys:
            missing = sorted(required_keys - set(payload))
            extra = sorted(set(payload) - required_keys)
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: Stage-1 response must "
                f"contain exactly {sorted(required_keys)}; missing={missing}, "
                f"extra={extra}"
            )

        raw_relevant = payload["is_relevant_to_answer"]
        raw_usable = payload["has_usable_answer_evidence"]
        if not isinstance(raw_relevant, bool):
            raise ReadingResponseError("is_relevant_to_answer must be boolean")
        if not isinstance(raw_usable, bool):
            raise ReadingResponseError(
                "has_usable_answer_evidence must be boolean"
            )
        effective_relevant = bool(raw_relevant or resolved_target_owner)
        compound_component_only = bool(
            raw_relevant
            and not raw_usable
            and not resolved_target_owner
            and _candidate_is_only_a_compound_name_component(
                query.question, candidate, allowed_records.values()
            )
        )
        if compound_component_only:
            effective_relevant = False
        open_ended_paper_list = bool(
            not resolved_target_owner
            and re.match(r"(?i)^\s*which\b.+\bpapers?\b", query.question)
        )
        if open_ended_paper_list and not raw_usable:
            effective_relevant = False
        raw_chunk_ids = payload["evidence_chunk_ids"]
        if not isinstance(raw_chunk_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in raw_chunk_ids
        ):
            raise ReadingResponseError(
                "evidence_chunk_ids must be a list of non-empty strings"
            )
        if len(raw_chunk_ids) != len(set(raw_chunk_ids)):
            raise ReadingResponseError("evidence_chunk_ids must not contain duplicates")
        if effective_relevant and raw_usable and not raw_chunk_ids:
            raise ReadingResponseError(
                "has_usable_answer_evidence=true requires at least one exact "
                "visible evidence chunk ID"
            )

        validated_ids: list[str] = []
        if raw_usable:
            for chunk_id in raw_chunk_ids:
                record = allowed_records.get(chunk_id)
                if record is None:
                    raise JudgmentEvidenceChunkError(
                        f"{query.query_id}/{candidate.paper_id}: model invented or "
                        f"cross-cited chunk_id {chunk_id!r}"
                    )
                if str(record.get("paper_id") or "") != candidate.paper_id:
                    raise JudgmentEvidenceChunkError(
                        f"{query.query_id}: chunk {chunk_id!r} belongs to another paper"
                    )
                validated_ids.append(chunk_id)

        attached_images = {
            str(Path(path).resolve()) for path in (attached_image_paths or [])
        }
        question_type = judgment_question_type(query)
        explicit_figure_keys = {
            key
            for key in _explicit_object_keys(query.question)
            if key[0] == "figure"
        }
        attached_exact_visual_ids = [
            chunk_id
            for chunk_id, record in allowed_records.items()
            if (
                resolved_target_owner
                and question_type == "visual"
                and record_source_type(record) == "figure"
                and explicit_figure_keys.intersection(
                    _metadata_figure_object_keys(record)
                )
                and (path := readable_image_path(record))
                and str(Path(path).resolve()) in attached_images
            )
        ]
        deterministic_visual_handoff = bool(attached_exact_visual_ids)
        if (
            deterministic_visual_handoff
            and raw_usable
            and not set(validated_ids).intersection(attached_exact_visual_ids)
        ):
            raise ReadingResponseError(
                "visual has_usable_answer_evidence=true for a resolved owning "
                "paper requires the actually attached exact requested Figure "
                "chunk ID"
            )
        candidate_visual_evidence = any(
            record_source_type(allowed_records[chunk_id]) == "figure"
            for chunk_id in validated_ids
        )
        if (
            effective_relevant
            and raw_usable
            and question_type == "visual"
            and candidate_visual_evidence
        ):
            cited_attached_image = any(
                (path := readable_image_path(allowed_records[chunk_id]))
                and str(Path(path).resolve()) in attached_images
                for chunk_id in validated_ids
            )
            if not cited_attached_image:
                raise ReadingResponseError(
                    "visual has_usable_answer_evidence=true requires a cited chunk "
                    "mapped to an actually attached image"
                )

        if deterministic_visual_handoff and not raw_usable:
            validated_ids = attached_exact_visual_ids
        send_to_answer_agent = bool(
            effective_relevant
            and (raw_usable or deterministic_visual_handoff)
            and validated_ids
        )
        effective_ids = validated_ids if send_to_answer_agent else []
        evidence = [
            {
                "chunk_id": chunk_id,
                "source_type": record_source_type(allowed_records[chunk_id]),
                "locator": coarse_locator(allowed_records[chunk_id]),
                "purpose": "answer",
                "quote_or_value": "",
            }
            for chunk_id in effective_ids
        ]
        routing_adjustments: list[str] = []
        if open_ended_paper_list and raw_relevant and not raw_usable:
            routing_adjustments.append(
                "relevance was set false because an open-ended paper list "
                "requires usable evidence that this candidate satisfies every "
                "inclusion condition"
            )
        if compound_component_only:
            routing_adjustments.append(
                "relevance was set false because the candidate title occurred "
                "only as a base/component inside a longer requested compound "
                "method name, and the supplied context did not report that full name"
            )
        if resolved_target_owner and not raw_relevant:
            routing_adjustments.append(
                "relevance was set true because canonical metadata resolved this "
                "paper as the requested owner"
            )
        if raw_usable and not effective_relevant:
            routing_adjustments.append(
                "usable evidence was suppressed because the paper was not relevant"
            )
        if not raw_usable and raw_chunk_ids:
            routing_adjustments.append(
                "evidence IDs were cleared because usable evidence was false"
            )
        if deterministic_visual_handoff and not raw_usable:
            routing_adjustments.append(
                "usable visual evidence was set true because the uniquely "
                "resolved owner supplied an actually attached image mapped to "
                "the exact numbered object requested by the query"
            )

        if effective_relevant:
            paper_role = (
                "target_owner"
                if resolved_target_owner
                else "answer_source" if send_to_answer_agent else "constraint_source"
            )
            label = "partial_answer" if send_to_answer_agent else "supporting_only"
        else:
            paper_role = "topic_only"
            label = "irrelevant"
        # Preserve the v30 Stage-1 checkpoint contract. Candidate-local visual
        # enforcement is derived at the answer boundary from the cited source
        # type, where a compound query's ordinary Table paper can be separated
        # safely from its Figure-reading paper without invalidating completed
        # pairwise judgments.
        visual_required = bool(
            effective_relevant
            and send_to_answer_agent
            and question_type == "visual"
        )
        visual_status = (
            "inspected"
            if visual_required and send_to_answer_agent
            else "missing" if visual_required else "not_needed"
        )
        return {
            "is_relevant_to_answer": effective_relevant,
            "has_usable_answer_evidence": send_to_answer_agent,
            "send_to_answer_agent": send_to_answer_agent,
            "model_is_relevant_to_answer": raw_relevant,
            "model_has_usable_answer_evidence": raw_usable,
            "model_evidence_chunk_ids": list(raw_chunk_ids),
            "routing_adjustments": routing_adjustments,
            "question_type": question_type,
            "question_type_version": JUDGMENT_QUESTION_TYPE_VERSION,
            "paper_role": paper_role,
            "label": label,
            "answerable_from_this_paper": send_to_answer_agent,
            "satisfied_constraints": [],
            "missing_constraints": [],
            "blocking_mismatches": [],
            "visual": {"required": visual_required, "status": visual_status},
            "evidence": evidence,
            "evidence_chunk_ids": effective_ids,
            "candidate_answer": {"units": [], "rows": []},
            "confidence": 0.0,
            "reason": "Stage-1 routing was derived from the validated three-field response.",
        }

    def _parse_judgment(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        payload_text: str,
        allowed_records: dict[str, Record],
        attached_image_paths: Iterable[str] | None = None,
        resolved_target_owner: bool = False,
        allow_legacy_schema: bool = True,
    ) -> dict[str, Any]:
        payload = parse_json_object(payload_text)
        if not isinstance(payload, dict):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: "
                "judgment response is not a JSON object"
            )
        if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY:
            return self._parse_selected_evidence(
                query=query,
                candidate=candidate,
                payload=payload,
                allowed_records=allowed_records,
                attached_image_paths=attached_image_paths,
            )
        simple_keys = {
            "is_relevant_to_answer",
            "has_usable_answer_evidence",
            "evidence_chunk_ids",
        }
        if simple_keys.intersection(payload):
            return self._parse_simple_judgment(
                query=query,
                candidate=candidate,
                payload=payload,
                allowed_records=allowed_records,
                attached_image_paths=attached_image_paths,
                resolved_target_owner=resolved_target_owner,
            )
        if not allow_legacy_schema:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: current Stage-1 response "
                "must use exactly is_relevant_to_answer, "
                "has_usable_answer_evidence, and evidence_chunk_ids"
            )
        label = str(payload.get("label") or "")
        if label not in JUDGMENT_LABELS:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: "
                f"invalid judgment label {label!r}"
            )
        answerable = payload.get("answerable_from_this_paper")
        if not isinstance(answerable, bool):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: "
                "answerable_from_this_paper must be boolean"
            )
        reported_paper_role = str(payload.get("paper_role") or "uncertain")
        paper_role = reported_paper_role
        if paper_role not in {
            "target_owner",
            "answer_source",
            "comparison_source",
            "constraint_source",
            "option_source",
            "distractor",
            "topic_only",
            "uncertain",
        }:
            raise ReadingResponseError(f"invalid paper_role {paper_role!r}")
        if resolved_target_owner:
            # Candidate metadata is authoritative for identity.  Preserve the
            # model's relevance/evidence decision, but do not let a mistaken
            # role label suppress the safe Stage-2 target-owner recheck.
            paper_role = "target_owner"
        blocking_mismatches = _string_list(payload.get("blocking_mismatches") or [])
        if resolved_target_owner:
            blocking_mismatches = [
                mismatch
                for mismatch in blocking_mismatches
                if not _identity_only_blocking_mismatch(mismatch)
            ]
        if paper_role in {"distractor", "topic_only"} and label in RELEVANT_LABELS:
            raise ReadingResponseError(
                f"paper_role={paper_role} is incompatible with relevant label={label}"
            )
        if paper_role == "distractor" and not blocking_mismatches:
            raise ReadingResponseError(
                "paper_role=distractor requires a specific blocking_mismatch"
            )
        if label in RELEVANT_LABELS and blocking_mismatches:
            raise ReadingResponseError(
                f"{label} is incompatible with blocking_mismatches"
            )
        visual = payload.get("visual") or {"required": False, "status": "not_needed"}
        if not isinstance(visual, dict) or not isinstance(visual.get("required"), bool):
            raise ReadingResponseError("judgment visual must contain boolean required")
        visual_status = str(visual.get("status") or "")
        if visual_status not in {"not_needed", "inspected", "missing", "unreadable"}:
            raise ReadingResponseError("judgment visual.status is invalid")
        visual_required = visual["required"]
        if (
            paper_role == "distractor"
            and label == "irrelevant"
            and answerable is False
            and blocking_mismatches
            and visual_status == "not_needed"
        ):
            # ``status=not_needed`` already says that this decision did not use
            # a visual.  For an authoritative wrong-owner rejection only, make
            # the redundant boolean agree with that candidate-local status.
            # Never apply this to a possible owner or an answer-bearing label:
            # doing so could silently waive genuinely required visual evidence.
            visual_required = False
        if visual_required and visual_status == "not_needed":
            raise ReadingResponseError(
                "visual.required=true is incompatible with status=not_needed"
            )
        if not visual_required and visual_status != "not_needed":
            raise ReadingResponseError(
                "visual.required=false requires status=not_needed"
            )
        attached_images = {
            str(Path(path).resolve()) for path in (attached_image_paths or [])
        }
        if visual_status == "inspected" and not attached_images:
            raise ReadingResponseError(
                "visual.status=inspected requires an actually attached image"
            )
        if label == "direct_answer" and not answerable:
            raise ReadingResponseError(
                "direct_answer requires answerable_from_this_paper=true"
            )
        if label == "partial_answer" and not answerable:
            raise ReadingResponseError(
                "partial_answer requires answerable_from_this_paper=true because "
                "it contributes at least one requested answer unit"
            )
        if label == "direct_answer" and payload.get("missing_constraints"):
            raise ReadingResponseError(
                "direct_answer is incompatible with missing_constraints"
            )
        if label not in RELEVANT_LABELS and answerable:
            raise ReadingResponseError(
                f"{label} requires answerable_from_this_paper=false"
            )
        if (
            label in {"direct_answer", "partial_answer"}
            and visual_required
            and visual_status in {"missing", "unreadable"}
        ):
            raise ReadingResponseError(
                f"{label} is incompatible with unavailable required visual evidence"
            )
        evidence: list[dict[str, Any]] = []
        evidence_by_chunk_id: dict[str, dict[str, Any]] = {}
        raw_evidence = payload.get("evidence") or []
        if not isinstance(raw_evidence, list):
            raise ReadingResponseError("judgment evidence must be a list")
        for item in raw_evidence:
            if not isinstance(item, dict):
                raise ReadingResponseError("judgment evidence item must be an object")
            chunk_id = str(item.get("chunk_id") or "")
            record = allowed_records.get(chunk_id)
            if record is None:
                raise JudgmentEvidenceChunkError(
                    f"{query.query_id}/{candidate.paper_id}: model invented or cross-cited "
                    f"chunk_id {chunk_id!r}"
                )
            if record.get("paper_id") != candidate.paper_id:
                raise JudgmentEvidenceChunkError(
                    f"{query.query_id}: chunk {chunk_id!r} belongs to another paper"
                )
            normalized_evidence = {
                "chunk_id": chunk_id,
                "source_type": record_source_type(record),
                "locator": coarse_locator(record),
                "purpose": str(item.get("purpose") or "answer"),
                "quote_or_value": str(item.get("quote_or_value") or "").strip(),
            }
            previous = evidence_by_chunk_id.get(chunk_id)
            if previous is not None:
                if previous == normalized_evidence:
                    continue
                raise ReadingResponseError(
                    f"{query.query_id}/{candidate.paper_id}: duplicate evidence "
                    f"chunk_id {chunk_id!r} has conflicting purpose or quote; "
                    "emit each chunk_id once and use purpose=answer when one "
                    "chunk supports both a constraint and the answer"
                )
            evidence_by_chunk_id[chunk_id] = normalized_evidence
            evidence.append(normalized_evidence)
        if label in RELEVANT_LABELS and not evidence:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: {label} requires evidence"
            )
        if label == "mention_only" and any(
            item["purpose"] == "answer" for item in evidence
        ):
            raise ReadingResponseError(
                "mention_only cannot cite answer-purpose evidence; when the "
                "candidate directly reports a requested operand, preserve it as "
                "partial_answer with answerable_from_this_paper=true and a "
                "candidate_answer unit"
            )
        if label in {"direct_answer", "partial_answer"} and not any(
            item["purpose"] == "answer"
            and submission_evidence_eligible(allowed_records[item["chunk_id"]])
            for item in evidence
        ):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: {label} requires at least "
                "one submission-eligible answer evidence chunk"
            )
        if visual_status == "inspected":
            cited_images = {
                str(Path(path).resolve())
                for item in evidence
                if (record := allowed_records.get(str(item["chunk_id"]))) is not None
                if (path := readable_image_path(record))
            }
            if not cited_images.intersection(attached_images):
                raise ReadingResponseError(
                    "visual.status=inspected requires evidence from the actually "
                    "attached source image"
                )
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ReadingResponseError("judgment confidence must be a number")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ReadingResponseError("judgment confidence must be between 0 and 1")
        candidate_answer = payload.get("candidate_answer") or {}
        if not isinstance(candidate_answer, dict):
            raise ReadingResponseError("candidate_answer must be an object")
        if label in {"direct_answer", "partial_answer"}:
            _validate_stage1_visual_subfigure_count(
                query=query,
                candidate_answer=candidate_answer,
            )
        if (
            label in {"direct_answer", "partial_answer"}
            and is_aggregate_citation_count_query(query)
        ):
            _validate_stage1_citation_count(
                query=query,
                candidate_answer=candidate_answer,
                evidence=evidence,
                allowed_records=allowed_records,
            )
        if (
            label in {"direct_answer", "partial_answer"}
            and "multiple_choice" in query.answer_types
        ):
            units = candidate_answer.get("units")
            if not isinstance(units, list) or not units:
                raise ReadingResponseError(
                    f"{label} for a multiple-choice query requires "
                    "candidate_answer.units with matched_option_labels"
                )
            matched_labels: list[str] = []
            for unit_index, unit in enumerate(units):
                if not isinstance(unit, dict):
                    raise ReadingResponseError(
                        f"candidate_answer.units[{unit_index}] must be an object"
                    )
                if not str(unit.get("name") or "").strip():
                    raise ReadingResponseError(
                        f"candidate_answer.units[{unit_index}] requires a name"
                    )
                unit_value = unit.get("value")
                if unit_value is None or (
                    isinstance(unit_value, str) and not unit_value.strip()
                ):
                    raise ReadingResponseError(
                        f"candidate_answer.units[{unit_index}] requires a value"
                    )
                raw_labels = unit.get("matched_option_labels") or []
                if not isinstance(raw_labels, list) or any(
                    not isinstance(value, str) or not value
                    for value in raw_labels
                ):
                    raise ReadingResponseError(
                        "candidate_answer matched_option_labels must be a list "
                        "of non-empty released labels"
                    )
                matched_labels.extend(raw_labels)
            distinct_labels = set(matched_labels)
            unknown_labels = sorted(distinct_labels - set(query.option_labels))
            if unknown_labels:
                raise ReadingResponseError(
                    "candidate_answer matched_option_labels contains labels absent "
                    f"from the released options: {unknown_labels}"
                )
            if label == "direct_answer" and len(distinct_labels) != 1:
                raise ReadingResponseError(
                    "direct_answer for a multiple-choice query requires exactly "
                    "one released label across candidate_answer matched_option_labels; "
                    "if the owning paper still supplies requested components but no "
                    "complete option is uniquely identified, return partial_answer "
                    "and preserve those units with matched_option_labels=[]"
                )
            if label == "partial_answer" and len(distinct_labels) > 1:
                raise ReadingResponseError(
                    "partial_answer candidate units cannot claim multiple released "
                    "labels; use matched_option_labels=[] when an operand or set of "
                    "components does not identify one complete option"
                )
        if label in {"direct_answer", "partial_answer"} and "table" in query.answer_types:
            rows = candidate_answer.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ReadingResponseError(
                    f"{label} for a table query requires at least one complete "
                    "candidate_answer row"
                )
            required_columns = [
                str(column.get("name") or "")
                for column in query.table_schema or []
                if isinstance(column, dict) and column.get("name")
            ]
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ReadingResponseError(
                        f"candidate_answer.rows[{row_index}] must be an object"
                    )
                missing_or_blank = [
                    column
                    for column in required_columns
                    if column not in row
                    or row[column] is None
                    or (isinstance(row[column], str) and not row[column].strip())
                ]
                if missing_or_blank:
                    raise ReadingResponseError(
                        f"candidate_answer.rows[{row_index}] has missing or blank "
                        f"required cells: {missing_or_blank}"
                    )
        return {
            "paper_role": paper_role,
            "label": label,
            "answerable_from_this_paper": answerable,
            "satisfied_constraints": _string_list(payload.get("satisfied_constraints")),
            "missing_constraints": _string_list(payload.get("missing_constraints")),
            "blocking_mismatches": blocking_mismatches,
            "visual": {"required": visual_required, "status": visual_status},
            "evidence": evidence,
            "evidence_chunk_ids": [item["chunk_id"] for item in evidence],
            "candidate_answer": candidate_answer,
            "confidence": confidence,
            "reason": str(payload.get("reason") or "").strip(),
        }

    def _judgment_evidence_repair_prompt(
        self,
        *,
        original_prompt: str,
        rejected_response: str,
        error: ReadingResponseError,
        allowed_chunk_ids: list[str],
        eligible_chunk_ids: list[str],
    ) -> str:
        if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY:
            return (
                original_prompt
                + "\n\nREPAIR\n\nYour previous fixed-selected evidence "
                "extraction was rejected by the deterministic validator. "
                "Correct it once. Return exactly one JSON object whose only "
                "field is evidence_facts. Every item must contain exactly "
                "chunk_id, purpose, fact, and source_excerpt. Copy chunk IDs "
                "from the allowed list. Copy source_excerpt from the cited "
                "chunk text; use an empty excerpt only with purpose=visual_fact "
                "for an actually attached image. Do not decide paper relevance "
                "and do not answer the "
                "whole query. If no requested fact is visible, return "
                '{"evidence_facts": []}.\n'
                f"Validation error: {error}\n"
                f"Allowed selected-context chunk_ids: "
                f"{_json_dumps(allowed_chunk_ids)}\n"
                f"Submission-eligible chunk_ids: "
                f"{_json_dumps(eligible_chunk_ids)}\n"
                "<rejected_response>\n"
                + html.escape(rejected_response, quote=False)
                + "\n</rejected_response>\nReturn only the corrected one-field JSON."
            )
        return (
            original_prompt
            + "\n\nREPAIR\n\nYour previous Stage-1 response was rejected by "
            "the deterministic validator. Correct it once. Return exactly one JSON "
            "object with exactly these fields: is_relevant_to_answer (boolean), "
            "has_usable_answer_evidence (boolean), and evidence_chunk_ids (array "
            "of exact strings). Do not add an explanation or any other field. "
            "When usable evidence is true, copy the smallest sufficient set of "
            "chunk IDs exactly from the allowed list. When usable evidence is "
            "false, return an empty array. For a visual question, usable evidence "
            "must cite a chunk mapped to an actually attached image.\n"
            f"Validation error: {error}\n"
            f"Allowed selected-context chunk_ids: {_json_dumps(allowed_chunk_ids)}\n"
            f"Submission-eligible chunk_ids: {_json_dumps(eligible_chunk_ids)}\n"
            "<rejected_response>\n"
            + html.escape(rejected_response, quote=False)
            + "\n</rejected_response>\nReturn only the corrected three-field JSON."
        )

    # ---- Stage 2: accepted evidence -> answer -----------------------------

    def answer_cache_key(
        self, query: Query, judgments: list[dict[str, Any]]
    ) -> str:
        require_production_query(query)
        stable_judgments = [
            {
                key: judgment.get(key)
                for key in (
                    "checkpoint_kind",
                    "paper_id",
                    "rank",
                    "cache_key",
                    "question_type",
                    "question_type_version",
                    "is_relevant_to_answer",
                    "has_usable_answer_evidence",
                    "send_to_answer_agent",
                    "evidence_chunk_ids",
                    "extracted_facts",
                    "label",
                    "relevant",
                    "satisfied_constraints",
                    "missing_constraints",
                    "blocking_mismatches",
                    "evidence",
                    "candidate_answer",
                    "reason",
                    "paper_role",
                    "identity_conflict",
                    "visual",
                )
            }
            for judgment in sorted(
                judgments, key=lambda item: (int(item.get("rank") or 0), item.get("paper_id"))
            )
        ]
        return _json_sha256(
            {
                "prompt_version": self.answer_prompt_version,
                "paper_set_policy": self.paper_set_policy,
                "fixed_selected_answer_fallback_version": (
                    FIXED_SELECTED_ANSWER_FALLBACK_VERSION
                    if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
                    else None
                ),
                "answer_derivation_version": ANSWER_DERIVATION_VERSION,
                "citation_locator_version": CITATION_LOCATOR_VERSION,
                "split_caption_table_locator_version": (
                    SPLIT_CAPTION_TABLE_LOCATOR_VERSION
                ),
                "image_handoff_version": ANSWER_IMAGE_HANDOFF_VERSION,
                "submission_paper_selection_version": (
                    SUBMISSION_PAPER_SELECTION_VERSION
                ),
                "query_requirements_version": QUERY_REQUIREMENTS_VERSION,
                "few_shot_examples": example_manifest(query)["answer"],
                "query": _production_query_payload(query),
                "judgments": stable_judgments,
                "limits": {
                    "answer_context_chars": self.answer_context_chars,
                    "answer_neighbor_chunks": self.answer_neighbor_chunks,
                    "max_answer_images": self.max_answer_images,
                    "max_answer_papers": self.max_answer_papers,
                    "max_evidence": self.max_evidence,
                    "max_evidence_per_paper": self.max_evidence_per_paper,
                    "answer_max_completion_tokens": (
                        self.answer_max_completion_tokens
                    ),
                },
            }
        )

    def _fixed_selected_answer_pool(
        self,
        query: Query,
        candidates: list[CandidatePaper],
        judgments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep every fixed paper readable even when extraction returned empty.

        The model's empty extraction is preserved in the checkpoint. For the
        answer call only, Python supplies up to three query-ranked original
        chunks from that same externally selected paper. This cannot change the
        submitted paper set and does not assert that a fallback chunk is answer
        evidence; Stage 2 must still ground every submitted fact and locator.
        """

        by_paper: dict[str, dict[str, Any]] = {}
        for judgment in judgments:
            paper_id = str(judgment.get("paper_id") or "")
            if not paper_id or paper_id in by_paper:
                raise ValueError(
                    f"{query.query_id}: fixed-selected judgments must contain "
                    "each selected paper exactly once"
                )
            by_paper[paper_id] = judgment
        candidate_ids = [candidate.paper_id for candidate in candidates]
        if set(by_paper) != set(candidate_ids) or len(by_paper) != len(candidate_ids):
            raise ValueError(
                f"{query.query_id}: fixed-selected judgments do not exactly match "
                "the externally selected paper set"
            )

        output: list[dict[str, Any]] = []
        for candidate in candidates:
            judgment = by_paper[candidate.paper_id]
            if (
                judgment.get("is_relevant_to_answer") is True
                and judgment.get("has_usable_answer_evidence") is True
                and judgment.get("send_to_answer_agent") is True
                and judgment.get("evidence_chunk_ids")
            ):
                output.append(judgment)
                continue

            records = recover_split_caption_table_records(
                self.chunk_store.load_paper(candidate.paper_id)
            )
            if not records:
                raise FileNotFoundError(
                    f"{query.query_id}: selected paper is absent from MinerU "
                    f"corpus: {candidate.paper_id}"
                )
            scores, reasons = _paper_record_scores(
                records,
                _query_selection_focus(query),
                source_question=query.question,
            )
            image_indices = _ranked_image_record_indices(
                records,
                scores=scores,
                reasons=reasons,
                limit=min(3, self.max_answer_images),
                question=query.question,
            )
            explicit_indices = [
                index
                for index, item_reasons in enumerate(reasons)
                if "explicit_object_reference" in item_reasons
            ]
            eligible_indices = sorted(
                (
                    index
                    for index, record in enumerate(records)
                    if submission_evidence_eligible(record)
                ),
                key=lambda index: (-scores[index], index),
            )
            ranked_indices = sorted(
                range(len(records)), key=lambda index: (-scores[index], index)
            )
            selected_indices = _ordered_unique_ints(
                [
                    *image_indices,
                    *explicit_indices,
                    *eligible_indices,
                    *ranked_indices,
                ]
            )[:3]
            if not selected_indices:
                raise ReadingResponseError(
                    f"{query.query_id}/{candidate.paper_id}: no deterministic "
                    "fallback chunk is available"
                )
            evidence = [
                {
                    "chunk_id": str(records[index]["chunk_id"]),
                    "source_type": record_source_type(records[index]),
                    "locator": coarse_locator(records[index]),
                    "purpose": "answer",
                    "quote_or_value": "",
                }
                for index in selected_indices
            ]
            fallback = dict(judgment)
            fallback.update(
                {
                    "has_usable_answer_evidence": True,
                    "send_to_answer_agent": True,
                    "evidence_chunk_ids": [
                        item["chunk_id"] for item in evidence
                    ],
                    "evidence": evidence,
                    "paper_role": "answer_source",
                    "label": "partial_answer",
                    "answerable_from_this_paper": True,
                    "fixed_selected_answer_fallback_version": (
                        FIXED_SELECTED_ANSWER_FALLBACK_VERSION
                    ),
                    "fixed_selected_extraction_was_empty": True,
                    "routing_adjustments": [
                        *list(judgment.get("routing_adjustments") or []),
                        "deterministic query-ranked original context supplied "
                        "to Stage 2 after an empty extraction",
                    ],
                }
            )
            output.append(fallback)
        return output

    def answer_from_judgments(
        self,
        query: Query,
        candidates: Iterable[CandidatePaper],
        judgments: list[dict[str, Any]],
        *,
        attempt_callback: Callable[[dict[str, Any]], None] | None = None,
        provider_attempt_callback: ProviderAttemptCallback | None = None,
    ) -> tuple[Prediction, dict[str, Any]]:
        """Answer one query from validated Stage-1 handoff chunks only."""

        require_production_query(query)
        candidate_list = list(candidates)
        candidate_by_id = {item.paper_id: item for item in candidate_list}
        fixed_selected = self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
        if (
            fixed_selected
            and "table" in query.answer_types
            and len(candidate_list) > self.max_evidence
        ):
            raise ValueError(
                f"{query.query_id}: fixed-selected table has "
                f"{len(candidate_list)} selected papers but max_evidence="
                f"{self.max_evidence}; max_evidence must be at least the selected "
                "paper count because every selected paper requires a grounded "
                "evidence chunk"
            )
        relevant = (
            self._fixed_selected_answer_pool(query, candidate_list, judgments)
            if fixed_selected
            else _answer_review_pool(judgments)
        )
        uses_simple_stage1 = any(
            "is_relevant_to_answer" in item for item in judgments
        )
        stage1_relevant_paper_ids = (
            _ordered_unique(
                str(item.get("paper_id") or "")
                for item in sorted(
                    judgments, key=lambda value: int(value.get("rank") or 0)
                )
                if item.get("is_relevant_to_answer") is True
                and str(item.get("paper_id") or "")
            )
            if uses_simple_stage1 and not fixed_selected
            else None
        )
        submission_paper_ids = (
            [item.paper_id for item in candidate_list]
            if fixed_selected
            else None
        )
        if stage1_relevant_paper_ids is not None:
            unknown_relevant_ids = [
                paper_id
                for paper_id in stage1_relevant_paper_ids
                if paper_id not in candidate_by_id
            ]
            if unknown_relevant_ids:
                raise ValueError(
                    f"{query.query_id}: Stage-1 relevance references non-candidate "
                    f"papers: {unknown_relevant_ids}"
                )
        if not relevant:
            raise NoRelevantCandidatesError(
                f"{query.query_id}: Stage 1 produced no validated answer handoff"
            )
        if any(str(item.get("paper_id") or "") not in candidate_by_id for item in relevant):
            raise ValueError(f"{query.query_id}: judgment references a non-candidate paper")

        context = self._answer_context(query, relevant)
        prompt = self._answer_prompt(query, relevant, context)
        relevant_paper_ids = {str(item["paper_id"]) for item in relevant}
        required_visual_paper_ids = {
            str(item["paper_id"])
            for item in relevant
            if isinstance(item.get("visual"), dict)
            and item["visual"].get("required") is True
            and any(
                str(evidence.get("source_type") or "") == "figure"
                for evidence in item.get("evidence") or []
                if isinstance(evidence, dict)
            )
        }
        required_singleton_eligibility_chunk_ids = (
            _required_singleton_eligibility_chunk_ids(query, relevant)
            if fixed_selected
            and has_explicit_singleton_eligibility_filter(query.question)
            else set()
        )
        completion = self._complete(
            prompt,
            context["image_paths"],
            max_completion_tokens=self.answer_max_completion_tokens,
            provider_attempt_callback=provider_attempt_callback,
            semantic_phase="answer_initial",
        )
        attempts: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        for repair_attempt in range(MAX_ANSWER_REPAIR_ATTEMPTS + 1):
            try:
                payload = self._parse_answer(
                    query=query,
                    payload_text=completion["text"],
                    relevant_paper_ids=relevant_paper_ids,
                    context_records=context["records_by_id"],
                    canonical_paper_titles={
                        item.paper_id: item.title for item in candidate_list
                    },
                    required_singleton_eligibility_chunk_ids=(
                        required_singleton_eligibility_chunk_ids
                    ),
                    attached_image_paths=_completion_attached_paths(
                        completion, context["image_paths"]
                    ),
                    required_visual_paper_ids=required_visual_paper_ids,
                )
            except ReadingResponseError as exc:
                _finalize_provider_response_attempt(
                    completion,
                    callback=provider_attempt_callback,
                    semantic_phase=str(
                        completion.get("provider_semantic_phase")
                        or "answer_initial"
                    ),
                    logical_attempt_index=repair_attempt + 1,
                    parse_error=str(exc),
                )
                attempt_record = {
                    "raw_response": completion["text"],
                    "parse_error": str(exc),
                    "call": {
                        key: value
                        for key, value in completion.items()
                        if key != "text"
                    },
                }
                attempts.append(attempt_record)
                if attempt_callback is not None:
                    attempt_callback(attempt_record)
                if repair_attempt >= MAX_ANSWER_REPAIR_ATTEMPTS:
                    raise
                repair_prompt = self._answer_locator_repair_prompt(
                    query=query,
                    original_prompt=prompt,
                    rejected_response=completion["text"],
                    error=exc,
                    context_records=context["records_by_id"],
                    repair_attempt=repair_attempt + 1,
                    prior_errors=[
                        str(attempt["parse_error"])
                        for attempt in attempts
                        if attempt.get("parse_error")
                    ],
                )
                completion = self._complete(
                    repair_prompt,
                    context["image_paths"],
                    max_completion_tokens=self.answer_max_completion_tokens,
                    provider_attempt_callback=provider_attempt_callback,
                    semantic_phase=f"answer_repair_{repair_attempt + 1}",
                )
                continue

            _finalize_provider_response_attempt(
                completion,
                callback=provider_attempt_callback,
                semantic_phase=str(
                    completion.get("provider_semantic_phase") or "answer_initial"
                ),
                logical_attempt_index=repair_attempt + 1,
                parse_error=None,
            )
            attempt_record = {
                "raw_response": completion["text"],
                "parse_error": None,
                "call": {
                    key: value
                    for key, value in completion.items()
                    if key != "text"
                },
            }
            attempts.append(attempt_record)
            if attempt_callback is not None:
                attempt_callback(attempt_record)
            break
        if payload is None:
            raise AssertionError("validated answer payload was not produced")
        actual_answer_image_paths = _completion_attached_paths(
            completion, context["image_paths"]
        )
        actual_answer_image_path_set = set(actual_answer_image_paths)
        actual_answer_evidence_image_chunk_ids = [
            chunk_id
            for chunk_id in context["answer_evidence_image_chunk_ids"]
            if (
                (record := context["records_by_id"].get(chunk_id)) is not None
                and readable_image_path(record) in actual_answer_image_path_set
            )
        ]
        prediction = self._build_prediction(
            query=query,
            payload=payload,
            context_records=context["records_by_id"],
            candidate_ids=[item.paper_id for item in candidate_list],
            relevant=relevant,
            stage1_relevant_paper_ids=stage1_relevant_paper_ids,
            submission_paper_ids=submission_paper_ids,
            image_count=len(actual_answer_image_paths),
        )
        effective_submission_paper_ids = [
            str(item["paper_id"]) for item in prediction.gold_papers
        ]
        answer_record = {
            "query_id": query.query_id,
            "status": "complete",
            "prompt_version": self.answer_prompt_version,
            "paper_set_policy": self.paper_set_policy,
            "answer_derivation_version": ANSWER_DERIVATION_VERSION,
            "few_shot_example_ids": example_manifest(query)["answer"],
            "cache_key": self.answer_cache_key(query, judgments),
            "accepted_paper_ids": [str(item["paper_id"]) for item in relevant],
            "stage1_relevant_paper_ids": stage1_relevant_paper_ids,
            "submission_paper_ids": effective_submission_paper_ids,
            "context_chunk_ids": list(context["records_by_id"]),
            "stage1_handoff_chunk_ids": list(
                context["stage1_handoff_chunk_ids"]
            ),
            "python_supplemental_chunk_ids": list(
                context["python_supplemental_chunk_ids"]
            ),
            "fixed_selected_table_supplement_chunk_ids": list(
                context["fixed_selected_table_supplement_chunk_ids"]
            ),
            "requested_image_paths": list(context["image_paths"]),
            "image_paths": actual_answer_image_paths,
            "image_handoff_version": ANSWER_IMAGE_HANDOFF_VERSION,
            "answer_evidence_image_chunk_ids": list(
                context["answer_evidence_image_chunk_ids"]
            ),
            "attached_answer_evidence_image_chunk_ids": (
                actual_answer_evidence_image_chunk_ids
            ),
            "parsed_response": payload,
            "semantic_multiple_choice": payload.get("answer", {}).get(
                "multiple_choice"
            ),
            "paper_relevance": payload.get("paper_relevance"),
            "derivation": payload.get("derivation"),
            "completeness": payload.get("completeness"),
            "raw_response": completion["text"],
            "call": {key: value for key, value in completion.items() if key != "text"},
            "base_answer_call_count": 1,
            "logical_answer_attempt_count": len(attempts),
            "answer_call_count": sum(
                int(attempt["call"].get("provider_invocation_count") or 1)
                for attempt in attempts
            ),
            "provider_invocation_count": sum(
                int(attempt["call"].get("provider_invocation_count") or 1)
                for attempt in attempts
            ),
            "attempts": attempts,
            "prediction": prediction.to_dict(),
        }
        return prediction, answer_record

    def _answer_context(
        self, query: Query, relevant: list[dict[str, Any]]
    ) -> AnswerContext:
        primary: list[tuple[Record, str]] = []
        neighbours: list[tuple[Record, str]] = []
        seen_primary: set[str] = set()
        seen_neighbours: set[str] = set()
        evidence_quotes: dict[str, list[str]] = {}
        answer_image_records_by_paper: dict[str, list[Record]] = {}
        requested_handoff_chunk_ids: list[str] = []
        fixed_table_supplements_by_paper: dict[str, list[Record]] = {}
        fixed_multi_paper_table = (
            self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            and "table" in query.answer_types
            and len(relevant) > 1
        )
        option_text = " ".join(str(value) for value in (query.options or {}).values())
        compound_answer = bool(
            (
                "multiple_choice" in query.answer_types
                and re.search(r"(?i)\b(?:and|respectively)\b|[;,/]", option_text)
                and re.search(
                    r"(?i)\b(?:and|respectively|across|both|two|three)\b|;",
                    query.question,
                )
            )
            or (
                "table" in query.answer_types
                and sum(
                    column.get("is_row_key") is not True
                    for column in query.table_schema or []
                    if isinstance(column, dict)
                )
                >= 2
            )
            or (
                "freeform" in query.answer_types
                and re.search(
                    r"(?i)\b(?:respectively|both)\b|"
                    r"\b(?:and|versus|vs\.?)\b[^?]{0,120}"
                    r"\b(?:what|how|which)\b",
                    query.question,
                )
            )
        )
        mixed_visual_text_supplement = (
            requires_visual_image(query.question) and compound_answer
        )
        mixed_visual_text_focus = _query_selection_focus(query)

        def add_neighbour(record: Record) -> None:
            chunk_id = str(record.get("chunk_id") or "")
            if (
                chunk_id
                and chunk_id not in seen_primary
                and chunk_id not in seen_neighbours
            ):
                seen_neighbours.add(chunk_id)
                neighbours.append((record, ""))

        for judgment in relevant:
            paper_id = str(judgment["paper_id"])
            records = recover_split_caption_table_records(
                self.chunk_store.load_paper(paper_id)
            )
            by_id = {str(item.get("chunk_id") or ""): item for item in records}
            positions = {str(item.get("chunk_id") or ""): i for i, item in enumerate(records)}
            cited_records: list[Record] = []

            # An ordinal reference request is deterministic bibliography
            # addressing: "third reference" means the entry labeled [3], not
            # the third citation marker in body-reading order.  Make the exact
            # [N] entry available even when Stage 1 followed body citation order
            # and handed off a different reference chunk.  This supplement is
            # separately audited and does not mutate the durable Stage-1 result.
            for entry in requested_ordinal_bibliography_entries(
                query.question, records
            ):
                record = by_id.get(entry.chunk_id)
                if record is None:
                    continue
                evidence_quotes.setdefault(entry.chunk_id, []).append(entry.text)
                if entry.chunk_id not in seen_primary:
                    seen_primary.add(entry.chunk_id)
                    primary.append((record, entry.text))

            if judgment.get("checkpoint_kind") == FIXED_SELECTED_CHECKPOINT_KIND:
                # Preserve the extractor's atomic fact hints and exact copied
                # excerpts. Original chunks remain authoritative in Stage 2.
                handoff_evidence = list(judgment.get("evidence") or [])
            elif "send_to_answer_agent" in judgment:
                # For the current schema, the validated IDs are the sole handoff
                # authority. Rebuild the internal evidence objects so a stale or
                # hand-edited compatibility ``evidence`` array cannot change the
                # chunks Stage 2 receives.
                handoff_evidence = [
                    {
                        "chunk_id": chunk_id,
                        "purpose": "answer",
                        "quote_or_value": "",
                    }
                    for chunk_id in judgment.get("evidence_chunk_ids") or []
                ]
            else:
                handoff_evidence = list(judgment.get("evidence") or [])
            for evidence in handoff_evidence:
                chunk_id = str(evidence.get("chunk_id") or "")
                if chunk_id and chunk_id not in requested_handoff_chunk_ids:
                    requested_handoff_chunk_ids.append(chunk_id)
                record = by_id.get(chunk_id)
                if record is None:
                    raise ValueError(
                        f"{query.query_id}/{paper_id}: cached judgment cites missing "
                        f"chunk {chunk_id!r}"
                    )
                quote = str(evidence.get("quote_or_value") or "")
                cited_records.append(record)
                evidence_quotes.setdefault(chunk_id, []).append(quote)
                if (
                    str(evidence.get("purpose") or "answer") == "answer"
                    and readable_image_path(record)
                ):
                    answer_image_records_by_paper.setdefault(paper_id, []).append(
                        record
                    )
                if chunk_id not in seen_primary:
                    seen_primary.add(chunk_id)
                    primary.append((record, quote))
                position = positions[chunk_id]
                for offset in range(-self.answer_neighbor_chunks, self.answer_neighbor_chunks + 1):
                    neighbour_position = position + offset
                    if offset == 0 or not 0 <= neighbour_position < len(records):
                        continue
                    neighbour = records[neighbour_position]
                    add_neighbour(neighbour)

            visual_recheck = requires_visual_image(query.question) or (
                isinstance(judgment.get("visual"), dict)
                and judgment["visual"].get("required") is True
            )
            if visual_recheck:
                for record in cited_records:
                    if not readable_image_path(record):
                        continue
                    position = positions[str(record["chunk_id"])]
                    metadata = record.get("metadata") or {}
                    page = metadata.get("page")
                    source_type = record_source_type(record)
                    for sibling_index in range(
                        max(0, position - 3), min(len(records), position + 4)
                    ):
                        sibling = records[sibling_index]
                        sibling_metadata = sibling.get("metadata") or {}
                        if (
                            sibling_metadata.get("page") == page
                            and record_source_type(sibling) == source_type
                            and readable_image_path(sibling)
                        ):
                            add_neighbour(sibling)

            # A compound visual question can name a Figure as the identity
            # anchor while asking for a definition or categorical value that
            # is stated only in prose.  Keep the validated visual handoff and
            # add a small, deterministic same-paper reading companion selected
            # solely from the released query and options.  These records remain
            # ordinary context: Stage 2 must still cite and bind a companion if
            # it uses one, and none is promoted to evidence automatically.
            if mixed_visual_text_supplement and any(
                readable_image_path(record)
                and record_source_type(record) in {"figure", "table"}
                for record in cited_records
            ):
                scores, _ = _paper_record_scores(
                    records,
                    mixed_visual_text_focus,
                    source_question=query.question,
                )
                ranked_text_indices = sorted(
                    (
                        index
                        for index, record in enumerate(records)
                        if str(record.get("chunk_type") or "") == "text_span"
                        and record_source_type(record) == "text_span"
                    ),
                    key=lambda index: (-scores[index], index),
                )
                for index in ranked_text_indices[
                    :MIXED_VISUAL_TEXT_SUPPLEMENTS_PER_PAPER
                ]:
                    add_neighbour(records[index])

            # Stage 1 can correctly read an uncaptioned OCR table while that
            # chunk is unusable in the official submission.  Preserve the read
            # context, but also expose a small deterministic set of eligible
            # same-paper records and query-relevant images so Stage 2 can ground
            # the answer without inventing a table/figure ID.
            if cited_records and not any(
                submission_evidence_eligible(record) for record in cited_records
            ):
                rescue_focus = " ".join(
                    (
                        _query_selection_focus(query),
                        str(judgment.get("reason") or ""),
                        _json_dumps(judgment.get("candidate_answer") or {}),
                        " ".join(
                            str(item.get("quote_or_value") or "")
                            for item in judgment.get("evidence") or []
                        ),
                    )
                )
                scores, reasons = _paper_record_scores(
                    records, rescue_focus, source_question=query.question
                )
                eligible_indices = [
                    index
                    for index, record in enumerate(records)
                    if submission_evidence_eligible(record)
                ]
                eligible_index_set = set(eligible_indices)
                ranked_eligible = sorted(
                    eligible_indices, key=lambda index: (-scores[index], index)
                )
                image_indices = [
                    index
                    for index in _ranked_image_record_indices(
                        records,
                        scores=scores,
                        reasons=reasons,
                        limit=self.max_answer_images,
                        question=query.question,
                    )
                    if index in eligible_index_set
                ]
                rescue_indices = _ordered_unique_ints(
                    [*image_indices, *ranked_eligible[:4]]
                )
                for index in rescue_indices:
                    add_neighbour(records[index])

            # A fixed multi-paper table treats every selected paper as a
            # required source.  Extraction can still return only a citation or
            # prose constraint and miss the paper's answer-bearing table (for
            # example when OCR makes the table harder to summarize).  Supply a
            # small, deterministic, query-ranked set of eligible table chunks
            # from that same paper.  These are reading context, not automatic
            # evidence: Stage 2 must cite and bind any chunk it actually uses.
            if fixed_multi_paper_table and not any(
                submission_evidence_eligible(record)
                and record_source_type(record) == "table"
                for record in cited_records
            ):
                supplement_focus = " ".join(
                    [
                        _query_selection_focus(query),
                        *(
                            str(fact.get("source_excerpt") or fact.get("fact") or "")
                            for fact in judgment.get("extracted_facts") or []
                            if isinstance(fact, dict)
                        ),
                    ]
                )
                scores, _ = _paper_record_scores(
                    records,
                    supplement_focus,
                    source_question=query.question,
                )
                ranked_tables = sorted(
                    (
                        index
                        for index, record in enumerate(records)
                        if submission_evidence_eligible(record)
                        and record_source_type(record) == "table"
                    ),
                    key=lambda index: (-scores[index], index),
                )
                for index in ranked_tables[
                    :FIXED_SELECTED_TABLE_SUPPLEMENTS_PER_PAPER
                ]:
                    record = records[index]
                    chunk_id = str(record.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    fixed_table_supplements_by_paper.setdefault(
                        paper_id, []
                    ).append(record)
                    if chunk_id not in seen_primary:
                        seen_primary.add(chunk_id)
                        primary.append((record, ""))

        # Round-robin by paper. A rank-1 paper with many citations must not
        # consume the cap before rank-44 receives even one evidence chunk.
        by_paper: dict[str, list[tuple[Record, str]]] = {}
        for item in primary:
            by_paper.setdefault(str(item[0].get("paper_id") or ""), []).append(item)
        paper_image_priority = {
            str(item.get("paper_id") or ""): (
                str(item.get("label") or ""),
                int(item.get("rank") or 0),
            )
            for item in relevant
        }
        explicit_visual_query = requires_visual_image(query.question)
        visual_required_paper_ids = {
            str(item.get("paper_id") or "")
            for item in relevant
            if explicit_visual_query
            or item.get("visual") is None
            or (
                isinstance(item.get("visual"), dict)
                and item["visual"].get("required") is True
            )
        }
        visual_image_records_by_paper = {
            paper_id: [
                record
                for record, _ in paper_records
                if readable_image_path(record)
            ]
            for paper_id, paper_records in by_paper.items()
            if paper_id in visual_required_paper_ids
        }

        def prioritized_image_records(
            records_by_paper: dict[str, list[Record]],
        ) -> list[Record]:
            """Order images by semantic label, rank, and fair paper rotation."""

            prioritized: list[Record] = []
            for label in ("direct_answer", "partial_answer", "supporting_only"):
                label_papers = sorted(
                    (
                        paper_id
                        for paper_id, paper_records in records_by_paper.items()
                        if paper_records
                        and paper_image_priority.get(paper_id, ("", 0))[0] == label
                    ),
                    key=lambda paper_id: (
                        paper_image_priority[paper_id][1],
                        paper_id,
                    ),
                )
                # Preserve strict label priority, but share each label's image
                # budget fairly: every paper gets image 1 in rank order before
                # any paper gets image 2.
                image_position = 0
                while label_papers:
                    added = False
                    for paper_id in label_papers:
                        paper_records = records_by_paper[paper_id]
                        if image_position < len(paper_records):
                            prioritized.append(paper_records[image_position])
                            added = True
                    if not added:
                        break
                    image_position += 1
            return prioritized

        # Stage 1's purpose field is the strongest handoff signal.  Reserve
        # image slots for every answer-purpose table/figure before considering
        # other cited visuals or neighbouring context.  In particular, do not
        # discard these images merely because Stage 1 could also read the OCR
        # text and therefore reported visual.required=false.
        answer_image_records = prioritized_image_records(
            answer_image_records_by_paper
        )
        visual_image_records = prioritized_image_records(
            visual_image_records_by_paper
        )
        fixed_table_supplement_records = prioritized_image_records(
            fixed_table_supplements_by_paper
        )
        if len(by_paper) > self.max_answer_papers:
            raise ReadingResponseError(
                f"{query.query_id}: {len(by_paper)} accepted papers exceed "
                f"max_answer_papers={self.max_answer_papers}; do not silently "
                "drop papers"
            )
        context_evidence_limit = max(self.max_evidence, len(by_paper))
        round_robin: list[tuple[Record, str]] = []
        position = 0
        while len(round_robin) < context_evidence_limit:
            added = False
            for items in by_paper.values():
                if position < len(items):
                    round_robin.append(items[position])
                    added = True
                    if len(round_robin) >= context_evidence_limit:
                        break
            if not added:
                break
            position += 1
        # The ordinary evidence cap controls what the final answer may cite, not
        # whether Stage 2 can re-read the finite set of images already chosen as
        # Stage-1 answer evidence.  Prepend the image records that can fit in the
        # AOAI request, then retain the complete ordinary round-robin selection.
        # This prevents a late evidence item in one paper from losing its image
        # before attachment solely because text evidence filled the context cap.
        primary = []
        primary_chunk_ids: set[str] = set()
        for record in answer_image_records[: self.max_answer_images]:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in primary_chunk_ids:
                continue
            primary_chunk_ids.add(chunk_id)
            quotes = evidence_quotes.get(chunk_id) or [""]
            primary.append((record, quotes[0]))
        # Unlike generic neighbours, fixed-table supplements must remain
        # visible even when the ordinary context-evidence cap is saturated.
        # Fair paper rotation prevents a high-ranked paper from consuming this
        # small supplemental budget before later authoritative papers.
        for record in fixed_table_supplement_records:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in primary_chunk_ids:
                continue
            primary_chunk_ids.add(chunk_id)
            primary.append((record, ""))
        for record, quote in round_robin:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in primary_chunk_ids:
                continue
            primary_chunk_ids.add(chunk_id)
            primary.append((record, quote))
        if not primary:
            raise ReadingResponseError(
                f"{query.query_id}: accepted judgments contain no evidence chunks"
            )
        records_by_id: dict[str, Record] = {}
        parts: list[str] = []
        used_chars = 0
        # Give every cited chunk a fair share before adding neighbours. This is
        # essential for multi-paper tables where a rank-1 paper must not consume
        # the context before rank-44 is represented. The allocation includes
        # record headers and separators so the configured hard limit is real.
        focus = query.question + " " + " ".join(
            quote for _, quote in primary if quote
        )
        for primary_index, (record, quote) in enumerate(primary):
            chunk_id = str(record["chunk_id"])
            separator_chars = 2 if parts else 0
            remaining_items = len(primary) - primary_index
            remaining_chars = (
                self.answer_context_chars - used_chars - separator_chars
            )
            if remaining_chars < remaining_items:
                raise ReadingResponseError(
                    f"{query.query_id}: answer_context_chars="
                    f"{self.answer_context_chars} cannot represent "
                    f"{len(primary)} accepted evidence chunks"
                )
            fair_share = min(40_000, remaining_chars // remaining_items)
            formatted = _bounded_formatted_record(
                self,
                record,
                focus=focus + " " + quote,
                max_chars=fair_share,
                selected=True,
            )
            parts.append(formatted)
            used_chars += separator_chars + len(formatted)
            records_by_id[chunk_id] = record
        for record, _ in neighbours:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in records_by_id:
                continue
            separator_chars = 2 if parts else 0
            remaining = self.answer_context_chars - used_chars - separator_chars
            if remaining < 1_000:
                break
            excerpt = _focused_excerpt(
                str(record.get("text") or ""), query.question, min(20_000, remaining)
            )
            formatted = self._format_record(record, text=excerpt, selected=False)
            if (
                used_chars + separator_chars + len(formatted)
                > self.answer_context_chars
            ):
                continue
            parts.append(formatted)
            used_chars += separator_chars + len(formatted)
            records_by_id[chunk_id] = record

        if len("\n\n".join(parts)) > self.answer_context_chars:
            raise AssertionError("answer context exceeds configured character limit")

        image_paths: list[str] = []

        def add_images(records: Iterable[Record]) -> None:
            for record in records:
                if len(image_paths) >= self.max_answer_images:
                    return
                chunk_id = str(record.get("chunk_id") or "")
                if chunk_id not in records_by_id:
                    continue
                image_path = readable_image_path(record)
                if image_path and image_path not in image_paths:
                    image_paths.append(image_path)

        # Answer-purpose images are a lossless Stage-1-to-Stage-2 handoff and
        # therefore consume the finite AOAI image budget first.  Broader visual
        # evidence and neighbouring images are supplemental.
        add_images(answer_image_records)
        add_images(fixed_table_supplement_records)
        add_images(visual_image_records)
        add_images(record for record, _ in neighbours)
        attached_image_paths = set(image_paths)
        answer_evidence_image_chunk_ids = _ordered_unique(
            str(record.get("chunk_id") or "")
            for record in answer_image_records
            if str(record.get("chunk_id") or "")
        )
        attached_answer_evidence_image_chunk_ids = [
            chunk_id
            for chunk_id in answer_evidence_image_chunk_ids
            if (
                (record := records_by_id.get(chunk_id)) is not None
                and readable_image_path(record) in attached_image_paths
            )
        ]
        requested_handoff_chunk_id_set = set(requested_handoff_chunk_ids)
        fixed_table_supplement_chunk_ids = _ordered_unique(
            str(record.get("chunk_id") or "")
            for record in fixed_table_supplement_records
            if str(record.get("chunk_id") or "")
        )
        return {
            "text": "\n\n".join(parts),
            "records_by_id": records_by_id,
            "image_paths": image_paths,
            "stage1_handoff_chunk_ids": [
                chunk_id
                for chunk_id in requested_handoff_chunk_ids
                if chunk_id in records_by_id
            ],
            "python_supplemental_chunk_ids": [
                chunk_id
                for chunk_id in records_by_id
                if chunk_id not in requested_handoff_chunk_id_set
            ],
            "fixed_selected_table_supplement_chunk_ids": [
                chunk_id
                for chunk_id in fixed_table_supplement_chunk_ids
                if chunk_id in records_by_id
            ],
            "answer_evidence_image_chunk_ids": answer_evidence_image_chunk_ids,
            "attached_answer_evidence_image_chunk_ids": (
                attached_answer_evidence_image_chunk_ids
            ),
        }

    def _answer_prompt(
        self,
        query: Query,
        relevant: list[dict[str, Any]],
        context: AnswerContext,
    ) -> str:
        image_legend = self._image_legend(
            context["records_by_id"], context["image_paths"]
        )
        return render_answer_prompt(
            query=query,
            query_payload=_production_query_payload(query),
            accepted_summary=relevant,
            evidence_text=context["text"],
            image_legend=image_legend,
            answer_shape=answer_response_shape(query),
            max_evidence=self.max_evidence,
            max_evidence_per_paper=self.max_evidence_per_paper,
            paper_set_policy=self.paper_set_policy,
        )

    def _answer_locator_repair_prompt(
        self,
        *,
        query: Query | None = None,
        original_prompt: str,
        rejected_response: str,
        error: ReadingResponseError,
        context_records: dict[str, Record],
        repair_attempt: int,
        prior_errors: list[str] | None = None,
    ) -> str:
        eligible_chunk_ids = [
            chunk_id
            for chunk_id, record in context_records.items()
            if submission_evidence_eligible(record)
        ]
        error_text = str(error)
        error_corpus = "\n".join(dict.fromkeys(prior_errors or [error_text]))
        error_corpus_folded = error_corpus.casefold()
        direct_reported_optimum_context = bool(
            query is not None
            and _DIRECT_REPORTED_OPTIMUM_QUERY_RE.search(query.question)
        )
        axis_extent_context = bool(
            query is not None and is_axis_extent_lookup_query(query)
        )
        genuine_extremum_context = bool(
            query is not None and requires_extremum_operation(query)
        )
        axis_extent_repair = ""
        if axis_extent_context:
            axis_extent_repair = (
                " This query contains an axis-extent lookup. A phrase such as "
                "'roughly what is the highest value on the horizontal axis' asks "
                "for the terminal visible tick/limit, not a winner among methods. "
                "Delete any argmax/argmin for that clause. Re-read the actually "
                "attached plot, create one minimal numeric value_kind='visual' "
                "fact for the terminal tick, and bind that exact value with "
                "operations=[] for this clause. Keep every other compound answer "
                "part in its own fact and binding. A Table cell copied from visible "
                "extracted text may be value_kind='reported'; use "
                "value_kind='visual' only when that cell was read from an actually "
                "attached table image. If another clause independently asks which "
                "candidate is best/highest, retain a complete argmax/argmin only "
                "for that separate clause."
            )
        direct_reported_optimum_repair = ""
        if direct_reported_optimum_context:
            direct_reported_optimum_repair = (
                " This is a direct reported-optimum lookup with another atomic "
                "answer part, not a request to rank parameter magnitudes. First "
                "check whether the supplied source explicitly states the optimum or "
                "winner. If it does, delete every argmax/argmin for that clause, set "
                "operations=[] for that clause, create one minimal fact whose value is "
                "the answer-bearing parameter value and whose value_kind='reported', "
                "and bind its exact substring in the emitted answer. Keep each other "
                "requested effect or condition in a separate atomic fact and binding. "
                "A parameter such as gamma=0.98 is not a performance score: never rank "
                "the released option values or invent pseudo-scores. Only if the source "
                "does not state the optimum and instead supplies every raw candidate "
                "performance operand may you use a complete grounded argmax/argmin."
            )
        labeled_selection_context = (
            'id="A27_same_performance_requires_both_operands"' in original_prompt
        )
        labeled_selection_repair = ""
        if labeled_selection_context:
            labeled_selection_repair = (
                " This is a labeled comparison-selection task. Use one atomic "
                "numeric reported/visual fact for each side of every eligible "
                "labeled row. Then use exactly one kind='select_where' operation. "
                "Its fact_ids list contains every operand fact exactly once; its "
                "comparisons list contains one object per row with exactly label, "
                "left_fact_id, right_fact_id, left, and right. Copy left/right from "
                "the named facts, use operator='==' for a same/equal question or "
                "operator='!=' for a different question, and set result to the "
                "unique label satisfying that predicate. The operation-local "
                "expected and answer_fragment are that label, not boolean true/false. "
                "Bind both final freeform and multiple-choice outputs to the same "
                "select_where operation. Do not use a vector fact, reuse one fact on "
                "both sides, turn equality into a reported status fact, or bind an "
                "internal predicate boolean to the task label. Keep all labels and "
                "numbers grounded in the live evidence; never copy synthetic example "
                "values."
            )
        comparison_repair = ""
        if (
            not labeled_selection_context
            and not direct_reported_optimum_context
            and (not axis_extent_context or genuine_extremum_context)
            and (
                "candidate labels must be distinct" in error_corpus
                or "value must contain label and value" in error_corpus
                or "candidates do not match label/value pairs" in error_corpus
                or "does not express expected result" in error_corpus
                or "does not equal computed result" in error_corpus
                or "question explicitly requests argmax/argmin" in error_corpus
            )
        ):
            comparison_repair = (
                " This is an argmax/argmin comparison-contract error. For every "
                "referenced comparison fact, replace fact.value with an actual JSON "
                "object of exactly {\"label\":\"unique answer-aligned row identity\","
                "\"value\":numeric operand}; do not use a bare number, a source "
                "mean±deviation string, or a JSON-encoded string. Copy those same "
                "objects exactly into operation.candidates. Every label must be "
                "unique: distinguish only repeated family names with their source "
                "setting such as '(m = 9)' and '(m = 40)'. Keep labels that are "
                "already unique equal to the canonical query or option text whenever "
                "possible; for example keep a lone 'KS' as 'KS', not 'KS (m = 128)', "
                "so the computed winner is an exact substring of the final answer. "
                "Preserve this labeling scheme in all later corrections, then recompute "
                "result and bindings from the corrected candidates. If the original "
                "source directly and explicitly reports the requested winner or optimum "
                "rather than merely listing operands, do not invent a one-row or "
                "duplicate-row argmax: use the minimal winner value with "
                "value_kind='reported' and bind every final output to that reported fact. "
                "When the released question itself contains an explicit 'only' filter "
                "and the grounded eligible set truly reduces to one paper, use a unary "
                "argmax/argmin with exactly that one distinct candidate. Never pad it "
                "with a duplicate fact or an ineligible paper."
            )
        table_binding_repair = ""
        if (
            "answer.table.rows[" in error_corpus
            and "does not exactly equal sourced value" in error_corpus
        ):
            table_binding_repair = (
                " This is a table-binding shape error. A path such as "
                "answer.table.rows[0] resolves to the entire JSON row object. "
                "Use that whole-row path only when the referenced fact.value is "
                "exactly the same row object. When fact.value is a scalar cell, "
                "bind it to the exact leaf path, for example "
                "answer.table.rows[0].Paper Title. Keep row-level support allowed; "
                "derivation bindings and support paths have different purposes."
            )
        object_fact_repair = ""
        if "does not express sourced fact value {" in error_corpus:
            object_fact_repair = (
                " This is an object-valued fact versus text-fragment error. Never "
                "bind a whole JSON row-object fact directly to a freeform string "
                "fragment. Replace that row object with separate scalar facts for "
                "each answer-bearing cell, using unique fact ids and unique "
                "descriptive names. Bind each scalar fact independently to its "
                "exact freeform substring and to its exact table cell path, for "
                "example answer.table.rows[0].Method and "
                "answer.table.rows[0].Base Model. Alternatively, keep an object "
                "fact only for a whole-row table binding and add separate scalar "
                "facts for freeform; do not bind the object itself to text."
            )
        scalar_fact_repair = ""
        if (
            "does not express sourced fact value" in error_corpus
            or "answer_fragment is not an exact substring" in error_corpus
        ):
            scalar_fact_repair = (
                " This is a scalar fact/fragment error. Do not alternate between "
                "a long source sentence and the whole shorter option. Use the "
                "smallest source-grounded answer identifier that is also an exact "
                "substring of the emitted answer, preferably a JSON number for a "
                "numeric value. Split genuinely compound answers into separate "
                "atomic facts and bindings. Never copy option-only wording into a "
                "reported fact; every fact value must remain supported by its chunk. "
                "For a multiple-choice option that faithfully canonicalizes a directly "
                "stated qualitative effect, a text fact may use the option's minimal "
                "canonical phrase only when the cited source visibly states the same "
                "direction and polarity. For example, source text saying accuracy "
                "drops below a reference may ground 'reduces accuracy'. Never use this "
                "paraphrase rule to change a number, comparator, negation, condition, "
                "dataset, model, or setting."
            )
        visual_fact_repair = ""
        if (
            "visual fact" in error_corpus
            or "actually attached source image" in error_corpus
        ):
            visual_fact_repair = (
                " This is a visual-grounding error. Preserve the already derived "
                "scientific value and selected option. Create an atomic fact with "
                "value_kind='visual' and an actually attached figure/table chunk. "
                "Use a JSON number for a printed scalar when possible, or the "
                "smallest visible trend phrase. Keep that visual fact in every "
                "later correction."
            )
        evidence_set_repair = ""
        if "papers and support evidence disagree" in error_corpus:
            evidence_set_repair = (
                " This is a duplicated evidence-set error. The union of every "
                "fact's paper/chunk pair, papers/evidence_chunk_ids, and support "
                "paper/chunk pairs must be identical. Remove an unused duplicate "
                "text fact when one visual fact is sufficient, or add the retained "
                "fact's chunk to support for each answer path it supports."
            )
        non_ready_repair = ""
        if (
            "answer is not ready" in error_corpus
            or "omitted evidence papers" in error_corpus
        ):
            non_ready_repair = (
                " Re-read every supplied eligible chunk before returning a non-ready "
                "status. A query-relevant eligible rescue figure from the established "
                "owner may replace an ineligible OCR table, but it must be actually "
                "read and directly support the answer. Do not guess when none does."
            )
        citation_count_repair = ""
        if (
            "aggregate citation count" in error_corpus
            or "stable citation identity" in error_corpus
            or "citation identities" in error_corpus
        ):
            citation_count_repair = (
                " This is an aggregate citation-count inventory error. Rebuild one "
                "explicit list containing only distinct cited-paper identities in "
                "the form '[N]', 'FirstAuthor et al. (YYYY)', or single-author "
                "'FirstAuthor (YYYY)'. Remove method acronyms, the owning paper or "
                "method name, section/concept names, bare years, DOI, and URLs. "
                "Every item must be visible in the chunks cited by the referenced "
                "count facts. For an author-filtered query, the required author "
                "must be visible inside each counted bibliography entry. Copy "
                "exactly that same list into fact.value and "
                "operation.items, set result to its length, bind every final answer "
                "to that count operation, and select only a bare-numeric option "
                "equal to result. Do not use this inventory for a last-reference "
                "index lookup."
            )
        ordinal_reference_repair = ""
        ordinal = (
            requested_reference_ordinal(query.question)
            if query is not None
            else None
        )
        if ordinal is not None and "ordinal bibliography lookup" in error_corpus:
            target_chunks = _ordered_unique(
                entry.chunk_id
                for entry in requested_ordinal_bibliography_entries(
                    query.question, context_records.values()
                )
            )
            ordinal_reference_repair = (
                f" This is an ordinal bibliography-addressing error. The query's "
                f"ordinal means the bibliography entry labeled [{ordinal}]; it "
                "never means the same-numbered citation occurrence encountered "
                "while scanning body prose. Discard every body-order inference "
                "and every answer fact taken from another bibliography label. "
                f"Re-read the exact [{ordinal}] entry in these deterministic "
                f"supplemental chunk_ids: {_json_dumps(target_chunks)}. Source "
                f"each final answer-bound fact from within that [{ordinal}] entry, "
                "then make facts, papers, and support cite only the retained "
                "answer-bearing chunk pairs."
            )
        minimal_freeform_repair = ""
        if "minimal atomic freeform" in error_corpus:
            minimal_freeform_repair = (
                " This is a minimal-freeform surface error. Keep the supported "
                "fact or operation and its evidence, but replace freeform.text "
                "and final_semantic_answer with only the canonical atomic value "
                "or short phrase. Do not add a lead-in, explanation, redundant "
                "unit, or final period. Make answer_fragment the same minimal "
                "surface when it binds the whole answer."
            )
        visual_count_repair = ""
        if "visual subfigure count" in error_corpus:
            visual_count_repair = (
                " This is a visual subfigure-inventory error. Re-read the actual "
                "attached full figure and discard the previous count, count result, "
                "and selected option. Enumerate every independently bounded "
                "coordinate-axes region with a distinct spatial identifier such as "
                "'top-row col-1 axes' or '(a)-left axes', then recompute the count "
                "and option from that inventory. A row, model family, group heading, "
                "or bare label such as '(a)' is not itself a subfigure. Do not invent "
                "panel letters that are not visible in the pixels."
            )
        table_inventory_repair = ""
        if "explicitly requested table item" in error_corpus:
            table_inventory_repair = (
                " This is an explicit table-row completeness error. Re-read the "
                "Stage-1 candidate hypotheses and their original source chunks or "
                "attached table images. Account for every named item exactly once: "
                "return a fully source-grounded row, or copy the exact item name into "
                "completeness.missing when the supplied evidence truly cannot support "
                "it. A visibly printed dash is a source value and must remain a row; "
                "never invent a numeric replacement."
            )
        fixed_enumerative_row_repair = ""
        if (
            self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            and (
                "open-ended fixed-selected multi-paper table" in error_corpus
                or "fixed-selected multi-paper table must use every authoritative"
                in error_corpus
            )
        ):
            fixed_enumerative_row_repair = (
                " This is an authoritative enumerative-row error. Upstream has "
                "already selected every supplied paper as a qualifying answer "
                "unit. Do not classify any selected paper as ineligible or use "
                "a negative/constraint fact merely to account for it. Emit at "
                "least one source-grounded answer.table.rows[i] row for every "
                "selected paper. Bind a fact from that same paper directly to "
                "the complete row when its value is the row object, or bind its "
                "atomic facts to the corresponding leaf cells. Every binding's "
                "answer_fragment must be an exact substring of the leaf path it "
                "names; omit answer_fragment for an object-valued whole-row "
                "binding. The value in each emitted cell must answer that exact "
                "table-schema column. Never copy a value from a different source "
                "field, such as an eligibility or reference-status column, into "
                "the requested result column merely to satisfy row coverage."
            )
        filtered_singleton_repair = ""
        if "filtered-singleton eligibility evidence" in error_corpus_folded:
            filtered_singleton_repair = (
                " This is an explicit eligibility-filter evidence error. Keep "
                "a separate source-grounded fact for every listed "
                "eligibility chunk_ids that proves the query's hard condition, "
                "such as trained only on the named dataset or using the stated "
                "budget. Include that chunk in papers and support. Do not delete "
                "the eligibility fact merely because it is not an argmax numeric "
                "operand, and do not put its text value into operation.candidates. "
                "Keep the eligible label/value score fact separately in the "
                "argmax/argmin operation."
            )
        scaling_identifier_repair = ""
        if "canonical scaling identifier mismatch" in error_corpus_folded:
            scaling_identifier_repair = (
                " This is a deterministic scaling-identifier surface error. "
                "Copy every expected Method and Base Model value from the "
                "validation error exactly. Update the corresponding atomic "
                "fact.value, table cell, freeform substring, answer_fragment, "
                "and final_semantic_answer together; do not change the selected "
                "papers or scientific model identities. These are the only "
                "allowed canonical edits: source-body method punctuation, a "
                "family-size hyphen, omission of a trailing lowercase syntactic "
                "head 'model', and slash compression of co-introduced sizes of "
                "one identical family."
            )
        canonical_row_identifier_repair = ""
        if "canonical explicit row identifier mismatch" in error_corpus_folded:
            canonical_row_identifier_repair = (
                " This is a source-canonical row-identifier error caused by an "
                "obvious one-character spelling difference in the released "
                "query. Copy every expected row identifier from the validation "
                "error exactly into the row-key cell, its atomic fact.value, "
                "answer_fragment, and final semantic answer. Keep the requested "
                "value and evidence unchanged. Do not apply fuzzy matching when "
                "more than one source identifier is plausible."
            )
        one_row_per_method_repair = ""
        if "one-row-per-method table" in error_corpus:
            one_row_per_method_repair = (
                " This is a duplicate-method row error. The question asks for "
                "one answer for each canonical method, so emit exactly one row "
                "per method. If the source explicitly names several equally "
                "requested variants, preserve them together in that row's "
                "string-valued answer cell. Do not create one row per variant. "
                "For a singular base-model request, a later experiment described "
                "as an additional generalizability check is not a second base "
                "model row."
            )
        canonical_title_repair = ""
        if "canonical paper title" in error_corpus_folded:
            canonical_title_repair = (
                " This is a canonical-title copying error. Copy the expected "
                "Paper Title printed in the validation error byte-for-byte into "
                "the named row, its source fact, derivation binding, and final "
                "answer. A literal '&' is an ordinary JSON string character: "
                "preserve an HTML entity such as '&#x27;' literally. Do not "
                "decode it to an apostrophe and do not turn '&' into a numeric "
                "JSON escape such as '\\u00026'. Bind each Paper Title row to "
                "facts from exactly one selected source paper; never mix two "
                "papers in one title row or leave its source owner ambiguous."
            )
        fixed_selected_repair_boundary = (
            "For this fixed-selected workflow, never change the externally "
            "selected submission paper set. For an open-ended multi-paper table, "
            "do not omit a selected paper: emit its grounded answer row. "
            if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            else "If a paper has no eligible direct evidence, omit that paper "
            "rather than inventing a locator. "
        )
        error_history = _json_dumps(list(dict.fromkeys(prior_errors or [error_text])))
        return (
            original_prompt
            + "\n\nYour previous JSON response was rejected by deterministic validation. "
            "Correct the JSON once and recompute the scientific answer when the error says "
            "the derivation, comparison polarity, option mapping, table types, support, or "
            "evidence is inconsistent. Evidence used only while comparing may be omitted "
            "from papers/support. Every submitted chunk must come from the eligible list. "
            + fixed_selected_repair_boundary
            + "For a fact binding, fact.value must be the smallest "
            "answer-bearing typed value copied from evidence and expressed by the exact "
            "answer_fragment; never keep a surrounding evidence sentence as fact.value "
            "when the answer uses only its concise value. While shortening, preserve every "
            "negation, quantifier, comparator, number, unit, and model identifier needed "
            "for the scientific claim; a generic noun such as GPU is not an adequate "
            "replacement for a specific hardware value. Do not repeat the rejected structure.\n"
            + labeled_selection_repair
            + direct_reported_optimum_repair
            + axis_extent_repair
            + comparison_repair
            + table_binding_repair
            + object_fact_repair
            + scalar_fact_repair
            + visual_fact_repair
            + evidence_set_repair
            + non_ready_repair
            + citation_count_repair
            + ordinal_reference_repair
            + minimal_freeform_repair
            + visual_count_repair
            + table_inventory_repair
            + fixed_enumerative_row_repair
            + filtered_singleton_repair
            + scaling_identifier_repair
            + canonical_row_identifier_repair
            + one_row_per_method_repair
            + canonical_title_repair
            + "\n"
            f"Correction attempt: {repair_attempt}/{MAX_ANSWER_REPAIR_ATTEMPTS}\n"
            f"Validation error: {error}\n"
            f"All validation errors seen so far: {error_history}\n"
            "Do not reintroduce any earlier error while fixing the latest one.\n"
            f"Eligible evidence chunk_ids: {_json_dumps(eligible_chunk_ids)}\n"
            "<rejected_response>\n"
            + html.escape(rejected_response, quote=False)
            + "\n</rejected_response>\nReturn one corrected JSON object only."
        )

    def _parse_answer(
        self,
        *,
        query: Query,
        payload_text: str,
        relevant_paper_ids: set[str],
        context_records: dict[str, Record],
        canonical_paper_titles: Mapping[str, str] | None = None,
        required_singleton_eligibility_chunk_ids: set[str] | None = None,
        attached_image_paths: list[str] | None = None,
        required_visual_paper_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        payload = parse_json_object(payload_text)
        if not isinstance(payload, dict):
            raise ReadingResponseError(
                f"{query.query_id}: answer response is not a JSON object"
            )
        status = str(payload.get("status") or "")
        if status != "ready":
            if status not in {"needs_image", "insufficient_evidence"}:
                raise ReadingResponseError(
                    f"{query.query_id}: invalid answer status {status!r}"
                )
            raise ReadingResponseError(
                f"{query.query_id}: answer is not ready: {status}"
            )

        papers = payload.get("papers")
        if not isinstance(papers, list) or not papers:
            raise ReadingResponseError(f"{query.query_id}: answer omitted evidence papers")
        validated_papers: list[dict[str, Any]] = []
        seen_papers: set[str] = set()
        seen_chunks: set[str] = set()
        for item in papers:
            if not isinstance(item, dict):
                raise ReadingResponseError("answer paper item must be an object")
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in relevant_paper_ids:
                raise ReadingResponseError(
                    f"{query.query_id}: answer invented/non-relevant paper {paper_id!r}"
                )
            chunk_ids: list[str] = []
            raw_chunk_ids = item.get("evidence_chunk_ids") or []
            if not isinstance(raw_chunk_ids, list):
                raise ReadingResponseError("evidence_chunk_ids must be a list")
            for raw_chunk_id in raw_chunk_ids:
                chunk_id = str(raw_chunk_id)
                record = context_records.get(chunk_id)
                if record is None or str(record.get("paper_id") or "") != paper_id:
                    raise ReadingResponseError(
                        f"{query.query_id}: answer invented/cross-cited chunk {chunk_id!r}"
                    )
                if not submission_evidence_eligible(record):
                    raise AnswerEvidenceLocatorError(
                        f"{query.query_id}: answer cited chunk {chunk_id!r} without a "
                        "valid official page/table/figure locator"
                    )
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    chunk_ids.append(chunk_id)
            if not chunk_ids:
                raise ReadingResponseError(
                    f"{query.query_id}: answer paper {paper_id} has no valid evidence chunk"
                )
            if paper_id not in seen_papers:
                seen_papers.add(paper_id)
                validated_papers.append(
                    {"paper_id": paper_id, "evidence_chunk_ids": chunk_ids}
                )
            else:
                for existing in validated_papers:
                    if existing["paper_id"] == paper_id:
                        existing["evidence_chunk_ids"].extend(chunk_ids)
                        break
        per_paper_counts = {
            item["paper_id"]: len(item["evidence_chunk_ids"])
            for item in validated_papers
        }
        if any(count > self.max_evidence_per_paper for count in per_paper_counts.values()):
            raise ReadingResponseError(
                f"{query.query_id}: evidence exceeds max_evidence_per_paper="
                f"{self.max_evidence_per_paper}: {per_paper_counts}"
            )
        if len(seen_chunks) > self.max_evidence:
            raise ReadingResponseError(
                f"{query.query_id}: answer cited {len(seen_chunks)} chunks; "
                f"maximum is {self.max_evidence}"
            )

        support = payload.get("support")
        if not isinstance(support, list) or not support:
            raise ReadingResponseError(f"{query.query_id}: support mapping is empty")
        validated_support: list[dict[str, Any]] = []
        support_pairs: set[tuple[str, str]] = set()
        for index, item in enumerate(support):
            if not isinstance(item, dict):
                raise ReadingResponseError(f"support[{index}] must be an object")
            answer_path = str(item.get("answer_path") or "").strip()
            if not answer_path.startswith("answer."):
                raise ReadingResponseError(
                    f"support[{index}].answer_path must start with 'answer.'"
                )
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in relevant_paper_ids:
                raise ReadingResponseError(
                    f"support[{index}] references non-relevant paper {paper_id!r}"
                )
            raw_chunk_ids = item.get("chunk_ids")
            if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
                raise ReadingResponseError(f"support[{index}].chunk_ids must be non-empty")
            chunk_ids: list[str] = []
            item_seen_pairs: set[tuple[str, str]] = set()
            for raw_chunk_id in raw_chunk_ids:
                chunk_id = str(raw_chunk_id)
                record = context_records.get(chunk_id)
                if record is None or str(record.get("paper_id") or "") != paper_id:
                    raise ReadingResponseError(
                        f"support[{index}] invented/cross-cited chunk {chunk_id!r}"
                    )
                if not submission_evidence_eligible(record):
                    raise AnswerEvidenceLocatorError(
                        f"support[{index}] cited ineligible chunk {chunk_id!r}"
                    )
                pair = (paper_id, chunk_id)
                support_pairs.add(pair)
                # A single source chunk may legitimately support several answer
                # paths (for example all cells copied from one table row).  Only
                # deduplicate repetitions inside this one support item.
                if pair not in item_seen_pairs:
                    item_seen_pairs.add(pair)
                    chunk_ids.append(chunk_id)
            validated_support.append(
                {
                    "answer_path": answer_path,
                    "paper_id": paper_id,
                    "chunk_ids": chunk_ids,
                }
            )
        paper_pairs = {
            (item["paper_id"], chunk_id)
            for item in validated_papers
            for chunk_id in item["evidence_chunk_ids"]
        }
        if paper_pairs != support_pairs:
            unused = sorted(paper_pairs - support_pairs)
            missing = sorted(support_pairs - paper_pairs)
            raise ReadingResponseError(
                f"{query.query_id}: papers and support evidence disagree; "
                f"unused={unused}, missing={missing}"
            )

        paper_relevance = payload.get("paper_relevance")
        if not isinstance(paper_relevance, list) or not paper_relevance:
            raise ReadingResponseError(
                f"{query.query_id}: paper_relevance must be a non-empty list"
            )
        validated_relevance: list[dict[str, str]] = []
        seen_relevance: set[str] = set()
        allowed_relevance_roles = {
            "target_owner",
            "answer_source",
            "comparison_source",
            "constraint_source",
            "option_source",
        }
        for index, item in enumerate(paper_relevance):
            if not isinstance(item, dict):
                raise ReadingResponseError(f"paper_relevance[{index}] must be an object")
            paper_id = str(item.get("paper_id") or "")
            role = str(item.get("role") or "")
            if paper_id not in relevant_paper_ids:
                raise ReadingResponseError(
                    f"paper_relevance[{index}] uses non-accepted paper {paper_id!r}"
                )
            if role not in allowed_relevance_roles:
                raise ReadingResponseError(
                    f"paper_relevance[{index}] has invalid role {role!r}"
                )
            if paper_id not in seen_relevance:
                seen_relevance.add(paper_id)
                validated_relevance.append(
                    {
                        "paper_id": paper_id,
                        "role": role,
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )
        if not {item["paper_id"] for item in validated_papers}.issubset(seen_relevance):
            raise ReadingResponseError(
                f"{query.query_id}: every answer-support paper must appear in paper_relevance"
            )

        answer = payload.get("answer")
        if not isinstance(answer, dict):
            raise ReadingResponseError(f"{query.query_id}: answer object is missing")
        if (
            self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            and "table" in query.answer_types
            and re.search(r"(?i)\beach\s+method\b", query.question)
        ):
            method_columns = [
                str(column.get("name") or "")
                for column in query.table_schema or []
                if column.get("is_row_key") is True
                and str(column.get("name") or "").strip().casefold()
                in {"method", "methods"}
            ]
            answer_table = answer.get("table")
            answer_rows = (
                answer_table.get("rows")
                if isinstance(answer_table, dict)
                else None
            )
            if method_columns and isinstance(answer_rows, list):
                method_column = method_columns[0]
                seen_methods: dict[str, int] = {}
                duplicates: list[str] = []
                for row_index, row in enumerate(answer_rows):
                    if not isinstance(row, dict):
                        continue
                    method = str(row.get(method_column) or "").strip()
                    # A model/variant suffix must not be used to disguise two
                    # rows for one canonical method.  In an "each method"
                    # answer, a trailing parenthetical or bracketed qualifier
                    # belongs in its schema cell, not in the Method identity.
                    canonical_method = re.sub(
                        r"\s*(?:\([^()]+\)|\[[^\[\]]+\])\s*$",
                        "",
                        method,
                    ).strip()
                    key = canonical_method.casefold()
                    if key and key in seen_methods:
                        duplicates.append(method)
                    elif key:
                        seen_methods[key] = row_index
                if duplicates:
                    raise ReadingResponseError(
                        f"{query.query_id}: one-row-per-method table cannot "
                        "duplicate a canonical method across variant rows; "
                        f"duplicate_methods={duplicates}"
                    )
        derivation = payload.get("derivation")
        if isinstance(derivation, dict):
            for fact_index, fact in enumerate(derivation.get("facts") or []):
                if not isinstance(fact, dict):
                    continue
                paper_id = str(fact.get("paper_id") or "")
                if paper_id not in relevant_paper_ids:
                    raise ReadingResponseError(
                        f"derivation.facts[{fact_index}] uses non-accepted paper {paper_id!r}"
                    )
                for raw_chunk_id in fact.get("chunk_ids") or []:
                    chunk_id = str(raw_chunk_id)
                    record = context_records.get(chunk_id)
                    if record is None or str(record.get("paper_id") or "") != paper_id:
                        raise ReadingResponseError(
                            f"derivation.facts[{fact_index}] invented/cross-cited "
                            f"chunk {chunk_id!r}"
                        )
        try:
            validated_derivation = validate_answer_semantics(
                query,
                derivation=derivation,
                answer=answer,
            )
        except DerivationValidationError as exc:
            raise ReadingResponseError(f"{query.query_id}: {exc}") from exc
        if (
            self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            and requires_scaling_eligibility_output(query)
        ):
            _validate_scaling_identifier_rows(
                query=query,
                answer=answer,
                derivation=validated_derivation,
                context_records=context_records,
            )
        if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY:
            _validate_explicit_row_identifier_surfaces(
                query=query,
                answer=answer,
                context_records=context_records,
            )
        paper_title_columns = [
            str(column.get("name") or "")
            for column in query.table_schema or []
            if str(column.get("name") or "").strip().casefold() == "paper title"
        ]
        if paper_title_columns and canonical_paper_titles:
            title_column = paper_title_columns[0]
            facts_by_id_for_titles = {
                str(fact["id"]): fact for fact in validated_derivation["facts"]
            }
            row_papers: dict[int, set[str]] = {}
            for binding in validated_derivation["answer_bindings"]:
                if binding.get("source_type") != "fact":
                    continue
                match = re.fullmatch(
                    r"answer\.table\.rows\[(\d+)\](?:\..+)?",
                    str(binding.get("answer_path") or ""),
                )
                fact = facts_by_id_for_titles.get(str(binding.get("source_id") or ""))
                if match is None or fact is None:
                    continue
                row_papers.setdefault(int(match.group(1)), set()).add(
                    str(fact["paper_id"])
                )
            answer_table = answer.get("table")
            answer_rows = (
                answer_table.get("rows")
                if isinstance(answer_table, dict)
                else []
            )
            for row_index, row in enumerate(answer_rows):
                paper_ids = row_papers.get(row_index, set())
                if len(paper_ids) != 1:
                    raise ReadingResponseError(
                        f"{query.query_id}: canonical Paper Title row "
                        f"answer.table.rows[{row_index}] must be bound to exactly "
                        f"one selected source paper; bound_paper_ids="
                        f"{sorted(paper_ids)}"
                    )
                paper_id = next(iter(paper_ids))
                expected_title = str(canonical_paper_titles.get(paper_id) or "")
                actual_title = (
                    str(row.get(title_column) or "")
                    if isinstance(row, dict)
                    else ""
                )
                if expected_title and actual_title != expected_title:
                    raise ReadingResponseError(
                        f"{query.query_id}: canonical Paper Title mismatch for "
                        f"{paper_id} at answer.table.rows[{row_index}]."
                        f"{title_column}; expected={expected_title!r}, "
                        f"actual={actual_title!r}"
                    )
        _validate_ordinal_reference_support(
            query=query,
            derivation=validated_derivation,
            context_records=context_records,
        )
        if is_aggregate_citation_count_query(query):
            _validate_stage2_citation_count_support(
                query=query,
                derivation=validated_derivation,
                context_records=context_records,
            )
        fact_pairs = {
            (str(fact["paper_id"]), str(chunk_id))
            for fact in validated_derivation["facts"]
            for chunk_id in fact["chunk_ids"]
        }
        required_eligibility_ids = set(
            required_singleton_eligibility_chunk_ids or ()
        )
        if required_eligibility_ids:
            extremum_fact_ids = {
                str(fact_id)
                for operation in validated_derivation["operations"]
                if operation.get("kind") in {"argmax", "argmin"}
                for fact_id in operation.get("fact_ids", [])
            }
            independent_eligibility_facts = [
                fact
                for fact in validated_derivation["facts"]
                if str(fact["id"]) not in extremum_fact_ids
                and any(
                    str(chunk_id) in required_eligibility_ids
                    for chunk_id in fact["chunk_ids"]
                )
            ]
            independent_eligibility_ids = {
                str(fact["id"]) for fact in independent_eligibility_facts
            }
            independent_eligibility_chunk_ids = {
                str(chunk_id)
                for fact in independent_eligibility_facts
                for chunk_id in fact["chunk_ids"]
                if str(chunk_id) in required_eligibility_ids
            }
            required_support_ids = {
                chunk_id
                for _, chunk_id in support_pairs
                if chunk_id in required_eligibility_ids
            }
            missing_fact_chunks = sorted(
                required_eligibility_ids - independent_eligibility_chunk_ids
            )
            missing_support_chunks = sorted(
                required_eligibility_ids - required_support_ids
            )
            if missing_fact_chunks or missing_support_chunks:
                raise ReadingResponseError(
                    f"{query.query_id}: filtered-singleton eligibility evidence is "
                    "missing from an independent derivation fact and submitted "
                    "support; preserve the explicit hard-condition fact outside "
                    "argmax/argmin for every required chunk; "
                    f"missing_fact_chunks={missing_fact_chunks}, "
                    f"missing_support_chunks={missing_support_chunks}, "
                    f"required_chunk_ids={sorted(required_eligibility_ids)}"
                )
            top_level_bound_fact_ids = {
                str(binding.get("source_id") or "")
                for binding in validated_derivation["answer_bindings"]
                if binding.get("source_type") == "fact"
            }
            allowed_filtered_fact_ids = (
                extremum_fact_ids
                | independent_eligibility_ids
                | top_level_bound_fact_ids
            )
            unused_filtered_fact_ids = sorted(
                str(fact["id"])
                for fact in validated_derivation["facts"]
                if str(fact["id"]) not in allowed_filtered_fact_ids
            )
            if unused_filtered_fact_ids:
                raise ReadingResponseError(
                    f"{query.query_id}: filtered-singleton eligibility evidence "
                    "contains unused identity/background facts; keep only the "
                    "independent hard-condition fact, compared operands, and "
                    "directly answer-bound facts; unused_fact_ids="
                    f"{unused_filtered_fact_ids}"
                )
        if fact_pairs != paper_pairs:
            unsupported_facts = sorted(fact_pairs - paper_pairs)
            unrelated_evidence = sorted(paper_pairs - fact_pairs)
            raise ReadingResponseError(
                f"{query.query_id}: derivation facts and submitted evidence disagree; "
                f"unsupported_facts={unsupported_facts}, "
                f"unrelated_evidence={unrelated_evidence}"
            )
        fact_paper_ids = {paper_id for paper_id, _ in fact_pairs}
        if not fact_paper_ids.issubset(seen_relevance):
            raise ReadingResponseError(
                f"{query.query_id}: every derivation fact paper must appear in "
                "paper_relevance"
            )
        if (
            self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
            and "table" in query.answer_types
            and len(relevant_paper_ids) > 1
        ):
            support_paper_ids = {paper_id for paper_id, _ in support_pairs}
            missing_facts = sorted(relevant_paper_ids - fact_paper_ids)
            missing_support = sorted(relevant_paper_ids - support_paper_ids)
            missing_relevance = sorted(relevant_paper_ids - seen_relevance)
            if missing_facts or missing_support or missing_relevance:
                raise ReadingResponseError(
                    f"{query.query_id}: fixed-selected multi-paper table must "
                    "use every authoritative selected paper with at least one "
                    "grounded derivation fact, support mapping, and "
                    "paper_relevance entry; "
                    f"missing_facts={missing_facts}, "
                    f"missing_support={missing_support}, "
                    f"missing_relevance={missing_relevance}"
                )
            # Open-ended multi-paper table questions enumerate their answer
            # rows through the authoritative selected papers themselves.  A
            # paper-local negative or constraint fact that is never bound to
            # an emitted row must not satisfy that completeness contract.  The
            # derivation validator above has already resolved every path and
            # checked that each direct fact binding's typed value agrees with
            # the emitted row object or leaf, so this check can safely use the
            # normalized bindings as a value-validated paper-to-row mapping.
            #
            # Explicit named-row questions are deliberately exempt: a selected
            # paper may supply a shared constraint or comparison operand while
            # another owner paper supplies the named output row.
            if not explicit_table_row_items(query):
                facts_by_id = {
                    str(fact["id"]): fact
                    for fact in validated_derivation["facts"]
                }
                row_bound_paper_ids = {
                    str(facts_by_id[str(binding["source_id"])]["paper_id"])
                    for binding in validated_derivation["answer_bindings"]
                    if binding.get("source_type") == "fact"
                    and re.fullmatch(
                        r"answer\.table\.rows\[\d+\](?:\..+)?",
                        str(binding.get("answer_path") or ""),
                    )
                }
                missing_row_bindings = sorted(
                    relevant_paper_ids - row_bound_paper_ids
                )
                if missing_row_bindings:
                    raise ReadingResponseError(
                        f"{query.query_id}: open-ended fixed-selected "
                        "multi-paper table must bind at least one "
                        "value-validated derivation fact from every "
                        "authoritative selected paper directly to an emitted "
                        "answer.table.rows[i] row or leaf; "
                        f"missing_row_bindings={missing_row_bindings}"
                    )
        # Stage 1 is deliberately recall-oriented, so an accepted paper can be a
        # visual false positive that Stage 2 correctly omits.  Enforce Stage-1's
        # paper-specific visual requirement only for papers that survive into the
        # final submitted facts/evidence.  ``fact_pairs == paper_pairs`` above
        # guarantees these are exactly the evidence-support papers.  The separate
        # query-level explicit-visual check remains unconditional inside
        # ``_validate_visual_grounding``.
        used_required_visual_paper_ids = set(
            required_visual_paper_ids or ()
        ).intersection(fact_paper_ids)
        _validate_visual_grounding(
            query,
            validated_derivation,
            context_records=context_records,
            attached_image_paths=attached_image_paths or [],
            required_visual_paper_ids=used_required_visual_paper_ids,
        )
        _validate_support_coverage(query, answer, validated_support)
        completeness = payload.get("completeness")
        if not isinstance(completeness, dict) or set(completeness) != {
            "answered_parts",
            "missing",
        }:
            raise ReadingResponseError(
                "completeness must contain exactly answered_parts and missing"
            )
        answered_parts = completeness.get("answered_parts")
        missing_parts = completeness.get("missing")
        if not isinstance(answered_parts, list) or not all(
            isinstance(value, str) and value.strip() for value in answered_parts
        ):
            raise ReadingResponseError(
                "completeness.answered_parts must be a list of non-empty strings"
            )
        if not isinstance(missing_parts, list) or not all(
            isinstance(value, str) and value.strip() for value in missing_parts
        ):
            raise ReadingResponseError(
                "completeness.missing must be a list of non-empty strings"
            )
        if missing_parts and "table" not in query.answer_types:
            raise ReadingResponseError(
                "ready freeform/multiple-choice answers cannot have missing parts"
            )
        completeness = {
            "answered_parts": _ordered_unique(value.strip() for value in answered_parts),
            "missing": _ordered_unique(value.strip() for value in missing_parts),
        }
        unaccounted_items = unaccounted_explicit_table_items(
            query,
            answer.get("table", {}).get("rows", []),
            completeness["missing"],
        )
        if unaccounted_items:
            raise ReadingResponseError(
                f"{query.query_id}: explicitly requested table item(s) are absent "
                "from both answer.table.rows and completeness.missing: "
                f"{list(unaccounted_items)}"
            )
        payload["papers"] = validated_papers
        payload["paper_relevance"] = validated_relevance
        payload["support"] = validated_support
        payload["derivation"] = validated_derivation
        payload["completeness"] = completeness
        return payload

    def _build_prediction(
        self,
        *,
        query: Query,
        payload: dict[str, Any],
        context_records: dict[str, Record],
        candidate_ids: list[str],
        relevant: list[dict[str, Any]],
        stage1_relevant_paper_ids: list[str] | None = None,
        submission_paper_ids: list[str] | None = None,
        image_count: int,
    ) -> Prediction:
        support_paper_ids: list[str] = []
        support_items: list[tuple[str, str, Record]] = []
        for paper in payload["papers"]:
            paper_id = str(paper["paper_id"])
            support_paper_ids.append(paper_id)
            for chunk_id in paper["evidence_chunk_ids"]:
                record = context_records[chunk_id]
                if not submission_evidence_eligible(record):
                    raise ReadingResponseError(
                        f"{query.query_id}: validated answer contains an invalid locator"
                    )
                support_items.append((paper_id, chunk_id, record))

        paper_records = [
            record
            for paper_id in _ordered_unique(support_paper_ids)
            for record in recover_split_caption_table_records(
                self.chunk_store.load_paper(paper_id)
            )
        ]
        citation_overrides = infer_citation_locator_overrides(
            query,
            derivation=payload["derivation"],
            answer=payload["answer"],
            support_records=[record for _, _, record in support_items],
            paper_records=paper_records,
        )

        evidence = []
        object_locator_overrides: dict[str, dict[str, Any]] = {}
        for paper_id, chunk_id, record in support_items:
            citation_ids: tuple[str | None, ...] = citation_overrides.get(
                chunk_id, (None,)
            )
            for citation_id in citation_ids:
                metadata = dict(record.get("metadata") or {})
                original_locator = coarse_locator(record)
                prediction_locator = query_aware_prediction_locator(
                    record, query.question
                )
                for object_field in ("table_id", "figure_id"):
                    recovered_id = prediction_locator.get(object_field)
                    if recovered_id == original_locator.get(object_field):
                        continue
                    metadata[object_field] = recovered_id
                    object_locator_overrides.setdefault(
                        chunk_id,
                        {
                            "source_type": record_source_type(record),
                            "from": original_locator.get(object_field),
                            "to": recovered_id,
                        },
                    )
                if citation_id is not None:
                    metadata["citation_id"] = citation_id
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    score=0.0,
                    text=str(record.get("text") or ""),
                    chunk_type=record_source_type(record),
                    metadata=metadata,
                    source="aoai_pairwise_reader",
                )
                evidence.append(evidence_from_result(result))
        if not evidence:
            raise ReadingResponseError(
                f"{query.query_id}: answer cited no chunk with a valid official locator"
            )
        raw_answer = payload["answer"]
        freeform = raw_answer.get("freeform") if "freeform" in query.answer_types else None
        table = None
        if "table" in query.answer_types:
            table = {
                "schema": query.table_schema or [],
                "rows": raw_answer["table"]["rows"],
            }
        multiple_choice = None
        if "multiple_choice" in query.answer_types:
            multiple_choice = {
                "gold": raw_answer["multiple_choice"]["label"]
            }
        fixed_selected = self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY
        relevant_paper_ids = (
            list(submission_paper_ids)
            if fixed_selected and submission_paper_ids is not None
            else _ordered_unique(support_paper_ids)
        )
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in relevant_paper_ids],
            evidence=evidence,
            answer=Answer(
                freeform=freeform,
                multiple_choice=multiple_choice,
                table=table,
            ),
            trace=[
                {
                    "stage": (
                        "fixed_selected_evidence_extraction"
                        if fixed_selected
                        else "pairwise_candidate_judgment"
                    ),
                    "judged": len(candidate_ids),
                    "paper_set_policy": self.paper_set_policy,
                    "relevant_paper_ids": relevant_paper_ids,
                    "accepted_paper_ids": [item["paper_id"] for item in relevant],
                },
                {
                    "stage": "accepted_evidence_answer",
                    "paper_relevance": payload["paper_relevance"],
                    "support_paper_ids": support_paper_ids,
                    "image_count": image_count,
                    "citation_locator": {
                        "version": CITATION_LOCATOR_VERSION,
                        "overrides": {
                            chunk_id: list(citation_ids)
                            for chunk_id, citation_ids in citation_overrides.items()
                        },
                    },
                    "object_locator": {
                        "version": QUERY_AWARE_OBJECT_LOCATOR_VERSION,
                        "split_caption_version": (
                            SPLIT_CAPTION_TABLE_LOCATOR_VERSION
                        ),
                        "overrides": object_locator_overrides,
                    },
                    "derivation": payload.get("derivation"),
                    "semantic_multiple_choice": raw_answer.get("multiple_choice"),
                    "completeness": payload.get("completeness"),
                },
            ],
            candidate_papers=candidate_ids,
        )

    # ---- Shared helpers ---------------------------------------------------

    def _complete(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        *,
        max_completion_tokens: int | None = None,
        provider_attempt_callback: ProviderAttemptCallback | None = None,
        semantic_phase: str = "unspecified",
    ) -> CompletionResult:
        started = time.monotonic()
        provider_invocation_count = 0

        def invoke(
            effective_prompt: str, paths: list[str] | None
        ) -> CompletionResult:
            nonlocal provider_invocation_count
            attempt_id = str(uuid.uuid4())
            invocation_index = provider_invocation_count + 1
            prompt_sha256 = hashlib.sha256(
                effective_prompt.encode("utf-8")
            ).hexdigest()
            if provider_attempt_callback is not None:
                # The runner callback blocks until the coordinator has fsynced
                # this PREPARE row. A crash thereafter is reported as uncertain
                # rather than silently losing a potentially billable request.
                provider_attempt_callback(
                    {
                        "attempt_id": attempt_id,
                        "event_kind": "prepare",
                        "semantic_phase": semantic_phase,
                        "provider_invocation_index": invocation_index,
                        "provider_invocation_count": 1,
                        "prompt_sha256": prompt_sha256,
                        "prompt_characters": len(effective_prompt),
                        "requested_image_count": len(paths or []),
                        "max_completion_tokens": max_completion_tokens,
                    }
                )
            provider_invocation_count += 1
            try:
                complete_with_metadata = getattr(
                    self.llm, "complete_with_metadata", None
                )
                if callable(complete_with_metadata):
                    kwargs: dict[str, Any] = {"image_paths": paths or None}
                    if max_completion_tokens is not None and _accepts_keyword(
                        complete_with_metadata, "max_completion_tokens"
                    ):
                        kwargs["max_completion_tokens"] = max_completion_tokens
                    response = complete_with_metadata(effective_prompt, **kwargs)
                    if not isinstance(response, dict) or not isinstance(
                        response.get("text"), str
                    ):
                        raise ReadingResponseError(
                            "LLM metadata response must contain text"
                        )
                    result = cast(CompletionResult, dict(response))
                else:
                    complete = getattr(self.llm, "complete", None)
                    if callable(complete):
                        kwargs = {"image_paths": paths or None}
                        if max_completion_tokens is not None and _accepts_keyword(
                            complete, "max_completion_tokens"
                        ):
                            kwargs["max_completion_tokens"] = max_completion_tokens
                        raw = complete(effective_prompt, **kwargs)
                    else:
                        raw = self.llm(effective_prompt)
                    if not isinstance(raw, str):
                        raise ReadingResponseError("LLM response must be text")
                    result = {"text": raw}
            except Exception as exc:
                exc._littraceqa_provider_attempt_id = attempt_id  # type: ignore[attr-defined]
                if provider_attempt_callback is not None:
                    provider_attempt_callback(
                        {
                            "attempt_id": attempt_id,
                            "event_kind": "finalize",
                            "outcome": "provider_error",
                            "semantic_phase": semantic_phase,
                            "provider_invocation_index": invocation_index,
                            "provider_invocation_count": 1,
                            "prompt_sha256": prompt_sha256,
                            "prompt_characters": len(effective_prompt),
                            "requested_image_count": len(paths or []),
                            "max_completion_tokens": max_completion_tokens,
                            "_exception": exc,
                        }
                    )
                raise
            result["provider_attempt_id"] = attempt_id
            result["provider_semantic_phase"] = semantic_phase
            result["provider_invocation_index"] = invocation_index
            return result

        sent_prompt = prompt
        try:
            result = invoke(sent_prompt, image_paths)
        except Exception as exc:
            if not image_paths or not _is_image_content_policy_violation(exc):
                if _prompt_content_filter_categories(exc):
                    _record_prompt_filter_provider_invocation_count(
                        exc, provider_invocation_count, sent_prompt
                    )
                raise
            # Do not retry or transform rejected image content.  Preserve the
            # paper-level run by judging the same selected context from text alone, and
            # make the degraded modality explicit in the checkpoint metadata.
            if semantic_phase.startswith("judgment"):
                if self.paper_set_policy == FIXED_SELECTED_PAPER_POLICY:
                    modality_result_instruction = (
                        "Extract text-grounded atomic facts only. Omit every fact "
                        "that requires reading an image. Return exactly one JSON "
                        "object whose only field is evidence_facts; if no requested "
                        "text-grounded fact remains, return "
                        '{"evidence_facts": []}.'
                    )
                else:
                    modality_result_instruction = (
                        "Judge A independently from paper identity and text. If the "
                        "requested answer evidence depends on an image, set "
                        "has_usable_answer_evidence=false and evidence_chunk_ids=[] "
                        "in the required three-field JSON."
                    )
            else:
                modality_result_instruction = (
                    "If the requested answer depends on an image, return "
                    "status=needs_image in the required Stage-2 JSON."
                )
            fallback_prompt = (
                prompt
                + "\n\nMODALITY OVERRIDE FOR THIS RETRY: No image is attached. "
                "Ignore every earlier image mapping, do not claim visual inspection, "
                + modality_result_instruction
            )
            sent_prompt = fallback_prompt
            try:
                result = invoke(sent_prompt, None)
            except Exception as fallback_exc:
                if _prompt_content_filter_categories(fallback_exc):
                    _record_prompt_filter_provider_invocation_count(
                        fallback_exc, provider_invocation_count, sent_prompt
                    )
                raise
            result["image_fallback_reason"] = "content_policy_violation"
            result["requested_image_count"] = len(image_paths)
            result["attached_image_count"] = 0

        result.setdefault("latency_seconds", time.monotonic() - started)
        result.setdefault("requested_image_count", len(image_paths or []))
        result.setdefault("attached_image_count", len(image_paths or []))
        # This counts explicit adapter invocations, including the image-policy
        # text-only fallback. Provider-SDK HTTP retries remain visible through
        # provider telemetry rather than this application-level counter.
        result["provider_invocation_count"] = provider_invocation_count
        result["prompt_sha256"] = hashlib.sha256(
            sent_prompt.encode("utf-8")
        ).hexdigest()
        result["prompt_characters"] = len(sent_prompt)
        return result
    def _format_record(
        self,
        record: Record,
        *,
        text: str,
        selected: bool | None = None,
    ) -> str:
        metadata = record.get("metadata") or {}
        locator = {
            key: metadata.get(key)
            for key in (
                "page",
                "section",
                "table_id",
                "figure_id",
                "equation_id",
                "algorithm_id",
                "citation_id",
            )
            if metadata.get(key) is not None
        }
        header = {
            "paper_id": record.get("paper_id"),
            "chunk_id": record.get("chunk_id"),
            "source_type": record_source_type(record),
            "locator": locator,
            "submission_eligible": submission_evidence_eligible(record),
        }
        if selected is not None:
            header["stage1_selected"] = selected
        return "[chunk " + _json_dumps(header) + "]\n" + text

    def _image_legend(
        self, records_by_id: dict[str, Record], image_paths: list[str]
    ) -> str:
        lines: list[str] = []
        for index, path in enumerate(image_paths, start=1):
            chunk_ids = [
                chunk_id
                for chunk_id, record in records_by_id.items()
                if readable_image_path(record) == path
            ]
            mapped_records = [records_by_id[chunk_id] for chunk_id in chunk_ids]
            sources = _ordered_unique(
                record_source_type(record) for record in mapped_records
            )
            locators: list[dict[str, Any]] = []
            for record in mapped_records:
                locator = coarse_locator(record)
                if locator not in locators:
                    locators.append(locator)
            lines.append(
                f"Image {index}: chunk_ids={','.join(chunk_ids)} "
                f"source_types={_json_dumps(sources)} "
                f"locators={_json_dumps(locators)} file={Path(path).name}"
            )
        return "\n".join(lines)


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    """Whether a bound adapter method explicitly accepts a safe override."""

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _production_query_payload(query: Query) -> dict[str, Any]:
    return query.to_dict()


def _query_selection_focus(query: Query) -> str:
    """Build selector input exclusively from fields observable at test time."""

    parts = [query.question]
    parts.extend(str(value) for value in (query.options or {}).values())
    for column in query.table_schema or []:
        if isinstance(column, dict):
            parts.extend(
                str(column[key])
                for key in ("name", "type")
                if column.get(key) is not None
            )
    return "\n".join(parts)


def _selection_tokens(text: str) -> list[str]:
    """Tokenize compounds and add a few morphology-safe matching forms."""

    output: list[str] = []
    aliases = {
        "increase": ("improvement", "performance", "comparison", "delta"),
        "increases": ("improvement", "performance", "comparison", "delta"),
        "improve": ("improvement", "performance", "delta"),
        "improves": ("improvement", "performance", "delta"),
        "outperform": ("performance", "comparison", "delta"),
        "outperforms": ("performance", "comparison", "delta"),
        "avg": ("average",),
        "average": ("avg",),
    }
    for raw_token in _WORD_RE.findall(text):
        raw = raw_token.casefold().removesuffix("'s")
        pieces = re.findall(
            r"[a-z]+[0-9]*(?:\.[0-9]+)*|[0-9]+(?:\.[0-9]+)*(?:[a-z]+)?",
            raw,
        )
        candidates = [raw, *pieces]
        for token in list(candidates):
            if token.endswith("ing") and len(token) > 5:
                candidates.extend((token[:-3], token[:-3] + "e"))
            elif token.endswith("ed") and len(token) > 4:
                candidates.extend((token[:-2], token[:-1]))
            elif token.endswith("s") and len(token) > 4:
                candidates.append(token[:-1])
            candidates.extend(aliases.get(token, ()))
        for token in candidates:
            if (
                (len(token) >= 2 or token.isdigit())
                and token not in _QUERY_STOPWORDS
                and token not in output
            ):
                output.append(token)
    return output


def _required_singleton_eligibility_chunk_ids(
    query: Query,
    judgments: Iterable[dict[str, Any]],
) -> set[str]:
    """Select the Stage-1 fact that proves the query's explicit ``only`` clause.

    A paper may have several broad ``eligibility_condition`` facts.  Requiring
    any one of them is too weak: a generic fact such as "is VLM-based" must not
    replace the narrower condition "trained only on BaseSet (1000 clips)".
    Rank within each selected paper using only lexical overlap with the local
    query clause and retain every tied best fact.  If Stage 1 supplied no
    lexical match, fail conservatively by retaining all of that paper's
    eligibility facts.
    """

    clause = explicit_singleton_eligibility_clause(query.question)
    clause_tokens = set(_selection_tokens(clause or ""))
    required: set[str] = set()
    for judgment in judgments:
        eligible_facts = [
            fact
            for fact in (judgment.get("extracted_facts") or [])
            if isinstance(fact, dict)
            and str(fact.get("purpose") or "") == "eligibility_condition"
            and str(fact.get("chunk_id") or "")
        ]
        if not eligible_facts:
            continue
        scored: list[tuple[int, dict[str, Any]]] = []
        for fact in eligible_facts:
            # ``fact`` is free-form model prose.  ``source_excerpt`` has
            # already been verified as a contiguous visible source substring,
            # so only the latter may decide a hard deterministic requirement.
            fact_text = str(fact.get("source_excerpt") or "")
            fact_tokens = set(_selection_tokens(fact_text))
            overlap = clause_tokens & fact_tokens
            score = sum(3 if token.isdigit() else 1 for token in overlap)
            scored.append((score, fact))
        best_score = max(score for score, _ in scored)
        selected = (
            [fact for score, fact in scored if score == best_score]
            if best_score > 0
            else eligible_facts
        )
        required.update(str(fact["chunk_id"]) for fact in selected)
    return required


def _canonical_scaling_base_model(value: str) -> str:
    """Apply the narrow canonical surface allowed for a scaling Base Model cell."""

    canonical = unicodedata.normalize("NFKC", str(value or "")).strip()
    if re.search(r"\d+(?:\.\d+)?[bmk]\b", canonical, re.IGNORECASE):
        canonical = re.sub(r"\s+models?\s*$", "", canonical, flags=re.IGNORECASE)
    # Source prose often prints compact identifiers such as Infinity2B while
    # table schemas use the conventional family-size form Infinity-2B.
    canonical = re.sub(
        r"(?<=[A-Za-z])(?=\d+(?:\.\d+)?[BMK]\b)",
        "-",
        canonical,
    )
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:and|&)\s+|\s*,\s*", canonical)
        if part.strip()
    ]
    parsed: list[tuple[str, str]] = []
    for part in parts:
        match = re.fullmatch(
            r"(?P<family>.+?)-(?P<size>\d+(?:\.\d+)?[BMK])",
            part,
            re.IGNORECASE,
        )
        if match is None:
            parsed = []
            break
        parsed.append((match.group("family"), match.group("size")))
    if len(parsed) >= 2 and len({family.casefold() for family, _ in parsed}) == 1:
        family = parsed[0][0]
        canonical = f"{family}-{parsed[0][1]}/" + "/".join(
            size for _, size in parsed[1:]
        )
    return canonical


def _canonical_scaling_method(value: str) -> str:
    """Remove a procedural scaling suffix from an owning method identity."""

    canonical = unicodedata.normalize("NFKC", str(value or "")).strip()
    reduced = re.sub(
        r"\s*(?:\+|with|using)?\s*"
        r"(?:inference(?:[- ]time)?|test[- ]time)\s+scaling\s*$",
        "",
        canonical,
        flags=re.IGNORECASE,
    ).strip()
    return reduced or canonical


def _source_identifier_surface(value: str, records: Iterable[Record]) -> str | None:
    """Recover an equivalent identifier surface from cited source body text."""

    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)+|\d+[A-Za-z]+", value)
    if not tokens:
        return None
    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        + r"[\s._+-]*".join(re.escape(token) for token in tokens)
        + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str]] = []
    for record in records:
        body = _without_corpus_title_prefix(str(record.get("text") or ""))
        if value in body:
            return value
        for match in pattern.finditer(body):
            surface = match.group(0)
            window = body[max(0, match.start() - 80) : match.end() + 80]
            introduction = bool(
                re.search(
                    r"\b(?:propose|present|introduce|term(?:ed)?|call(?:ed)?)\b",
                    window,
                    re.IGNORECASE,
                )
            )
            punctuation = sum(character in "-._+" for character in surface)
            candidates.append((introduction, punctuation, surface))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _source_specific_base_model(value: str, records: Iterable[Record]) -> str:
    """Recover one unambiguous family+size identifier omitted by the model."""

    canonical = _canonical_scaling_base_model(value)
    if re.search(r"\d+(?:\.\d+)?[BMK]\b", canonical, re.IGNORECASE):
        return canonical
    family = canonical.strip()
    if not family or not re.fullmatch(r"[A-Za-z][A-Za-z0-9._+-]*", family):
        return canonical
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(family)}(?P<separator>[- ]?)"
        r"(?P<size>\d+(?:\.\d+)?[BMK])\b"
        r"(?P<version>\s+v\d+(?:\.\d+)*)?",
        re.IGNORECASE,
    )
    identifiers: dict[str, str] = {}
    cointroduced: dict[str, str] = {}
    for record in records:
        body = _without_corpus_title_prefix(str(record.get("text") or ""))
        record_matches = list(pattern.finditer(body))
        rendered_record: list[tuple[re.Match[str], str]] = []
        for match in record_matches:
            size = match.group("size")
            separator = match.group("separator")
            version = match.group("version") or ""
            # Preserve an explicit source-space when a versioned family name
            # such as SANA-1.5 is followed by its parameter size.  For compact
            # forms such as Infinity2B, use the conventional family-size dash.
            rendered_separator = separator or "-"
            identifier = f"{family}{rendered_separator}{size}{version}"
            identifiers.setdefault(identifier.casefold(), identifier)
            rendered_record.append((match, identifier))
        if len(rendered_record) >= 2:
            # Combine only variants introduced together inside one sentence.
            # Later ``also`` experiments normally live in another sentence or
            # chunk and must not silently expand a singular primary answer.
            contiguous = all(
                re.fullmatch(
                    r"\s*(?:,|and|&|,\s*and)\s*",
                    body[left.end() : right.start()],
                    re.IGNORECASE,
                )
                is not None
                for (left, _), (right, _) in zip(
                    rendered_record,
                    rendered_record[1:],
                    strict=False,
                )
            )
            if contiguous:
                combined = _canonical_scaling_base_model(
                    " and ".join(identifier for _, identifier in rendered_record)
                )
                cointroduced.setdefault(combined.casefold(), combined)
    if len(cointroduced) == 1:
        return next(iter(cointroduced.values()))
    if len(identifiers) != 1:
        return canonical
    return next(iter(identifiers.values()))


def _validate_scaling_identifier_rows(
    *,
    query: Query,
    answer: dict[str, Any],
    derivation: dict[str, Any],
    context_records: Mapping[str, Record],
) -> None:
    """Require deterministic canonical method/model surfaces for scaling rows."""

    table = answer.get("table")
    rows = table.get("rows") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return
    schema_names = {
        str(column.get("name") or "").strip().casefold(): str(
            column.get("name") or ""
        )
        for column in query.table_schema or []
    }
    method_column = schema_names.get("method") or schema_names.get("methods")
    base_column = schema_names.get("base model")
    if not method_column or not base_column:
        return
    facts_by_id = {str(fact["id"]): fact for fact in derivation["facts"]}
    cell_fact_ids: dict[tuple[int, str], list[str]] = {}
    for binding in derivation["answer_bindings"]:
        if binding.get("source_type") != "fact":
            continue
        match = re.fullmatch(
            r"answer\.table\.rows\[(\d+)\]\.([^\n]+)",
            str(binding.get("answer_path") or ""),
        )
        if match is None:
            continue
        cell_fact_ids.setdefault((int(match.group(1)), match.group(2)), []).append(
            str(binding.get("source_id") or "")
        )

    mismatches: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        actual_method = str(row.get(method_column) or "")
        canonical_method = _canonical_scaling_method(actual_method)
        method_records: list[Record] = []
        for fact_id in cell_fact_ids.get((row_index, method_column), []):
            fact = facts_by_id.get(fact_id)
            if fact is None:
                continue
            method_records.extend(
                context_records[chunk_id]
                for chunk_id in fact["chunk_ids"]
                if chunk_id in context_records
            )
        expected_method = _source_identifier_surface(canonical_method, method_records)
        expected_method = expected_method or canonical_method
        if expected_method and actual_method != expected_method:
            mismatches.append(
                f"answer.table.rows[{row_index}].{method_column}: "
                f"expected={expected_method!r}, actual={actual_method!r}"
            )

        actual_base = str(row.get(base_column) or "")
        base_records: list[Record] = []
        for fact_id in cell_fact_ids.get((row_index, base_column), []):
            fact = facts_by_id.get(fact_id)
            if fact is None:
                continue
            base_records.extend(
                context_records[chunk_id]
                for chunk_id in fact["chunk_ids"]
                if chunk_id in context_records
            )
        expected_base = _source_specific_base_model(actual_base, base_records)
        if actual_base != expected_base:
            mismatches.append(
                f"answer.table.rows[{row_index}].{base_column}: "
                f"expected={expected_base!r}, actual={actual_base!r}"
            )
    if mismatches:
        raise ReadingResponseError(
            f"{query.query_id}: canonical scaling identifier mismatch; "
            + "; ".join(mismatches)
        )


def _identifier_key(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        unicodedata.normalize("NFKC", str(value or "")).casefold(),
    )


def _one_edit_or_equal(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    skipped = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        skipped += 1
        long_index += 1
        if skipped > 1:
            return False
    return True


def _source_identifier_tokens(records: Iterable[Record]) -> dict[str, set[str]]:
    tokens: dict[str, set[str]] = {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9]*"
        r"(?:[-_][A-Za-z0-9]+)+|[A-Z][A-Z0-9]{2,})(?![A-Za-z0-9])"
    )
    for record in records:
        body = _without_corpus_title_prefix(str(record.get("text") or ""))
        for match in pattern.finditer(body):
            surface = match.group(0)
            key = _identifier_key(surface)
            if key:
                tokens.setdefault(key, set()).add(surface)
    return tokens


def _validate_explicit_row_identifier_surfaces(
    *,
    query: Query,
    answer: dict[str, Any],
    context_records: Mapping[str, Record],
) -> None:
    """Prefer one unique source spelling over a one-character query typo."""

    required = explicit_table_row_items(query)
    table = answer.get("table")
    rows = table.get("rows") if isinstance(table, dict) else None
    row_keys = [
        str(column.get("name") or "")
        for column in query.table_schema or []
        if column.get("is_row_key") is True
    ]
    if not required or not isinstance(rows, list) or len(row_keys) != 1:
        return
    row_key = row_keys[0]
    row_surfaces = table_row_key_texts(query, rows)
    source_tokens = _source_identifier_tokens(context_records.values())
    mismatches: list[str] = []
    used_rows: set[int] = set()
    for item in required:
        item_key = _identifier_key(item)
        if len(item_key) < 6 or re.search(r"\s", item.strip()):
            continue
        row_candidates = [
            index
            for index, surface in enumerate(row_surfaces)
            if index not in used_rows
            and _one_edit_or_equal(item_key, _identifier_key(surface))
        ]
        if len(row_candidates) != 1:
            continue
        row_index = row_candidates[0]
        used_rows.add(row_index)
        # Punctuation-only variation (RISurConv versus RISur-Conv) is not a
        # spelling correction: keep the released query's row surface because
        # the official string evaluator distinguishes it.  Correct only when
        # the alphanumeric keys themselves differ by exactly one character.
        if source_tokens.get(item_key):
            continue
        nearby = {
            surface
            for key, surfaces in source_tokens.items()
            if key != item_key and _one_edit_or_equal(item_key, key)
            for surface in surfaces
        }
        nearby_keys = {_identifier_key(surface) for surface in nearby}
        if len(nearby_keys) != 1:
            continue
        expected = sorted(nearby, key=lambda value: (len(value), value))[0]
        actual = str(rows[row_index].get(row_key) or "")
        if actual != expected:
            mismatches.append(
                f"answer.table.rows[{row_index}].{row_key}: "
                f"expected={expected!r}, actual={actual!r}, query_item={item!r}"
            )
    if mismatches:
        raise ReadingResponseError(
            f"{query.query_id}: canonical explicit row identifier mismatch; "
            + "; ".join(mismatches)
        )


def _canonical_object_kind(raw_kind: str) -> str:
    kind = raw_kind.casefold().rstrip(".")
    if kind in {"fig", "figure"}:
        return "figure"
    if kind in {"eq", "equation"}:
        return "equation"
    if kind in {"ref", "reference", "citation"}:
        return "reference"
    return kind


def _explicit_object_keys(text: str) -> set[tuple[str, str]]:
    return {
        (_canonical_object_kind(match.group(1)), match.group(2).casefold())
        for match in _OBJECT_REFERENCE_RE.finditer(text)
    }


def _record_object_keys(record: Record) -> set[tuple[str, str]]:
    metadata = record.get("metadata") or {}
    keys: set[tuple[str, str]] = set()
    for field, kind in (
        ("figure_id", "figure"),
        ("table_id", "table"),
        ("equation_id", "equation"),
        ("algorithm_id", "algorithm"),
        ("citation_id", "reference"),
    ):
        value = metadata.get(field)
        if value is not None:
            parsed = _explicit_object_keys(f"{kind} {value}")
            if parsed:
                keys.update(parsed)
            else:
                identifier = re.sub(r"(?i)^(?:figure|fig\.?|table|equation|eq\.?|"
                                    r"algorithm|reference|ref\.?|citation)\s*", "", str(value))
                if identifier:
                    keys.add((kind, identifier.strip().casefold()))

    source_type = record_source_type(record)
    if source_type in {"figure", "table", "equation_algorithm", "citation_context"}:
        keys.update(_explicit_object_keys(str(record.get("text") or "")[:2_000]))
    return keys


def _metadata_figure_object_keys(record: Record) -> set[tuple[str, str]]:
    """Return only exact Figure identifiers declared by record metadata.

    Deterministic visual handoff must not treat a Figure number merely mentioned
    in another Figure's caption or OCR text as ownership of that numbered object.
    """

    figure_id = (record.get("metadata") or {}).get("figure_id")
    if figure_id is None:
        return set()
    return {
        key
        for key in _explicit_object_keys(f"Figure {figure_id}")
        if key[0] == "figure"
    }


def _without_corpus_title_prefix(text: str) -> str:
    """Remove the synthetic corpus title line from a scoped Figure caption.

    MinerU chunks begin with ``[VENUE YEAR] paper title``.  Candidate metadata
    already supplies that identity; retaining it in a query that asks whether a
    term occurs *inside a Figure* would blur the location boundary.
    """

    first, separator, remainder = text.partition("\n")
    if separator and re.match(r"^\[[^\]]+\]\s+", first.strip()):
        return remainder
    return text


def _requested_source_types(focus: str) -> set[str]:
    lower = focus.casefold()
    requested: set[str] = set()
    if requires_visual_image(focus):
        requested.add("figure")
    if re.search(r"\b(?:table|row|column)s?\b", lower):
        requested.add("table")
    if re.search(r"\b(?:equation|formula)s?\b|\beq\.\s*\d", lower):
        requested.add("equation_algorithm")
    if re.search(r"\balgorithms?\b", lower):
        requested.add("equation_algorithm")
    if re.search(
        r"\b(?:cited|citation|citations|bibliography|bibliographic)\b|"
        r"\breferences?\b(?![- ]free)",
        lower,
    ):
        requested.add("citation_context")
    return requested


def _paper_record_scores(
    records: list[Record], focus: str, *, source_question: str | None = None
) -> tuple[list[float], list[list[str]]]:
    """Score records deterministically with query-only IDF-weighted overlap."""

    query_tokens = _selection_tokens(focus)
    query_counts: dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    record_token_sets: list[set[str]] = []
    record_token_counts: list[dict[str, int]] = []
    document_frequency: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        searchable = " ".join(
            (
                str(record.get("text") or ""),
                str(record_source_type(record)),
                str(metadata.get("section") or ""),
                str(metadata.get("table_id") or ""),
                str(metadata.get("figure_id") or ""),
                str(metadata.get("equation_id") or ""),
                str(metadata.get("algorithm_id") or ""),
                str(metadata.get("citation_id") or ""),
            )
        )
        counts: dict[str, int] = {}
        for token in _selection_tokens(searchable):
            counts[token] = counts.get(token, 0) + 1
        token_set = set(counts)
        record_token_counts.append(counts)
        record_token_sets.append(token_set)
        for token in token_set.intersection(query_counts):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    explicit_keys = _explicit_object_keys(focus)
    requested_sources = _requested_source_types(source_question or focus)
    count = len(records)
    scores: list[float] = []
    all_reasons: list[list[str]] = []
    for index, record in enumerate(records):
        score = 0.0
        reasons: list[str] = []
        for token, query_count in query_counts.items():
            term_count = record_token_counts[index].get(token, 0)
            if not term_count:
                continue
            inverse_frequency = math.log(
                (count + 1) / (document_frequency.get(token, 0) + 1)
            )
            numeric_boost = 4.0 if any(character.isdigit() for character in token) else 1.0
            score += (
                inverse_frequency
                * numeric_boost
                * min(2, query_count)
                * (1.0 + min(term_count, 4) * 0.15)
            )
        if score:
            reasons.append("query_lexical_overlap")
        source_type = record_source_type(record)
        if source_type in requested_sources:
            score += 80.0
            reasons.append("requested_source_type")
        if explicit_keys.intersection(_record_object_keys(record)):
            score += 1_000_000.0
            reasons.append("explicit_object_reference")
        if index == 0 or source_type == "title_abstract":
            score += 8.0
            reasons.append("paper_identity_anchor")
        scores.append(score)
        all_reasons.append(reasons)
    return scores, all_reasons


def _ranked_image_record_indices(
    records: list[Record],
    *,
    scores: list[float],
    reasons: list[list[str]],
    limit: int,
    question: str,
    scoped_source_type: str | None = None,
) -> list[int]:
    if limit <= 0:
        return []
    explicit_indices = {
        index
        for index, item_reasons in enumerate(reasons)
        if "explicit_object_reference" in item_reasons
        and readable_image_path(records[index])
        and (
            scoped_source_type is None
            or record_source_type(records[index]) == scoped_source_type
        )
    }
    # MinerU commonly emits the panels of one official figure as consecutive
    # images, while only the last panel carries the shared ``figure_id`` and
    # caption.  Keep readable, same-page, same-type siblings around an explicit
    # anchor so a Figure 4(b) panel is not dropped merely because Figure 4's ID
    # was attached to panel (c).
    explicit_group_indices = set(explicit_indices)
    requested_object_keys = _explicit_object_keys(question)
    for anchor in explicit_indices:
        anchor_metadata = records[anchor].get("metadata") or {}
        anchor_page = anchor_metadata.get("page")
        anchor_source_type = record_source_type(records[anchor])
        for index in range(max(0, anchor - 3), min(len(records), anchor + 4)):
            metadata = records[index].get("metadata") or {}
            sibling_object_keys = _record_object_keys(records[index])
            if (
                metadata.get("page") == anchor_page
                and record_source_type(records[index]) == anchor_source_type
                and readable_image_path(records[index])
                and (
                    not sibling_object_keys
                    or bool(sibling_object_keys.intersection(requested_object_keys))
                )
            ):
                explicit_group_indices.add(index)
    if explicit_indices:
        # When the requested object exists, attach only that exact object and
        # the nearby same-page panels that MinerU may have split from it. A
        # second high-scoring figure from another page can contain tempting
        # answer-like text and be incorrectly attributed to the requested
        # object. Broader visual fallback is reserved for papers where no exact
        # readable object survived preprocessing.
        selected: list[int] = []
        seen_paths: set[str] = set()
        for index in sorted(explicit_group_indices):
            path = readable_image_path(records[index])
            if not path or path in seen_paths:
                continue
            selected.append(index)
            seen_paths.add(path)
            if len(selected) >= limit:
                break
        return selected
    explicit_visual = requires_visual_image(question)
    explicit_image_source = bool(
        _requested_source_types(question).intersection({"figure", "table"})
    )
    implicit_visual_value = bool(
        re.search(
            r"(?i)\b(?:score|accuracy|performance|benchmark|value|rate|"
            r"percentage|percent|standard deviation|nrmse|fid|map|gamma|"
            r"how much|how many|"
            r"more(?:\s+[a-z0-9&-]+){0,4}\s+than|"
            r"(?:less|fewer)(?:\s+[a-z0-9&-]+){0,4}\s+than|"
            r"increase[sd]?|improv(?:e[sd]?|ement)|outperform(?:s|ed)?|"
            r"highest|lowest|largest|smallest|best|worst|avg(?:erage)?@?\d*)\b",
            question,
        )
    )
    if explicit_visual or explicit_image_source:
        # An unnumbered "main figure/table" can appear late in a long paper;
        # retain the configured safety cap while still ranking query-relevant
        # visuals first.
        selection_limit = limit
    elif implicit_visual_value:
        selection_limit = min(limit, 7)
    else:
        # A one-call design cannot inspect every decorative figure cheaply.
        # Text/citation questions get no speculative images; the selected
        # captions/table bodies remain available as text.
        return []

    contextual_scores: list[float] = []
    for index, score in enumerate(scores):
        neighbours = [
            scores[neighbour]
            for neighbour in range(max(0, index - 2), min(len(scores), index + 3))
            if neighbour != index
        ]
        contextual_scores.append(score + 0.65 * max(neighbours, default=0.0))
    candidates = sorted(
        (
            index
            for index, record in enumerate(records)
            if readable_image_path(record)
            and (
                scoped_source_type is None
                or record_source_type(record) == scoped_source_type
            )
        ),
        key=lambda index: (
            index not in explicit_group_indices,
            -contextual_scores[index],
            index,
        ),
    )
    if explicit_group_indices:
        candidates = _ordered_unique_ints(
            [*sorted(explicit_group_indices), *candidates]
        )
    structural_fallback_indices: set[int] = set()
    if not explicit_indices:
        requested_sources = _requested_source_types(question)
        early_source_types = (
            requested_sources.intersection({"figure", "table"})
            if explicit_image_source
            else {"figure", "table"}
        )
        early_structural = sorted(
            (
                index
                for index in candidates
                if record_source_type(records[index]) in early_source_types
            ),
            key=lambda index: (
                int((records[index].get("metadata") or {}).get("page") or 10**9),
                index,
            ),
        )
        if scoped_source_type == "table":
            structural_fallback_indices.update(early_structural[:2])
            candidates = _ordered_unique_ints([*early_structural[:2], *candidates])
        if explicit_visual or explicit_image_source:
            # "primary/main figure/table" often has a generic caption whose
            # lexical score is weak. Reserve three slots for early structural
            # objects, then fill the remaining configured cap by relevance.
            structural_fallback_indices.update(early_structural[:3])
            candidates = _ordered_unique_ints([*early_structural[:3], *candidates])
        elif implicit_visual_value:
            # Quantitative questions without an explicit locator get the five
            # strongest query matches plus two early-figure fallbacks. Numeric
            # tables normally rank lexically; charts whose labels only exist in
            # pixels need the extra structural coverage (for example Figure 2
            # with category counts).
            early_figures = [
                index
                for index in early_structural
                if record_source_type(records[index]) == "figure"
            ]
            structural_fallback_indices.update(early_figures[:2])
            candidates = _ordered_unique_ints(
                [*candidates[:5], *early_figures[:2], *candidates[5:]]
            )
    selected: list[int] = []
    seen_paths: set[str] = set()
    for index in candidates:
        if (
            index not in explicit_group_indices
            and index not in structural_fallback_indices
            and contextual_scores[index] <= 0.0
        ):
            continue
        path = readable_image_path(records[index])
        if not path or path in seen_paths:
            continue
        selected.append(index)
        seen_paths.add(path)
        if len(selected) >= selection_limit:
            break
    return selected


def _ordered_unique_ints(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _bounded_formatted_record(
    reader: PairwiseAOAIReader,
    record: Record,
    *,
    focus: str,
    max_chars: int,
    selected: bool | None = None,
    source_text: str | None = None,
) -> str:
    empty = reader._format_record(record, text="", selected=selected)
    if len(empty) >= max_chars:
        return empty[:max_chars]
    excerpt = _focused_excerpt(
        str(record.get("text") or "") if source_text is None else source_text,
        focus,
        max_chars - len(empty),
    )
    return reader._format_record(record, text=excerpt, selected=selected)[:max_chars]


def _validate_support_coverage(
    query: Query,
    answer: dict[str, Any],
    support: list[dict[str, Any]],
) -> None:
    paths = {str(item["answer_path"]) for item in support}
    allowed_paths: set[str] = set()
    required_paths: set[str] = set()
    if "freeform" in query.answer_types:
        required_paths.add("answer.freeform.text")
    if "multiple_choice" in query.answer_types:
        required_paths.add("answer.multiple_choice")
    if "table" in query.answer_types:
        rows = answer["table"]["rows"]
        for row_index, row in enumerate(rows):
            row_path = f"answer.table.rows[{row_index}]"
            cell_paths = {f"{row_path}.{column}" for column in row}
            allowed_paths.add(row_path)
            allowed_paths.update(cell_paths)
            if row_path not in paths and not cell_paths.issubset(paths):
                missing = sorted(cell_paths - paths)
                raise ReadingResponseError(
                    f"table row {row_index} needs row-level support or every cell "
                    f"support path; missing={missing}"
                )
    allowed_paths.update(required_paths)
    missing_required = sorted(required_paths - paths)
    if missing_required:
        raise ReadingResponseError(
            f"answer support mapping is missing required paths: {missing_required}"
        )
    unknown_paths = sorted(paths - allowed_paths)
    if unknown_paths:
        raise ReadingResponseError(
            f"answer support mapping contains unknown paths: {unknown_paths}"
        )


def _completion_attached_paths(
    completion: CompletionResult,
    requested_paths: list[str],
) -> list[str]:
    """Return the ordered image paths the completion actually received.

    Providers normally attach the complete requested list.  The content-policy
    fallback explicitly records zero; if an adapter reports a smaller count we
    conservatively expose only that prefix to deterministic validation.
    """

    raw_count = completion.get("attached_image_count", 0)
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        return []
    return list(requested_paths[: max(0, min(raw_count, len(requested_paths)))])


def _validate_visual_grounding(
    query: Query,
    derivation: Any,
    *,
    context_records: dict[str, Record],
    attached_image_paths: list[str],
    required_visual_paper_ids: set[str] | None = None,
) -> None:
    """Reject image-derived claims when their source image was not attached."""

    if not isinstance(derivation, dict):
        return
    facts = derivation.get("facts")
    if not isinstance(facts, list):
        return
    attached = {str(Path(path).resolve()) for path in attached_image_paths}
    visual_facts = [
        fact
        for fact in facts
        if isinstance(fact, dict) and fact.get("value_kind") == "visual"
    ]
    visual_fact_paper_ids = {
        str(fact.get("paper_id") or "") for fact in visual_facts
    }
    missing_visual_papers = sorted(
        set(required_visual_paper_ids or ()) - visual_fact_paper_ids
    )
    if missing_visual_papers:
        raise ReadingResponseError(
            "stage-1 marked visual evidence as required, but stage 2 has no "
            f"visual fact for papers: {missing_visual_papers}"
        )
    for index, fact in enumerate(visual_facts):
        source_images = {
            str(Path(path).resolve())
            for raw_chunk_id in fact.get("chunk_ids") or []
            if (record := context_records.get(str(raw_chunk_id))) is not None
            if (path := readable_image_path(record))
        }
        if not source_images.intersection(attached):
            raise ReadingResponseError(
                f"derivation visual fact {index} has no actually attached source image"
            )
    if requires_visual_image(query.question) and not visual_facts:
        raise ReadingResponseError(
            "the query explicitly requires visual reading but derivation has no "
            "value_kind=visual fact"
        )


def _candidate_payload(candidate: CandidatePaper) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "rank": candidate.rank,
        "title": candidate.title,
        "venue": candidate.venue,
        "year": candidate.year,
    }


def _focused_excerpt(text: str, focus: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    tokens = {
        token for token in _selection_tokens(focus) if len(token) >= 3
    }
    lower = text.lower()
    positions: list[int] = []
    for token in tokens:
        start = 0
        for _ in range(12):
            position = lower.find(token, start)
            if position < 0:
                break
            positions.append(position)
            start = position + max(1, len(token))
    if not positions:
        return text[:max_chars]

    def window_start(center: int) -> int:
        start = max(0, center - max_chars // 3)
        return max(0, min(start, len(text) - max_chars))

    candidate_starts = sorted({window_start(position) for position in positions})

    def window_score(start: int) -> tuple[float, int]:
        window = lower[start : start + max_chars]
        score = sum(
            (4.0 if any(character.isdigit() for character in token) else 1.0)
            * min(3.0, 0.5 + len(token) / 5.0)
            for token in tokens
            if token in window
        )
        return score, -start

    start = max(candidate_starts, key=window_score)
    prefix = "… " if start else ""
    content_budget = max_chars - len(prefix)
    end = min(len(text), start + content_budget)
    start = max(0, end - content_budget)
    suffix = " …" if end < len(text) and content_budget >= 2 else ""
    if suffix:
        end = max(start, end - len(suffix))
    return (prefix + text[start:end] + suffix)[:max_chars]


def _records_sha256(records: list[Record]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_json_dumps(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _image_content_sha256(records: list[Record]) -> str | None:
    """Hash every readable image that can affect a paper judgment."""

    digest = hashlib.sha256()
    seen: set[str] = set()
    count = 0
    for record in records:
        image_path = readable_image_path(record)
        if not image_path:
            continue
        resolved = str(Path(image_path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        count += 1
        digest.update(resolved.encode("utf-8"))
        digest.update(b"\0")
        with Path(image_path).open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest() if count else None


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ReadingResponseError("constraint field must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output
