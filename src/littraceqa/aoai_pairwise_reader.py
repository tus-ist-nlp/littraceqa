"""Reading-only LitTraceQA agent over the fixed paper ranking from PR #7.

The retrieval result is already fixed for each query.  This module never
searches, reranks, decomposes a query for retrieval, or reads development-only
labels.  It performs exactly two semantic stages:

1. Pair one observable query with one candidate paper hydrated from the MinerU
   corpus and ask an LLM whether that paper is useful, citing exact chunk IDs.
2. Give only the accepted original chunks (plus their immediate neighbours)
   back to the LLM and construct the answer.

Stage 1 sends exactly one selected paper context for each query-paper pair.
Long papers and image-heavy papers are compacted deterministically before that
single request; they are never partitioned into additional semantic calls.  A
failed API call, invalid JSON, or invented chunk ID raises an error; it is never
converted into an ``irrelevant`` decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Required, TypedDict, cast

from littraceqa.answer_derivation import (
    DerivationValidationError,
    validate_answer_semantics,
)
from littraceqa.candidate_handoff import (
    CandidatePaper,
    require_production_query,
)
from littraceqa.chunk_store import ChunkStore, Record
from littraceqa.corpus_preflight import requires_visual_image
from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.mineru_record import (
    MAX_AOAI_IMAGES_PER_REQUEST,
    coarse_locator,
    readable_image_path,
    record_source_type,
    submission_evidence_eligible,
)
from littraceqa.pairwise_prompts import (
    ANSWER_PROMPT_VERSION,
    JUDGMENT_PROMPT_VERSION,
    answer_response_shape,
    example_manifest,
    render_answer_prompt,
    render_judgment_prompt,
)

PAPER_CONTEXT_SELECTOR_VERSION = "query-lexical-v1"

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
    """Accepted evidence rendered for the final answer model call."""

    text: str
    records_by_id: dict[str, Record]
    image_paths: list[str]


class CompletionResult(TypedDict, total=False):
    """Serializable model response plus optional provider metadata."""

    text: Required[str]
    request_id: str | None
    model: str | None
    deployment: str
    usage: Any
    latency_seconds: float
    requested_image_count: int
    attached_image_count: int
    image_fallback_reason: str
    provider_invocation_count: int


class ReadingResponseError(RuntimeError):
    """The model returned a response that cannot safely drive the next stage."""


class JudgmentEvidenceChunkError(ReadingResponseError):
    """Stage 1 cited a chunk outside its selected candidate-paper context."""


class AnswerEvidenceLocatorError(ReadingResponseError):
    """Stage 2 cited evidence that cannot be serialized with an official locator."""


class NoRelevantCandidatesError(RuntimeError):
    """No candidate was accepted, so an evidence-grounded answer is impossible."""


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
    """Judge fixed candidate papers one by one, then answer from accepted chunks."""

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
        max_evidence: int = 32,
        max_evidence_per_paper: int | None = None,
    ) -> None:
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
        resolved_per_paper = (
            min(4, max_evidence)
            if max_evidence_per_paper is None
            else max_evidence_per_paper
        )
        if (
            answer_neighbor_chunks < 0
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
        self.max_evidence = max_evidence
        self.max_evidence_per_paper = resolved_per_paper

    # ---- Stage 1: one query x one candidate paper -------------------------

    def judgment_cache_key(
        self, query: Query, candidate: CandidatePaper, records: list[Record]
    ) -> str:
        require_production_query(query)
        payload = {
            "prompt_version": JUDGMENT_PROMPT_VERSION,
            "few_shot_examples": example_manifest(query)["judgment"],
            "query": _production_query_payload(query),
            "candidate": _candidate_payload(candidate),
            "paper_content_sha256": _records_sha256(records),
            "paper_image_content_sha256": _image_content_sha256(records),
            "limits": {
                "paper_context_selector_version": PAPER_CONTEXT_SELECTOR_VERSION,
                "max_paper_context_chars": self.max_paper_context_chars,
                "max_judgment_prompt_chars": self.max_judgment_prompt_chars,
                "max_paper_images": self.max_paper_images,
            },
        }
        return _json_sha256(payload)

    def judge_candidate(
        self, query: Query, candidate: CandidatePaper
    ) -> dict[str, Any]:
        """Run an independent relevance/evidence judgment for one candidate paper."""

        require_production_query(query)
        records = self.chunk_store.load_paper(candidate.paper_id)
        if not records:
            raise FileNotFoundError(
                f"{query.query_id}: candidate paper is absent from MinerU corpus: "
                f"{candidate.paper_id}"
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
        completion = self._complete(prompt, context["image_paths"])
        calls: list[dict[str, Any]] = []
        try:
            parsed = self._parse_judgment(
                query=query,
                candidate=candidate,
                payload_text=completion["text"],
                allowed_records=context["records_by_id"],
                attached_image_paths=_completion_attached_paths(
                    completion, context["image_paths"]
                ),
            )
        except ReadingResponseError as exc:
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
            )
            repair_completion = self._complete(
                repair_prompt, context["image_paths"]
            )
            parsed = self._parse_judgment(
                query=query,
                candidate=candidate,
                payload_text=repair_completion["text"],
                allowed_records=context["records_by_id"],
                attached_image_paths=_completion_attached_paths(
                    repair_completion, context["image_paths"]
                ),
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
            calls.append(
                _judgment_call_record(
                    completion,
                    phase="single_context",
                    attempt="initial",
                    parse_error=None,
                    context=context,
                )
            )

        parsed["relevant"] = parsed["label"] in RELEVANT_LABELS
        parsed["identity_conflict"] = parsed["paper_role"] == "distractor"
        parsed["visual_conflict"] = False

        result = {
            "query_id": query.query_id,
            **_candidate_payload(candidate),
            "status": "complete",
            "prompt_version": JUDGMENT_PROMPT_VERSION,
            "few_shot_example_ids": example_manifest(query)["judgment"],
            "cache_key": self.judgment_cache_key(query, candidate, records),
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
            "attached_image_count": len(context["image_paths"]),
            "attached_image_chunk_ids": context["selected_image_chunk_ids"],
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
        image_indices = _ranked_image_record_indices(
            records,
            scores=scores,
            reasons=reasons,
            limit=self.max_paper_images,
            question=query.question,
        )

        if len(full_text) <= context_char_limit:
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
            chunk_ids[index]: records[index] for index in selected_indices
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
        return render_judgment_prompt(
            query=query,
            query_payload=_production_query_payload(query),
            candidate_payload=_candidate_payload(candidate),
            paper_text=context["text"],
            image_legend=image_legend,
        )

    def _parse_judgment(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        payload_text: str,
        allowed_records: dict[str, Record],
        attached_image_paths: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        payload = parse_json_object(payload_text)
        if not isinstance(payload, dict):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: "
                "judgment response is not a JSON object"
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
        paper_role = str(payload.get("paper_role") or "uncertain")
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
        blocking_mismatches = _string_list(payload.get("blocking_mismatches") or [])
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
        if visual["required"] and visual_status == "not_needed":
            raise ReadingResponseError(
                "visual.required=true is incompatible with status=not_needed"
            )
        if not visual["required"] and visual_status != "not_needed":
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
            and visual["required"]
            and visual_status in {"missing", "unreadable"}
        ):
            raise ReadingResponseError(
                f"{label} is incompatible with unavailable required visual evidence"
            )
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
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
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "source_type": record_source_type(record),
                    "locator": coarse_locator(record),
                    "purpose": str(item.get("purpose") or "answer"),
                    "quote_or_value": str(item.get("quote_or_value") or "").strip(),
                }
            )
        if label in RELEVANT_LABELS and not evidence:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}: {label} requires evidence"
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
        return {
            "paper_role": paper_role,
            "label": label,
            "answerable_from_this_paper": answerable,
            "satisfied_constraints": _string_list(payload.get("satisfied_constraints")),
            "missing_constraints": _string_list(payload.get("missing_constraints")),
            "blocking_mismatches": blocking_mismatches,
            "visual": {"required": visual["required"], "status": visual_status},
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
    ) -> str:
        return (
            original_prompt
            + "\n\nYour previous stage-1 JSON response was rejected by the deterministic "
            "validator. Correct the JSON once. Preserve the semantic judgment only when "
            "it is consistent with the validation error and allowed evidence. Every "
            "evidence chunk_id must be "
            "copied exactly from the allowed list below; never invent or approximate an ID. "
            "If no allowed chunk supports a relevant label, return the appropriate "
            "non-relevant label with empty evidence instead.\n"
            f"Validation error: {error}\n"
            f"Allowed selected-context chunk_ids: {_json_dumps(allowed_chunk_ids)}\n"
            "<rejected_response>\n"
            + rejected_response
            + "\n</rejected_response>\nReturn one corrected JSON object only."
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
                    "paper_id",
                    "rank",
                    "cache_key",
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
                "prompt_version": ANSWER_PROMPT_VERSION,
                "few_shot_examples": example_manifest(query)["answer"],
                "query": _production_query_payload(query),
                "judgments": stable_judgments,
                "limits": {
                    "answer_context_chars": self.answer_context_chars,
                    "answer_neighbor_chunks": self.answer_neighbor_chunks,
                    "max_answer_images": self.max_answer_images,
                    "max_evidence": self.max_evidence,
                    "max_evidence_per_paper": self.max_evidence_per_paper,
                },
            }
        )

    def answer_from_judgments(
        self,
        query: Query,
        candidates: Iterable[CandidatePaper],
        judgments: list[dict[str, Any]],
    ) -> tuple[Prediction, dict[str, Any]]:
        """Answer one query from stage-1 accepted original chunks only."""

        require_production_query(query)
        candidate_by_id = {item.paper_id: item for item in candidates}
        relevant = [
            item
            for item in sorted(
                judgments, key=lambda value: int(value.get("rank") or 0)
            )
            if item.get("relevant") is True
            and item.get("label") in RELEVANT_LABELS
        ]
        if not relevant:
            raise NoRelevantCandidatesError(
                f"{query.query_id}: stage 1 accepted no candidate paper"
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
        }
        completion = self._complete(prompt, context["image_paths"])
        attempts: list[dict[str, Any]] = []
        try:
            payload = self._parse_answer(
                query=query,
                payload_text=completion["text"],
                relevant_paper_ids=relevant_paper_ids,
                context_records=context["records_by_id"],
                attached_image_paths=_completion_attached_paths(
                    completion, context["image_paths"]
                ),
                required_visual_paper_ids=required_visual_paper_ids,
            )
        except ReadingResponseError as exc:
            attempts.append(
                {
                    "raw_response": completion["text"],
                    "parse_error": str(exc),
                    "call": {
                        key: value for key, value in completion.items() if key != "text"
                    },
                }
            )
            repair_prompt = self._answer_locator_repair_prompt(
                original_prompt=prompt,
                rejected_response=completion["text"],
                error=exc,
                context_records=context["records_by_id"],
            )
            completion = self._complete(repair_prompt, context["image_paths"])
            payload = self._parse_answer(
                query=query,
                payload_text=completion["text"],
                relevant_paper_ids=relevant_paper_ids,
                context_records=context["records_by_id"],
                attached_image_paths=_completion_attached_paths(
                    completion, context["image_paths"]
                ),
                required_visual_paper_ids=required_visual_paper_ids,
            )
        attempts.append(
            {
                "raw_response": completion["text"],
                "parse_error": None,
                "call": {
                    key: value for key, value in completion.items() if key != "text"
                },
            }
        )
        prediction = self._build_prediction(
            query=query,
            payload=payload,
            context_records=context["records_by_id"],
            candidate_ids=[item.paper_id for item in candidates],
            relevant=relevant,
            image_count=len(_completion_attached_paths(completion, context["image_paths"])),
        )
        answer_record = {
            "query_id": query.query_id,
            "status": "complete",
            "prompt_version": ANSWER_PROMPT_VERSION,
            "few_shot_example_ids": example_manifest(query)["answer"],
            "cache_key": self.answer_cache_key(query, judgments),
            "accepted_paper_ids": [str(item["paper_id"]) for item in relevant],
            "context_chunk_ids": list(context["records_by_id"]),
            "image_paths": list(context["image_paths"]),
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

        for judgment in relevant:
            paper_id = str(judgment["paper_id"])
            records = self.chunk_store.load_paper(paper_id)
            by_id = {str(item.get("chunk_id") or ""): item for item in records}
            positions = {str(item.get("chunk_id") or ""): i for i, item in enumerate(records)}
            for evidence in judgment.get("evidence") or []:
                chunk_id = str(evidence.get("chunk_id") or "")
                record = by_id.get(chunk_id)
                if record is None:
                    raise ValueError(
                        f"{query.query_id}/{paper_id}: cached judgment cites missing "
                        f"chunk {chunk_id!r}"
                    )
                quote = str(evidence.get("quote_or_value") or "")
                evidence_quotes.setdefault(chunk_id, []).append(quote)
                if chunk_id not in seen_primary:
                    seen_primary.add(chunk_id)
                    primary.append((record, quote))
                position = positions[chunk_id]
                for offset in range(-self.answer_neighbor_chunks, self.answer_neighbor_chunks + 1):
                    neighbour_position = position + offset
                    if offset == 0 or not 0 <= neighbour_position < len(records):
                        continue
                    neighbour = records[neighbour_position]
                    neighbour_id = str(neighbour.get("chunk_id") or "")
                    if (
                        neighbour_id
                        and neighbour_id not in seen_primary
                        and neighbour_id not in seen_neighbours
                    ):
                        seen_neighbours.add(neighbour_id)
                        neighbours.append((neighbour, ""))

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
        image_records_by_paper = {
            paper_id: [
                record
                for record, _ in paper_records
                if readable_image_path(record)
            ]
            for paper_id, paper_records in by_paper.items()
        }
        primary_image_records: list[Record] = []
        for label in ("direct_answer", "partial_answer", "supporting_only"):
            label_papers = sorted(
                (
                    paper_id
                    for paper_id in by_paper
                    if image_records_by_paper[paper_id]
                    and paper_image_priority.get(paper_id, ("", 0))[0] == label
                ),
                key=lambda paper_id: (
                    paper_image_priority[paper_id][1],
                    paper_id,
                ),
            )
            # Preserve strict label priority (all direct images precede all
            # partial images), but share each label's image budget fairly:
            # every paper gets image 1 in rank order before any gets image 2.
            image_position = 0
            while label_papers:
                added = False
                for paper_id in label_papers:
                    paper_records = image_records_by_paper[paper_id]
                    if image_position < len(paper_records):
                        primary_image_records.append(paper_records[image_position])
                        added = True
                if not added:
                    break
                image_position += 1
        if len(by_paper) > self.max_evidence:
            raise ReadingResponseError(
                f"{query.query_id}: {len(by_paper)} accepted papers exceed "
                f"max_evidence={self.max_evidence}; do not silently drop papers"
            )
        round_robin: list[tuple[Record, str]] = []
        position = 0
        while len(round_robin) < self.max_evidence:
            added = False
            for items in by_paper.values():
                if position < len(items):
                    round_robin.append(items[position])
                    added = True
                    if len(round_robin) >= self.max_evidence:
                        break
            if not added:
                break
            position += 1
        primary = round_robin
        if not primary:
            raise ReadingResponseError(
                f"{query.query_id}: accepted judgments contain no evidence chunks"
            )
        records_by_id: dict[str, Record] = {}
        parts: list[str] = []
        used_chars = 0
        # Give every cited chunk a fair share before adding neighbours.  This is
        # essential for multi-paper tables where a rank-1 paper must not consume
        # the context before rank-44 is represented.
        per_primary = max(
            3_000,
            min(40_000, self.answer_context_chars // max(1, len(primary))),
        )
        focus = query.question + " " + " ".join(
            quote for _, quote in primary if quote
        )
        for record, quote in primary:
            chunk_id = str(record["chunk_id"])
            excerpt = _focused_excerpt(
                str(record.get("text") or ""),
                focus + " " + quote,
                per_primary,
            )
            formatted = self._format_record(record, text=excerpt, selected=True)
            parts.append(formatted)
            used_chars += len(formatted)
            records_by_id[chunk_id] = record
        for record, _ in neighbours:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in records_by_id:
                continue
            remaining = self.answer_context_chars - used_chars
            if remaining < 1_000:
                break
            excerpt = _focused_excerpt(
                str(record.get("text") or ""), query.question, min(20_000, remaining)
            )
            formatted = self._format_record(record, text=excerpt, selected=False)
            if used_chars + len(formatted) > self.answer_context_chars:
                continue
            parts.append(formatted)
            used_chars += len(formatted)
            records_by_id[chunk_id] = record

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

        # Images explicitly cited by stage 1 are more valuable than images from
        # neighbouring context.  Prioritize direct answers, then partial answers,
        # then supporting papers; rank breaks ties while evidence order remains
        # stable within each paper.  This prevents a direct paper's later figure
        # from being starved by the first figure of many accepted papers.
        add_images(primary_image_records)
        add_images(record for record, _ in neighbours)
        return {
            "text": "\n\n".join(parts),
            "records_by_id": records_by_id,
            "image_paths": image_paths,
        }

    def _answer_prompt(
        self,
        query: Query,
        relevant: list[dict[str, Any]],
        context: AnswerContext,
    ) -> str:
        relevant_summary: list[dict[str, Any]] = []
        for item in relevant:
            summary_item = {
                "paper_id": item["paper_id"],
                "title": str(item.get("title") or ""),
                "rank": item["rank"],
                "label": item["label"],
                "paper_role": str(item.get("paper_role") or "uncertain"),
                "satisfied_constraints": item.get("satisfied_constraints") or [],
                "missing_constraints": item.get("missing_constraints") or [],
                "blocking_mismatches": item.get("blocking_mismatches") or [],
                "candidate_answer": item.get("candidate_answer") or {},
                "reason": str(item.get("reason") or ""),
            }
            relevant_summary.append(summary_item)
        image_legend = self._image_legend(
            context["records_by_id"], context["image_paths"]
        )
        return render_answer_prompt(
            query=query,
            query_payload=_production_query_payload(query),
            accepted_summary=relevant_summary,
            evidence_text=context["text"],
            image_legend=image_legend,
            answer_shape=answer_response_shape(query),
            max_evidence=self.max_evidence,
            max_evidence_per_paper=self.max_evidence_per_paper,
        )

    def _answer_locator_repair_prompt(
        self,
        *,
        original_prompt: str,
        rejected_response: str,
        error: ReadingResponseError,
        context_records: dict[str, Record],
    ) -> str:
        eligible_chunk_ids = [
            chunk_id
            for chunk_id, record in context_records.items()
            if submission_evidence_eligible(record)
        ]
        return (
            original_prompt
            + "\n\nYour previous JSON response was rejected by deterministic validation. "
            "Correct the JSON once and recompute the scientific answer when the error says "
            "the derivation, comparison polarity, option mapping, table types, support, or "
            "evidence is inconsistent. Evidence used only while comparing may be omitted "
            "from papers/support. Every submitted chunk must come from the eligible list. "
            "If a paper has no eligible direct evidence, omit that paper rather than "
            "inventing a locator.\n"
            f"Validation error: {error}\n"
            f"Eligible evidence chunk_ids: {_json_dumps(eligible_chunk_ids)}\n"
            "<rejected_response>\n"
            + rejected_response
            + "\n</rejected_response>\nReturn one corrected JSON object only."
        )

    def _parse_answer(
        self,
        *,
        query: Query,
        payload_text: str,
        relevant_paper_ids: set[str],
        context_records: dict[str, Record],
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
        fact_pairs = {
            (str(fact["paper_id"]), str(chunk_id))
            for fact in validated_derivation["facts"]
            for chunk_id in fact["chunk_ids"]
        }
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
        image_count: int,
    ) -> Prediction:
        evidence = []
        support_paper_ids: list[str] = []
        for paper in payload["papers"]:
            paper_id = str(paper["paper_id"])
            support_paper_ids.append(paper_id)
            for chunk_id in paper["evidence_chunk_ids"]:
                record = context_records[chunk_id]
                if not submission_evidence_eligible(record):
                    raise ReadingResponseError(
                        f"{query.query_id}: validated answer contains an invalid locator"
                    )
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    score=0.0,
                    text=str(record.get("text") or ""),
                    chunk_type=record_source_type(record),
                    metadata=dict(record.get("metadata") or {}),
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
        relevant_paper_ids = [
            str(item["paper_id"]) for item in payload["paper_relevance"]
        ]
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
                    "stage": "pairwise_candidate_judgment",
                    "judged": len(relevant),
                    "accepted_paper_ids": [item["paper_id"] for item in relevant],
                },
                {
                    "stage": "accepted_evidence_answer",
                    "paper_relevance": payload["paper_relevance"],
                    "support_paper_ids": support_paper_ids,
                    "image_count": image_count,
                    "derivation": payload.get("derivation"),
                    "semantic_multiple_choice": raw_answer.get("multiple_choice"),
                    "completeness": payload.get("completeness"),
                },
            ],
            candidate_papers=candidate_ids,
        )

    # ---- Shared helpers ---------------------------------------------------

    def _complete(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> CompletionResult:
        started = time.monotonic()
        provider_invocation_count = 0

        def invoke(
            effective_prompt: str, paths: list[str] | None
        ) -> CompletionResult:
            nonlocal provider_invocation_count
            provider_invocation_count += 1
            complete_with_metadata = getattr(
                self.llm, "complete_with_metadata", None
            )
            if callable(complete_with_metadata):
                response = complete_with_metadata(
                    effective_prompt, image_paths=paths or None
                )
                if not isinstance(response, dict) or not isinstance(
                    response.get("text"), str
                ):
                    raise ReadingResponseError(
                        "LLM metadata response must contain text"
                    )
                return cast(CompletionResult, dict(response))

            complete = getattr(self.llm, "complete", None)
            if callable(complete):
                raw = complete(effective_prompt, image_paths=paths or None)
            else:
                raw = self.llm(effective_prompt)
            if not isinstance(raw, str):
                raise ReadingResponseError("LLM response must be text")
            return {"text": raw}

        sent_prompt = prompt
        try:
            result = invoke(sent_prompt, image_paths)
        except Exception as exc:
            if not image_paths or not _is_image_content_policy_violation(exc):
                raise
            # Do not retry or transform rejected image content.  Preserve the
            # paper-level run by judging the same selected context from text alone, and
            # make the degraded modality explicit in the checkpoint metadata.
            fallback_prompt = (
                prompt
                + "\n\nMODALITY OVERRIDE FOR THIS RETRY: No image is attached. "
                "Ignore every earlier image mapping, do not claim visual inspection, "
                "and return unreadable/needs_image when the requested fact depends on "
                "an image."
            )
            sent_prompt = fallback_prompt
            result = invoke(sent_prompt, None)
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
        }
        if selected is not None:
            header["stage1_selected"] = selected
            header["submission_eligible"] = submission_evidence_eligible(record)
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
            lines.append(
                f"Image {index}: chunk_ids={','.join(chunk_ids)} file={Path(path).name}"
            )
        return "\n".join(lines)


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
) -> list[int]:
    if limit <= 0:
        return []
    explicit_indices = {
        index
        for index, item_reasons in enumerate(reasons)
        if "explicit_object_reference" in item_reasons
        and readable_image_path(records[index])
    }
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
    if explicit_indices:
        selection_limit = min(limit, max(len(explicit_indices), 2))
    elif explicit_visual or explicit_image_source:
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
        ),
        key=lambda index: (
            "explicit_object_reference" not in reasons[index],
            -contextual_scores[index],
            index,
        ),
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
            index not in explicit_indices
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
) -> str:
    empty = reader._format_record(record, text="")
    if len(empty) >= max_chars:
        return empty[:max_chars]
    excerpt = _focused_excerpt(
        str(record.get("text") or ""), focus, max_chars - len(empty)
    )
    return reader._format_record(record, text=excerpt)[:max_chars]


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
