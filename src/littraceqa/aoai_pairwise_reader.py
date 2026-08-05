"""Reading-only LitTraceQA agent over the fixed paper ranking from PR #7.

The retrieval result is already fixed for each query.  This module never
searches, reranks, decomposes a query for retrieval, or reads development-only
labels.  It performs exactly two semantic stages:

1. Pair one observable query with one candidate paper hydrated from the MinerU
   corpus and ask an LLM whether that paper is useful, citing exact chunk IDs.
2. Give only the accepted original chunks (plus their immediate neighbours)
   back to the LLM and construct the answer.

Long papers are split deterministically inside stage 1.  A failed API call,
invalid JSON, or invented chunk ID raises an error; it is never converted into
an ``irrelevant`` decision.
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

SUMMARY_MAX_BATCH_ANSWERS = 64
SUMMARY_MAX_BATCH_ANSWER_CHARS = 32_000
SUMMARY_MAX_SINGLE_BATCH_ANSWER_CHARS = 4_000

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
JUDGMENT_IMAGE_MODES = frozenset({"full", "text_then_relevant_images"})
IMAGE_REFINE_LABELS = frozenset(
    {*RELEVANT_LABELS, "mention_only", "unreadable"}
)
_LABEL_PRIORITY = {label: index for index, label in enumerate(reversed(JUDGMENT_LABELS))}
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+%/^-]*")


class PaperBatch(TypedDict):
    """One bounded paper slice sent to the judgment model."""

    text: str
    records_by_id: dict[str, Record]
    image_paths: list[str]


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


class ReadingResponseError(RuntimeError):
    """The model returned a response that cannot safely drive the next stage."""


class JudgmentEvidenceChunkError(ReadingResponseError):
    """Stage 1 cited a chunk outside its current candidate-paper batch."""


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
    batch_index: int,
    batch_count: int,
    batch: PaperBatch,
) -> dict[str, Any]:
    """Build the durable audit record for one judgment model call."""
    return {
        **{key: value for key, value in completion.items() if key != "text"},
        "usage": completion.get("usage"),
        "phase": phase,
        "attempt": attempt,
        "batch_index": batch_index,
        "batch_count": batch_count,
        "chunk_ids": list(batch["records_by_id"]),
        "image_paths": list(batch["image_paths"]),
        "raw_response": completion["text"],
        "parse_error": parse_error,
    }


def merge_batch_judgments(
    judgments: list[dict[str, Any]],
    *,
    prefer_later_on_label_tie: bool = False,
) -> dict[str, Any]:
    """Merge paper batches while giving explicit owner conflicts precedence.

    A positive finding in one slice must not override another slice that proves
    the candidate is the wrong owning paper.  Ordinary local misses do not veto
    later direct evidence, and a constraint found later resolves the same
    normalized item in ``missing_constraints``.
    """
    # Hybrid pair merging supplies text_screen first and visual_refine second.
    # On an equal label, the image-grounded candidate answer wins while evidence
    # remains unioned in its original text-then-visual order.
    best_candidates = reversed(judgments) if prefer_later_on_label_tie else judgments
    ranked_best = max(best_candidates, key=lambda item: _LABEL_PRIORITY[item["label"]])
    # ``paper_role`` is the structured owner-identity decision.  Do not make
    # this safety veto depend on the model choosing one of a few English
    # phrasings in ``blocking_mismatches``.
    identity_conflicts = [
        mismatch
        for item in judgments
        if str(item.get("paper_role") or "") == "distractor"
        for mismatch in item.get("blocking_mismatches") or []
    ]
    if identity_conflicts:
        best = next(
            item
            for item in judgments
            if str(item.get("paper_role") or "") == "distractor"
            and bool(item.get("blocking_mismatches"))
        )
        final_label = "irrelevant"
    else:
        best = ranked_best
        final_label = str(best["label"])
    evidence_judgments = (
        [item for item in judgments if item["label"] in RELEVANT_LABELS]
        if final_label in RELEVANT_LABELS
        else ([best] if identity_conflicts else judgments)
    )
    evidence: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for item in evidence_judgments:
        for evidence_item in item["evidence"]:
            chunk_id = evidence_item["chunk_id"]
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                evidence.append(evidence_item)
    satisfied_constraints = _ordered_unique(
        value
        for item in evidence_judgments
        for value in item.get("satisfied_constraints") or []
    )
    satisfied_keys = {_normalized_constraint(value) for value in satisfied_constraints}
    missing_constraints = _ordered_unique(
        value
        for item in evidence_judgments
        for value in item.get("missing_constraints") or []
        if _normalized_constraint(value) not in satisfied_keys
    )
    blocking_mismatches = _ordered_unique(
        value
        for item in judgments
        for value in item.get("blocking_mismatches") or []
    )
    return {
        "paper_role": str(best.get("paper_role") or "uncertain"),
        "label": final_label,
        "relevant": final_label in RELEVANT_LABELS,
        "answerable_from_this_paper": final_label in RELEVANT_LABELS
        and any(bool(item["answerable_from_this_paper"]) for item in judgments),
        "satisfied_constraints": satisfied_constraints,
        "missing_constraints": missing_constraints,
        "blocking_mismatches": blocking_mismatches,
        "identity_conflict": bool(identity_conflicts),
        "visual": dict(best.get("visual") or {}),
        "evidence": evidence,
        "evidence_chunk_ids": [item["chunk_id"] for item in evidence],
        "candidate_answer": dict(best["candidate_answer"]),
        "candidate_answers_by_batch": [
            item["candidate_answer"]
            for item in evidence_judgments
            if item["candidate_answer"]
        ],
        "confidence": max(float(item["confidence"]) for item in evidence_judgments),
        "reason": " | ".join(
            str(item["reason"]) for item in judgments if item["reason"]
        ),
    }


class PairwiseAOAIReader:
    """Judge fixed candidate papers one by one, then answer from accepted chunks."""

    def __init__(
        self,
        chunk_store: ChunkStore,
        llm: LLMClient,
        max_batch_chars: int = 160_000,
        batch_overlap_chars: int = 1_000,
        max_images_per_batch: int = 6,
        answer_context_chars: int = 220_000,
        answer_neighbor_chunks: int = 1,
        max_answer_images: int = 12,
        max_evidence: int = 32,
        max_evidence_per_paper: int | None = None,
        judgment_image_mode: str = "full",
        image_refine_labels: Iterable[str] | None = None,
    ) -> None:
        if max_batch_chars < 8_000:
            raise ValueError("max_batch_chars must be at least 8000")
        if batch_overlap_chars < 0 or batch_overlap_chars >= max_batch_chars // 2:
            raise ValueError("batch_overlap_chars is out of range")
        if max_images_per_batch < 0 or max_answer_images < 0:
            raise ValueError("image limits must be non-negative")
        if answer_context_chars < 8_000:
            raise ValueError("answer_context_chars must be at least 8000")
        resolved_per_paper = min(4, max_evidence) if max_evidence_per_paper is None else max_evidence_per_paper
        if (
            answer_neighbor_chunks < 0
            or max_evidence < 1
            or resolved_per_paper < 1
            or resolved_per_paper > max_evidence
        ):
            raise ValueError("invalid answer context limits")
        if judgment_image_mode not in JUDGMENT_IMAGE_MODES:
            raise ValueError(
                "judgment_image_mode must be full or text_then_relevant_images"
            )
        requested_refine_labels = (
            set(IMAGE_REFINE_LABELS)
            if image_refine_labels is None
            else set(image_refine_labels)
        )
        if not requested_refine_labels or not requested_refine_labels.issubset(
            IMAGE_REFINE_LABELS
        ):
            raise ValueError(
                "image_refine_labels must be a non-empty subset of direct_answer, "
                "partial_answer, supporting_only, mention_only, and unreadable"
            )
        self.chunk_store = chunk_store
        self.llm = llm
        self.max_batch_chars = max_batch_chars
        self.batch_overlap_chars = batch_overlap_chars
        self.max_images_per_batch = max_images_per_batch
        self.answer_context_chars = answer_context_chars
        self.answer_neighbor_chunks = answer_neighbor_chunks
        self.max_answer_images = max_answer_images
        self.max_evidence = max_evidence
        self.max_evidence_per_paper = resolved_per_paper
        self.judgment_image_mode = judgment_image_mode
        self.image_refine_labels = tuple(
            label for label in JUDGMENT_LABELS if label in requested_refine_labels
        )

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
                "max_batch_chars": self.max_batch_chars,
                "batch_overlap_chars": self.batch_overlap_chars,
                "max_images_per_batch": self.max_images_per_batch,
                "judgment_image_mode": self.judgment_image_mode,
                "image_refine_labels": list(self.image_refine_labels),
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
        text_judgment: dict[str, Any] | None = None
        visual_judgment: dict[str, Any] | None = None
        visual_refinement_status = "not_applicable_full"
        visual_conflict = False

        if self.judgment_image_mode == "full":
            batches = self._paper_batches(records)
            batch_judgments, calls = self._judge_batches(
                query=query,
                candidate=candidate,
                batches=batches,
                phase="full",
            )
            merged = merge_batch_judgments(batch_judgments)
        else:
            text_batches = self._paper_batches(records, include_images=False)
            text_batch_judgments, text_calls = self._judge_batches(
                query=query,
                candidate=candidate,
                batches=text_batches,
                phase="text_screen",
            )
            text_judgment = merge_batch_judgments(text_batch_judgments)
            merged = text_judgment
            batch_judgments = list(text_batch_judgments)
            calls = list(text_calls)

            should_refine = any(
                item["label"] in self.image_refine_labels
                for item in text_batch_judgments
            )
            if not should_refine:
                visual_refinement_status = "skipped_label"
            else:
                visual_batches = [
                    batch
                    for batch in self._paper_batches(records, include_images=True)
                    if batch["image_paths"]
                ]
                if not visual_batches:
                    visual_refinement_status = "skipped_no_images"
                else:
                    visual_batch_judgments, visual_calls = self._judge_batches(
                        query=query,
                        candidate=candidate,
                        batches=visual_batches,
                        phase="visual_refine",
                    )
                    visual_judgment = merge_batch_judgments(visual_batch_judgments)
                    visual_refinement_status = "complete"
                    visual_conflict = not bool(visual_judgment["relevant"])
                    merged = merge_batch_judgments(
                        [text_judgment, visual_judgment],
                        prefer_later_on_label_tie=True,
                    )
                    batch_judgments.extend(visual_batch_judgments)
                    calls.extend(visual_calls)

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
            "batch_count": len(batch_judgments),
            "judgment_image_mode": self.judgment_image_mode,
            "visual_refinement_status": visual_refinement_status,
            "visual_conflict": visual_conflict,
            **merged,
            "batch_judgments": batch_judgments,
            "calls": calls,
        }
        if text_judgment is not None:
            result["text_judgment"] = text_judgment
        if visual_judgment is not None:
            result["visual_judgment"] = visual_judgment
        return result

    def _judge_batches(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        batches: list[PaperBatch],
        phase: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        batch_judgments: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            prompt = self._judgment_prompt(
                query=query,
                candidate=candidate,
                batch=batch,
                batch_index=batch_index,
                batch_count=len(batches),
            )
            completion = self._complete(prompt, batch["image_paths"])

            try:
                parsed = self._parse_judgment(
                    query=query,
                    candidate=candidate,
                    payload_text=completion["text"],
                    allowed_records=batch["records_by_id"],
                    batch_index=batch_index,
                    attached_image_count=int(
                        completion.get("attached_image_count") or 0
                    ),
                )
            except ReadingResponseError as exc:
                calls.append(
                    _judgment_call_record(
                        completion,
                        phase=phase,
                        attempt="initial",
                        parse_error=str(exc),
                        batch_index=batch_index,
                        batch_count=len(batches),
                        batch=batch,
                    )
                )
                repair_prompt = self._judgment_evidence_repair_prompt(
                    original_prompt=prompt,
                    rejected_response=completion["text"],
                    error=exc,
                    allowed_chunk_ids=list(batch["records_by_id"]),
                )
                repair_completion = self._complete(
                    repair_prompt, batch["image_paths"]
                )
                # Parse with the exact same validator. A second invalid ID or
                # any other malformed repair remains a hard failure.
                parsed = self._parse_judgment(
                    query=query,
                    candidate=candidate,
                    payload_text=repair_completion["text"],
                    allowed_records=batch["records_by_id"],
                    batch_index=batch_index,
                    attached_image_count=int(
                        repair_completion.get("attached_image_count") or 0
                    ),
                )
                calls.append(
                    _judgment_call_record(
                        repair_completion,
                        phase=phase,
                        attempt="evidence_repair",
                        parse_error=None,
                        batch_index=batch_index,
                        batch_count=len(batches),
                        batch=batch,
                    )
                )
            else:
                calls.append(
                    _judgment_call_record(
                        completion,
                        phase=phase,
                        attempt="initial",
                        parse_error=None,
                        batch_index=batch_index,
                        batch_count=len(batches),
                        batch=batch,
                    )
                )
            parsed["phase"] = phase
            batch_judgments.append(parsed)
        return batch_judgments, calls

    def _paper_batches(
        self, records: list[Record], *, include_images: bool = True
    ) -> list[PaperBatch]:
        segments: list[tuple[Record, str]] = []
        # Reserve space for instructions and the query.  A single oversized
        # MinerU chunk is sliced, while retaining its original chunk_id so the
        # evidence locator remains mechanically verifiable.
        segment_chars = max(4_000, self.max_batch_chars - 12_000)
        for record in records:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id:
                raise ValueError(
                    f"MinerU record has no chunk_id: paper_id={record.get('paper_id')!r}"
                )
            text = str(record.get("text") or "")
            pieces = _split_text(text, segment_chars, self.batch_overlap_chars)
            for piece_index, piece in enumerate(pieces, start=1):
                segments.append(
                    (
                        record,
                        self._format_record(
                            record,
                            text=piece,
                            segment=(piece_index, len(pieces)),
                        ),
                    )
                )

        batches: list[PaperBatch] = []
        parts: list[str] = []
        records_by_id: dict[str, Record] = {}
        image_paths: list[str] = []
        used_chars = 0

        def flush() -> None:
            nonlocal parts, records_by_id, image_paths, used_chars
            if not parts:
                return
            batches.append(
                {
                    "text": "\n\n".join(parts),
                    "records_by_id": records_by_id,
                    "image_paths": image_paths,
                }
            )
            parts = []
            records_by_id = {}
            image_paths = []
            used_chars = 0

        for record, formatted in segments:
            image_path = readable_image_path(record) if include_images else ""
            adds_image = bool(image_path and image_path not in image_paths)
            exceeds_chars = bool(parts and used_chars + len(formatted) > self.max_batch_chars)
            exceeds_images = bool(
                parts
                and adds_image
                and len(image_paths) >= self.max_images_per_batch
            )
            if exceeds_chars or exceeds_images:
                flush()
            parts.append(formatted)
            used_chars += len(formatted)
            chunk_id = str(record["chunk_id"])
            records_by_id[chunk_id] = record
            if (
                image_path
                and image_path not in image_paths
                and len(image_paths) < self.max_images_per_batch
            ):
                image_paths.append(image_path)
        flush()
        if not batches:
            raise ValueError("paper has no readable MinerU chunks")
        return batches

    def _judgment_prompt(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        batch: PaperBatch,
        batch_index: int,
        batch_count: int,
    ) -> str:
        image_legend = self._image_legend(batch["records_by_id"], batch["image_paths"])
        return render_judgment_prompt(
            query=query,
            query_payload=_production_query_payload(query),
            candidate_payload=_candidate_payload(candidate),
            paper_text=batch["text"],
            batch_index=batch_index,
            batch_count=batch_count,
            image_legend=image_legend,
        )

    def _parse_judgment(
        self,
        *,
        query: Query,
        candidate: CandidatePaper,
        payload_text: str,
        allowed_records: dict[str, Record],
        batch_index: int,
        attached_image_count: int = 0,
    ) -> dict[str, Any]:
        payload = parse_json_object(payload_text)
        if not isinstance(payload, dict):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}/batch {batch_index}: "
                "judgment response is not a JSON object"
            )
        label = str(payload.get("label") or "")
        if label not in JUDGMENT_LABELS:
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}/batch {batch_index}: "
                f"invalid judgment label {label!r}"
            )
        answerable = payload.get("answerable_from_this_paper")
        if not isinstance(answerable, bool):
            raise ReadingResponseError(
                f"{query.query_id}/{candidate.paper_id}/batch {batch_index}: "
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
        if visual_status == "inspected" and attached_image_count < 1:
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
            f"Allowed batch chunk_ids: {_json_dumps(allowed_chunk_ids)}\n"
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
                **{
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
                        "visual_conflict",
                        "paper_role",
                        "identity_conflict",
                        "visual",
                    )
                },
                "candidate_answers_by_batch": _bounded_candidate_answers_by_batch(
                    judgment.get("candidate_answers_by_batch")
                ),
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
            batch_answers = _bounded_candidate_answers_by_batch(
                item.get("candidate_answers_by_batch")
            )
            distinct_batch_answers = {_json_dumps(answer) for answer in batch_answers}
            batch_answer_conflict = len(distinct_batch_answers) > 1
            summary_item = {
                "paper_id": item["paper_id"],
                "title": str(item.get("title") or ""),
                "rank": item["rank"],
                "label": item["label"],
                "paper_role": str(item.get("paper_role") or "uncertain"),
                "satisfied_constraints": item.get("satisfied_constraints") or [],
                "missing_constraints": item.get("missing_constraints") or [],
                "blocking_mismatches": item.get("blocking_mismatches") or [],
                # A lossy merged answer can anchor stage 2 to the wrong visual
                # batch.  When batches disagree, expose the disagreement but
                # deliberately withhold that representative shortcut.
                "candidate_answer": (
                    {} if batch_answer_conflict else item.get("candidate_answer") or {}
                ),
                "candidate_answers_by_batch": batch_answers,
                "reason": str(item.get("reason") or ""),
                "visual_conflict": bool(item.get("visual_conflict", False)),
            }
            if batch_answer_conflict:
                summary_item["batch_answer_conflict"] = True
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
        _validate_visual_grounding(
            query,
            derivation,
            context_records=context_records,
            attached_image_paths=attached_image_paths or [],
        )
        try:
            validated_derivation = validate_answer_semantics(
                query,
                derivation=derivation,
                answer=answer,
            )
        except DerivationValidationError as exc:
            raise ReadingResponseError(f"{query.query_id}: {exc}") from exc
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

        def invoke(
            effective_prompt: str, paths: list[str] | None
        ) -> CompletionResult:
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
            # paper-level run by judging the same batch from text alone, and
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
        segment: tuple[int, int] | None = None,
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
        if segment and segment[1] > 1:
            header["segment"] = f"{segment[0]}/{segment[1]}"
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


def _normalized_constraint(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


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


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def _focused_excerpt(text: str, focus: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    tokens = {
        token.lower()
        for token in _WORD_RE.findall(focus)
        if len(token) >= 3
    }
    lower = text.lower()
    positions: list[int] = []
    for token in tokens:
        position = lower.find(token)
        if position >= 0:
            positions.append(position)
    if not positions:
        return text[:max_chars] + " …"
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    return ("… " if start else "") + text[start:end] + (" …" if end < len(text) else "")


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


def _bounded_candidate_answers_by_batch(value: Any) -> list[dict[str, Any]]:
    """Keep stage-1 answer hints useful without allowing prompt-size blow-ups."""

    if not isinstance(value, list):
        return []

    output: list[dict[str, Any]] = []
    serialized_chars = 2  # JSON list brackets.
    for candidate in value:
        if len(output) >= SUMMARY_MAX_BATCH_ANSWERS:
            break
        if not isinstance(candidate, dict):
            continue
        try:
            rendered = _json_dumps(candidate)
        except (TypeError, ValueError):
            continue

        bounded_candidate = candidate
        if len(rendered) > SUMMARY_MAX_SINGLE_BATCH_ANSWER_CHARS:
            # Binary-search the longest escaped JSON prefix that still fits the
            # per-answer cap. The marker makes it impossible to mistake the
            # diagnostic prefix for a complete stage-1 answer.
            low = 0
            high = len(rendered)
            bounded_candidate = {"_truncated": True, "_json_prefix": ""}
            bounded_rendered = _json_dumps(bounded_candidate)
            while low <= high:
                middle = (low + high) // 2
                proposal = {
                    "_truncated": True,
                    "_json_prefix": rendered[:middle],
                }
                proposal_rendered = _json_dumps(proposal)
                if len(proposal_rendered) <= SUMMARY_MAX_SINGLE_BATCH_ANSWER_CHARS:
                    bounded_candidate = proposal
                    bounded_rendered = proposal_rendered
                    low = middle + 1
                else:
                    high = middle - 1
            rendered = bounded_rendered

        separator_chars = 1 if output else 0
        if (
            serialized_chars + separator_chars + len(rendered)
            > SUMMARY_MAX_BATCH_ANSWER_CHARS
        ):
            break
        output.append(bounded_candidate)
        serialized_chars += separator_chars + len(rendered)
    return output


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
