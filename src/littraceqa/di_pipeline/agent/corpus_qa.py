"""Read student-produced MinerU chunks and answer from a ranked paper handoff.

This is deliberately separate from the legacy Azure RAG implementation and
from ``ReadingAgent``.  Retrieval chooses papers; this agent hydrates those
papers from ``ChunkStore``, selects a bounded amount of context, identifies
evidence and generates every requested answer representation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from littraceqa.candidate_handoff import CandidatePaper, require_production_query
from littraceqa.chunk_store import ChunkStore, Record
from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.submission import deterministic_mc_letter


OFFICIAL_SOURCE_TYPES = frozenset(
    {"text_span", "table", "figure", "citation_context", "equation_algorithm"}
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-/]*")
_LOCATOR_RE = re.compile(
    r"\b(table|figure|fig\.?|equation|eq\.?|algorithm|reference|ref\.?)\s*([A-Za-z0-9.\-]+)",
    re.IGNORECASE,
)
_ENUMERATION_RE = re.compile(
    r"\b(each|all|among|across|compare|comparison|respectively|papers?|methods?|models?|approaches?)\b",
    re.IGNORECASE,
)
_ORDINAL_REFERENCE_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\s+(?:reference|citation)\b", re.IGNORECASE
)
_LAST_REFERENCE_RE = re.compile(
    r"\b(?:last|final)\s+(?:reference|citation)\b", re.IGNORECASE
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "paper",
        "papers",
        "that",
        "the",
        "their",
        "to",
        "under",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)


class CorpusQAAgent:
    """Two-stage paper reader over the MinerU corpus.

    Stage 1 presents a small, lexically selected bundle for each candidate
    paper and asks the LLM to choose the complete paper set.  Stage 2 expands
    only those papers, asks for exact evidence chunks and answer values, and
    validates all identifiers against the corpus before building a Prediction.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        llm: LLMClient,
        max_candidate_papers: int = 50,
        detailed_candidate_papers: int = 20,
        brief_chunks_per_paper: int = 3,
        answer_chunks_per_paper: int = 8,
        max_selected_papers: int = 10,
        snippet_chars: int = 1600,
        tail_snippet_chars: int = 600,
        max_context_chars: int = 120_000,
        max_evidence: int = 24,
        max_images: int = 6,
    ) -> None:
        self.chunk_store = chunk_store
        self.llm = llm
        self.max_candidate_papers = max_candidate_papers
        self.detailed_candidate_papers = detailed_candidate_papers
        self.brief_chunks_per_paper = brief_chunks_per_paper
        self.answer_chunks_per_paper = answer_chunks_per_paper
        self.max_selected_papers = max_selected_papers
        self.snippet_chars = snippet_chars
        self.tail_snippet_chars = tail_snippet_chars
        self.max_context_chars = max_context_chars
        self.max_evidence = max_evidence
        self.max_images = max_images

    def run(
        self, query: Query, candidate_papers: tuple[CandidatePaper, ...] | list[CandidatePaper]
    ) -> Prediction:
        require_production_query(query)
        candidates = list(candidate_papers)[: self.max_candidate_papers]
        if not candidates:
            raise ValueError(f"{query.query_id}: no candidate papers")

        records_by_paper: dict[str, list[Record]] = {}
        missing_papers: list[str] = []
        for candidate in candidates:
            records = self.chunk_store.load_paper(candidate.paper_id)
            if records:
                records_by_paper[candidate.paper_id] = records
            else:
                missing_papers.append(candidate.paper_id)
        available = [paper for paper in candidates if paper.paper_id in records_by_paper]
        if not available:
            raise ValueError(f"{query.query_id}: none of the candidates exist in the corpus")

        plan = self._plan_query(query)
        ranked_by_paper = {
            paper.paper_id: self._rank_records(
                query, records_by_paper[paper.paper_id], plan
            )
            for paper in available
        }
        selection = self._select_papers(query, plan, available, ranked_by_paper)
        selected_ids = self._validated_selection(selection, available)
        if not selected_ids:
            selected_ids = self._fallback_paper_ids(query, plan, available)

        answer_payload, context_records, image_paths = self._read_selected_papers(
            query, plan, selected_ids, ranked_by_paper
        )

        evidence = self._build_evidence(
            answer_payload,
            selected_ids,
            context_records,
        )
        if not evidence:
            raise RuntimeError(
                f"{query.query_id}: answer stage returned no valid evidence chunk"
            )
        answer = self._build_answer(query, answer_payload)
        trace = [
            {
                "stage": "query_plan",
                "plan": plan,
            },
            {
                "stage": "candidate_handoff",
                "candidate_count": len(candidates),
                "available_count": len(available),
                "missing_paper_ids": missing_papers,
            },
            {
                "stage": "paper_selection",
                "selected_paper_ids": selected_ids,
                "unresolved_targets": selection.get("unresolved_targets", []),
            },
            {
                "stage": "corpus_reading",
                "context_chunk_count": len(context_records),
                "image_count": len(image_paths),
                "semantic_multiple_choice": answer_payload.get(
                    "semantic_multiple_choice"
                ),
            },
        ]
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in selected_ids],
            evidence=evidence,
            answer=answer,
            trace=trace,
            candidate_papers=[paper.paper_id for paper in candidates],
        )

    # ---- Stage 0: observable query plan ----------------------------------

    def _plan_query(self, query: Query) -> dict[str, Any]:
        prompt = (
            "Turn the scientific QA request into an explicit retrieval/reading plan. "
            "Use only the question, answer_types and table_schema below. Split enumerations "
            "into atomic targets (method, model, paper, dataset or requested table row). "
            "Extract venue/year constraints only when explicitly written. Infer one or more "
            "evidence modalities from table, figure, equation, citation and text cues.\n\n"
            f"Question: {query.question}\n"
            f"Answer types: {query.answer_types}\n"
            f"Table schema: {query.table_schema or []}\n\n"
            "Return JSON only:\n"
            '{"targets": [{"name": "...", "search_terms": ["..."]}], '
            '"venues": ["..."], "years": [2025], '
            '"modalities": ["text_span"], "requires_multiple_papers": false}'
        )
        raw = self._ask_json(prompt, strict=True)
        targets: list[dict[str, Any]] = []
        for item in raw.get("targets") or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            terms = item.get("search_terms") or []
            if not isinstance(terms, list):
                terms = []
            targets.append(
                {
                    "name": str(item["name"]),
                    "search_terms": [str(term) for term in terms[:8] if term],
                }
            )
            if len(targets) >= 12:
                break
        if not targets:
            targets = [{"name": query.question, "search_terms": []}]

        modalities = [
            str(item)
            for item in raw.get("modalities") or []
            if str(item) in OFFICIAL_SOURCE_TYPES
        ]
        if not modalities:
            modalities = sorted(_expected_modalities(query))
        years: list[int] = []
        for item in raw.get("years") or []:
            try:
                year = int(item)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100 and year not in years:
                years.append(year)
        venues = [str(item) for item in raw.get("venues") or [] if item]
        return {
            "targets": targets,
            "venues": venues[:10],
            "years": years[:10],
            "modalities": list(dict.fromkeys(modalities)),
            "requires_multiple_papers": bool(raw.get("requires_multiple_papers")),
        }

    # ---- Stage 1: paper selection -----------------------------------------

    def _select_papers(
        self,
        query: Query,
        plan: dict[str, Any],
        candidates: list[CandidatePaper],
        ranked_by_paper: dict[str, list[tuple[float, Record]]],
    ) -> dict[str, Any]:
        listings: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            header = (
                f"[rank={candidate.rank} paper_id={candidate.paper_id}] "
                f"{candidate.title} ({candidate.venue} {candidate.year or ''})"
            )
            # Keep all top-50 paper titles visible, while hydrating only the
            # first group in this pass. Tail papers selected by title are fully
            # loaded during Stage 2, so @50 coverage does not require a 250k+
            # character prompt.
            focus_text = _plan_focus_text(query, plan)
            if candidate_index < self.detailed_candidate_papers:
                chunks = ranked_by_paper[candidate.paper_id][
                    : self.brief_chunks_per_paper
                ]
                body = "\n".join(
                    self._format_record(record, focus_text=focus_text)
                    for _, record in chunks
                )
            else:
                # A short target-centered excerpt keeps rank 21-50 usable
                # without expanding the selection prompt beyond the model
                # context window.
                tail = ranked_by_paper[candidate.paper_id][:1]
                body = "\n".join(
                    self._format_record(
                        record,
                        max_chars=self.tail_snippet_chars,
                        focus_text=focus_text,
                    )
                    for _, record in tail
                )
            listings.append(header + ("\n" + body if body else ""))

        prompt = (
            "Select the complete set of candidate papers needed by the question.\n"
            "Use only the observable question and requested output schema. If the question "
            "enumerates or compares methods, keep one relevant owner paper for every target; "
            "do not stop after finding a single paper. Prefer a precise evidence-owning set, "
            "but do not assume that a named-paper question is necessarily scored as one paper. "
            "Never invent a paper_id.\n\n"
            f"Question: {query.question}\n"
            f"Answer types: {query.answer_types}\n"
            f"Table schema: {query.table_schema or []}\n\n"
            f"Observable query plan: {plan}\n\n"
            "Candidate paper bundles:\n"
            + "\n\n".join(listings)
            + "\n\nReturn JSON only:\n"
            '{"paper_ids": ["..."], "unresolved_targets": ["..."], '
            '"reason": "short explanation"}'
        )
        return self._ask_json(prompt, strict=True)

    def _validated_selection(
        self, payload: dict[str, Any], candidates: list[CandidatePaper]
    ) -> list[str]:
        allowed = {paper.paper_id for paper in candidates}
        output: list[str] = []
        raw_ids = payload.get("paper_ids")
        if not isinstance(raw_ids, list):
            return output
        for raw_id in raw_ids:
            paper_id = str(raw_id)
            if paper_id in allowed and paper_id not in output:
                output.append(paper_id)
            if len(output) >= self.max_selected_papers:
                break
        return output

    def _fallback_paper_ids(
        self,
        query: Query,
        plan: dict[str, Any],
        candidates: list[CandidatePaper],
    ) -> list[str]:
        count = 1
        if (
            plan.get("requires_multiple_papers")
            or len(plan.get("targets") or []) > 1
            or "table" in query.answer_types
            or _ENUMERATION_RE.search(query.question)
        ):
            count = min(4, self.max_selected_papers)
        return [paper.paper_id for paper in candidates[:count]]

    # ---- Stage 2: evidence and answer -------------------------------------

    def _read_selected_papers(
        self,
        query: Query,
        plan: dict[str, Any],
        selected_ids: list[str],
        ranked_by_paper: dict[str, list[tuple[float, Record]]],
    ) -> tuple[dict[str, Any], dict[str, Record], list[str]]:
        per_paper = {
            paper_id: self._target_covered_records(
                query, plan, ranked_by_paper[paper_id]
            )
            for paper_id in selected_ids
        }
        # Round-robin prevents early papers from consuming the global character
        # budget before later selected papers receive any context.
        selected_records: list[Record] = []
        for position in range(self.answer_chunks_per_paper):
            for paper_id in selected_ids:
                records = per_paper[paper_id]
                if position < len(records):
                    selected_records.append(records[position])

        context_parts: list[str] = []
        context_records: dict[str, Record] = {}
        used_chars = 0
        focus_text = _plan_focus_text(query, plan)
        for record in selected_records:
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id or chunk_id in context_records:
                continue
            formatted = self._format_record(record, focus_text=focus_text)
            if context_parts and used_chars + len(formatted) > self.max_context_chars:
                break
            context_records[chunk_id] = record
            context_parts.append(formatted)
            used_chars += len(formatted)

        image_items = self._image_items(
            query, plan, list(context_records.values())
        )
        image_paths = [item["path"] for item in image_items]
        image_legend = "\n".join(
            f"Image {index} (attached in this order): paper_id={item['paper_id']} "
            f"chunk_id={item['chunk_id']} type={item['chunk_type']}"
            for index, item in enumerate(image_items, start=1)
        )
        requested_answer = self._answer_json_spec(query)
        prompt = (
            "Read the corpus chunks and answer the question using only this evidence. "
            "Identify the exact chunk_ids that support the answer. Preserve short numeric "
            "answers exactly. For tables, output every requested row and use column names "
            "exactly as supplied. Do not invent chunk_ids or paper_ids.\n\n"
            f"Question: {query.question}\n"
            f"Answer types: {query.answer_types}\n"
            f"Table schema: {query.table_schema or []}\n\n"
            f"Observable query plan: {plan}\n\n"
            + (
                "Attached image mapping:\n" + image_legend + "\n\n"
                if image_legend
                else ""
            )
            +
            "Corpus chunks:\n"
            + "\n\n".join(context_parts)
            + "\n\nReturn JSON only with this shape:\n"
            "{\n"
            '  "papers": [{"paper_id": "...", "evidence_chunk_ids": ["..."]}],\n'
            f'  "answer": {requested_answer},\n'
            '  "semantic_multiple_choice": {"text": "meaning-level answer, not a letter"}\n'
            "}\n"
            "The option-to-letter mapping is unavailable. Do not guess an A/B/C/D letter; "
            "give the meaning-level answer in semantic_multiple_choice instead."
        )
        payload = self._ask_json(prompt, image_paths=image_paths, strict=True)
        self._validate_answer_payload(query, plan, payload)
        return payload, context_records, image_paths

    def _build_evidence(
        self,
        payload: dict[str, Any],
        selected_ids: list[str],
        context_records: dict[str, Record],
    ) -> list:
        requested_ids: list[str] = []
        raw_papers = payload.get("papers")
        if isinstance(raw_papers, list):
            for item in raw_papers:
                if not isinstance(item, dict):
                    continue
                paper_id = str(item.get("paper_id") or "")
                if paper_id not in selected_ids:
                    continue
                for raw_id in item.get("evidence_chunk_ids") or []:
                    chunk_id = str(raw_id)
                    record = context_records.get(chunk_id)
                    if (
                        record is not None
                        and record.get("paper_id") == paper_id
                        and self._valid_evidence_record(record)
                        and chunk_id not in requested_ids
                    ):
                        requested_ids.append(chunk_id)

        evidence = []
        for order, chunk_id in enumerate(requested_ids[: self.max_evidence]):
            record = context_records[chunk_id]
            result = self._to_retrieval_result(record, score=float(-order))
            evidence.append(evidence_from_result(result))
        return evidence

    def _build_answer(self, query: Query, payload: dict[str, Any]) -> Answer:
        raw_answer = payload.get("answer")
        if not isinstance(raw_answer, dict):
            raise RuntimeError(f"{query.query_id}: answer payload is not an object")

        freeform = None
        if "freeform" in query.answer_types:
            raw_freeform = raw_answer.get("freeform")
            if isinstance(raw_freeform, str):
                raw_freeform = {"text": raw_freeform}
            if not isinstance(raw_freeform, dict):
                raise RuntimeError(f"{query.query_id}: freeform answer is missing")
            freeform = {"text": str(raw_freeform.get("text") or "").strip()}

        table = None
        if "table" in query.answer_types:
            raw_table = raw_answer.get("table")
            rows = raw_table.get("rows") if isinstance(raw_table, dict) else []
            table = {"schema": query.table_schema or [], "rows": rows or []}

        multiple_choice = None
        if "multiple_choice" in query.answer_types:
            # The four-field contract does not contain option text.  Preserve the
            # semantic answer in trace, but use an unbiased deterministic letter
            # so the official record is non-empty and reproducible.
            multiple_choice = {"gold": deterministic_mc_letter(query.query_id)}

        return Answer(
            freeform=freeform,
            multiple_choice=multiple_choice,
            table=table,
        )

    def _validate_answer_payload(
        self, query: Query, plan: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        papers = payload.get("papers")
        if not isinstance(papers, list) or not papers:
            raise RuntimeError(f"{query.query_id}: answer stage omitted evidence papers")
        raw_answer = payload.get("answer")
        if not isinstance(raw_answer, dict):
            raise RuntimeError(f"{query.query_id}: answer stage omitted answer object")
        if "freeform" in query.answer_types:
            freeform = raw_answer.get("freeform")
            text = (
                str(freeform.get("text") or "").strip()
                if isinstance(freeform, dict)
                else str(freeform or "").strip()
            )
            if not text:
                raise RuntimeError(f"{query.query_id}: freeform answer is empty")
        if "table" in query.answer_types:
            table = raw_answer.get("table")
            rows = table.get("rows") if isinstance(table, dict) else None
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(f"{query.query_id}: table answer has no rows")
            target_count = len(plan.get("targets") or [])
            if (
                plan.get("requires_multiple_papers")
                and target_count > 1
                and len(rows) < target_count
            ):
                raise RuntimeError(
                    f"{query.query_id}: table answer covers {len(rows)} rows for "
                    f"{target_count} planned targets"
                )
        if "multiple_choice" in query.answer_types:
            semantic = payload.get("semantic_multiple_choice")
            text = (
                str(semantic.get("text") or "").strip()
                if isinstance(semantic, dict)
                else ""
            )
            if not text:
                raise RuntimeError(
                    f"{query.query_id}: meaning-level multiple-choice answer is empty"
                )

    # ---- Local chunk selection -------------------------------------------

    def _rank_records(
        self,
        query: Query,
        records: list[Record],
        plan: dict[str, Any],
        *,
        focus_only: bool = False,
    ) -> list[tuple[float, Record]]:
        query_text = " ".join(
            ([] if focus_only else [query.question])
            + (
                []
                if focus_only
                else [
                    str(column.get("name") or "")
                    for column in query.table_schema or []
                    if isinstance(column, dict)
                ]
            )
            + [
                str(target.get("name") or "")
                + " "
                + " ".join(str(term) for term in target.get("search_terms") or [])
                for target in plan.get("targets") or []
                if isinstance(target, dict)
            ]
        )
        query_tokens = [token for token in _tokens(query_text) if token not in _STOPWORDS]
        token_counts = Counter(query_tokens)
        modalities = set(plan.get("modalities") or _expected_modalities(query))
        locator_terms = {
            f"{kind.lower()} {identifier.lower()}"
            for kind, identifier in _LOCATOR_RE.findall(query.question)
        }
        ordinal_references = [
            int(match) for match in _ORDINAL_REFERENCE_RE.findall(query.question)
        ]
        wants_last_reference = bool(_LAST_REFERENCE_RE.search(query.question))

        scored: list[tuple[float, Record]] = []
        for position, record in enumerate(records):
            text = str(record.get("text") or "")
            lower = text.lower()
            record_tokens = set(_tokens(text))
            overlap = sum(
                (1.0 + math.log1p(count))
                for token, count in token_counts.items()
                if token in record_tokens
            )
            coverage = (
                len(set(query_tokens).intersection(record_tokens))
                / max(1, len(set(query_tokens)))
            )
            score = overlap + 8.0 * coverage
            chunk_type = _record_source_type(record)
            if chunk_type in modalities:
                score += 5.0
            metadata = record.get("metadata") or {}
            locator_text = " ".join(
                str(metadata.get(key) or "")
                for key in ("table_id", "figure_id", "equation_id", "citation_id")
            ).lower()
            if any(term in lower or term in locator_text for term in locator_terms):
                score += 12.0
            if chunk_type == "citation_context":
                for reference_number in ordinal_references:
                    marker = re.compile(
                        rf"(?:^|\n|\[|\()\s*{reference_number}\s*(?:[.\])]|\))"
                    )
                    if marker.search(text):
                        score += 24.0
                if wants_last_reference:
                    score += 12.0 * position / max(1, len(records) - 1)
            if str(record.get("chunk_type") or "") == "title_abstract":
                score += 0.25
            score -= position * 1e-6
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _target_covered_records(
        self,
        query: Query,
        plan: dict[str, Any],
        ranked_records: list[tuple[float, Record]],
    ) -> list[Record]:
        records = [record for _, record in ranked_records]
        selected: list[Record] = []
        seen: set[str] = set()
        targets = [
            target
            for target in plan.get("targets") or []
            if isinstance(target, dict)
        ][: self.answer_chunks_per_paper]
        for target in targets:
            target_plan = dict(plan)
            target_plan["targets"] = [target]
            reranked = self._rank_records(
                query, records, target_plan, focus_only=True
            )
            for _, record in reranked:
                chunk_id = str(record.get("chunk_id") or "")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    selected.append(record)
                    break
        for _, record in ranked_records:
            chunk_id = str(record.get("chunk_id") or "")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                selected.append(record)
            if len(selected) >= self.answer_chunks_per_paper:
                break
        return selected[: self.answer_chunks_per_paper]

    def _valid_evidence_record(self, record: Record) -> bool:
        source_type = _record_source_type(record)
        metadata = record.get("metadata") or {}
        page = metadata.get("page")
        if (
            source_type not in OFFICIAL_SOURCE_TYPES
            or isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
        ):
            return False
        if source_type == "table" and not _nonempty_string(metadata.get("table_id")):
            return False
        if source_type == "figure" and not _nonempty_string(metadata.get("figure_id")):
            return False
        return True

    def _to_retrieval_result(self, record: Record, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=str(record.get("chunk_id") or ""),
            paper_id=str(record.get("paper_id") or ""),
            score=score,
            text=str(record.get("text") or ""),
            chunk_type=_record_source_type(record),
            metadata=dict(record.get("metadata") or {}),
            source="corpus_qa",
        )

    def _image_items(
        self, query: Query, plan: dict[str, Any], records: list[Record]
    ) -> list[dict[str, str]]:
        modalities = set(plan.get("modalities") or []).union(
            _expected_modalities(query)
        )
        if not {"table", "figure"}.intersection(modalities):
            return []
        items: list[dict[str, str]] = []
        paths: set[str] = set()
        for record in records:
            if str(record.get("chunk_type") or "") not in {"table", "figure"}:
                continue
            image_path = str((record.get("metadata") or {}).get("image_path") or "")
            if image_path and Path(image_path).is_file() and image_path not in paths:
                paths.add(image_path)
                items.append(
                    {
                        "path": image_path,
                        "paper_id": str(record.get("paper_id") or ""),
                        "chunk_id": str(record.get("chunk_id") or ""),
                        "chunk_type": str(record.get("chunk_type") or ""),
                    }
                )
            if len(items) >= self.max_images:
                break
        return items

    # ---- Prompt helpers ---------------------------------------------------

    def _format_record(
        self,
        record: Record,
        max_chars: int | None = None,
        focus_text: str = "",
    ) -> str:
        metadata = record.get("metadata") or {}
        locator = ", ".join(
            f"{key}={metadata[key]}"
            for key in (
                "page",
                "section",
                "table_id",
                "figure_id",
                "equation_id",
                "citation_id",
            )
            if metadata.get(key) is not None
        )
        max_chars = max_chars or self.snippet_chars
        if _record_source_type(record) in {
            "table",
            "citation_context",
            "equation_algorithm",
        }:
            multiplier = 4 if _record_source_type(record) == "table" else 2
            max_chars = min(max_chars * multiplier, self.max_context_chars)
        text = _focused_excerpt(
            str(record.get("text") or ""), focus_text, max_chars
        )
        return (
            f"[paper_id={record.get('paper_id')} chunk_id={record.get('chunk_id')} "
            f"type={_record_source_type(record)} {locator}]\n{text}"
        )

    def _answer_json_spec(self, query: Query) -> str:
        fields: list[str] = []
        if "freeform" in query.answer_types:
            fields.append('"freeform": {"text": "short extractive answer"}')
        if "table" in query.answer_types:
            fields.append('"table": {"rows": [{"exact schema column": "value"}]}')
        return "{" + ", ".join(fields) + "}"

    def _ask_json(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        try:
            complete = getattr(self.llm, "complete", None)
            raw = (
                complete(prompt, image_paths=image_paths)
                if image_paths and callable(complete)
                else self.llm(prompt)
            )
            parsed = parse_json_object(raw)
            if parsed is None and strict:
                raise RuntimeError("LLM response was not a JSON object")
            return parsed or {}
        except Exception as exc:
            if strict:
                raise RuntimeError("corpus QA answer-stage LLM call failed") from exc
            return {}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _official_chunk_type(chunk_type: str) -> str:
    if chunk_type == "title_abstract":
        return "text_span"
    return chunk_type


def _record_source_type(record: Record) -> str:
    chunk_type = _official_chunk_type(str(record.get("chunk_type") or ""))
    metadata = record.get("metadata") or {}
    section = str(metadata.get("section") or "").strip().lower()
    if chunk_type == "text_span" and (
        metadata.get("citation_id")
        or section in {"references", "bibliography"}
        or section.startswith("references ")
    ):
        return "citation_context"
    return chunk_type


def _plan_focus_text(query: Query, plan: dict[str, Any]) -> str:
    parts = [query.question]
    for target in plan.get("targets") or []:
        if not isinstance(target, dict):
            continue
        parts.append(str(target.get("name") or ""))
        parts.extend(str(term) for term in target.get("search_terms") or [])
    return " ".join(parts)


def _focused_excerpt(text: str, focus_text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lower = text.lower()
    focus_tokens = {
        token
        for token in _tokens(focus_text)
        if token not in _STOPWORDS and len(token) >= 3
    }
    for token in tuple(focus_tokens):
        match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", token)
        if match:
            focus_tokens.add(match.group(1))
    search_start = 0
    first_line_end = text.find("\n", 0, 500)
    if text.startswith("[") and first_line_end >= 0:
        # MinerU prefixes every chunk with venue/year/title. Matching a paper
        # name there must not pin a long table excerpt to its first rows.
        search_start = first_line_end + 1
    candidates: list[int] = []
    for token in focus_tokens:
        cursor = search_start
        for _ in range(12):
            position = lower.find(token, cursor)
            if position < 0:
                break
            candidates.append(position)
            cursor = position + max(1, len(token))
    if not candidates:
        return text[:max_chars] + " …"
    def window_score(center: int) -> tuple[float, int]:
        start = max(search_start, center - max_chars // 3)
        window = lower[start : start + max_chars]
        coverage = sum(
            1.0 + min(len(token), 20) / 20.0
            for token in focus_tokens
            if token in window
        )
        return coverage, center

    center = max(candidates, key=window_score)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "… " if start else ""
    suffix = " …" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _expected_modalities(query: Query) -> set[str]:
    text = query.question.lower()
    modalities: set[str] = {"text_span"}
    if "table" in query.answer_types or any(word in text for word in ("table", "row", "column")):
        modalities.add("table")
    if any(word in text for word in ("figure", "fig.", "plot", "chart", "diagram")):
        modalities.add("figure")
    if any(
        word in text
        for word in ("equation", "objective", "loss function", "formula", "algorithm")
    ):
        modalities.add("equation_algorithm")
    if any(word in text for word in ("citation", "cited", "reference", "bibliography")):
        modalities.add("citation_context")
    return modalities
