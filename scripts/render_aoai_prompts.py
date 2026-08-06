"""Render the pairwise-reader prompts without calling Azure OpenAI.

This is an inspection tool, not an inference shortcut.  It projects an
organizer input record through the same production-safe query loader used by
the reader, selects the versioned synthetic few-shot examples, and renders the
Stage-1 and/or Stage-2 prompt as Markdown or JSON.  Paper/evidence text can be
provided explicitly; otherwise conspicuous synthetic preview text is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from littraceqa.candidate_handoff import (
    CandidatePaper,
    candidate_papers_from_record,
    production_query_from_record,
    read_jsonl,
)
from littraceqa.di_pipeline.contracts import Query
from littraceqa.pairwise_prompts import (
    ANSWER_PROMPT_VERSION,
    JUDGMENT_PROMPT_VERSION,
    PAIRWISE_SYSTEM_PROMPT,
    answer_response_shape,
    example_manifest,
    render_answer_prompt,
    render_judgment_prompt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the exact versioned prompt template used by the pairwise "
            "reader, without making an API call."
        )
    )
    parser.add_argument("--queries", required=True, help="Organizer input JSONL")
    parser.add_argument(
        "--query-id",
        default=None,
        help="Query to render; defaults to the first record in --queries",
    )
    parser.add_argument(
        "--stage",
        choices=("judgment", "answer", "all"),
        default="all",
        help="Render Stage 1, Stage 2, or both",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Optional sanitized query_id + candidate_papers JSONL",
    )
    parser.add_argument(
        "--paper-id",
        default=None,
        help="Candidate to preview; defaults to the first candidate",
    )
    parser.add_argument(
        "--paper-text-file",
        default=None,
        help="Optional already-formatted Stage-1 paper text",
    )
    parser.add_argument(
        "--accepted-summary-file",
        default=None,
        help="Optional Stage-2 accepted-summary JSON or JSONL",
    )
    parser.add_argument(
        "--evidence-file",
        default=None,
        help="Optional already-formatted Stage-2 evidence text",
    )
    parser.add_argument(
        "--image-legend-file",
        default=None,
        help="Optional image mapping text to display in the rendered prompt",
    )
    parser.add_argument("--max-evidence", type=int, default=32)
    parser.add_argument("--max-evidence-per-paper", type=int, default=4)
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write to this file instead of stdout",
    )
    return parser


def build_preview(args: argparse.Namespace) -> dict[str, Any]:
    """Build a serializable prompt preview from parsed command-line arguments."""

    if args.max_evidence < 1 or args.max_evidence_per_paper < 1:
        raise ValueError("evidence limits must be positive")

    query, query_position = _load_query(args.queries, args.query_id)
    candidate, candidate_source = _load_candidate(
        args.candidates, query.query_id, args.paper_id
    )
    query_payload = query.to_dict()
    candidate_payload = _candidate_payload(candidate)
    image_legend = _optional_text(args.image_legend_file)

    prompts: list[dict[str, Any]] = []
    if args.stage in {"judgment", "all"}:
        paper_text = (
            _required_nonempty_text(args.paper_text_file, "paper text")
            if args.paper_text_file
            else _sample_paper_text(candidate.paper_id)
        )
        prompt = render_judgment_prompt(
            query=query,
            query_payload=query_payload,
            candidate_payload=candidate_payload,
            paper_text=paper_text,
            image_legend=image_legend,
        )
        prompts.append(_prompt_record("judgment", JUDGMENT_PROMPT_VERSION, prompt))

    if args.stage in {"answer", "all"}:
        accepted_summary = (
            _load_json_collection(args.accepted_summary_file)
            if args.accepted_summary_file
            else [_sample_accepted_summary(candidate)]
        )
        evidence_text = (
            _required_nonempty_text(args.evidence_file, "answer evidence")
            if args.evidence_file
            else _sample_evidence_text(candidate.paper_id)
        )
        prompt = render_answer_prompt(
            query=query,
            query_payload=query_payload,
            accepted_summary=accepted_summary,
            evidence_text=evidence_text,
            image_legend=image_legend,
            answer_shape=answer_response_shape(query),
            max_evidence=args.max_evidence,
            max_evidence_per_paper=args.max_evidence_per_paper,
        )
        prompts.append(_prompt_record("answer", ANSWER_PROMPT_VERSION, prompt))

    return {
        "schema_version": 1,
        "query_id": query.query_id,
        "query_position": query_position,
        "query_payload": query_payload,
        "candidate_source": candidate_source,
        "candidate_payload": candidate_payload,
        "synthetic_paper_text": args.paper_text_file is None,
        "synthetic_answer_context": (
            args.accepted_summary_file is None or args.evidence_file is None
        ),
        "few_shot_examples": example_manifest(query),
        "prompts": prompts,
    }


def render_output(preview: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_format != "markdown":
        raise ValueError(f"unsupported output format: {output_format}")

    lines = [
        "# Pairwise AOAI prompt preview",
        "",
        f"- Query: `{preview['query_id']}`",
        f"- Candidate source: `{preview['candidate_source']}`",
        f"- Candidate: `{preview['candidate_payload']['paper_id']}`",
        f"- Synthetic paper text: `{str(preview['synthetic_paper_text']).lower()}`",
        (
            "- Synthetic answer context: `"
            + str(preview["synthetic_answer_context"]).lower()
            + "`"
        ),
        "",
        "## Selected synthetic few-shot examples",
        "",
        "- Judgment: " + ", ".join(preview["few_shot_examples"]["judgment"]),
        "- Answer: " + ", ".join(preview["few_shot_examples"]["answer"]),
    ]
    for prompt in preview["prompts"]:
        title = "Stage 1: candidate judgment" if prompt["stage"] == "judgment" else "Stage 2: final answer"
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- Prompt version: `{prompt['prompt_version']}`",
                f"- System SHA-256: `{prompt['system_sha256']}`",
                f"- SHA-256: `{prompt['sha256']}`",
                f"- Characters: `{prompt['characters']}`",
                "",
                "### System message",
                "",
                *_markdown_fence(prompt["system"]),
                "",
                "### User message",
                "",
                *_markdown_fence(prompt["text"]),
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preview = build_preview(args)
    rendered = render_output(preview, args.format)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


def _load_query(path: str | Path, query_id: str | None) -> tuple[Query, int]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"query input is empty: {path}")
    matches = [
        (position, record)
        for position, record in enumerate(records, start=1)
        if query_id is None or record.get("query_id") == query_id
    ]
    if not matches:
        raise ValueError(f"query_id is absent from {path}: {query_id}")
    if query_id is not None and len(matches) > 1:
        raise ValueError(f"duplicate query_id in {path}: {query_id}")
    position, record = matches[0]
    return production_query_from_record(record), position


def _load_candidate(
    path: str | Path | None,
    query_id: str,
    paper_id: str | None,
) -> tuple[CandidatePaper, str]:
    if path is None:
        return (
            CandidatePaper(
                paper_id=paper_id or "preview_paper",
                rank=1,
                title="Synthetic preview candidate",
                venue="PREVIEW",
                year=2025,
            ),
            "synthetic",
        )

    records = read_jsonl(path)
    matches = [record for record in records if record.get("query_id") == query_id]
    if not matches:
        raise ValueError(f"candidate sidecar has no query_id {query_id!r}: {path}")
    if len(matches) > 1:
        raise ValueError(f"duplicate candidate query_id in {path}: {query_id}")
    candidates = candidate_papers_from_record(matches[0])
    if paper_id is None:
        return candidates[0], "sidecar"
    for candidate in candidates:
        if candidate.paper_id == paper_id:
            return candidate, "sidecar"
    raise ValueError(f"paper_id {paper_id!r} is not a candidate for {query_id}")


def _load_json_collection(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = read_jsonl(source)
    if isinstance(value, dict):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise TypeError(
            "accepted summary must be a non-empty JSON object or list/JSONL of objects"
        )
    return value


def _candidate_payload(candidate: CandidatePaper) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "rank": candidate.rank,
        "title": candidate.title,
        "venue": candidate.venue,
        "year": candidate.year,
    }


def _sample_paper_text(paper_id: str) -> str:
    header = {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}#preview",
        "source_type": "text_span",
        "locator": {"page": 1, "section": "Preview"},
    }
    return (
        "[chunk "
        + json.dumps(header, ensure_ascii=False, separators=(",", ":"))
        + "]\nSynthetic preview paper text. Supply --paper-text-file to inspect a "
        "real formatted selected paper context."
    )


def _sample_accepted_summary(candidate: CandidatePaper) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "title": candidate.title,
        "rank": candidate.rank,
        "label": "direct_answer",
        "candidate_answer": {
            "units": [
                {
                    "name": "preview answer unit",
                    "value": "synthetic preview value",
                    "value_kind": "text",
                    "matched_option_labels": [],
                }
            ],
            "rows": [],
        },
        "reason": "Synthetic preview only.",
    }


def _sample_evidence_text(paper_id: str) -> str:
    header = {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}#preview",
        "source_type": "text_span",
        "locator": {"page": 1, "section": "Preview"},
        "stage1_selected": True,
        "submission_eligible": True,
    }
    return (
        "[chunk "
        + json.dumps(header, ensure_ascii=False, separators=(",", ":"))
        + "]\nSynthetic preview evidence. Supply --evidence-file and "
        "--accepted-summary-file to inspect a real Stage-2 context."
    )


def _prompt_record(stage: str, version: str, prompt: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "prompt_version": version,
        "system_sha256": hashlib.sha256(
            PAIRWISE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "characters": len(prompt),
        "system": PAIRWISE_SYSTEM_PROMPT,
        "text": prompt,
        "messages": [
            {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }


def _optional_text(path: str | Path | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def _required_nonempty_text(path: str | Path, label: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"supplied {label} file is empty: {path}")
    return text


def _markdown_fence(value: str) -> list[str]:
    """Fence arbitrary MinerU Markdown without an embedded fence ending it."""

    longest = max((len(item) for item in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + "text", value, fence]


if __name__ == "__main__":
    raise SystemExit(main())
