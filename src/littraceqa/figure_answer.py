#!/usr/bin/env python3
"""Figure vision second pass over an existing predictions file.

Figure-primary questions are often unanswerable from caption-only figure
chunks (e.g. "how many subfigures", values read off a plot). This module
revises a ``run_rag`` predictions file by actually looking at the figures:
for every question whose ``primary_evidence_type`` is ``figure`` it fetches
the predicted papers' PDFs, renders the candidate figure pages to images,
and asks the vision-capable AOAI chat deployment to answer from the images.
Only answers the model marks as confident overwrite the original row; every
other row is copied through unchanged.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .azure_config import OpenAISettings, build_openai_client, load_environment
from .common import (
    DEFAULT_METADATA,
    ROOT,
    Record,
    compact_text,
    read_jsonl,
    retry_chat_completion,
    try_parse_json_object,
    write_jsonl,
)
from .fix_chunk_locators import normalize_object_id
from .run_rag import (
    extract_choice_value,
    freeform_text,
    load_options_map,
    normalize_choice,
    options_for_sample,
    render_options,
)


CHUNKS_DIR = ROOT / "artifacts" / "docint" / "chunks"
DEFAULT_PDF_CACHE = ROOT / "artifacts" / "pdf_cache"

PDF_MAGIC = b"%PDF-"
RENDER_SCALE = 2.0
DOWNLOAD_ATTEMPTS = 3  # 1 try + 2 retries
DOWNLOAD_DELAY_SECONDS = 3.0
CONNECT_TIMEOUT = 15.0
READ_TIMEOUT = 60.0

# Same identification as scripts/download_pdfs.py.
USER_AGENT = (
    "LitTraceQA-PDF-Fetcher/1.0 "
    "(+https://github.com/; academic dataset assembly; contact: maintainer)"
)

SYSTEM_PROMPT = """You are solving LitTraceQA, a scientific-paper grounded QA task.
You are shown rendered PDF pages containing figures. Answer the question by reading the figures in the images.
Copy exact values verbatim from the source; never paraphrase numbers.
Return JSON only. Do not include markdown."""

# Copied verbatim from run_rag.answer_rules so freeform answers stay extractive.
FREEFORM_RULE = (
    '- For "freeform": output ONLY the shortest verbatim value/name/phrase exactly as printed in the source. '
    "Preserve the printed formatting exactly (answer 14.70, not 14.7). "
    "Always emit the value as a quoted JSON string, never a bare JSON number, so trailing zeros survive. "
    "No sentence, no explanation, no trailing period, no units unless printed as part of the value. "
    'Examples: an F1 score question -> "14.70"; an author-name question -> "Jane Smith"; '
    'a hardware question -> "a single NVIDIA A100 GPU".'
)


def log(message: str) -> None:
    print(message, file=sys.stderr)


def predicted_paper_ids(row: Record) -> list[str]:
    papers = row.get("gold_papers")
    ids: list[str] = []
    if isinstance(papers, list):
        for paper in papers:
            if isinstance(paper, dict) and paper.get("paper_id"):
                paper_id = str(paper["paper_id"])
                if paper_id not in ids:
                    ids.append(paper_id)
    return ids


def coerce_page(value: Any) -> Optional[int]:
    try:
        page = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def load_figure_chunks(paper_id: str) -> list[Record]:
    """Ordered [{figure_id, page}] parsed from the paper's chunk file (read-only)."""
    path = CHUNKS_DIR / f"{paper_id}.jsonl"
    if not path.exists():
        return []
    figures: list[Record] = []
    try:
        for chunk in read_jsonl(path):
            if chunk.get("source_type") != "figure":
                continue
            try:
                locator = json.loads(chunk.get("locator_json") or "{}")
            except json.JSONDecodeError:
                locator = {}
            if not isinstance(locator, dict):
                locator = {}
            page = coerce_page(locator.get("page"))
            if page is None:
                pages = chunk.get("page_numbers") or []
                page = coerce_page(pages[0]) if pages else None
            if page is None:
                continue
            figures.append(
                {"figure_id": str(locator.get("figure_id") or ""), "page": page}
            )
    except Exception as exc:  # noqa: BLE001 - a bad chunk file must not kill the run.
        log(f"WARN {paper_id}: could not read chunk file: {exc}")
        return []
    return figures


def evidence_page_candidates(row: Record, paper_ids: list[str]) -> list[tuple[str, int]]:
    """(paper_id, page) pairs from the row's evidence items, figure items first."""
    wanted = set(paper_ids)
    figure_pairs: list[tuple[str, int]] = []
    other_pairs: list[tuple[str, int]] = []
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        if paper_id not in wanted:
            continue
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        page = coerce_page(locator.get("page"))
        if page is None:
            continue
        pair = (paper_id, page)
        if item.get("source_type") == "figure":
            figure_pairs.append(pair)
        else:
            other_pairs.append(pair)
    return figure_pairs + other_pairs


def plan_candidate_pages(
    row: Record,
    paper_ids: list[str],
    figure_index: dict[str, list[Record]],
    max_pages: int,
) -> list[tuple[str, int]]:
    """Ordered (paper_id, page) list: evidence pages first, then figure-chunk
    pages per paper in chunk order, deduplicated and capped at ``max_pages``."""
    ordered: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(pair: tuple[str, int]) -> None:
        if pair not in seen and len(ordered) < max_pages:
            seen.add(pair)
            ordered.append(pair)

    for pair in evidence_page_candidates(row, paper_ids):
        add(pair)
    for paper_id in paper_ids:
        for figure in figure_index.get(paper_id, []):
            add((paper_id, int(figure["page"])))
    return ordered


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    return session


def looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return PDF_MAGIC in handle.read(1024)
    except OSError:
        return False


def fetch_pdf(
    session: requests.Session,
    paper_id: str,
    pdf_url: str,
    cache_dir: Path,
) -> Optional[Path]:
    """Download (or reuse) a paper's PDF; None on failure. Never raises."""
    dest = cache_dir / f"{paper_id}.pdf"
    if dest.exists() and looks_like_pdf(dest):
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pdf.part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with session.get(
                pdf_url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status_code != 200:
                    log(f"WARN {paper_id}: HTTP {resp.status_code} for {pdf_url}")
                    if resp.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                        return None  # permanent failure: do not retry
                else:
                    with tmp.open("wb") as out:
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            if chunk:
                                out.write(chunk)
                    if looks_like_pdf(tmp):
                        tmp.replace(dest)
                        time.sleep(DOWNLOAD_DELAY_SECONDS)  # polite pacing
                        return dest
                    tmp.unlink(missing_ok=True)
                    log(f"WARN {paper_id}: response is not a PDF ({pdf_url})")
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            log(f"WARN {paper_id}: network error: {exc.__class__.__name__}: {exc}")
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            log(f"WARN {paper_id}: io error: {exc}")
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(DOWNLOAD_DELAY_SECONDS * attempt)
    tmp.unlink(missing_ok=True)
    return None


def render_pages(pdf_path: Path, pages: list[int]) -> dict[int, bytes]:
    """Render 1-based ``pages`` of ``pdf_path`` to PNG bytes; skips bad pages."""
    import pypdfium2 as pdfium

    rendered: dict[int, bytes] = {}
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(document)
        for page_number in pages:
            index = page_number - 1
            if index < 0 or index >= page_count:
                log(f"WARN {pdf_path.name}: page {page_number} out of range (1-{page_count})")
                continue
            try:
                page = document[index]
                bitmap = page.render(scale=RENDER_SCALE)
                image = bitmap.to_pil()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                rendered[page_number] = buffer.getvalue()
            except Exception as exc:  # noqa: BLE001 - skip an unrenderable page.
                log(f"WARN {pdf_path.name}: could not render page {page_number}: {exc}")
    finally:
        document.close()
    return rendered


def page_manifest_lines(
    candidates: list[tuple[str, int]],
    figure_index: dict[str, list[Record]],
) -> list[str]:
    lines: list[str] = []
    for position, (paper_id, page) in enumerate(candidates, start=1):
        figure_ids = [
            figure["figure_id"]
            for figure in figure_index.get(paper_id, [])
            if int(figure["page"]) == page and figure["figure_id"]
        ]
        note = f" (figures on this page: {', '.join(figure_ids)})" if figure_ids else ""
        lines.append(f"Image {position}: paper_id={paper_id} page={page}{note}")
    return lines


def vision_user_content(
    sample: Record,
    options: Optional[dict[str, str]],
    candidates: list[tuple[str, int]],
    images: dict[tuple[str, int], bytes],
    figure_index: dict[str, list[Record]],
) -> list[Record]:
    answer_types = sample.get("answer_types") or []
    rules = [FREEFORM_RULE]
    if "multiple_choice" in answer_types:
        if options:
            rules.append(
                '- For "multiple_choice": answer with exactly one of the letter keys listed under Options. '
                "Output the letter only."
            )
        else:
            rules.append('- For "multiple_choice": answer with a single letter A-D.')
    rules.append(
        '- Set "confident" to true ONLY if the answer is actually visible in the provided images; '
        "if the relevant figure is missing, cropped, or unreadable, set it to false."
    )
    rules.append(
        '- In "figure" identify the single figure that answers the question, using the paper_id and '
        "page of the image it appears in and its printed label (e.g. \"Figure 3\")."
    )
    sections = [
        f"Question:\n{sample.get('question')}",
        f"Answer types: {answer_types}",
    ]
    if options and "multiple_choice" in answer_types:
        sections.append(render_options(options))
    sections.append("Rendered PDF pages:\n" + "\n".join(page_manifest_lines(candidates, figure_index)))
    sections.append(
        "Return a JSON object shaped like:\n"
        '{ "freeform": "...", "multiple_choice": "B", '
        '"figure": {"paper_id": "...", "figure_id": "Figure 3", "page": 5}, '
        '"confident": true }'
    )
    sections.append("Rules:\n" + "\n".join(rules))

    content: list[Record] = [{"type": "text", "text": "\n\n".join(sections) + "\n"}]
    for paper_id, page in candidates:
        png = images.get((paper_id, page))
        if png is None:
            continue
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
        )
    return content


def call_vision_model(
    client: Any,
    settings: OpenAISettings,
    content: list[Record],
    *,
    max_tokens: int,
) -> Record:
    kwargs: Record = {
        "model": settings.chat_deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = retry_chat_completion(client, kwargs)
    payload, _ = try_parse_json_object(response.choices[0].message.content or "")
    return payload


def resolve_figure_ref(
    payload: Record,
    paper_ids: list[str],
    figure_index: dict[str, list[Record]],
) -> Optional[Record]:
    """Match the model's figure reference against known figure chunks.

    Returns {"paper_id", "figure_id", "page"} using the chunk's canonical
    values, or None when the reference does not correspond to any known
    figure-chunk page (the caller then keeps the original evidence)."""
    figure = payload.get("figure")
    if not isinstance(figure, dict):
        return None
    paper_id = str(figure.get("paper_id") or "")
    if paper_id not in paper_ids:
        if len(paper_ids) == 1:
            paper_id = paper_ids[0]  # tolerate a mangled id for single-paper rows
        else:
            return None
    entries = figure_index.get(paper_id, [])
    wanted_id = normalize_object_id(figure.get("figure_id"), "figure")
    if wanted_id:
        for entry in entries:
            if normalize_object_id(entry["figure_id"], "figure") == wanted_id:
                return {
                    "paper_id": paper_id,
                    "figure_id": entry["figure_id"] or figure.get("figure_id"),
                    "page": int(entry["page"]),
                }
    page = coerce_page(figure.get("page"))
    if page is not None:
        for entry in entries:
            if int(entry["page"]) == page:
                return {
                    "paper_id": paper_id,
                    "figure_id": str(entry["figure_id"] or figure.get("figure_id") or ""),
                    "page": page,
                }
    return None


def merge_answer(
    row: Record,
    sample: Record,
    payload: Record,
    options: Optional[dict[str, str]],
    figure_ref: Optional[Record],
) -> bool:
    """Apply a confident vision answer to ``row`` in place; True if revised."""
    answer_types = set(sample.get("answer_types") or [])
    text = freeform_text(payload)
    choice = ""
    if "multiple_choice" in answer_types:
        choice = normalize_choice(extract_choice_value(payload), options, fallback_texts=(text,))
    wrote = False
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    if "freeform" in answer_types and text:
        answer["freeform"] = {"text": text}
        wrote = True
    if "multiple_choice" in answer_types and choice:
        answer["multiple_choice"] = {"gold": choice}
        wrote = True
    if not wrote:
        return False
    row["answer"] = answer
    if figure_ref is not None:
        snippet = text or (f"Answer: {choice}" if choice else "")
        row["evidence"] = [
            {
                "evidence_id": "ev_001",
                "paper_id": figure_ref["paper_id"],
                "source_type": "figure",
                "evidence_text_or_value": compact_text(
                    snippet or f"Read from {figure_ref.get('figure_id') or 'figure'}",
                    max_chars=300,
                ),
                "locator": {
                    "page": figure_ref["page"],
                    "figure_id": figure_ref.get("figure_id") or "",
                },
            }
        ]
    return True


def revise_row(
    row: Record,
    sample: Record,
    *,
    client: Any,
    settings: OpenAISettings,
    session: requests.Session,
    metadata_by_id: dict[str, Record],
    options: Optional[dict[str, str]],
    args: argparse.Namespace,
) -> bool:
    """Run the vision pass for one figure question; True if the row was revised."""
    query_id = row.get("query_id")
    paper_ids = predicted_paper_ids(row)
    if not paper_ids:
        log(f"KEEP {query_id}: no predicted papers in the row")
        return False

    figure_index = {paper_id: load_figure_chunks(paper_id) for paper_id in paper_ids}
    candidates = plan_candidate_pages(row, paper_ids, figure_index, args.max_pages)
    if not candidates:
        log(f"KEEP {query_id}: no candidate figure pages")
        return False

    if args.dry_run:
        for paper_id, page in candidates:
            pdf_url = str(metadata_by_id.get(paper_id, {}).get("pdf_url") or "")
            cached = (args.pdf_cache_dir / f"{paper_id}.pdf").exists()
            log(
                f"DRY {query_id}: paper={paper_id} page={page} "
                f"pdf={'cached' if cached else pdf_url or 'NO pdf_url'}"
            )
        return False

    images: dict[tuple[str, int], bytes] = {}
    for paper_id in paper_ids:
        pages = sorted({page for pid, page in candidates if pid == paper_id})
        if not pages:
            continue
        pdf_url = str(metadata_by_id.get(paper_id, {}).get("pdf_url") or "")
        if not pdf_url:
            log(f"WARN {query_id}: no pdf_url for {paper_id}")
            continue
        pdf_path = fetch_pdf(session, paper_id, pdf_url, args.pdf_cache_dir)
        if pdf_path is None:
            log(f"WARN {query_id}: could not fetch PDF for {paper_id}")
            continue
        for page, png in render_pages(pdf_path, pages).items():
            images[(paper_id, page)] = png
    if not images:
        log(f"KEEP {query_id}: no pages could be rendered")
        return False
    # Keep the prompt's image manifest in lockstep with the images actually
    # attached: drop candidates whose page failed to download/render.
    candidates = [pair for pair in candidates if pair in images]

    content = vision_user_content(sample, options, candidates, images, figure_index)
    payload = call_vision_model(client, settings, content, max_tokens=args.max_answer_tokens)
    if not payload:
        log(f"KEEP {query_id}: vision model returned no parseable JSON")
        return False
    if payload.get("confident") is not True:
        log(f"KEEP {query_id}: model not confident")
        return False

    figure_ref = resolve_figure_ref(payload, paper_ids, figure_index)
    if figure_ref is None:
        log(f"KEEP-EVIDENCE {query_id}: figure ref does not match any known figure chunk page")
    revised = merge_answer(row, sample, payload, options, figure_ref)
    if not revised:
        log(f"KEEP {query_id}: confident response carried no usable answer value")
    return revised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revise figure-primary predictions by rendering figure pages and "
            "asking the vision-capable AOAI chat deployment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--predictions", type=Path, required=True, help="run_rag output JSONL.")
    parser.add_argument("--inputs", type=Path, required=True, help="Questions JSONL (validation_inputs shape).")
    parser.add_argument("--output", type=Path, required=True, help="Revised predictions JSONL.")
    parser.add_argument(
        "--options-file",
        type=Path,
        default=None,
        help=(
            "JSONL joining multiple-choice options by query_id (gold rows or "
            "bare sidecar records); only answer.multiple_choice.options is "
            "read, never answers. Default: data/validation.jsonl, applied "
            "ONLY when --inputs is the default data/validation_inputs.jsonl "
            "(same semantics as run_rag)."
        ),
    )
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        dest="query_ids",
        help="Revise only these query_ids (repeatable); everything else is copied through.",
    )
    parser.add_argument(
        "--pdf-cache-dir",
        type=Path,
        default=DEFAULT_PDF_CACHE,
        help="Where downloaded PDFs are cached (created on use).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Cap on rendered candidate pages per question, across papers.",
    )
    parser.add_argument("--max-answer-tokens", type=int, default=500)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which questions/pages/PDFs would be used; no AOAI calls, no downloads, no output file.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_pages < 1:
        args.max_pages = 1

    if args.options_file is None:
        default_options = ROOT / "data" / "validation.jsonl"
        default_inputs = ROOT / "data" / "validation_inputs.jsonl"
        # Same guard as run_rag: only join validation options onto the
        # validation inputs; any other input must opt in explicitly.
        if default_options.exists() and args.inputs.resolve() == default_inputs.resolve():
            args.options_file = default_options
    options_map = load_options_map(args.options_file)

    samples_by_id = {
        str(sample.get("query_id") or ""): sample for sample in read_jsonl(args.inputs)
    }
    rows = read_jsonl(args.predictions)
    metadata_by_id: dict[str, Record] = {}
    for record in read_jsonl(args.metadata_file):
        paper_id = str(record.get("paper_id") or "")
        if paper_id:
            metadata_by_id[paper_id] = record

    wanted_ids = {str(query_id) for query_id in args.query_ids} if args.query_ids else None

    client: Any = None
    settings: Optional[OpenAISettings] = None
    if not args.dry_run:
        load_environment(args.env_file)
        settings = OpenAISettings.from_env()
        client = build_openai_client(settings, max_retries=6)
    session = build_http_session()

    examined = 0
    revised = 0
    output_rows: list[Record] = []
    for row in rows:
        query_id = str(row.get("query_id") or "")
        sample = samples_by_id.get(query_id)
        eligible = (
            sample is not None
            and str(sample.get("primary_evidence_type") or "") == "figure"
            and (wanted_ids is None or query_id in wanted_ids)
        )
        if eligible:
            examined += 1
            try:
                options = options_for_sample(sample, options_map)
                if revise_row(
                    row,
                    sample,
                    client=client,
                    settings=settings,
                    session=session,
                    metadata_by_id=metadata_by_id,
                    options=options,
                    args=args,
                ):
                    revised += 1
            except Exception as exc:  # noqa: BLE001 - keep the original row.
                log(f"KEEP {query_id}: unexpected error: {exc.__class__.__name__}: {exc}")
        output_rows.append(row)

    if args.dry_run:
        log(f"dry-run: would examine {examined} figure questions; no output written")
        return 0

    write_jsonl(args.output, output_rows)
    log(
        f"wrote {len(output_rows)} rows to {args.output}; "
        f"figure questions examined={examined}, revised={revised}, kept={examined - revised}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
