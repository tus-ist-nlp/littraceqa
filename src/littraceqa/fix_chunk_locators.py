#!/usr/bin/env python3
"""Renumber table_id/figure_id locators in existing per-paper chunk files.

The original chunker assigned ``Table {index}`` / ``Figure {index}`` from
Document Intelligence enumeration order, which drifts from the caption numbers
printed in the paper (e.g. an author-affiliation block detected as ``Table 1``
shifts every later table). The official evaluator matches evidence on the
normalized caption number, so drifted ids can never score.

This post-processor repairs the ids without re-running Document Intelligence:

* Figures: chunk content looks like ``"Figure 3: Figure 4: <caption...>"``
  (positional prefix + original caption that starts with the true label).
  The true label is parsed from the caption part, ``figure_id`` is updated,
  and the duplicated positional prefix is stripped from the content.
* Tables: table chunks carry no caption text, so captions are recovered from
  the paper's text_span chunks (lines starting with ``Table <num><punct>``)
  together with the page they were seen on. The k-th table chunk on page P is
  aligned with the k-th caption found on page P; when some per-page counts
  disagree, whole-document order alignment is used if the total counts match
  exactly, otherwise an order-preserving greedy page-window alignment fixes
  only the confidently matched chunks and the paper is counted as a conflict
  (unmatched chunks keep their original ids).

Default mode is a dry-run report. Use ``--output-dir`` to write fixed copies
of the changed papers, or ``--in-place`` to rewrite the chunk files atomically.
``--check-gold`` scores gold table/figure evidence items against the chunks on
the evaluator's coarse key before and after the proposed fix; ``--details``
additionally writes a per-item CSV. Chunks whose locator already carries a
``caption`` entry come from the new caption-based chunker and are left alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import ROOT, Record, read_jsonl, write_jsonl


FIGURE_POSITIONAL_PREFIX_RE = re.compile(r"^Figure\s+(\d+[a-z]?)\s*:\s*", re.IGNORECASE)
FIGURE_LABEL_RE = re.compile(r"^(?:Figure|Fig\.?)\s*(\d+[a-z]?)\s*[:.\s]", re.IGNORECASE)
# Label group matches the new chunker's TABLE_LABEL_PATTERN ("4", "A2", "S1",
# "Tab. 3"); the trailing [:.] keeps prose references ("Tab. 3 shows") out.
TABLE_CAPTION_LINE_RE = re.compile(
    r"^\s*Tab(?:le)?\.?\s*([A-Za-z]?\d+[a-z]?)\s*[:.]", re.IGNORECASE
)
TABLE_CONTENT_HEAD_RE = re.compile(r"^Table\s+\d+[a-z]?\s*$", re.IGNORECASE)


@dataclass
class ScanStats:
    papers_scanned: int = 0
    papers_changed: int = 0
    table_chunks: int = 0
    figure_chunks: int = 0
    tables_renumbered: int = 0
    figures_renumbered: int = 0
    figure_prefixes_stripped: int = 0
    table_conflicts: int = 0
    conflict_paper_ids: list[str] = field(default_factory=list)

    def add(self, other: "ScanStats") -> None:
        self.papers_scanned += other.papers_scanned
        self.papers_changed += other.papers_changed
        self.table_chunks += other.table_chunks
        self.figure_chunks += other.figure_chunks
        self.tables_renumbered += other.tables_renumbered
        self.figures_renumbered += other.figures_renumbered
        self.figure_prefixes_stripped += other.figure_prefixes_stripped
        self.table_conflicts += other.table_conflicts
        self.conflict_paper_ids.extend(other.conflict_paper_ids)


def parse_locator(chunk: Record) -> Record:
    raw = chunk.get("locator_json") or "{}"
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def chunk_page(chunk: Record, locator: Optional[Record] = None) -> Optional[int]:
    locator = parse_locator(chunk) if locator is None else locator
    page = locator.get("page")
    if isinstance(page, int):
        return page
    pages = chunk.get("page_numbers") or []
    if isinstance(pages, list) and pages and isinstance(pages[0], int):
        return pages[0]
    return None


def with_locator(chunk: Record, locator: Record) -> Record:
    updated = dict(chunk)
    updated["locator_json"] = json.dumps(locator, ensure_ascii=False)
    return updated


def normalize_object_id(value: Any, prefix: str) -> str:
    """Match evaluate.py's normalization: 'Table 4' == 'table 4' == '4'."""
    text = str(value or "").strip().lower().strip("\"'“”‘’`")
    text = re.sub(r"\s+", " ", text).strip()
    match = re.fullmatch(rf"{prefix}\s*(\d+[a-z]?)", text)
    if match:
        return f"{prefix} {match.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{prefix} {text}"
    return text


def fix_figure_chunk(chunk: Record) -> tuple[Record, bool, bool]:
    """Return (chunk, renumbered, prefix_stripped) for one figure chunk."""
    content = str(chunk.get("content") or "")
    locator = parse_locator(chunk)
    prefix_match = FIGURE_POSITIONAL_PREFIX_RE.match(content)
    caption = content[prefix_match.end() :] if prefix_match else content
    label_match = FIGURE_LABEL_RE.match(caption)
    if not label_match:
        # No true label recoverable from the caption; leave the chunk alone.
        return chunk, False, False

    new_id = f"Figure {label_match.group(1)}"
    old_id = str(locator.get("figure_id") or "")
    renumbered = normalize_object_id(new_id, "figure") != normalize_object_id(old_id, "figure")
    prefix_stripped = prefix_match is not None and caption != content
    if not renumbered and not prefix_stripped:
        return chunk, False, False

    locator["figure_id"] = new_id
    updated = with_locator(chunk, locator)
    updated["content"] = caption
    return updated, renumbered, prefix_stripped


@dataclass(frozen=True)
class Caption:
    label: str
    page: Optional[int]
    allowed_pages: frozenset[int]


def collect_table_captions(chunks: list[Record]) -> list[Caption]:
    """Scan text_span chunks (in chunk order) for 'Table N<punct>' caption lines.

    Captions are returned in document order and deduplicated by label so the
    chunker's overlap window does not produce duplicates. ``page`` is the
    containing text chunk's locator page; ``allowed_pages`` also covers the
    chunk's other pages plus the following page, because captions often sit on
    the page before the table body that Document Intelligence records.
    """
    captions: list[Caption] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.get("source_type") != "text_span":
            continue
        page = chunk_page(chunk)
        pages = {
            int(value)
            for value in (chunk.get("page_numbers") or [])
            if isinstance(value, int)
        }
        if page is not None:
            pages.add(page)
        allowed = frozenset(pages | {value + 1 for value in pages})
        for line in str(chunk.get("content") or "").split("\n"):
            match = TABLE_CAPTION_LINE_RE.match(line)
            if not match:
                continue
            label = match.group(1)
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            captions.append(Caption(label=label, page=page, allowed_pages=allowed))
    return captions


def greedy_align(
    table_indices: list[int],
    chunks: list[Record],
    captions: list[Caption],
) -> dict[int, str]:
    """Order-preserving greedy caption assignment with a page window.

    Both the table chunks and the caption labels appear in document order, so
    walk the two sequences together: a chunk matches the next unmatched caption
    when the chunk page falls inside the caption's allowed page window. Chunks
    before the window (spurious early detections such as author blocks) are
    skipped and keep their ids; captions whose table was never detected are
    skipped as well.
    """
    mapping: dict[int, str] = {}
    position = 0
    for index in table_indices:
        page = chunk_page(chunks[index])
        if page is None:
            continue
        while position < len(captions):
            caption = captions[position]
            if not caption.allowed_pages:
                position += 1
                continue
            if page in caption.allowed_pages:
                mapping[index] = caption.label
                position += 1
                break
            if page < min(caption.allowed_pages):
                break
            position += 1
    return mapping


def align_tables(
    table_indices: list[int],
    chunks: list[Record],
    captions: list[Caption],
) -> tuple[dict[int, str], bool]:
    """Map chunk-list index -> caption label.

    Returns (mapping, conflict). When every page's chunk/caption counts agree
    the k-th chunk on page P maps to the k-th caption on page P. Otherwise
    whole-document order alignment is used if the total counts match exactly.
    Failing both, an order-preserving greedy page-window alignment maps only
    the confidently matched chunks and ``conflict`` is True (unmatched chunks
    keep their original ids).
    """
    if not table_indices:
        return {}, False

    pages = [chunk_page(chunks[index]) for index in table_indices]
    chunks_by_page: dict[Optional[int], list[int]] = defaultdict(list)
    for index, page in zip(table_indices, pages):
        chunks_by_page[page].append(index)
    captions_by_page: dict[Optional[int], list[str]] = defaultdict(list)
    for caption in captions:
        captions_by_page[caption.page].append(caption.label)

    per_page_ok = None not in chunks_by_page and all(
        len(captions_by_page.get(page, [])) == len(indices)
        for page, indices in chunks_by_page.items()
    )
    if per_page_ok:
        mapping: dict[int, str] = {}
        for page, indices in chunks_by_page.items():
            for index, label in zip(indices, captions_by_page[page]):
                mapping[index] = label
        return mapping, False

    if len(captions) == len(table_indices):
        return {
            index: caption.label
            for index, caption in zip(table_indices, captions)
        }, False

    return greedy_align(table_indices, chunks, captions), True


def fix_table_content(content: str, new_id: str) -> str:
    head, sep, rest = content.partition("\n")
    if TABLE_CONTENT_HEAD_RE.match(head):
        return f"{new_id}{sep}{rest}"
    return content


def plan_paper(paper_id: str, chunks: list[Record]) -> tuple[list[Record], ScanStats]:
    """Compute the fixed chunk list for one paper without touching disk."""
    stats = ScanStats(papers_scanned=1)
    fixed = list(chunks)

    for index, chunk in enumerate(fixed):
        if chunk.get("source_type") != "figure":
            continue
        stats.figure_chunks += 1
        if "caption" in parse_locator(chunk):
            # New-chunker output: the id was already derived from the DI
            # caption; the positional-prefix heuristic does not apply.
            continue
        updated, renumbered, prefix_stripped = fix_figure_chunk(chunk)
        if renumbered:
            stats.figures_renumbered += 1
        if prefix_stripped:
            stats.figure_prefixes_stripped += 1
        fixed[index] = updated

    all_table_indices = [
        index for index, chunk in enumerate(fixed) if chunk.get("source_type") == "table"
    ]
    stats.table_chunks += len(all_table_indices)
    # Skip new-chunker output (locator carries "caption"): its table_id was
    # already read from the DI caption and must not be re-aligned. Also skip
    # uncaptioned tables (fragments, ToC/author-block detections): assigning a
    # printed label to them creates duplicate or phantom coarse keys — audit
    # found all four corpus-wide fixer corruptions came from this path.
    def _is_uncaptioned(chunk: Record) -> bool:
        loc_id = str(parse_locator(chunk).get("table_id") or "")
        first_line = str(chunk.get("content") or "").split("\n", 1)[0]
        return "(uncaptioned)" in loc_id or "(uncaptioned)" in first_line

    table_indices = [
        index
        for index in all_table_indices
        if "caption" not in parse_locator(fixed[index])
        and not _is_uncaptioned(fixed[index])
    ]
    if table_indices:
        captions = collect_table_captions(fixed)
        mapping, conflict = align_tables(table_indices, fixed, captions)
        if conflict:
            stats.table_conflicts += 1
            stats.conflict_paper_ids.append(paper_id)
        # Labels already held by tables outside the mapping (per page): never
        # reassign a printed label away from — or duplicate it against — an
        # unmapped chunk on another page.
        owned: dict[str, set[str]] = {}
        for idx2 in all_table_indices:
            if idx2 in mapping:
                continue
            loc2 = parse_locator(fixed[idx2])
            nid2 = normalize_object_id(str(loc2.get("table_id") or ""), "table")
            owned.setdefault(nid2, set()).add(str(loc2.get("page") or ""))
        for index, label in mapping.items():
            chunk = fixed[index]
            locator = parse_locator(chunk)
            new_id = f"Table {label}"
            old_id = str(locator.get("table_id") or "")
            if normalize_object_id(new_id, "table") == normalize_object_id(old_id, "table"):
                continue
            page_here = str(locator.get("page") or "")
            if any(pg != page_here for pg in owned.get(normalize_object_id(new_id, "table"), set())):
                continue
            locator["table_id"] = new_id
            updated = with_locator(chunk, locator)
            updated["content"] = fix_table_content(
                str(chunk.get("content") or ""), new_id
            )
            fixed[index] = updated
            stats.tables_renumbered += 1

    changed = any(before is not after for before, after in zip(chunks, fixed))
    stats.papers_changed = int(changed)
    return fixed, stats


def object_key_set(chunks: list[Record]) -> set[tuple[str, str, str]]:
    """Coarse evaluator keys (source_type, page, normalized id) for table/figure chunks."""
    keys: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        source_type = str(chunk.get("source_type") or "")
        if source_type not in ("table", "figure"):
            continue
        locator = parse_locator(chunk)
        page = str(locator.get("page", "")).strip()
        if not page:
            continue
        raw_id = locator.get("table_id") if source_type == "table" else locator.get("figure_id")
        keys.add((source_type, page, normalize_object_id(raw_id, source_type)))
    return keys


def gold_object_items(gold_records: list[Record]) -> list[tuple[str, str, str, str, str]]:
    """(query_id, paper_id, source_type, page, normalized id) for gold table/figure evidence."""
    items: list[tuple[str, str, str, str, str]] = []
    for record in gold_records:
        query_id = str(record.get("query_id") or "")
        evidence = record.get("evidence") or []
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source_type = str(item.get("source_type") or "")
            if source_type not in ("table", "figure"):
                continue
            locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
            page = str(locator.get("page", "")).strip()
            raw_id = (
                locator.get("table_id") if source_type == "table" else locator.get("figure_id")
            )
            items.append(
                (
                    query_id,
                    str(item.get("paper_id") or ""),
                    source_type,
                    page,
                    normalize_object_id(raw_id, source_type),
                )
            )
    return items


def report_check_gold(
    gold_path: Path,
    before_keys: dict[str, set[tuple[str, str, str]]],
    after_keys: dict[str, set[tuple[str, str, str]]],
    details_path: Optional[Path] = None,
) -> None:
    gold_records = read_jsonl(gold_path)
    items = gold_object_items(gold_records)
    totals = {"table": 0, "figure": 0}
    matched_before = {"table": 0, "figure": 0}
    matched_after = {"table": 0, "figure": 0}
    missing_papers: set[str] = set()
    detail_rows: list[dict[str, str]] = []
    for query_id, paper_id, source_type, page, object_id in items:
        totals[source_type] += 1
        before = after = False
        if paper_id not in before_keys:
            missing_papers.add(paper_id)
        else:
            key = (source_type, page, object_id)
            before = key in before_keys[paper_id]
            after = key in after_keys[paper_id]
            matched_before[source_type] += int(before)
            matched_after[source_type] += int(after)
        detail_rows.append(
            {
                "query_id": query_id,
                "paper_id": paper_id,
                "source_type": source_type,
                "page": page,
                "object_id": object_id,
                "matched_before": str(before).lower(),
                "matched_after": str(after).lower(),
            }
        )

    if details_path is not None:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "paper_id",
                    "source_type",
                    "page",
                    "object_id",
                    "matched_before",
                    "matched_after",
                ],
            )
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"per-item details written: {details_path} ({len(detail_rows)} rows)")

    print(f"check-gold against {gold_path}:")
    for source_type in ("table", "figure"):
        print(
            f"  {source_type} evidence coarse-key matches: "
            f"{matched_before[source_type]}/{totals[source_type]} before -> "
            f"{matched_after[source_type]}/{totals[source_type]} after"
        )
    if missing_papers:
        print(
            f"  gold papers without a chunk file: {len(missing_papers)} "
            f"({', '.join(sorted(missing_papers))})"
        )


def gold_paper_ids(gold_path: Path) -> set[str]:
    paper_ids: set[str] = set()
    for record in read_jsonl(gold_path):
        for item in record.get("gold_papers") or []:
            if isinstance(item, dict) and item.get("paper_id"):
                paper_ids.add(str(item["paper_id"]))
        for item in record.get("evidence") or []:
            if isinstance(item, dict) and item.get("paper_id"):
                paper_ids.add(str(item["paper_id"]))
    return paper_ids


def select_files(args: argparse.Namespace) -> list[Path]:
    if args.paper_id:
        return [args.chunks_dir / f"{paper_id}.jsonl" for paper_id in args.paper_id]
    if args.check_gold and not args.output_dir and not args.in_place:
        # Pure check mode: only the gold-referenced papers matter.
        wanted = gold_paper_ids(args.check_gold)
        return sorted(
            path
            for path in args.chunks_dir.glob("*.jsonl")
            if path.stem in wanted
        )
    return sorted(args.chunks_dir.glob("*.jsonl"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Renumber table_id/figure_id locators in per-paper chunk files "
            "to match the papers' visible caption numbers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Default mode is a dry-run report. --output-dir writes fixed copies "
            "of changed papers only; --in-place rewrites changed files atomically. "
            "When --check-gold is given without a write mode or --paper-id, only "
            "gold-referenced papers are scanned."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=ROOT / "artifacts" / "docint" / "chunks",
        help="Directory of per-paper chunk JSONL files.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Restrict to specific paper ids (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write fixed copies of changed papers into this directory.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite changed chunk files in place (atomic temp+replace).",
    )
    parser.add_argument(
        "--check-gold",
        type=Path,
        default=None,
        help="Gold JSONL (e.g. data/validation.jsonl): report evidence coarse-key matches before/after.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=None,
        help=(
            "With --check-gold: write a CSV with one row per gold table/figure "
            "evidence item (query_id, paper_id, gold key, matched before/after) "
            "so per-item regressions from the alignment heuristic are visible."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.in_place and args.output_dir:
        print("--in-place and --output-dir are mutually exclusive", file=sys.stderr)
        return 2
    if args.details and not args.check_gold:
        print("--details requires --check-gold", file=sys.stderr)
        return 2
    if not args.chunks_dir.is_dir():
        print(f"chunks dir not found: {args.chunks_dir}", file=sys.stderr)
        return 2

    files = select_files(args)
    missing = [path for path in files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing chunk file: {path}", file=sys.stderr)
        files = [path for path in files if path.is_file()]

    check_papers: set[str] = set()
    if args.check_gold:
        check_papers = gold_paper_ids(args.check_gold)
    before_keys: dict[str, set[tuple[str, str, str]]] = {}
    after_keys: dict[str, set[tuple[str, str, str]]] = {}

    totals = ScanStats()
    for index, path in enumerate(files, start=1):
        paper_id = path.stem
        chunks = read_jsonl(path)
        fixed, stats = plan_paper(paper_id, chunks)
        totals.add(stats)
        if paper_id in check_papers:
            before_keys[paper_id] = object_key_set(chunks)
            after_keys[paper_id] = object_key_set(fixed)
        if stats.papers_changed:
            if args.output_dir:
                write_jsonl(args.output_dir / path.name, fixed)
            elif args.in_place:
                write_jsonl(path, fixed)
        if index % 1000 == 0:
            print(f"[{index}/{len(files)}] scanned", file=sys.stderr)

    mode = "in-place" if args.in_place else ("output-dir" if args.output_dir else "dry-run")
    print(f"mode: {mode}")
    print(f"papers scanned: {totals.papers_scanned}")
    print(f"papers changed: {totals.papers_changed}")
    print(f"table chunks: {totals.table_chunks} (renumbered {totals.tables_renumbered})")
    print(
        f"figure chunks: {totals.figure_chunks} "
        f"(renumbered {totals.figures_renumbered}, prefixes stripped {totals.figure_prefixes_stripped})"
    )
    print(
        "table alignment conflicts (papers with pages left unchanged): "
        f"{totals.table_conflicts}"
    )
    if totals.conflict_paper_ids and len(totals.conflict_paper_ids) <= 20:
        print(f"  conflict papers: {', '.join(totals.conflict_paper_ids)}")

    if args.check_gold:
        report_check_gold(args.check_gold, before_keys, after_keys, args.details)
    return 0


if __name__ == "__main__":
    sys.exit(main())
