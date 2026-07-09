#!/usr/bin/env python3
"""Query-rewrite prompt lab: measure gold-paper recall of rewritten queries.

DEV-ONLY measurement tool. It reads gold papers from data/validation.jsonl to
compute recall@K (the same gold-access policy as compare_runs). It must NEVER
be part of the submission path: nothing it writes feeds a submission file.

The iteration loop (see RUNBOOK.md, section "Query-rewrite experimentation"):
edit prompts/query_rewrite.txt -> run this module -> read
runs/query_lab/<name>/results.md.

Per question the lab makes ONE Azure OpenAI chat call with the formatted
prompt (JSON mode), parses {"queries": [...]}, always keeps the original
question as query 0, runs the same hybrid search run_rag uses for every
query, merges the hits (deduped by chunk id, best rank wins), ranks papers
with run_rag.rank_papers and compares the ranked pool against the gold
papers.

Baseline convention: run once with ``--name _baseline --baseline-only`` to
write runs/query_lab/_baseline/results.json (original-question single
search); later runs then automatically show a baseline column in their
results.md.

run_rag imports :func:`load_prompt_template` / :func:`rewrite_queries` from
here (lazily, inside its ``--rewrite-prompt-file`` branch) so the exact same
rewrite call — and therefore the exact same queries — can be replayed inside
a full RAG run.

FIDELITY CAVEAT — the merge differs between lab and pipeline. The lab merges
per-query hit lists with :func:`merge_query_hits` (rank-interleaved: best
per-query rank wins, like the decompose merge), while run_rag's
``--rewrite-prompt-file`` hook appends extra-query hits AFTER the organic
hits via ``merge_hit_lists`` (precision-conservative concatenation). Because
:func:`run_rag.rank_papers` RRF-weights by list position, a gold paper found
only by a rewritten query at search rank 1 scores ~1/(60+1) here but only
~1/(60+top_chunks+1) end-to-end. Lab recall gains are therefore an
OPTIMISTIC UPPER BOUND on what the flag delivers; always re-score a winning
prompt with scripts/evaluate.py + compare_runs (see RUNBOOK).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from .azure_config import (
    OpenAISettings,
    SearchSettings,
    build_openai_client,
    build_search_client,
    load_environment,
)
from ..common import (
    ROOT,
    Record,
    compact_text,
    read_json,
    read_jsonl,
    retry_chat_completion,
    temp_sibling,
    try_parse_json_object,
    write_json,
    write_jsonl,
)
from . import run_rag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_NAME = "_baseline"
DEFAULT_PROMPT_FILE = ROOT / "prompts" / "query_rewrite.txt"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "query_lab"

# The three tokens substituted into the prompt template. Only these are
# replaced (literal replacement, not str.format), so JSON braces in the
# template survive untouched.
PROMPT_PLACEHOLDERS = ("question", "task_family", "primary_evidence_type")

# Cap on extra queries kept from the model (the original question is always
# query 0 on top of these). run_rag's --rewrite-prompt-file branch uses the
# same default.
DEFAULT_MAX_QUERIES = 5
# max_tokens for the rewrite call: 5 short queries fit comfortably.
REWRITE_MAX_TOKENS = 400
# Vector leg K for each hybrid search; matches run_rag's --vector-k default.
VECTOR_K = 50

# results.md/console cell budgets.
QUERIES_CELL_CHARS = 200
MISSING_CELL_CHARS = 120

_PROMPT_CACHE: dict[tuple[str, int], str] = {}


# ---------------------------------------------------------------------------
# Prompt loading and the one-call rewrite (shared with run_rag)
# ---------------------------------------------------------------------------


def load_prompt_template(path: Path) -> str:
    """Load a rewrite prompt template, dropping '#' comment lines.

    Any line whose first non-blank character is '#' is a comment for the
    prompt author and never reaches the model. The result must contain the
    required ``{question}`` placeholder. Cached by (path, mtime) so run_rag
    workers do not re-read the file per question.
    """
    path = Path(path)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_mtime_ns)
    cached = _PROMPT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    lines = path.read_text(encoding="utf-8").splitlines()
    template = "\n".join(
        line for line in lines if not line.lstrip().startswith("#")
    ).strip()
    if "{question}" not in template:
        raise ValueError(
            f"prompt template {path} must contain the {{question}} placeholder"
        )
    _PROMPT_CACHE[cache_key] = template
    return template


def format_rewrite_prompt(template: str, sample: Record) -> str:
    """Substitute the supported placeholders (literal token replacement).

    Unlike str.format, literal braces elsewhere in the template (e.g. the
    JSON output-contract example) are left alone.
    """
    text = template
    for name in PROMPT_PLACEHOLDERS:
        text = text.replace("{" + name + "}", str(sample.get(name) or ""))
    return text


def rewrite_queries(
    client: Any,
    settings: OpenAISettings,
    template: str,
    sample: Record,
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> tuple[list[str], str, bool]:
    """One JSON-mode rewrite call; returns (queries, raw_content, ok).

    ``queries`` ALWAYS starts with the original question (query 0). On any
    call/parse failure it degrades to the original question only, with
    ``ok=False`` and the raw model output (or error text) preserved so the
    row can be marked. Never raises.
    """
    question = str(sample.get("question") or "")
    try:
        kwargs: Record = {
            "model": settings.chat_deployment,
            "messages": [
                {"role": "user", "content": format_rewrite_prompt(template, sample)}
            ],
            "temperature": 0,
            "max_tokens": REWRITE_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        response = retry_chat_completion(client, kwargs)
        content = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - the rewrite must never break a run.
        return [question], f"ERROR: {exc}", False
    payload, ok = try_parse_json_object(content)
    raw_queries = payload.get("queries") if ok else None
    if not isinstance(raw_queries, list):
        return [question], content, False
    extras: list[str] = []
    seen = {question.strip().lower()}
    for item in raw_queries:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        extras.append(text)
        if len(extras) >= max_queries:
            break
    if not extras:
        return [question], content, False
    return [question, *extras], content, True


# ---------------------------------------------------------------------------
# Per-question retrieval measurement
# ---------------------------------------------------------------------------


def merge_query_hits(hits_per_query: list[list[Record]]) -> list[Record]:
    """Merge per-query hit lists, deduping by chunk id; best rank wins.

    Ties on rank are broken by search score, mirroring
    run_rag.search_with_decomposition's merge (but without truncation: the
    full merged list feeds rank_papers so every query's coverage counts).
    """
    merged: dict[str, tuple[int, Record]] = {}
    for hits in hits_per_query:
        for position, hit in enumerate(hits, start=1):
            key = run_rag._chunk_key(hit)  # noqa: SLF001 - canonical chunk identity.
            existing = merged.get(key)
            if existing is None or position < existing[0]:
                merged[key] = (position, hit)
    ordered = sorted(
        merged.values(),
        key=lambda pair: (pair[0], -float(pair[1].get("search_score") or 0.0)),
    )
    return [hit for _, hit in ordered]


def evaluate_question(
    sample: Record,
    queries: list[str],
    gold_ids: list[str],
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    top_chunks: int,
    recall_ks: list[int],
) -> Record:
    """Search every query, merge, rank papers, compare against gold."""
    hits_per_query: list[list[Record]] = []
    for query in queries:
        hits_per_query.append(
            run_rag.search_chunks(
                search_client,
                openai_client,
                openai_settings,
                query=query,
                top_chunks=top_chunks,
                vector_k=VECTOR_K,
                query_type="hybrid",
            )
        )
    pool = max(recall_ks)
    ranked = run_rag.rank_papers(merge_query_hits(hits_per_query), pool)
    rank_by_paper = {
        str(paper.get("paper_id") or ""): index
        for index, paper in enumerate(ranked, start=1)
    }
    first_query_index: dict[str, int] = {}
    for query_index, hits in enumerate(hits_per_query):
        for hit in hits:
            paper_id = str(hit.get("paper_id") or "")
            if paper_id and paper_id not in first_query_index:
                first_query_index[paper_id] = query_index

    found: list[Record] = []
    missing: list[str] = []
    for paper_id in gold_ids:
        rank = rank_by_paper.get(paper_id)
        if rank is None:
            missing.append(paper_id)
            continue
        query_index = first_query_index.get(paper_id, 0)
        found.append(
            {
                "paper_id": paper_id,
                "rank": rank,
                "first_query_index": query_index,
                "first_query": queries[query_index] if query_index < len(queries) else "",
            }
        )
    recall_at = {
        str(k): (
            sum(1 for item in found if int(item["rank"]) <= k) / len(gold_ids)
            if gold_ids
            else 0.0
        )
        for k in recall_ks
    }
    return {
        "query_id": str(sample.get("query_id") or ""),
        "task_family": str(sample.get("task_family") or ""),
        "question": str(sample.get("question") or ""),
        "queries": queries,
        "gold_papers": gold_ids,
        "found": found,
        "missing": missing,
        "recall_at": recall_at,
    }


def error_row(sample: Record, queries: list[str], gold_ids: list[str], recall_ks: list[int], error: str) -> Record:
    """Row for a question whose searches failed; scored as zero recall."""
    return {
        "query_id": str(sample.get("query_id") or ""),
        "task_family": str(sample.get("task_family") or ""),
        "question": str(sample.get("question") or ""),
        "queries": queries,
        "gold_papers": gold_ids,
        "found": [],
        "missing": list(gold_ids),
        "recall_at": {str(k): 0.0 for k in recall_ks},
        "error": error,
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def aggregate_rows(rows: list[Record], recall_ks: list[int]) -> Record:
    """Aggregate per-question rows: mean recall@K and full-coverage counts."""
    questions = len(rows)
    recall_at: dict[str, float] = {}
    fully_covered_at: dict[str, int] = {}
    for k in recall_ks:
        key = str(k)
        values = [float((row.get("recall_at") or {}).get(key) or 0.0) for row in rows]
        recall_at[key] = sum(values) / questions if questions else 0.0
        fully_covered_at[key] = sum(1 for value in values if value >= 1.0)
    return {
        "questions": questions,
        "recall_at": recall_at,
        "fully_covered_at": fully_covered_at,
        "rewrite_failures": sum(1 for row in rows if row.get("rewrite_failed")),
        "search_errors": sum(1 for row in rows if row.get("error")),
    }


def load_baseline_aggregates(name: str) -> Optional[Record]:
    """Aggregates of the _baseline run, when present and not this run itself."""
    if name == BASELINE_NAME:
        return None
    path = DEFAULT_OUTPUT_ROOT / BASELINE_NAME / "results.json"
    if not path.exists():
        return None
    try:
        payload = read_json(path)
        aggregates = payload.get("aggregates")
        return aggregates if isinstance(aggregates, dict) else None
    except Exception:  # noqa: BLE001 - a corrupt baseline never blocks a run.
        return None


def report_k(recall_ks: list[int]) -> int:
    """K used for the per-question found@K column (8 when measured)."""
    return 8 if 8 in recall_ks else recall_ks[len(recall_ks) // 2]


def _md_cell(text: Any, max_chars: int) -> str:
    return compact_text(text, max_chars=max_chars).replace("|", "\\|") or "-"


def aggregate_table_lines(
    aggregates: Record, baseline: Optional[Record], recall_ks: list[int]
) -> list[str]:
    """Markdown aggregate table, with baseline/delta columns when available."""
    lines: list[str] = []
    if baseline:
        lines.append("| K | recall@K | fully covered | baseline | delta |")
        lines.append("|---|---------|---------------|----------|-------|")
    else:
        lines.append("| K | recall@K | fully covered |")
        lines.append("|---|---------|---------------|")
    questions = int(aggregates.get("questions") or 0)
    for k in recall_ks:
        key = str(k)
        recall = float((aggregates.get("recall_at") or {}).get(key) or 0.0)
        covered = int((aggregates.get("fully_covered_at") or {}).get(key) or 0)
        row = f"| {k} | {recall:.3f} | {covered}/{questions} |"
        if baseline:
            base = (baseline.get("recall_at") or {}).get(key)
            if base is None:
                row += " n/a | n/a |"
            else:
                row += f" {float(base):.3f} | {recall - float(base):+.3f} |"
        lines.append(row)
    return lines


def render_results_md(
    name: str,
    args: argparse.Namespace,
    prompt_sha: str,
    rows: list[Record],
    aggregates: Record,
    baseline: Optional[Record],
    recall_ks: list[int],
) -> str:
    k_report = report_k(recall_ks)
    lines: list[str] = [f"# query_lab: {name}", ""]
    if args.baseline_only:
        lines.append("- mode: **baseline-only** (original-question single search)")
    elif args.rewrites_file:
        lines.append(f"- mode: reused rewrites from `{args.rewrites_file}`")
    else:
        lines.append(f"- prompt: `{args.prompt_file}` (sha256 `{prompt_sha[:12]}`)")
    lines.append(
        f"- questions: {aggregates['questions']} (family={args.family}), "
        f"rewrite failures: {aggregates['rewrite_failures']}, "
        f"search errors: {aggregates['search_errors']}"
    )
    lines.append(
        f"- paper pool: top {max(recall_ks)} (RRF over merged hits), "
        f"top-chunks per search: {args.top_chunks}, max extra queries: {args.max_queries}"
    )
    if baseline:
        lines.append(f"- baseline column: `runs/query_lab/{BASELINE_NAME}/results.json`")
        base_questions = int(baseline.get("questions") or 0)
        if base_questions != int(aggregates.get("questions") or 0):
            lines.append(
                f"- **WARNING**: baseline covers {base_questions} questions, this run "
                f"{aggregates['questions']} — the delta is not apples-to-apples "
                "(use the same --family/--limit/--query-id scope for real comparisons)"
            )
    lines += ["", "## Aggregate recall", ""]
    lines += aggregate_table_lines(aggregates, baseline, recall_ks)
    lines += ["", "## Per-question", ""]
    lines.append(f"| query_id | gold | found@{k_report} | missing | rewritten queries |")
    lines.append("|----------|------|---------|---------|-------------------|")
    for row in rows:
        gold_n = len(row.get("gold_papers") or [])
        found_k = sum(
            1 for item in row.get("found") or [] if int(item["rank"]) <= k_report
        )
        flags = ""
        if row.get("rewrite_failed"):
            flags = " (rewrite FAILED)"
        if row.get("error"):
            flags += " (search ERROR)"
        extras = (row.get("queries") or [])[1:]
        queries_cell = _md_cell(" / ".join(extras), QUERIES_CELL_CHARS) + flags
        lines.append(
            f"| {row['query_id']} | {gold_n} | {found_k}/{gold_n} | "
            f"{_md_cell(', '.join(row.get('missing') or []), MISSING_CELL_CHARS)} | "
            f"{queries_cell} |"
        )
    lines += [
        "",
        "For each found gold paper, `results.json` records its merged rank and "
        "which query surfaced it first (`found[].first_query`).",
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    """Atomic UTF-8 text write (temp sibling + replace), matching common.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_sibling(path)
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def print_console_summary(
    rows: list[Record],
    aggregates: Record,
    baseline: Optional[Record],
    recall_ks: list[int],
) -> None:
    questions = int(aggregates.get("questions") or 0)
    print(f"questions: {questions}")
    header = f"{'K':>4} {'recall@K':>9} {'covered':>9}"
    if baseline:
        header += f" {'baseline':>9} {'delta':>8}"
    print(header)
    for k in recall_ks:
        key = str(k)
        recall = float(aggregates["recall_at"][key])
        covered = int(aggregates["fully_covered_at"][key])
        line = f"{k:>4} {recall:>9.3f} {covered:>6}/{questions:<2}"
        if baseline:
            base = (baseline.get("recall_at") or {}).get(key)
            if base is None:
                line += f" {'n/a':>9} {'':>8}"
            else:
                line += f" {float(base):>9.3f} {recall - float(base):>+8.3f}"
        print(line)
    max_key = str(max(recall_ks))
    worst = sorted(
        rows,
        key=lambda row: (
            float((row.get("recall_at") or {}).get(max_key) or 0.0),
            str(row.get("query_id")),
        ),
    )[:5]
    print()
    print(f"5 worst questions by recall@{max_key}:")
    for row in worst:
        recall = float((row.get("recall_at") or {}).get(max_key) or 0.0)
        missing = compact_text(", ".join(row.get("missing") or []), max_chars=90)
        print(
            f"  {row['query_id']} recall={recall:.2f} "
            f"missing=[{missing}] q={compact_text(row.get('question'), max_chars=90)}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_recall_at(raw: str) -> list[int]:
    """'5,8,20' -> sorted unique positive ints."""
    values: set[int] = set()
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"--recall-at values must be positive, got {value}")
        values.add(value)
    if not values:
        raise ValueError("--recall-at must contain at least one integer")
    return sorted(values)


def load_gold_papers(path: Path) -> dict[str, list[str]]:
    """query_id -> ordered gold paper ids (papers only; never answers)."""
    gold: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        query_id = str(record.get("query_id") or "")
        if not query_id:
            continue
        papers = [
            str(paper.get("paper_id") or "")
            for paper in record.get("gold_papers") or []
            if isinstance(paper, dict) and paper.get("paper_id")
        ]
        if papers:
            gold[query_id] = papers
    return gold


def load_rewrites_file(path: Path) -> dict[str, list[str]]:
    """query_id -> queries from a previous run's rewrites.jsonl."""
    rewrites: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        query_id = str(record.get("query_id") or "")
        queries = [str(q) for q in record.get("queries") or [] if str(q or "").strip()]
        if query_id and queries:
            rewrites[query_id] = queries
    return rewrites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query-rewrite prompt lab (DEV-ONLY; reads gold papers to measure "
            "recall@K). Edit the prompt file, run with a fresh --name, read "
            "runs/query_lab/<name>/results.md."
        ),
        epilog=(
            "Baseline convention: run once with `--name _baseline "
            "--baseline-only` to write the original-question reference under "
            f"runs/query_lab/{BASELINE_NAME}/; every later run then shows a "
            "baseline column and delta in its results.md automatically."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Rewrite prompt template ('#' lines stripped, {question} required).",
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "validation_inputs.jsonl")
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "validation.jsonl",
        help="Gold JSONL; only gold_papers are read (recall measurement).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name; results land in runs/query_lab/<name>/.",
    )
    parser.add_argument(
        "--family",
        choices=["multi_paper", "hidden_source_single_paper", "all"],
        default="multi_paper",
        help="Filter questions by task_family.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        dest="query_ids",
        help="Run only these query_ids (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--top-chunks", type=int, default=24, help="Chunk hits fetched per search query."
    )
    parser.add_argument(
        "--recall-at",
        default="5,8,20",
        help="Comma-separated K values; the max also sets the ranked paper pool size.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip rewriting; measure the original-question single search.",
    )
    parser.add_argument(
        "--rewrites-file",
        type=Path,
        default=None,
        help="Reuse a previous run's rewrites.jsonl instead of calling AOAI.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=DEFAULT_MAX_QUERIES,
        help="Cap on extra rewritten queries per question (original is always kept).",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recall_ks = parse_recall_at(args.recall_at)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    load_environment(args.env_file)
    search_settings = SearchSettings.from_env()
    openai_settings = OpenAISettings.from_env()
    search_client = build_search_client(search_settings)
    openai_client = build_openai_client(openai_settings, max_retries=6)

    samples = read_jsonl(args.input)
    if args.family != "all":
        samples = [
            sample
            for sample in samples
            if str(sample.get("task_family") or "") == args.family
        ]
    if args.query_ids:
        wanted = {str(query_id) for query_id in args.query_ids}
        samples = [sample for sample in samples if str(sample.get("query_id")) in wanted]
    if args.limit:
        samples = samples[: args.limit]

    gold_by_id = load_gold_papers(args.gold)
    skipped_no_gold = [
        str(sample.get("query_id")) for sample in samples
        if str(sample.get("query_id")) not in gold_by_id
    ]
    if skipped_no_gold:
        print(
            f"WARN: skipping {len(skipped_no_gold)} question(s) without gold papers: "
            + ", ".join(skipped_no_gold),
            file=sys.stderr,
        )
    samples = [
        sample for sample in samples if str(sample.get("query_id")) in gold_by_id
    ]
    if not samples:
        print("ERROR: no questions left to evaluate", file=sys.stderr)
        return 1

    template = ""
    prompt_text = ""
    prompt_sha = ""
    use_rewrite_call = not args.baseline_only and args.rewrites_file is None
    if use_rewrite_call:
        prompt_text = args.prompt_file.read_text(encoding="utf-8")
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        template = load_prompt_template(args.prompt_file)

    reused_rewrites: dict[str, list[str]] = {}
    if args.rewrites_file is not None:
        reused_rewrites = load_rewrites_file(args.rewrites_file)

    out_dir = DEFAULT_OUTPUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_lock = threading.Lock()
    completed = 0

    def process_sample(sample: Record) -> tuple[Record, Record]:
        """Returns (rewrites row, results row) for one question."""
        nonlocal completed
        query_id = str(sample.get("query_id") or "")
        question = str(sample.get("question") or "")
        gold_ids = gold_by_id[query_id]
        rewrite_failed = False
        raw = ""
        if args.baseline_only:
            queries = [question]
        elif args.rewrites_file is not None:
            queries = reused_rewrites.get(query_id) or [question]
            if query_id not in reused_rewrites:
                rewrite_failed = True
                raw = f"missing from {args.rewrites_file}"
        else:
            queries, raw, ok = rewrite_queries(
                openai_client,
                openai_settings,
                template,
                sample,
                max_queries=args.max_queries,
            )
            rewrite_failed = not ok
        rewrite_row: Record = {"query_id": query_id, "queries": queries}
        if rewrite_failed:
            rewrite_row["rewrite_failed"] = True
            rewrite_row["raw"] = raw
        try:
            row = evaluate_question(
                sample,
                queries,
                gold_ids,
                search_client=search_client,
                openai_client=openai_client,
                openai_settings=openai_settings,
                top_chunks=args.top_chunks,
                recall_ks=recall_ks,
            )
        except Exception as exc:  # noqa: BLE001 - one bad question must not abort.
            print(f"FAIL {query_id}: {exc}", file=sys.stderr)
            row = error_row(sample, queries, gold_ids, recall_ks, str(exc))
        if rewrite_failed:
            row["rewrite_failed"] = True
        with progress_lock:
            completed += 1
            print(f"[{completed}/{len(samples)}] {query_id}", file=sys.stderr)
        return rewrite_row, row

    rewrite_rows: dict[str, Record] = {}
    result_rows: dict[str, Record] = {}
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in samples]
        for future in as_completed(futures):
            rewrite_row, row = future.result()
            rewrite_rows[str(rewrite_row["query_id"])] = rewrite_row
            result_rows[str(row["query_id"])] = row

    ordered_ids = [str(sample.get("query_id")) for sample in samples]
    rows = [result_rows[query_id] for query_id in ordered_ids]
    aggregates = aggregate_rows(rows, recall_ks)
    baseline = load_baseline_aggregates(args.name)

    write_jsonl(out_dir / "rewrites.jsonl", [rewrite_rows[qid] for qid in ordered_ids])
    write_json(
        out_dir / "results.json",
        {
            "name": args.name,
            "recall_at_ks": recall_ks,
            "aggregates": aggregates,
            "per_question": rows,
        },
    )
    write_text(
        out_dir / "results.md",
        render_results_md(
            args.name, args, prompt_sha, rows, aggregates, baseline, recall_ks
        ),
    )
    write_json(
        out_dir / "meta.json",
        {
            "prompt_file": str(args.prompt_file),
            "prompt_sha256": prompt_sha,
            "prompt_text": prompt_text,
            "args": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "counts": {
                "questions": aggregates["questions"],
                "rewrite_failures": aggregates["rewrite_failures"],
                "search_errors": aggregates["search_errors"],
                "skipped_no_gold": len(skipped_no_gold),
            },
        },
    )

    print_console_summary(rows, aggregates, baseline, recall_ks)
    print(f"\nwrote {out_dir / 'results.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
