"""Fail-closed adjudication and composition for table-only answer candidates.

This module keeps candidate generation separate from promotion: it renders
every table answer for source review, seals all inputs with hashes, and only
composes decisions that carry an explicit source check.  Composition replaces
``answer.table``—nothing else—from a selected candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from littraceqa.submission import OFFICIAL_SOURCE_TYPES, TOP_LEVEL_KEYS


Record = dict[str, Any]
CONTRACT_VERSION = "littraceqa-table-adjudication-v1"
SUPPORTED_ANSWER_TYPES = frozenset({"freeform", "multiple_choice", "table"})
SUPPORTED_COLUMN_TYPES = frozenset({"string", "number", "boolean"})
SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class AdjudicationError(ValueError):
    """Raised when an adjudication artifact violates the sealed contract."""


@dataclass(frozen=True)
class JsonlDocument:
    path: Path
    sha256: str
    records: tuple[Record, ...]
    raw_lines: tuple[bytes, ...]


@dataclass(frozen=True)
class CandidateDocument:
    source_id: str
    document: JsonlDocument
    coverage: str
    tables: dict[str, Record]


@dataclass(frozen=True)
class PreparedSources:
    inputs: JsonlDocument
    base: JsonlDocument
    candidates: tuple[CandidateDocument, ...]
    input_by_id: dict[str, Record]
    base_by_id: dict[str, Record]
    input_ids: tuple[str, ...]
    table_ids: tuple[str, ...]
    query_order_sha256: str
    table_schema_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise AdjudicationError(f"non-finite JSON constant is forbidden: {value}")


def _load_jsonl(path: Path) -> JsonlDocument:
    path = path.resolve()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AdjudicationError(f"cannot read JSONL: {path}: {exc}") from exc
    if not payload:
        raise AdjudicationError(f"JSONL is empty: {path}")

    raw_lines = tuple(payload.splitlines(keepends=True))
    records: list[Record] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise AdjudicationError(f"{path}:{line_number}: blank lines are forbidden")
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AdjudicationError(
                f"{path}:{line_number}: line is not UTF-8"
            ) from exc
        try:
            record = json.loads(text, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise AdjudicationError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise AdjudicationError(
                f"{path}:{line_number}: every line must be a JSON object"
            )
        records.append(record)
    return JsonlDocument(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        records=tuple(records),
        raw_lines=raw_lines,
    )


def _load_json(path: Path) -> Any:
    payload, _ = _load_json_with_sha256(path)
    return payload


def _load_json_with_sha256(path: Path) -> tuple[Any, str]:
    """Parse one JSON document and hash the exact bytes that were parsed."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AdjudicationError(f"cannot read JSON: {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdjudicationError(f"{path}: JSON is not UTF-8") from exc
    try:
        parsed = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise AdjudicationError(f"{path}: invalid JSON: {exc.msg}") from exc
    return parsed, hashlib.sha256(raw).hexdigest()


def _record_ids(document: JsonlDocument, label: str) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for position, record in enumerate(document.records, start=1):
        query_id = record.get("query_id")
        if (
            not isinstance(query_id, str)
            or not query_id
            or query_id != query_id.strip()
        ):
            raise AdjudicationError(
                f"{label} record {position}: query_id must be a canonical string"
            )
        if query_id in seen:
            raise AdjudicationError(f"{label}: duplicate query_id: {query_id}")
        seen.add(query_id)
        output.append(query_id)
    return tuple(output)


def _answer_types(record: Record) -> tuple[str, ...]:
    raw = record.get("answer_types")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item for item in raw)
    ):
        raise AdjudicationError(
            f"{record.get('query_id', '<unknown>')}: answer_types must be a non-empty string list"
        )
    answer_types = tuple(raw)
    if len(answer_types) != len(set(answer_types)):
        raise AdjudicationError(
            f"{record.get('query_id', '<unknown>')}: answer_types contains duplicates"
        )
    unsupported = sorted(set(answer_types) - SUPPORTED_ANSWER_TYPES)
    if unsupported:
        raise AdjudicationError(
            f"{record.get('query_id', '<unknown>')}: unsupported answer types: {unsupported}"
        )
    return answer_types


def _validate_schema(query_id: str, schema: Any) -> list[Record]:
    if not isinstance(schema, list) or not schema:
        raise AdjudicationError(f"{query_id}: table_schema must be a non-empty list")
    output: list[Record] = []
    seen_names: set[str] = set()
    for position, column in enumerate(schema):
        if not isinstance(column, dict):
            raise AdjudicationError(
                f"{query_id}: table_schema[{position}] must be an object"
            )
        name = column.get("name")
        column_type = column.get("type")
        is_row_key = column.get("is_row_key")
        if not isinstance(name, str) or not name or name != name.strip():
            raise AdjudicationError(
                f"{query_id}: table_schema[{position}].name must be canonical"
            )
        if name in seen_names:
            raise AdjudicationError(f"{query_id}: duplicate schema column: {name}")
        seen_names.add(name)
        if column_type not in SUPPORTED_COLUMN_TYPES:
            raise AdjudicationError(
                f"{query_id}: unsupported schema type for {name}: {column_type!r}"
            )
        if not isinstance(is_row_key, bool):
            raise AdjudicationError(
                f"{query_id}: is_row_key for {name} must be boolean"
            )
        output.append(copy.deepcopy(column))
    return output


def _normalize_row_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    return re.sub(r"\s+", " ", text)


def _validate_cell(query_id: str, column: Record, value: Any) -> None:
    if value is None:
        return
    column_type = str(column["type"])
    column_name = str(column["name"])
    if column_type == "string" and not isinstance(value, str):
        raise AdjudicationError(
            f"{query_id}: {column_name} must be a JSON string or null"
        )
    if column_type == "boolean" and not isinstance(value, bool):
        raise AdjudicationError(
            f"{query_id}: {column_name} must be JSON true/false or null"
        )
    if column_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdjudicationError(
                f"{query_id}: {column_name} must be a JSON number or null"
            )
        if not math.isfinite(float(value)):
            raise AdjudicationError(
                f"{query_id}: {column_name} must be finite"
            )


def _validated_table(query_id: str, table: Any, schema: Any) -> Record:
    columns = _validate_schema(query_id, schema)
    if not isinstance(table, dict) or set(table) != {"rows"}:
        raise AdjudicationError(
            f"{query_id}: answer.table must contain exactly the rows field"
        )
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AdjudicationError(f"{query_id}: answer.table.rows must be non-empty")

    column_names = [str(column["name"]) for column in columns]
    row_key_columns = [
        str(column["name"]) for column in columns if column["is_row_key"]
    ]
    effective_row_keys = row_key_columns or column_names[:1]
    columns_by_name = {str(column["name"]): column for column in columns}
    seen_keys: set[tuple[str, ...]] = set()
    output_rows: list[Record] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(column_names):
            raise AdjudicationError(
                f"{query_id}: table row {row_index} must contain exactly {column_names}"
            )
        for column_name in column_names:
            _validate_cell(query_id, columns_by_name[column_name], row[column_name])
        if any(row[column] in (None, "") for column in row_key_columns):
            raise AdjudicationError(
                f"{query_id}: table row {row_index} has an empty declared row key"
            )
        key = tuple(_normalize_row_key(row[column]) for column in effective_row_keys)
        if key in seen_keys:
            raise AdjudicationError(
                f"{query_id}: duplicate row key after official normalization: {key}"
            )
        seen_keys.add(key)
        output_rows.append(copy.deepcopy(row))
    return {"rows": output_rows}


def _extract_table(record: Record, input_record: Record, source_id: str) -> Record:
    query_id = str(input_record["query_id"])
    answer = record.get("answer")
    if not isinstance(answer, dict) or "table" not in answer:
        raise AdjudicationError(
            f"{source_id}:{query_id}: candidate has no answer.table"
        )
    return _validated_table(query_id, answer["table"], input_record["table_schema"])


def _validate_papers(query_id: str, record: Record) -> set[str]:
    papers = record.get("gold_papers")
    if not isinstance(papers, list) or not papers:
        raise AdjudicationError(f"base:{query_id}: gold_papers must be non-empty")
    output: set[str] = set()
    for item in papers:
        if not isinstance(item, dict) or set(item) != {"paper_id"}:
            raise AdjudicationError(
                f"base:{query_id}: each paper must contain only paper_id"
            )
        paper_id = item.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise AdjudicationError(f"base:{query_id}: invalid paper_id")
        if paper_id in output:
            raise AdjudicationError(
                f"base:{query_id}: duplicate paper_id: {paper_id}"
            )
        output.add(paper_id)
    return output


def _validate_evidence(query_id: str, record: Record, paper_ids: set[str]) -> None:
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AdjudicationError(f"base:{query_id}: evidence must be non-empty")
    for position, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {
            "paper_id",
            "source_type",
            "locator",
        }:
            raise AdjudicationError(
                f"base:{query_id}: evidence[{position}] has a non-official shape"
            )
        if item.get("paper_id") not in paper_ids:
            raise AdjudicationError(
                f"base:{query_id}: evidence[{position}] paper is not submitted"
            )
        if item.get("source_type") not in OFFICIAL_SOURCE_TYPES:
            raise AdjudicationError(
                f"base:{query_id}: evidence[{position}] has invalid source_type"
            )
        if not isinstance(item.get("locator"), dict) or not item["locator"]:
            raise AdjudicationError(
                f"base:{query_id}: evidence[{position}] locator is empty"
            )
    _validate_review_locator(
        query_id,
        evidence,
        paper_ids,
        label="base evidence",
    )


def _valid_option_labels(input_record: Record) -> set[str]:
    options = input_record.get("multiple_choice_options")
    if not isinstance(options, list) or not options:
        return set()
    labels = {
        str(option.get("label") or "").strip().upper()
        for option in options
        if isinstance(option, dict)
    }
    if not labels or "" in labels:
        raise AdjudicationError(
            f"{input_record['query_id']}: invalid multiple_choice_options"
        )
    return labels


def _validate_base_record(input_record: Record, base_record: Record) -> None:
    query_id = str(input_record["query_id"])
    if set(base_record) != set(TOP_LEVEL_KEYS):
        raise AdjudicationError(
            f"base:{query_id}: top-level fields must be exactly {sorted(TOP_LEVEL_KEYS)}"
        )
    paper_ids = _validate_papers(query_id, base_record)
    _validate_evidence(query_id, base_record, paper_ids)

    answer_types = _answer_types(input_record)
    answer = base_record.get("answer")
    if not isinstance(answer, dict) or set(answer) != set(answer_types):
        raise AdjudicationError(
            f"base:{query_id}: answer keys must exactly match {list(answer_types)}"
        )
    if "freeform" in answer_types:
        freeform = answer.get("freeform")
        if (
            not isinstance(freeform, dict)
            or set(freeform) != {"text"}
            or not isinstance(freeform.get("text"), str)
            or not freeform["text"].strip()
        ):
            raise AdjudicationError(f"base:{query_id}: invalid freeform answer")
    if "multiple_choice" in answer_types:
        multiple_choice = answer.get("multiple_choice")
        if not isinstance(multiple_choice, dict) or set(multiple_choice) != {"gold"}:
            raise AdjudicationError(
                f"base:{query_id}: invalid multiple_choice answer shape"
            )
        label = multiple_choice.get("gold")
        labels = _valid_option_labels(input_record)
        if not isinstance(label, str) or not label or (labels and label not in labels):
            raise AdjudicationError(
                f"base:{query_id}: multiple_choice label is not released"
            )
    if "table" in answer_types:
        _validated_table(query_id, answer.get("table"), input_record["table_schema"])


def _prepare_sources(
    inputs_path: Path,
    base_path: Path,
    candidate_specs: Sequence[tuple[str, Path]],
) -> PreparedSources:
    inputs = _load_jsonl(inputs_path)
    base = _load_jsonl(base_path)
    input_ids = _record_ids(inputs, "inputs")
    base_ids = _record_ids(base, "base")
    if base_ids != input_ids:
        if set(base_ids) == set(input_ids):
            raise AdjudicationError("base query order differs from official inputs")
        raise AdjudicationError("base query_id coverage differs from official inputs")

    input_by_id = dict(zip(input_ids, inputs.records, strict=True))
    base_by_id = dict(zip(base_ids, base.records, strict=True))
    table_ids: list[str] = []
    schema_payload: list[Record] = []
    for query_id in input_ids:
        input_record = input_by_id[query_id]
        answer_types = _answer_types(input_record)
        if "table" in answer_types:
            schema = _validate_schema(query_id, input_record.get("table_schema"))
            table_ids.append(query_id)
            schema_payload.append({"query_id": query_id, "table_schema": schema})
        _validate_base_record(input_record, base_by_id[query_id])
    if not table_ids:
        raise AdjudicationError("official inputs contain no table queries")

    candidate_documents: list[CandidateDocument] = []
    used_source_ids = {"base"}
    for source_id, path in candidate_specs:
        if source_id in used_source_ids:
            raise AdjudicationError(f"duplicate or reserved candidate id: {source_id}")
        if not SOURCE_ID_RE.fullmatch(source_id):
            raise AdjudicationError(
                f"invalid candidate id {source_id!r}; use letters, digits, dot, dash, underscore"
            )
        used_source_ids.add(source_id)
        document = _load_jsonl(path)
        candidate_ids = _record_ids(document, source_id)
        if candidate_ids == input_ids:
            coverage = "full"
        elif candidate_ids == tuple(table_ids):
            coverage = "table_only"
        elif set(candidate_ids) in (set(input_ids), set(table_ids)):
            raise AdjudicationError(
                f"{source_id}: candidate query order differs from official inputs"
            )
        else:
            raise AdjudicationError(
                f"{source_id}: candidate must contain exactly all queries or all table queries"
            )
        records_by_id = dict(zip(candidate_ids, document.records, strict=True))
        tables = {
            query_id: _extract_table(
                records_by_id[query_id], input_by_id[query_id], source_id
            )
            for query_id in table_ids
        }
        candidate_documents.append(
            CandidateDocument(
                source_id=source_id,
                document=document,
                coverage=coverage,
                tables=tables,
            )
        )
    if not candidate_documents:
        raise AdjudicationError("at least one --candidate is required")

    return PreparedSources(
        inputs=inputs,
        base=base,
        candidates=tuple(candidate_documents),
        input_by_id=input_by_id,
        base_by_id=base_by_id,
        input_ids=input_ids,
        table_ids=tuple(table_ids),
        query_order_sha256=_canonical_sha256(list(input_ids)),
        table_schema_sha256=_canonical_sha256(schema_payload),
    )


def _row_key_values(table: Record, schema: list[Record]) -> list[list[Any]]:
    row_keys = [
        str(column["name"])
        for column in schema
        if isinstance(column, dict) and column.get("is_row_key") is True
    ]
    if not row_keys:
        row_keys = [str(schema[0]["name"])]
    return [[row.get(column) for column in row_keys] for row in table["rows"]]


def _source_contract(prepared: PreparedSources) -> list[Record]:
    return [
        {
            "source_id": candidate.source_id,
            "sha256": candidate.document.sha256,
            "coverage": candidate.coverage,
        }
        for candidate in prepared.candidates
    ]


def _review_payload(prepared: PreparedSources) -> Record:
    queries: list[Record] = []
    for query_id in prepared.table_ids:
        input_record = prepared.input_by_id[query_id]
        schema = input_record["table_schema"]
        base_table = copy.deepcopy(
            prepared.base_by_id[query_id]["answer"]["table"]
        )
        versions: list[Record] = [
            {
                "source_id": "base",
                "table_sha256": _canonical_sha256(base_table),
                "row_count": len(base_table["rows"]),
                "row_keys": _row_key_values(base_table, schema),
                "table": base_table,
            }
        ]
        for candidate in prepared.candidates:
            table = copy.deepcopy(candidate.tables[query_id])
            versions.append(
                {
                    "source_id": candidate.source_id,
                    "table_sha256": _canonical_sha256(table),
                    "row_count": len(table["rows"]),
                    "row_keys": _row_key_values(table, schema),
                    "table": table,
                }
            )
        groups: dict[str, list[str]] = {}
        for version in versions:
            groups.setdefault(version["table_sha256"], []).append(
                version["source_id"]
            )
        queries.append(
            {
                "query_id": query_id,
                "question": input_record.get("question", ""),
                "answer_types": copy.deepcopy(input_record["answer_types"]),
                "table_schema": copy.deepcopy(schema),
                "table_schema_sha256": _canonical_sha256(schema),
                "frozen_gold_papers": copy.deepcopy(
                    prepared.base_by_id[query_id]["gold_papers"]
                ),
                "frozen_evidence": copy.deepcopy(
                    prepared.base_by_id[query_id]["evidence"]
                ),
                "frozen_context_sha256": _frozen_context_sha256(
                    prepared.base_by_id[query_id]
                ),
                "versions": versions,
                "identical_groups": list(groups.values()),
            }
        )
    return {
        "version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "path": str(prepared.inputs.path),
            "sha256": prepared.inputs.sha256,
        },
        "base": {
            "path": str(prepared.base.path),
            "sha256": prepared.base.sha256,
        },
        "query_order_sha256": prepared.query_order_sha256,
        "table_schema_sha256": prepared.table_schema_sha256,
        "table_query_count": len(prepared.table_ids),
        "candidates": _source_contract(prepared),
        "queries": queries,
    }


def _decision_template(prepared: PreparedSources) -> Record:
    return {
        "version": CONTRACT_VERSION,
        "inputs_sha256": prepared.inputs.sha256,
        "base_sha256": prepared.base.sha256,
        "query_order_sha256": prepared.query_order_sha256,
        "table_schema_sha256": prepared.table_schema_sha256,
        "candidates": _source_contract(prepared),
        "decisions": [
            {
                "query_id": query_id,
                "selected_candidate": "base",
                "source_checked": False,
                "notes": "",
                "locator": [],
            }
            for query_id in prepared.table_ids
        ],
    }


def _markdown_json_block(value: Any) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", payload)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return [f"{fence}json", payload, fence]


def _markdown_inline_code(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False)
    escaped = html.escape(payload, quote=True).replace("|", "&#124;")
    return f"<code>{escaped}</code>"


def _markdown_review(review: Record) -> str:
    lines = [
        "# LitTraceQA table answer adjudication",
        "",
        f"- Contract: `{review['version']}`",
        f"- Inputs SHA-256: `{review['inputs']['sha256']}`",
        f"- Base SHA-256: `{review['base']['sha256']}`",
        f"- Table queries: {review['table_query_count']}",
        "",
        "Copy the decision template to a new file and edit the copy, not this "
        "report or the generated template. Every table decision must "
        "set `source_checked=true`, explain the choice in `notes`, and list at "
        "least one locator shown in its frozen evidence block.",
        "",
    ]
    for query in review["queries"]:
        lines.extend(
            [
                f"## Query {_markdown_inline_code(query['query_id'])}",
                "",
                "Question:",
                "",
                *_markdown_json_block(query["question"]),
                "",
                "Schema:",
                "",
                *_markdown_json_block(query["table_schema"]),
                "",
                "Frozen papers and evidence (the decision locator must be an exact "
                "member of `frozen_evidence`):",
                "",
                *_markdown_json_block(
                    {
                        "gold_papers": query["frozen_gold_papers"],
                        "evidence": query["frozen_evidence"],
                    }
                ),
                "",
                "| source | rows | table SHA-256 | row keys |",
                "|---|---:|---|---|",
            ]
        )
        for version in query["versions"]:
            row_keys = _markdown_inline_code(version["row_keys"])
            lines.append(
                f"| `{version['source_id']}` | {version['row_count']} | "
                f"`{version['table_sha256']}` | {row_keys} |"
            )
        lines.append("")
        emitted_hashes: set[str] = set()
        for version in query["versions"]:
            table_hash = version["table_sha256"]
            if table_hash in emitted_hashes:
                continue
            emitted_hashes.add(table_hash)
            same_sources = next(
                group
                for group in query["identical_groups"]
                if version["source_id"] in group
            )
            lines.extend(
                [
                    f"### {' / '.join(same_sources)}",
                    "",
                    *_markdown_json_block(version["table"]),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _absolute_unresolved(path: Path) -> Path:
    """Return an absolute path without following a final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _portable_path_key(path: Path) -> str:
    """Normalize a path conservatively for case-insensitive filesystems."""

    absolute = _absolute_unresolved(path)
    return unicodedata.normalize("NFC", str(absolute)).casefold()


def _paths_may_alias(left: Path, right: Path) -> bool:
    if _portable_path_key(left) == _portable_path_key(right):
        return True
    if _portable_path_key(left.resolve(strict=False)) == _portable_path_key(
        right.resolve(strict=False)
    ):
        return True
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return False


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(_absolute_unresolved(path)))


def _cleanup_temporary(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # A failed cleanup must not hide the original error or invalidate a
        # successfully published artifact. A later housekeeping pass may remove
        # the uniquely named private temporary file.
        pass


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path = _absolute_unresolved(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        _cleanup_temporary(temporary)
        raise


def _fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(_absolute_unresolved(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_new(temporary: Path, path: Path) -> tuple[int, int]:
    """Publish a staged file without ever replacing an existing artifact."""

    path = _absolute_unresolved(path)
    stat = temporary.lstat()
    identity = stat.st_dev, stat.st_ino
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        raise AdjudicationError(f"artifact already exists: {path}") from exc
    return identity


def _unlink_if_owned(path: Path, identity: tuple[int, int] | None) -> bool:
    if identity is None:
        return True
    try:
        stat = _absolute_unresolved(path).lstat()
    except FileNotFoundError:
        return True
    if (stat.st_dev, stat.st_ino) == identity:
        absolute = _absolute_unresolved(path)
        absolute.unlink()
        try:
            _fsync_parent_directory(absolute)
        except OSError:
            return False
    return True


def _encoded_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def create_review(
    *,
    inputs_path: Path,
    base_path: Path,
    candidate_specs: Sequence[tuple[str, Path]],
    output_dir: Path,
) -> Record:
    prepared = _prepare_sources(inputs_path, base_path, candidate_specs)
    output_dir = _absolute_unresolved(output_dir)
    review = _review_payload(prepared)
    template = _decision_template(prepared)
    review_json = output_dir / "table_review.json"
    review_markdown = output_dir / "table_review.md"
    decision_template = output_dir / "table_decisions.template.json"
    artifacts = (review_json, review_markdown, decision_template)
    protected_paths = (
        prepared.inputs.path,
        prepared.base.path,
        *(candidate.document.path for candidate in prepared.candidates),
    )
    if any(
        _paths_may_alias(path, protected)
        for path in artifacts
        for protected in protected_paths
    ):
        raise AdjudicationError("review artifact path must not overwrite an input")
    if any(
        _paths_may_alias(left, right)
        for index, left in enumerate(artifacts)
        for right in artifacts[index + 1 :]
    ):
        raise AdjudicationError("review artifact paths must be distinct")
    existing = [str(path) for path in artifacts if _path_lexists(path)]
    if existing:
        raise AdjudicationError(
            "review artifact already exists; use a new output directory: "
            + ", ".join(existing)
        )
    artifact_payloads = (
        (review_json, _encoded_json(review)),
        (review_markdown, _markdown_review(review).encode("utf-8")),
        (decision_template, _encoded_json(template)),
    )
    staged: list[tuple[Path, Path, tuple[int, int]]] = []
    try:
        for path, payload in artifact_payloads:
            temporary = _stage_bytes(path, payload)
            stat = temporary.lstat()
            staged.append((path, temporary, (stat.st_dev, stat.st_ino)))
    except BaseException:
        for _path, temporary, _identity in staged:
            _cleanup_temporary(temporary)
        raise
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for path, temporary, identity in staged:
            # Register ownership before linking so even a post-link interrupt is
            # rolled back using the staged inode identity.
            created.append((path, identity))
            _publish_staged_new(temporary, path)
            _fsync_parent_directory(path)
    except BaseException as original_error:
        cleanup_error: BaseException | None = None
        for path, identity in reversed(created):
            try:
                _unlink_if_owned(path, identity)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise cleanup_error from original_error
        raise
    finally:
        for _path, temporary, _identity in staged:
            _cleanup_temporary(temporary)
    return {
        "review_json": str(review_json),
        "review_markdown": str(review_markdown),
        "decision_template": str(decision_template),
        "table_query_count": len(prepared.table_ids),
    }


def _validate_review_locator(
    query_id: str,
    value: Any,
    papers: set[str],
    *,
    label: str = "decision",
) -> None:
    prefix = f"{label}:{query_id}"
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise AdjudicationError(
            f"{prefix}: locator must be an object or non-empty list"
        )
    if not items:
        raise AdjudicationError(f"{prefix}: locator is required")
    for position, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {
            "paper_id",
            "source_type",
            "locator",
        }:
            raise AdjudicationError(
                f"{prefix}: locator[{position}] must use official evidence fields"
            )
        paper_id = item.get("paper_id")
        if paper_id not in papers:
            raise AdjudicationError(
                f"{prefix}: locator[{position}] paper is not in frozen gold_papers"
            )
        if item.get("source_type") not in OFFICIAL_SOURCE_TYPES:
            raise AdjudicationError(
                f"{prefix}: locator[{position}] has invalid source_type"
            )
        locator = item.get("locator")
        if not isinstance(locator, dict) or not locator:
            raise AdjudicationError(
                f"{prefix}: locator[{position}].locator is empty"
            )
        source_type = str(item["source_type"])
        page = locator.get("page")
        page_valid = isinstance(page, int) and not isinstance(page, bool) and page >= 1
        section_valid = isinstance(locator.get("section"), str) and bool(
            locator["section"].strip()
        )
        location_valid = page_valid or section_valid
        allowed_keys = {"page", "section"}
        if source_type == "table":
            allowed_keys.add("table_id")
            object_valid = isinstance(locator.get("table_id"), str) and bool(
                locator["table_id"].strip()
            )
            location_valid = location_valid and object_valid
        elif source_type == "figure":
            allowed_keys.add("figure_id")
            object_valid = isinstance(locator.get("figure_id"), str) and bool(
                locator["figure_id"].strip()
            )
            location_valid = location_valid and object_valid
        elif source_type == "equation_algorithm":
            allowed_keys.update({"equation_id", "algorithm_id"})
            location_valid = location_valid or any(
                isinstance(locator.get(key), str) and bool(locator[key].strip())
                for key in ("equation_id", "algorithm_id")
            )
        elif source_type == "citation_context":
            allowed_keys.add("citation_id")
            location_valid = location_valid or (
                isinstance(locator.get("citation_id"), str)
                and bool(locator["citation_id"].strip())
            )
        if not set(locator).issubset(allowed_keys):
            raise AdjudicationError(
                f"{prefix}: locator[{position}] has non-official fields"
            )
        if not location_valid:
            raise AdjudicationError(
                f"{prefix}: locator[{position}] is incomplete for {source_type}"
            )


def _load_decisions(
    path: Path, prepared: PreparedSources
) -> tuple[list[Record], str]:
    payload, payload_sha256 = _load_json_with_sha256(path)
    if not isinstance(payload, dict):
        raise AdjudicationError("decision file must contain one JSON object")
    required_keys = {
        "version",
        "inputs_sha256",
        "base_sha256",
        "query_order_sha256",
        "table_schema_sha256",
        "candidates",
        "decisions",
    }
    if set(payload) != required_keys:
        raise AdjudicationError(
            f"decision file keys must be exactly {sorted(required_keys)}"
        )
    sealed_values = {
        "version": CONTRACT_VERSION,
        "inputs_sha256": prepared.inputs.sha256,
        "base_sha256": prepared.base.sha256,
        "query_order_sha256": prepared.query_order_sha256,
        "table_schema_sha256": prepared.table_schema_sha256,
        "candidates": _source_contract(prepared),
    }
    for key, expected in sealed_values.items():
        if payload.get(key) != expected:
            raise AdjudicationError(f"decision sealed field mismatch: {key}")

    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise AdjudicationError("decision decisions must be a list")
    decision_ids = [
        decision.get("query_id") if isinstance(decision, dict) else None
        for decision in decisions
    ]
    if tuple(decision_ids) != prepared.table_ids:
        if set(decision_ids) == set(prepared.table_ids):
            raise AdjudicationError("decision query order differs from official inputs")
        raise AdjudicationError("decision query coverage differs from table queries")

    valid_sources = {"base", *(item.source_id for item in prepared.candidates)}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
            "query_id",
            "selected_candidate",
            "source_checked",
            "notes",
            "locator",
        }:
            raise AdjudicationError(
                "each decision must contain query_id, selected_candidate, "
                "source_checked, notes, locator"
            )
        query_id = str(decision["query_id"])
        if decision.get("selected_candidate") not in valid_sources:
            raise AdjudicationError(
                f"decision:{query_id}: unknown selected_candidate"
            )
        if decision.get("source_checked") is not True:
            raise AdjudicationError(
                f"decision:{query_id}: source_checked must be true"
            )
        notes = decision.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            raise AdjudicationError(f"decision:{query_id}: notes are required")
        papers = {
            str(item["paper_id"])
            for item in prepared.base_by_id[query_id]["gold_papers"]
        }
        decision_locator = decision.get("locator")
        _validate_review_locator(query_id, decision_locator, papers)
        locator_items = (
            [decision_locator]
            if isinstance(decision_locator, dict)
            else decision_locator
        )
        frozen_evidence = {
            _canonical_json(item)
            for item in prepared.base_by_id[query_id]["evidence"]
        }
        if any(
            _canonical_json(item) not in frozen_evidence
            for item in locator_items
        ):
            raise AdjudicationError(
                f"decision:{query_id}: every review locator must be present "
                "in frozen evidence"
            )
    return copy.deepcopy(decisions), payload_sha256


def _frozen_context(record: Record) -> Record:
    answer = record.get("answer")
    other_answers = {
        key: copy.deepcopy(value)
        for key, value in answer.items()
        if key != "table"
    }
    return {
        "query_id": record.get("query_id"),
        "gold_papers": copy.deepcopy(record.get("gold_papers")),
        "evidence": copy.deepcopy(record.get("evidence")),
        "other_answers": other_answers,
    }


def _frozen_context_sha256(record: Record) -> str:
    return _canonical_sha256(_frozen_context(record))


def _line_ending(raw_line: bytes) -> bytes:
    if raw_line.endswith(b"\r\n"):
        return b"\r\n"
    if raw_line.endswith(b"\n"):
        return b"\n"
    if raw_line.endswith(b"\r"):
        return b"\r"
    return b""


def _selected_table(
    prepared: PreparedSources, query_id: str, selected_candidate: str
) -> Record:
    if selected_candidate == "base":
        return copy.deepcopy(prepared.base_by_id[query_id]["answer"]["table"])
    for candidate in prepared.candidates:
        if candidate.source_id == selected_candidate:
            return copy.deepcopy(candidate.tables[query_id])
    raise AssertionError(f"validated candidate disappeared: {selected_candidate}")


def compose_submission(
    *,
    inputs_path: Path,
    base_path: Path,
    candidate_specs: Sequence[tuple[str, Path]],
    decisions_path: Path,
    output_path: Path,
    audit_path: Path | None = None,
) -> Record:
    prepared = _prepare_sources(inputs_path, base_path, candidate_specs)
    decisions, decisions_sha256 = _load_decisions(decisions_path, prepared)
    output_path = _absolute_unresolved(output_path)
    protected_paths = (
        prepared.inputs.path,
        prepared.base.path,
        decisions_path.resolve(),
        *(candidate.document.path for candidate in prepared.candidates),
    )
    if any(_paths_may_alias(output_path, path) for path in protected_paths):
        raise AdjudicationError("output path must not overwrite an input artifact")
    if audit_path is None:
        audit_path = output_path.with_suffix(".audit.json")
    audit_path = _absolute_unresolved(audit_path)
    if any(
        _paths_may_alias(audit_path, path) for path in protected_paths
    ) or _paths_may_alias(audit_path, output_path):
        raise AdjudicationError("audit path must be a distinct new artifact")
    if _path_lexists(output_path):
        raise AdjudicationError(f"output already exists: {output_path}")
    if _path_lexists(audit_path):
        raise AdjudicationError(f"audit already exists: {audit_path}")

    decisions_by_id = {str(item["query_id"]): item for item in decisions}
    output_records: list[Record] = []
    output_lines: list[bytes] = []
    changed_query_ids: list[str] = []
    decision_audit: list[Record] = []
    for position, query_id in enumerate(prepared.input_ids):
        base_record = prepared.base_by_id[query_id]
        output_record = copy.deepcopy(base_record)
        if query_id in decisions_by_id:
            decision = decisions_by_id[query_id]
            selected_table = _selected_table(
                prepared, query_id, str(decision["selected_candidate"])
            )
            output_record["answer"]["table"] = selected_table
            if _frozen_context(output_record) != _frozen_context(base_record):
                raise AssertionError(f"frozen context changed for {query_id}")
            before_hash = _canonical_sha256(base_record["answer"]["table"])
            after_hash = _canonical_sha256(selected_table)
            changed = before_hash != after_hash
            if changed:
                changed_query_ids.append(query_id)
            decision_audit.append(
                {
                    "query_id": query_id,
                    "selected_candidate": decision["selected_candidate"],
                    "source_checked": True,
                    "notes": decision["notes"],
                    "locator": decision["locator"],
                    "before_table_sha256": before_hash,
                    "after_table_sha256": after_hash,
                    "frozen_context_sha256": _frozen_context_sha256(base_record),
                    "changed": changed,
                }
            )
        elif output_record != base_record:
            raise AssertionError(f"non-table record changed for {query_id}")

        raw_base_line = prepared.base.raw_lines[position]
        if _canonical_json(output_record) == _canonical_json(base_record):
            raw_output_line = raw_base_line
        else:
            raw_output_line = json.dumps(
                output_record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + _line_ending(raw_base_line)
        output_records.append(output_record)
        output_lines.append(raw_output_line)

    # Redundant postconditions are intentional: this is the final promotion gate.
    for query_id, base_record, output_record in zip(
        prepared.input_ids, prepared.base.records, output_records, strict=True
    ):
        if output_record["query_id"] != base_record["query_id"]:
            raise AssertionError(f"query_id changed for {query_id}")
        if output_record["gold_papers"] != base_record["gold_papers"]:
            raise AssertionError(f"gold_papers changed for {query_id}")
        if output_record["evidence"] != base_record["evidence"]:
            raise AssertionError(f"evidence changed for {query_id}")
        if _frozen_context(output_record) != _frozen_context(base_record):
            raise AssertionError(f"non-table answer changed for {query_id}")
        if query_id not in prepared.table_ids and output_record != base_record:
            raise AssertionError(f"non-table record changed for {query_id}")

    staged_output = _stage_bytes(output_path, b"".join(output_lines))
    staged_audit: Path | None = None
    output_identity: tuple[int, int] | None = None
    audit_identity: tuple[int, int] | None = None
    try:
        written = _load_jsonl(staged_output)
        if tuple(_canonical_json(item) for item in written.records) != tuple(
            _canonical_json(item) for item in output_records
        ):
            raise AssertionError("written output does not round-trip semantically")
        if _record_ids(written, "output") != prepared.input_ids:
            raise AssertionError("written output query order changed")
        non_table_positions = [
            index
            for index, query_id in enumerate(prepared.input_ids)
            if query_id not in prepared.table_ids
        ]
        if any(
            written.raw_lines[index] != prepared.base.raw_lines[index]
            for index in non_table_positions
        ):
            raise AssertionError("a non-table raw JSONL line changed")

        audit = {
            "version": CONTRACT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inputs_sha256": prepared.inputs.sha256,
            "base_sha256": prepared.base.sha256,
            "decisions_sha256": decisions_sha256,
            "candidates": _source_contract(prepared),
            "query_order_sha256": prepared.query_order_sha256,
            "table_schema_sha256": prepared.table_schema_sha256,
            "output": {
                "path": str(output_path),
                "sha256": written.sha256,
                "records": len(written.records),
            },
            "changed_query_ids": changed_query_ids,
            "freeze_checks": {
                "query_order_unchanged": True,
                "non_table_records_semantically_identical": True,
                "non_table_raw_lines_byte_identical": True,
                "all_gold_papers_unchanged": True,
                "all_evidence_unchanged": True,
                "all_non_table_answer_components_unchanged": True,
                "only_answer_table_may_change": True,
            },
            "decisions": decision_audit,
        }
        staged_audit = _stage_bytes(audit_path, _encoded_json(audit))
        # The audit is durably published first. The output is the commit marker:
        # whenever it is durable and visible, its corresponding audit is too.
        audit_stat = staged_audit.lstat()
        audit_identity = audit_stat.st_dev, audit_stat.st_ino
        _publish_staged_new(staged_audit, audit_path)
        _fsync_parent_directory(audit_path)
        output_stat = staged_output.lstat()
        output_identity = output_stat.st_dev, output_stat.st_ino
        _publish_staged_new(staged_output, output_path)
        _fsync_parent_directory(output_path)
    except BaseException as original_error:
        _cleanup_temporary(staged_output)
        _cleanup_temporary(staged_audit)
        try:
            output_cleanup_durable = _unlink_if_owned(
                output_path, output_identity
            )
        except BaseException as cleanup_error:
            # If the output cannot be removed, retain its already-published audit.
            raise cleanup_error from original_error
        if not output_cleanup_durable:
            raise AdjudicationError(
                "output cleanup could not be durably confirmed; retaining audit"
            ) from original_error
        _unlink_if_owned(audit_path, audit_identity)
        raise
    _cleanup_temporary(staged_output)
    _cleanup_temporary(staged_audit)
    return {
        "output": str(output_path),
        "output_sha256": written.sha256,
        "audit": str(audit_path),
        "changed_query_ids": changed_query_ids,
    }


def _parse_candidate_specs(values: Iterable[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    used: set[str] = set()
    for raw in values:
        if "=" in raw:
            source_id, raw_path = raw.split("=", 1)
        else:
            raw_path = raw
            source_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(raw).stem)
        source_id = source_id.strip()
        raw_path = raw_path.strip()
        if not source_id or not raw_path:
            raise AdjudicationError(
                f"invalid --candidate {raw!r}; use SOURCE_ID=PATH"
            )
        if source_id in used:
            raise AdjudicationError(f"duplicate candidate id: {source_id}")
        used.add(source_id)
        output.append((source_id, Path(raw_path)))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review and safely compose LitTraceQA table-only reruns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_sources(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--inputs", type=Path, required=True)
        subparser.add_argument("--base", type=Path, required=True)
        subparser.add_argument(
            "--candidate",
            action="append",
            required=True,
            metavar="SOURCE_ID=PATH",
            help="Repeat for each full or table-only candidate JSONL.",
        )

    review = subparsers.add_parser(
        "review", help="Generate JSON/Markdown comparison and decision template."
    )
    add_sources(review)
    review.add_argument("--output-dir", type=Path, required=True)

    compose = subparsers.add_parser(
        "compose", help="Compose a reviewed table-only promotion."
    )
    add_sources(compose)
    compose.add_argument("--decisions", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--audit", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidate_specs = _parse_candidate_specs(args.candidate)
        if args.command == "review":
            result = create_review(
                inputs_path=args.inputs,
                base_path=args.base,
                candidate_specs=candidate_specs,
                output_dir=args.output_dir,
            )
        else:
            result = compose_submission(
                inputs_path=args.inputs,
                base_path=args.base,
                candidate_specs=candidate_specs,
                decisions_path=args.decisions,
                output_path=args.output,
                audit_path=args.audit,
            )
    except AdjudicationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
