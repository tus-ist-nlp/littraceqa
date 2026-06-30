"""Shared helpers for the LitTraceQA Azure pipeline scripts."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, TypeVar


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "data" / "paper_metadata.jsonl"


Record = dict[str, Any]
T = TypeVar("T")


def read_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def append_jsonl(path: Path, record: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temp_path.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Path = DEFAULT_METADATA) -> list[Record]:
    return read_jsonl(path)


def load_metadata_by_id(path: Path = DEFAULT_METADATA) -> dict[str, Record]:
    return {
        str(record["paper_id"]): record
        for record in load_metadata(path)
        if record.get("paper_id")
    }


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_text(value: Any, *, max_chars: int = 0) -> str:
    text = re.sub(r"\s+", " ", clean_text(value))
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def parse_json_object(text: Any) -> Record:
    """Best-effort JSON object parser for model responses."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def normalize_alias(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def azure_document_key(*parts: Any) -> str:
    raw = ":".join(str(part) for part in parts)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def retry_chat_completion(
    client: Any,
    kwargs: Record,
    *,
    attempts: int = 4,
) -> Any:
    """Call chat completions while adapting to Azure/OpenAI model quirks.

    Some deployments reject ``max_tokens`` in favor of ``max_completion_tokens``;
    others do not support ``response_format`` or fixed ``temperature``. The caller
    supplies normal chat kwargs and this helper removes or renames only when the
    service explicitly asks for it.
    """
    request = dict(kwargs)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - SDK error classes vary.
            last_error = exc
            message = str(exc)
            lower = message.lower()
            changed = False
            if "max_tokens" in message and "max_completion_tokens" in message:
                if "max_tokens" in request:
                    request["max_completion_tokens"] = request.pop("max_tokens")
                    changed = True
            if "response_format" in message and "unsupported" in lower:
                if request.pop("response_format", None) is not None:
                    changed = True
            if "temperature" in message and "unsupported" in lower:
                if request.pop("temperature", None) is not None:
                    changed = True
            if not changed:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion was not attempted")
