"""scripts/merge_chunks.py の結合ロジックのテスト（実際にCLIとして起動して検証）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "merge_chunks.py"


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def _chunk(chunk_id: str, paper_id: str = "p1") -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "text": f"text of {chunk_id}",
        "chunk_type": "text_span",
        "metadata": {},
    }


def _run_merge(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_merges_multiple_files_without_duplicates(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(a, [_chunk("p1#c0000"), _chunk("p1#c0001")])
    _write_jsonl(b, [_chunk("p1#fig0001")])

    result = _run_merge(["--inputs", str(a), str(b), "--output", str(output)])

    assert result.returncode == 0, result.stderr
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    chunk_ids = {json.loads(line)["chunk_id"] for line in lines}
    assert chunk_ids == {"p1#c0000", "p1#c0001", "p1#fig0001"}


def test_duplicate_chunk_id_warns_and_keeps_first_by_default(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(a, [_chunk("p1#c0000")])
    _write_jsonl(b, [_chunk("p1#c0000")])  # 重複

    result = _run_merge(["--inputs", str(a), str(b), "--output", str(output)])

    assert result.returncode == 0, result.stderr
    assert "重複" in result.stderr
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_duplicate_chunk_id_raises_with_strict_flag(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(a, [_chunk("p1#c0000")])
    _write_jsonl(b, [_chunk("p1#c0000")])

    result = _run_merge(
        ["--inputs", str(a), str(b), "--output", str(output), "--strict"]
    )

    assert result.returncode != 0
    assert not output.exists()
