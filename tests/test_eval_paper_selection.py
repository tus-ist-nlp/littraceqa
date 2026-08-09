from __future__ import annotations

import subprocess
import sys


def test_evidence_coverage_requires_production_questions(tmp_path):
    mineru = tmp_path / "mineru"
    mineru.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_paper_selection.py",
            "--retrieval",
            str(tmp_path / "missing-retrieval.json"),
            "--gold",
            str(tmp_path / "missing-gold.jsonl"),
            "--evidence-coverage-mineru-dir",
            str(mineru),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--questions is required with evidence coverage" in result.stderr
