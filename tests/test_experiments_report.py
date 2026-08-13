"""実験一覧ページ（scripts/experiments_report.py）のテスト。

- 全行が載る（旧形式でも metrics は記録済みなので表示できる）
- 部分実行はバッジ付きで、ベスト値の太字競争から外れる
- チャート（cr@20）はフル網羅実行のみ
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiments_report import GOLD_TOTAL, build_experiments_html


def _row(ts: str, search: str, cr20: float, n_queries: int = GOLD_TOTAL) -> dict:
    return {
        "timestamp": ts,
        "process": "configs/process_style/mineru.yaml",
        "search": f"configs/search_style/{search}.yaml",
        "agent": "configs/agent_style/reading.yaml",
        "n_queries": n_queries,
        # 一覧・チャートの主役は candidate_recall（提出物側の指標は evaluate.py が
        # 既定で出さないので、新しい行には入らない）。
        "metrics": {
            "candidate_recall_at20_total_macro": cr20,
            "evidence_candidate_recall_at20_total_macro": 0.5,
        },
        "tuned_params": {"per_index_k": 100},
        "output": f"predictions_{search}.jsonl",
    }


def test_build_experiments_html() -> None:
    rows = [
        _row("2026-07-20T10:00:00", "bm25", 0.500),
        _row("2026-07-21T10:00:00", "bm25_qwen3", 0.640),
        _row("2026-07-22T10:00:00", "bm25_half", 0.900, n_queries=28),  # 部分実行
    ]
    page = build_experiments_html(rows, audit_url="https://example.invalid/audit")
    assert "20260721_100000_mineru_bm25_qwen3_reading" in page
    # 部分実行にはバッジが付く
    assert f'28/{GOLD_TOTAL}件' in page
    # ベスト太字はフル網羅の 0.640。部分実行の 0.900 は太字にならない
    assert '<span class="best">0.640</span>' in page
    assert '<span class="best">0.900</span>' not in page
    # チャートはフル網羅のみ（2本）
    assert page.count('class="cbar"') == 2
    # 監査ページへのリンク
    assert 'https://example.invalid/audit' in page
    # 設定値が展開行に入る
    assert "per_index_k" in page
