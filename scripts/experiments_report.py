#!/usr/bin/env python3
"""results/experiments.jsonl を「実験ごとの結果と設定値」の単一 HTML にする。

report/*.md（1実行1枚の Markdown）と同じ粒度で1行1実験を並べ、行を開くと
実行時に記録された実際の設定値（tuned_params / 合成済み config）と全指標を
表示する。audits/query_audit.html（クエリ品質監査）とは別ページ。

- 監査ビューアと違い candidate_papers は不要なので、旧形式の実行も全部載る
  （指標は実行時に計算済みの値をそのまま出すだけで、再計算しない）
- 網羅が55件に満たない実行（分割の片割れ等）は警告バッジを付け、
  ベスト値の太字判定からも除外する（薄まったマクロ平均を比較させない）

実行例:
    uv run python scripts/experiments_report.py --output report/experiments.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from audit_report import experiment_label, read_jsonl

GOLD_TOTAL = 55  # data/validation.jsonl の件数。これ未満の実行は部分実行として扱う。

# 一覧表に出す主要指標（キー, 表示名）。詳細展開では全指標を出す。
#
# **我々が上げるのは candidate_recall** なので、一覧はその系列だけにしてある。
# 提出物側（paper_* / evidence_* / 回答系）は evaluate.py が既定で出さなくなった
# （選定も回答生成も読解チーム側の担当。`--metrics all` で足せる）。過去の行には
# 残っているので、詳細展開すればそのまま読める。
KEY_METRICS = [
    ("candidate_recall_at5_total_macro", "cr@5"),
    ("candidate_recall_at20_total_macro", "cr@20"),
    ("candidate_recall_at50_total_macro", "cr@50"),
    ("evidence_candidate_recall_at20_total_macro", "ecr@20"),
    ("evidence_candidate_recall_at50_total_macro", "ecr@50"),
    ("evidence_candidate_recall_at20_multi_macro", "ecr@20 multi"),
]

_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#667; --line:#ddd; --card:#f6f7f9;
  --accent:#0969da; --warn:#9a6700; --best:#1a7f37; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --card:#161b22;
    --accent:#4493f8; } }
:root[data-theme="dark"] { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d;
  --card:#161b22; --accent:#4493f8; }
:root[data-theme="light"] { --bg:#fff; --fg:#1a1a1a; --muted:#667; --line:#ddd;
  --card:#f6f7f9; --accent:#0969da; }
* { box-sizing:border-box; }
body { margin:0; padding:16px 24px; background:var(--bg); color:var(--fg);
  font:14px/1.6 -apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif; }
h1 { font-size:18px; } h2 { font-size:15px; margin:6px 0; }
a { color:var(--accent); }
.note { color:var(--muted); font-size:13px; }
.chart { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 18px; margin:14px 0; max-width:860px; }
.cbar { display:grid; grid-template-columns:minmax(200px,42%) 1fr; gap:10px;
  align-items:center; padding:3px 0; }
.cbar .lbl { font-size:12px; text-align:right; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; direction:rtl; }
.cbar .bar { display:flex; align-items:center; gap:6px; }
.cbar .bar i { display:block; height:13px; border-radius:0 4px 4px 0; min-width:2px;
  background:var(--accent); }
.cbar .bar b { font-size:12px; font-variant-numeric:tabular-nums; }
.tblwrap { overflow-x:auto; }
table.runs { border-collapse:collapse; font-size:13px; width:100%; }
.runs th, .runs td { border:1px solid var(--line); padding:4px 9px; text-align:right;
  font-variant-numeric:tabular-nums; white-space:nowrap; }
.runs th { background:var(--card); color:var(--muted); font-weight:600; }
.runs td.name { text-align:left; font-family:monospace; font-size:12px; cursor:pointer;
  color:var(--accent); max-width:420px; overflow:hidden; text-overflow:ellipsis; }
.runs tr.details-row td { text-align:left; white-space:normal; background:var(--card); }
.runs tr.details-row { display:none; }
.runs tr.details-row.open { display:table-row; }
.best { color:var(--best); font-weight:700; }
.badge { display:inline-block; padding:0 7px; border-radius:9px; font-size:11px;
  color:#fff; background:var(--warn); }
pre { background:var(--bg); border:1px solid var(--line); border-radius:6px;
  padding:8px 12px; font-size:12px; overflow-x:auto; max-height:420px; overflow-y:auto; }
details.legend { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:10px 16px; margin:14px 0; max-width:980px; }
details.legend summary { cursor:pointer; font-weight:600; }
.legend table { border-collapse:collapse; font-size:13px; margin:8px 0; }
.legend th, .legend td { border:1px solid var(--line); padding:4px 10px; text-align:left;
  white-space:normal; }
.legend th { background:var(--bg); color:var(--muted); font-weight:600; }
.legend h2 { margin:12px 0 2px; }
.detail-grid { display:flex; flex-wrap:wrap; gap:18px; }
.detail-grid > div { flex:1 1 380px; min-width:0; }
.kv { margin:2px 0; } .kv b { color:var(--muted); font-weight:600; margin-right:6px; }
.mono { font-family:monospace; }
"""

_JS = """
function toggle(id){ document.getElementById(id).classList.toggle('open'); }
"""


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return "—" if value is None else html.escape(str(value))


def covered_of(row: dict[str, Any]) -> int:
    """採点に使われた gold の件数。--merge-with 実行は coverage が正。"""
    coverage = row.get("coverage") or {}
    if isinstance(coverage, dict) and coverage.get("covered"):
        return int(coverage["covered"])
    return int(row.get("n_queries") or 0)


def metrics_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metrics") if isinstance(row.get("metrics"), dict) else {}


def config_section(row: dict[str, Any]) -> str:
    """展開行に出す設定値。tuned_params（実際に効いた値）を主、config 全体を従で。"""
    parts = ['<div class="detail-grid">']
    meta = [
        ("process", row.get("process")),
        ("search", row.get("search")),
        ("agent", row.get("agent")),
        ("queries", row.get("queries")),
        ("output", row.get("output")),
        ("git_sha", row.get("git_sha")),
        ("production_input", row.get("production_input")),
        ("options_joined", row.get("options_joined")),
    ]
    meta_html = "".join(
        f'<div class="kv"><b>{html.escape(k)}</b><span class="mono">{html.escape(str(v))}</span></div>'
        for k, v in meta
        if v is not None
    )
    parts.append(f"<div><h2>実行メタ</h2>{meta_html}</div>")

    tuned = row.get("tuned_params")
    if tuned:
        parts.append(
            "<div><h2>設定値（tuned_params: 実際に効いた値）</h2><pre>"
            + html.escape(json.dumps(tuned, ensure_ascii=False, indent=1))
            + "</pre></div>"
        )
    config = row.get("config")
    if config:
        parts.append(
            "<div><h2>合成済み config 全体</h2><pre>"
            + html.escape(json.dumps(config, ensure_ascii=False, indent=1))
            + "</pre></div>"
        )
    if not tuned and not config:
        parts.append(
            '<div><p class="note">この実行は設定値の記録が始まる前のもの'
            "（yaml のファイル名のみ記録）。</p></div>"
        )

    metrics = metrics_of(row)
    if metrics:
        rows_html = "".join(
            f"<tr><td style='text-align:left'>{html.escape(k)}</td><td>{fmt(v)}</td></tr>"
            for k, v in sorted(metrics.items())
            if v is not None
        )
        parts.append(
            "<div><h2>全指標</h2><div class='tblwrap'><table class='runs'>"
            f"{rows_html}</table></div></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def metrics_legend_html() -> str:
    """各指標の定義（分子/分母）。evaluate.py の実装と一致させておくこと。"""

    def table(rows: list[tuple[str, str, str]]) -> str:
        body = "".join(
            f"<tr><td class='mono'>{html.escape(m)}</td><td>{html.escape(nu)}</td>"
            f"<td>{html.escape(de)}</td></tr>"
            for m, nu, de in rows
        )
        return (
            "<div class='tblwrap'><table><tr><th>指標</th><th>分子</th><th>分母</th></tr>"
            f"{body}</table></div>"
        )

    return (
        '<details class="legend"><summary>指標の説明（何を分子・分母にしているか）</summary>'
        "<p class='note'>macro = クエリごとに値を出して55件で単純平均。"
        "gold も提出も空なら 1.0、提出だけ空なら 0.0（prf の仕様）。</p>"
        "<h2>論文選定（提出物への採点。**evaluate.py の既定では出さない**——提出論文の選定は読解チーム側の担当。`--metrics all` で足せる）</h2>"
        + table(
            [
                ("paper_precision_macro", "提出∩gold の論文数", "提出した論文数"),
                ("paper_recall_macro", "提出∩gold の論文数", "gold の論文数（evidence 無しの gold も含む全件）"),
                ("paper_f1_macro", "上2つの調和平均", "—"),
            ]
        )
        + "<h2>検索力（LLM 絞り込み前の候補列 candidate_papers への診断）</h2>"
        + table(
            [
                (
                    "candidate_recall_at{k}_{single/multi/total}",
                    "候補列（reranker 通過後）の上位 k 本に入った gold 論文数",
                    "gold の論文数（全件）。single/multi は gold の task_family で振り分け",
                ),
                (
                    "evidence_candidate_recall_at{k}（ecr）",
                    "同上",
                    "evidence が1件以上紐づく gold のみ（根拠なしピア論文を除外。"
                    "根拠付き gold が0本のクエリは集計から除外）",
                ),
            ]
        )
        + "<h2>evidence（粗いキー (paper_id, source_type, ページ, 表/図番号) の完全一致で判定。"
        "evidence テキスト自体は比較しない）</h2>"
        + table(
            [
                ("evidence_precision_macro", "提出 evidence のうち gold と一致した件数", "提出した evidence 件数"),
                ("evidence_recall_macro", "同上", "gold の evidence 件数"),
                ("evidence_f1_macro", "調和平均", "—"),
            ]
        )
        + "<h2>回答（提出物側。上と同じく既定では出さない）</h2>"
        + table(
            [
                ("multiple_choice_accuracy", "正解した選択肢問題数", "multiple_choice を持つクエリ数（micro）"),
                (
                    "freeform_exact_match",
                    "正規化後（小文字化・空白圧縮・引用符除去）の完全一致数",
                    "freeform を持つクエリ数（micro）",
                ),
                (
                    "table_row_f1_macro",
                    "行キー（is_row_key 列の正規化値）集合の P/R から F1",
                    "table を持つクエリで macro",
                ),
                (
                    "table_cell_accuracy_macro",
                    "行キーが一致した gold 行の非キー列セル一致数",
                    "gold 行の全比較セル数（予測に行が無いセルは不正解扱い）。クエリ macro",
                ),
                ("table_cell_accuracy_micro", "一致セル数の全クエリ合算", "全比較セル数の合算"),
            ]
        )
        + "</details>"
    )


def build_experiments_html(
    rows: list[dict[str, Any]], audit_url: str | None = None
) -> str:
    rows = sorted(rows, key=lambda r: str(r.get("timestamp") or ""), reverse=True)

    # ベスト値の太字はフル網羅の実行だけで競わせる（部分実行の値は薄まっているため）。
    full_rows = [r for r in rows if covered_of(r) >= GOLD_TOTAL]
    best: dict[str, float] = {}
    for key, _ in KEY_METRICS:
        values = [
            metrics_of(r).get(key)
            for r in full_rows
            if isinstance(metrics_of(r).get(key), (int, float))
        ]
        if values:
            best[key] = max(values)

    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in KEY_METRICS)
    body_rows = []
    for i, row in enumerate(rows):
        name = experiment_label(row)
        covered = covered_of(row)
        partial = covered < GOLD_TOTAL
        badge = f' <span class="badge">{covered}/{GOLD_TOTAL}件</span>' if partial else ""
        metrics = metrics_of(row)
        cells = []
        for key, _ in KEY_METRICS:
            value = metrics.get(key)
            text = fmt(value)
            if (
                not partial
                and isinstance(value, (int, float))
                and key in best
                and abs(value - best[key]) < 1e-9
            ):
                text = f'<span class="best">{text}</span>'
            cells.append(f"<td>{text}</td>")
        body_rows.append(
            f'<tr><td class="name" onclick="toggle(\'d{i}\')" '
            f'title="クリックで設定値と全指標を展開">{html.escape(name)}{badge}</td>'
            f"<td>{covered}</td>{''.join(cells)}</tr>"
            f'<tr class="details-row" id="d{i}"><td colspan="{len(KEY_METRICS) + 2}">'
            f"{config_section(row)}</td></tr>"
        )

    # cr@20 の横棒（フル網羅のみ・降順）。単一系列なので accent 1色 + 直接ラベル。
    chart_key = "candidate_recall_at20_total_macro"
    chart_rows = sorted(
        (
            (experiment_label(r), metrics_of(r).get(chart_key))
            for r in full_rows
            if isinstance(metrics_of(r).get(chart_key), (int, float))
        ),
        key=lambda x: -x[1],
    )
    max_value = max((v for _, v in chart_rows), default=1.0) or 1.0
    chart_html = "".join(
        f'<div class="cbar"><span class="lbl" title="{html.escape(n)}">{html.escape(n)}</span>'
        f'<span class="bar"><i style="width:{v / max_value * 100:.1f}%"></i><b>{v:.3f}</b></span></div>'
        for n, v in chart_rows
    )

    audit_link = (
        f'<p class="note"><a href="{html.escape(audit_url)}">→ クエリ品質監査ページ</a></p>'
        if audit_url
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>実験一覧（結果と設定値）</title>
<style>{_CSS}</style></head>
<body>
<h1>実験一覧（結果と設定値）</h1>
<p class="note">results/experiments.jsonl の全実行。report/ の各レポートと同じ命名・同じ粒度。
実験名をクリックすると実行時に記録された設定値（tuned_params / 合成済み config）と全指標が開く。
太字はフル網羅（{GOLD_TOTAL}件）実行内のベスト値。</p>
{audit_link}
{metrics_legend_html()}
<div class="chart"><h2>cr@20（フル網羅実行のみ・降順）</h2>{chart_html}</div>
<div class="tblwrap"><table class="runs">
<tr><th style="text-align:left">実験（クリックで展開）</th><th>件数</th>{header}</tr>
{''.join(body_rows)}
</table></div>
<script>{_JS}</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="実験ログを結果・設定値の HTML 一覧にする。")
    parser.add_argument("--experiments", default="results/experiments.jsonl")
    parser.add_argument("--output", default="report/experiments.html")
    parser.add_argument(
        "--audit-url",
        default=None,
        help="クエリ品質監査ページへのリンク（Artifact の URL 等）",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.experiments))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_experiments_html(rows, audit_url=args.audit_url), encoding="utf-8"
    )
    print(f"wrote {output} ({len(rows)} 実験)", file=sys.stderr)


if __name__ == "__main__":
    main()
