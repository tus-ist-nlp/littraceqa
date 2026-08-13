#!/usr/bin/env python3
"""audits/nongold_audit.jsonl を集計して Markdown レポートを書く。

判定（audit_nongold.py）と集計・描画（本スクリプト）を分けるのは
audit_queries.py / audit_report.py と同じ方針。ここに判定ロジックは置かない。

実行例:
    uv run python scripts/nongold_report.py \
      --audit audits/nongold_audit.jsonl \
      --output report/nongold_analysis.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

RELATION_JA = {
    "same_topic_different_finding": "同じ主題・別の知見",
    "same_method_different_task": "同じ手法・別タスク",
    "cited_or_baseline": "gold の引用/ベースライン",
    "peer_of_gold": "gold と同クラスタのピア",
    "shared_terminology_only": "用語が重なるだけ",
    "unrelated": "無関係",
    "possible_gold": "gold になりうる（付与漏れ疑い）",
}

CAUSE_JA = {
    "query_terms_verbatim": "質問文の語がそのまま出る",
    "topic_centroid": "その主題の代表論文",
    "gold_evidence_is_deep": "gold の根拠が本文の奥にある",
    "gold_is_unnamed_peer": "gold が名指しされないピア",
    "question_is_underspecified": "質問が gold を一意に指せない",
    "not_applicable": "gold より上位ではない",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dist_table(rows: list[dict[str, Any]], key: str, ja: dict[str, str]) -> str:
    counts = Counter(r.get(key) for r in rows)
    total = sum(counts.values())
    lines = ["| 区分 | 件数 | 割合 |", "|---|---|---|"]
    for value, n in counts.most_common():
        label = ja.get(str(value), "判定できず" if value is None else str(value))
        lines.append(f"| {label} | {n} | {n / total:.1%} |")
    return "\n".join(lines)


def cross_table(rows: list[dict[str, Any]]) -> str:
    """relation × outranks_some_gold。関係の薄い論文がどれだけ gold を追い越したか。"""
    counts: dict[str, list[int]] = {}
    for r in rows:
        label = RELATION_JA.get(str(r.get("relation")), "判定できず")
        cell = counts.setdefault(label, [0, 0])
        cell[0 if r.get("outranks_some_gold") else 1] += 1
    lines = ["| 関係 | gold を追い越した | 追い越していない | 計 |", "|---|---|---|---|"]
    for label, (a, b) in sorted(counts.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        lines.append(f"| {label} | {a} | {b} | {a + b} |")
    return "\n".join(lines)


def examples(rows: list[dict[str, Any]], n: int = 8) -> str:
    """gold を追い越した事例を順位の高い順に並べる。"""
    picked = sorted(
        [r for r in rows if r.get("outranks_some_gold")],
        key=lambda r: (r.get("rank") or 99, str(r.get("query_id"))),
    )[:n]
    out = []
    for r in picked:
        paper = r.get("paper") or {}
        out += [
            f"**{r['query_id']} / {r['rank']}位 — {paper.get('title', '')[:70]}**"
            f"（{paper.get('venue')} {paper.get('year')}, `{r['paper_id']}`）",
            "",
            f"- 質問: {str(r.get('question') or '')[:160]}",
            f"- 関係: **{RELATION_JA.get(str(r.get('relation')), '判定できず')}** — "
            f"{r.get('relation_detail') or ''}",
            f"- 上位化の理由: **{CAUSE_JA.get(str(r.get('outrank_cause')), '判定できず')}** — "
            f"{r.get('outrank_detail') or ''}",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="非 gold 論文の判定 JSONL を集計する。")
    parser.add_argument("--audit", default="audits/nongold_audit.jsonl")
    parser.add_argument("--pred", default="predictions_8b_chunk_expand_fused_offline.jsonl")
    parser.add_argument("--output", default="report/nongold_analysis.md")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.audit))
    outranking = [r for r in rows if r.get("outranks_some_gold")]
    multi = [r for r in rows if r.get("task_family") == "multi_paper"]
    single = [r for r in rows if r.get("task_family") != "multi_paper"]
    could = [r for r in rows if r.get("could_be_gold")]

    doc = f"""# 検索上位に入った「gold でない論文」の分析

候補上位5本のうち gold として登録されていない論文について、
(1) 質問とどんな関係があるのか (2) なぜ gold より上位に来たのか を LLM に判定させた。

- 判定: `scripts/audit_nongold.py` → `{args.audit}`（1クエリ1呼び出し。gold の
  タイトル・abstract・根拠の所在を同じプロンプトに入れて比較させている）
- 集計: `scripts/nongold_report.py`（判定ロジックは持たない）
- 予測: `{args.pred}`
- 対象: **{len(rows)}件**（55クエリ × 上位5本 − gold 本数）。
  うち **{len(outranking)}件**が少なくとも1本の gold より上位
  （single {sum(1 for r in outranking if r.get("task_family") != "multi_paper")}件 /
  multi {sum(1 for r in outranking if r.get("task_family") == "multi_paper")}件）

## 質問との関係

{dist_table(rows, "relation", RELATION_JA)}

task_family 別の内訳:

- single_paper {len(single)}件: {", ".join(f"{RELATION_JA.get(str(k), '判定できず')} {v}" for k, v in Counter(r.get("relation") for r in single).most_common(4))}
- multi_paper {len(multi)}件: {", ".join(f"{RELATION_JA.get(str(k), '判定できず')} {v}" for k, v in Counter(r.get("relation") for r in multi).most_common(4))}

## gold より上位に来た理由

**gold を追い越した {len(outranking)}件だけ**を分母にした内訳
（追い越していない論文は `not_applicable` になるので分母から外す）:

{dist_table(outranking, "outrank_cause", CAUSE_JA)}

## 関係 × gold を追い越したか

{cross_table(rows)}

## gold 付与漏れの疑い

`could_be_gold = true`（その論文だけで質問に答えられると判定された）は **{len(could)}件**、
`relation = possible_gold` は **{sum(1 for r in rows if r.get("relation") == "possible_gold")}件**。

{chr(10).join(f"- **{r['query_id']} / {r['rank']}位** `{r['paper_id']}` {(r.get('paper') or {}).get('title', '')[:60]}{chr(10)}  - {(r.get('relation_detail') or '')[:200]}" for r in could[:10]) or "- なし"}

## 事例（gold を追い越したもの、順位の高い順）

{examples(rows)}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
