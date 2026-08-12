"""gold 論文が各手法の候補列で何位に出たかを、1枚の HTML 表にする。

`audit_report.py` が「1クエリを深掘りする」ビューアなのに対し、こちらは
**gold 論文1本を1行にして手法を横に並べる**。どの構成でどの gold が沈んだかを
実験どうしで見比べるための表で、判定ロジックは持たない（`audits/query_audit.jsonl`
と `results/experiments.jsonl` のビューア）。

    uv run python scripts/rank_table.py --output report/rank_by_method.html

列は `results/experiments.jsonl` のうち予測 JSONL が実在する行が古い順に並ぶ
（`audit_report.py` と同じ `load_experiments()` を使うので、セレクタの並びと一致する）。
`--pred` で記録に無い予測ファイルを足せる。

テンプレート（CSS と描画スクリプト）は `templates/rank_table.html` に置いてある。
このスクリプトが差し込むのは `const DATA = {...};` の1行だけ。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from audit_report import (  # noqa: E402
    candidate_rank_map,
    group_by_query,
    load_experiments,
    read_jsonl,
)

TEMPLATE = Path(__file__).parent.parent / "templates" / "rank_table.html"
PLACEHOLDER = "/*__DATA__*/"


def relevance_group(relevance: str) -> str:
    """relevance を表のフィルタ用に3値へ畳む。

    no_evidence は「その論文由来の evidence_id が無い」というデータセットの事実で、
    supporting は「回答を支持する」判定。その中間（partial / irrelevant /
    contradicting）は evidence は持つので other_evidence にまとめる。
    """
    if relevance == "supporting":
        return "supporting"
    if relevance == "no_evidence":
        return "no_evidence"
    return "other_evidence"


def build_payload(
    grouped: dict[str, list[dict[str, Any]]],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    rank_maps = [
        {qid: candidate_rank_map(pred) for qid, pred in exp["pred_by_id"].items()}
        for exp in experiments
    ]

    lines: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        for record in grouped[query_id]:
            paper_id = str(record.get("paper_id") or "")
            if not paper_id:
                continue
            paper = record.get("paper") or {}
            lines.append(
                {
                    "q": query_id,
                    "p": paper_id,
                    "f": record.get("task_family") or "",
                    "g": relevance_group(str(record.get("relevance") or "")),
                    "rel": record.get("relevance"),
                    "t": paper.get("title") or paper_id,
                    # 候補列に無い gold は null（表では「圏外」）。
                    "r": [rm.get(query_id, {}).get(paper_id) for rm in rank_maps],
                }
            )

    return {
        "experiments": [
            {"full": exp["name"], "date": _short_date(exp["timestamp"])}
            for exp in experiments
        ],
        "lines": lines,
    }


def _short_date(timestamp: str) -> str:
    """2026-08-03T14:37:41 -> 08/03。--pred 由来の '~' はそのまま出す。"""
    if len(timestamp) >= 10 and timestamp[4] == "-":
        return f"{timestamp[5:7]}/{timestamp[8:10]}"
    return timestamp or "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="audits/query_audit.jsonl")
    parser.add_argument(
        "--pred",
        action="append",
        default=None,
        help="experiments.jsonl に載っていない予測 JSONL を列に足す（複数可）",
    )
    parser.add_argument("--experiments", default="results/experiments.jsonl")
    parser.add_argument("--output", default="report/rank_by_method.html")
    args = parser.parse_args()

    grouped = group_by_query(read_jsonl(Path(args.audit)))
    experiments = load_experiments(
        Path(args.experiments) if args.experiments else None,
        [Path(p) for p in args.pred or []],
        grouped,
    )
    if not experiments:
        raise SystemExit("列に並べる実験が1つも無い（予測 JSONL が実在するか確認する）")

    payload = build_payload(grouped, experiments)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        PLACEHOLDER,
        "const DATA = " + json.dumps(payload, ensure_ascii=False) + ";",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(
        f"{len(payload['lines'])} 本の gold × {len(experiments)} 手法 -> {output}",
        file=sys.stderr,
    )
    for exp in experiments:
        print(f"  {_short_date(exp['timestamp'])}  {exp['name']}", file=sys.stderr)


if __name__ == "__main__":
    main()
