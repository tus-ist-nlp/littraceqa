#!/usr/bin/env python3
"""図・数式などモダリティ別に、根拠の gold ページが現行 retrieval の上位K候補に

含まれているか（= 引けているか）を計測する M1 用スクリプト。

「根拠 F1 が低いのは retrieval が引けていないからか、引けているのに
reading（読み取り）ができていないからか」を切り分けるための最初の一歩。
チャンク・索引・エージェントは一切変更しない。既存の索引を読み込んで
検索するだけの読み取り専用ツール。

使い方:
    uv run python scripts/measure_modality_gap.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/abstract_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --gold data/validation.jsonl \\
      --output results_modality.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from littraceqa.di_pipeline.config import build_pipeline, compose_config, load_config
from littraceqa.di_pipeline.contracts import RetrievalResult

_SOURCE_TYPES = ["figure", "equation_algorithm", "table", "text_span", "citation_context"]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_modality_hit_rates(
    gold_records: list[dict],
    retrieve_fn: Callable[[str], list[RetrievalResult]],
) -> dict[str, dict]:
    """source_type ごとに hit/total/miss一覧/no_page_locator件数を集計する。

    1クエリにつき retrieve_fn は1回だけ呼ぶ（同じ質問に対する複数 evidence item で
    検索を再実行しない）。evidence item の locator.page が無いものは
    hit/miss に数えず no_page_locator として別集計する。
    """
    stats: dict[str, dict] = {
        source_type: {"hit": 0, "total": 0, "no_page_locator": 0, "misses": []}
        for source_type in _SOURCE_TYPES
    }

    for record in gold_records:
        evidence_items = record.get("evidence") or []
        if not evidence_items:
            continue

        results = retrieve_fn(record["question"])
        retrieved_pages_by_paper: dict[str, set] = defaultdict(set)
        for result in results:
            page = result.metadata.get("page")
            if page is not None:
                retrieved_pages_by_paper[result.paper_id].add(page)

        for evidence in evidence_items:
            source_type = evidence.get("source_type")
            if source_type not in stats:
                continue

            locator = evidence.get("locator") or {}
            page = locator.get("page")
            if page is None:
                stats[source_type]["no_page_locator"] += 1
                continue

            stats[source_type]["total"] += 1
            hit = page in retrieved_pages_by_paper.get(evidence.get("paper_id"), set())
            if hit:
                stats[source_type]["hit"] += 1
            else:
                identifier = (
                    locator.get("figure_id")
                    or locator.get("table_id")
                    or locator.get("equation_id")
                    or locator.get("section")
                    or ""
                )
                stats[source_type]["misses"].append(
                    (record.get("query_id", ""), evidence.get("evidence_id", ""), identifier)
                )

    return stats


def render_report(stats: dict[str, dict], meta: dict) -> str:
    lines = ["# モダリティ別 retrieval hit率 (M1)", ""]
    lines.append(f"- 実行日時: {meta['timestamp']}")
    lines.append(f"- paths: {meta['paths']}")
    lines.append(f"- process: {meta['process']}")
    lines.append(f"- search: {meta['search']}")
    lines.append(f"- agent: {meta['agent']} (top_k={meta['top_k']})")
    lines.append(f"- gold: {meta['gold']} ({meta['n_queries']} 件)")
    lines.append("")
    lines.append("| source_type | n | hit | hit率 | no_page_locator |")
    lines.append("|---|---|---|---|---|")
    for source_type in _SOURCE_TYPES:
        s = stats[source_type]
        total = s["total"]
        hit_rate = f"{s['hit'] / total:.1%}" if total else "-"
        lines.append(
            f"| {source_type} | {total} | {s['hit']} | {hit_rate} | {s['no_page_locator']} |"
        )
    lines.append("")

    for source_type in ("figure", "equation_algorithm"):
        misses = stats[source_type]["misses"]
        lines.append(f"## {source_type} の miss 一覧 ({len(misses)}件)")
        if misses:
            for query_id, evidence_id, identifier in misses:
                lines.append(f"- {query_id} / {evidence_id} ({identifier})")
        else:
            lines.append("- (なし)")
        lines.append("")

    lines.append("## 読み方")
    lines.append(
        "- hit率が高い(引けている)場合: 根拠F1の低さは reading 律速の可能性が高い"
        " → M2(参照マップ)に進む価値がある。"
    )
    lines.append(
        "- hit率が低い(引けていない)場合: retrieval 律速。M2 は保留し、"
        "ColPali 系の視覚ページ索引などを別途検討する。"
    )
    lines.append("- 最終判断はこの数値を見て人が行うこと。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--output", default="results_modality.md")
    args = parser.parse_args()

    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=load_config(args.search),
        agent=load_config(args.agent),
    )
    _preprocessor, retriever, _agent = build_pipeline(cfg)

    print("既存の索引を読み込み中...")
    for indexer in retriever.indexers:
        try:
            indexer.load()
        except Exception as exc:
            print(
                f"エラー: {indexer.name} の索引読み込みに失敗しました: {exc}\n"
                f"先に scripts/run_search.py --build を実行して索引を構築してください。",
                file=sys.stderr,
            )
            sys.exit(1)
    print("読み込み完了")

    top_k = cfg["agent"].get("params", {}).get("top_k", 20)
    gold_records = read_jsonl(Path(args.gold))

    print(f"{len(gold_records)} 件の gold に対してモダリティ別 hit率を計測中 (top_k={top_k})...")
    stats = compute_modality_hit_rates(
        gold_records, retrieve_fn=lambda question: retriever.retrieve(question, top_k)
    )

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "paths": args.paths,
        "process": args.process,
        "search": args.search,
        "agent": args.agent,
        "top_k": top_k,
        "gold": args.gold,
        "n_queries": len(gold_records),
    }
    report = render_report(stats, meta)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"結果を {output_path} に書き出しました")
    print(report)


if __name__ == "__main__":
    main()
