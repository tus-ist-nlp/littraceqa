#!/usr/bin/env python3
"""検索手法(indexer + fuser + reranker)単体の recall@k / precision@k を測るスクリプト。

scripts/run_search.py が出す paper_recall_macro は、reading/simple などの agent が
LLM 判断（paper_cutoff）で提出論文数をクエリごとに変えた後の「可変長の提出セット」に
対する recall であり、固定 k での recall@k ではない（agent の絞り込み判断と検索力が
混ざった数字になる）。ここでは agent を経由せず HybridRetriever.retrieve() の生の
ランキングを使って、純粋な検索手法の recall@k / precision@k を測る。

使い方:
    uv run python scripts/eval_retrieval.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25.yaml \\
      --ks 5,10,20,50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from littraceqa.di_pipeline.config import build_pipeline, compose_config, load_config
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers


def load_gold(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def gold_paper_ids(record: dict) -> set[str]:
    return {p["paper_id"] for p in record.get("gold_papers", []) if p.get("paper_id")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever単体の recall@k / precision@k を測る")
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument(
        "--queries",
        default="data/validation.jsonl",
        help="question と gold_papers を含む jsonl（デフォルトは正解付きの validation.jsonl）",
    )
    parser.add_argument("--ks", default="5,10,20,50")
    args = parser.parse_args()

    ks = sorted({int(k) for k in args.ks.split(",")})

    # agent はここでは使わないが build_pipeline() の戻り値に必要なのでダミーで組む
    # （llm を渡さないので Azure 等の資格情報は不要）。
    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=load_config(args.search),
        agent={"name": "simple", "params": {}},
    )
    _, retriever, _ = build_pipeline(cfg)

    print("既存の索引を読み込み中...")
    for indexer in retriever.indexers:
        indexer.load()
    print("読み込み完了")

    # indexer.search() は retrieve() に渡す top_k に関わらず常に per_index_k 件を
    # 返すので、fuse 後にチャンク単位で切り詰められる前に十分な件数を要求しておく。
    # そうしないと、上位チャンクが同じ論文に偏っている場合に
    # 「論文としては k 件に満たない」まま recall@k を過小評価してしまう。
    request_k = retriever.per_index_k * max(1, len(retriever.indexers))
    max_k = max(ks)

    records = load_gold(Path(args.queries))
    recall_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    hit_query_counts = {k: 0 for k in ks}

    for record in records:
        gold = gold_paper_ids(record)
        results = retriever.retrieve(record["question"], request_k)
        ranked_papers = to_gold_papers(results, max_papers=max_k)
        for k in ks:
            topk = set(ranked_papers[:k])
            hit = gold & topk
            recall = (len(hit) / len(gold)) if gold else 1.0
            precision = len(hit) / k if k else 0.0
            recall_sums[k] += recall
            precision_sums[k] += precision
            if hit:
                hit_query_counts[k] += 1

    n = len(records)
    print(f"\n{n} 件のクエリで評価（gold_papers は {args.queries} 由来）\n")
    header = f"{'k':>5} | {'recall@k':>10} | {'precision@k':>12} | {'hit_rate@k':>10}"
    print(header)
    print("-" * len(header))
    for k in ks:
        print(
            f"{k:>5} | {recall_sums[k] / n:>10.4f} | {precision_sums[k] / n:>12.4f} | "
            f"{hit_query_counts[k] / n:>10.4f}"
        )


if __name__ == "__main__":
    main()
