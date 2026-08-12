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

`--agent` を渡すと、**候補列の並べ替えだけで完結する部分**（A/B の RRF 統合＝
論文→論文展開と、`paper_score_skip_chunk_types`）を生ランキングに載せる。
どちらも LLM を呼ばないので、ここで agent 込みの構成を測れる。載らないのは
サブクエリ分解・反復検索・読解（evidence）と `anchor_from: verdict` の4つ。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from littraceqa.di_pipeline.agent.reading import CANDIDATE_PAPERS_LIMIT, ReadingAgent
from littraceqa.di_pipeline.config import (
    build_paper_expander,
    build_pipeline,
    compose_config,
    load_config,
)
from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.hybrid import paper_scores, to_gold_papers


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


def evidence_backed_paper_ids(record: dict) -> set[str]:
    """gold のうち evidence が1件以上紐づいている論文だけ（evaluate.py と同じ定義）。

    multi_paper の gold には evidence がまったく無い論文が混ざる（検証55件では
    gold 120本中29本）。それらは「質問文が名指ししていない同トピックのピア論文」で、
    質問文をどう検索しても近傍に来ないのに、gold 全体を分母にすると常に混ざって
    天井を押し下げる。索引や reranker を変えた効果を読むときはこちらを見る。
    """
    backed = {
        item.get("paper_id")
        for item in (record.get("evidence") or [])
        if isinstance(item, dict) and item.get("paper_id")
    }
    return gold_paper_ids(record) & backed


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
    parser.add_argument(
        "--agent",
        help=(
            "configs/agent_style/*.yaml。渡すと A/B の RRF 統合（論文→論文展開）と "
            "`paper_score_skip_chunk_types` だけを生ランキングに載せる。"
            "サブクエリ分解・反復検索・読解は走らないので LLM は呼ばない"
        ),
    )
    args = parser.parse_args()

    ks = sorted({int(k) for k in args.ks.split(",")})

    paths, process, search = (
        load_config(args.paths),
        load_config(args.process),
        load_config(args.search),
    )
    # agent はここでは使わないが build_pipeline() の戻り値に必要なのでダミーで組む
    # （llm を渡さないので Azure 等の資格情報は不要）。
    cfg = compose_config(
        paths=paths,
        process=process,
        search=search,
        agent={"name": "simple", "params": {}},
    )
    _, retriever, _ = build_pipeline(cfg)

    # --agent があれば、展開・統合に必要な部分だけを持つ ReadingAgent を建てる
    # （replay_expansion.py と同じ形）。**本体のメソッドをそのまま呼ぶ**ので、
    # ここで出る候補列は本走行の `_build_prediction()` と同じ規則で並ぶ。
    #
    # `anchor_from: verdict` だけは載らない——起点にする「読解 LLM が確認した論文」が
    # 読解を走らせない限り存在しないため。`_anchor_papers()` が None を返して
    # 従来の起点（候補1位）に落ちるので、実質 `rrfk10.yaml` 相当として読む。
    # **その穴を埋めるのが `anchor_from: score`**（リランカのスコアで起点を選ぶ
    # LLM 不要版）。こちらはここでもそのまま効く。
    agent = None
    skip_chunk_types: tuple[str, ...] = ()
    if args.agent:
        agent_cfg = compose_config(paths, process, search, load_config(args.agent))["agent"]
        expander = build_paper_expander(agent_cfg)
        if expander is None:
            raise SystemExit(f"{args.agent} に expansion ブロックが無い")
        agent = ReadingAgent(
            retriever=retriever,
            llm=None,
            paper_expander=expander,
            **agent_cfg.get("params", {}),
        )
        skip_chunk_types = agent.paper_score_skip_chunk_types
        if getattr(expander, "anchor_from", None) == "verdict":
            print(
                "注意: anchor_from: verdict は読解を走らせないと起点が取れないので、"
                "候補1位起点に落ちる",
                file=sys.stderr,
            )

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
    # evidence_candidate_recall: 分母を根拠付き gold に絞った版。根拠付きが1本も
    # 無いクエリは分母が空なので集計から外す（recall を 1.0 扱いして水増ししない）。
    evidence_recall_sums = {k: 0.0 for k in ks}
    evidence_query_count = 0

    # single / multi の内訳。**打ち手の効きは multi にしか乗らない**（single は
    # cr@20 が 1.000 で飽和済み）ので、総合だけ見ていると改善が薄まって見える。
    # 振り分けは evaluate.py と同じ task_family 基準。
    by_scenario: dict[str, dict[str, dict[int, float] | int]] = {
        name: {
            "recall": {k: 0.0 for k in ks},
            "evidence": {k: 0.0 for k in ks},
            "n": 0,
            "n_evidence": 0,
        }
        for name in ("single", "multi")
    }

    for record in records:
        gold = gold_paper_ids(record)
        backed = evidence_backed_paper_ids(record)
        scenario = "multi" if "multi" in (record.get("task_family") or "") else "single"
        bucket = by_scenario[scenario]
        bucket["n"] += 1
        if backed:
            bucket["n_evidence"] += 1
        results = retriever.retrieve(record["question"], request_k)
        if agent is None:
            ranked_papers = to_gold_papers(results, max_papers=max_k)
        else:
            # RRF 統合では**切る前の全長**を A に使う（`_build_prediction()` と同じ）。
            # 深い順位の論文をランキングB が押し上げられるようにするため。
            ranked_papers = to_gold_papers(results, skip_chunk_types=skip_chunk_types)
            if ranked_papers:
                trace: list[dict] = []
                if getattr(agent.paper_expander, "combine", None) == "rrf":
                    ranked_papers = agent._combine_rrf(
                        ranked_papers,
                        trace,
                        paper_scores=paper_scores(
                            results, skip_chunk_types=skip_chunk_types
                        ),
                    )
                else:
                    query = Query(
                        query_id=record["query_id"],
                        question=record["question"],
                        answer_types=[],
                    )
                    ranked_papers = agent._expand_candidates(
                        query, ranked_papers[:CANDIDATE_PAPERS_LIMIT], {}, trace
                    )
            ranked_papers = ranked_papers[:max_k]
        if backed:
            evidence_query_count += 1
        for k in ks:
            topk = set(ranked_papers[:k])
            hit = gold & topk
            recall = (len(hit) / len(gold)) if gold else 1.0
            precision = len(hit) / k if k else 0.0
            recall_sums[k] += recall
            precision_sums[k] += precision
            if hit:
                hit_query_counts[k] += 1
            bucket["recall"][k] += recall
            if backed:
                evidence_recall_sums[k] += len(backed & topk) / len(backed)
                bucket["evidence"][k] += len(backed & topk) / len(backed)

    n = len(records)
    print(f"\n{n} 件のクエリで評価（gold_papers は {args.queries} 由来）")
    print(
        f"evidence_recall@k は gold のうち evidence が紐づく論文だけを分母にした値"
        f"（対象 {evidence_query_count} 件）\n"
    )
    header = (
        f"{'k':>5} | {'recall@k':>10} | {'evidence_recall@k':>18} | "
        f"{'precision@k':>12} | {'hit_rate@k':>10}"
    )
    print(header)
    print("-" * len(header))
    for k in ks:
        evidence_recall = (
            f"{evidence_recall_sums[k] / evidence_query_count:>18.4f}"
            if evidence_query_count
            else f"{'-':>18}"
        )
        print(
            f"{k:>5} | {recall_sums[k] / n:>10.4f} | {evidence_recall} | "
            f"{precision_sums[k] / n:>12.4f} | {hit_query_counts[k] / n:>10.4f}"
        )

    # single / multi の内訳。**看板は multi**（single は @20 で飽和しているので、
    # 総合だけ見ると打ち手の効果が半分に薄まって見える）。
    for name in ("single", "multi"):
        bucket = by_scenario[name]
        count = bucket["n"]
        if not count:
            continue
        print(f"\n[{name}] n={count}（うち根拠付き gold あり {bucket['n_evidence']} 件）")
        print(f"{'k':>5} | {'recall@k':>10} | {'evidence_recall@k':>18}")
        print("-" * 40)
        for k in ks:
            evidence = (
                f"{bucket['evidence'][k] / bucket['n_evidence']:>18.4f}"
                if bucket["n_evidence"]
                else f"{'-':>18}"
            )
            print(f"{k:>5} | {bucket['recall'][k] / count:>10.4f} | {evidence}")


if __name__ == "__main__":
    main()
