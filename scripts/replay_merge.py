"""--dump-runs で落としたサブクエリ単位の検索結果に対し、統合だけをやり直す。

本走行は1構成4〜5時間かかるが、サブクエリ間マージ（仕様4）・検索の深さ（7）・
プールの剪定（6）は **`runs` から候補列を組み直すだけ**の処理で、LLM も検索索引も
要らない。太い走行1回（`reading_normal/fat.yaml`, retrieve_top_k=100）を土台にすれば、
100以下の任意の `retrieve_top_k` と各設定を数十秒で振れる。

    uv run python scripts/replay_merge.py \
      --runs runs_fat.jsonl --pred predictions_fat.jsonl \
      --agent configs/agent_style/reading.yaml \
      --set subquery_merge=rrf --ks 5,10,20,50

**ReadingAgent 本体のメソッドをそのまま呼ぶ。** `_merged_results` / `_depth_for` /
`to_gold_papers` を再実装しない（`scripts/replay_expansion.py` と同じ流儀）。
ロジックを書き写すと、オフラインで出した結論が本走行に効かなくなる。

限界（数字は必ず**下限**として読む）:

* **仕様5（grounded_refine）は再生できない。** サブクエリ自体が変わるので
  フル走行が要る。この土台に含まれるのは「あの走行が投げたサブクエリ」だけ。
* 4・6・7 も「サブクエリを固定した条件下」の数字。本走行では読解 LLM が読む候補が
  変われば選定も変わり、`_refine()` の入力も変わるので、同じ値にはならない。
* `pool_rescore` は reranker（GPU）が要るので**ここでは再生しない**。
  `pool_prune_to` だけ振れる。
* 土台の `runs` に載っているのは走行時の `retrieve_top_k` 件まで。それより深い
  設定は試せない（だから土台は太く取る）。

`--pred` は candidate_papers を差し替える先。他のフィールド（gold_papers /
evidence / trace）はそのまま持ち越すので、採点の paper_f1 は土台の値のままになる。
**この再生で読むのは candidate_recall / evidence_candidate_recall だけ。**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from littraceqa.di_pipeline.agent.reading import (
    CANDIDATE_PAPERS_LIMIT,
    ReadingAgent,
    SubqueryRun,
)
from littraceqa.di_pipeline.config import load_config
from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_overrides(params: dict, overrides: list[str]) -> dict:
    """--set key=value を agent yaml の params に反映する（振るとき用）。"""
    params = dict(params)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set は key=value の形で指定する: {item!r}")
        params[key.strip()] = yaml.safe_load(raw)
    return params


def load_runs(path: Path) -> dict[str, list[SubqueryRun]]:
    """--dump-runs の JSONL を query_id -> [SubqueryRun] に戻す。

    ダンプにテキストは載っていない（chunk_id から引ける／土台が数百MBになるため）。
    統合に要るのは chunk_id / paper_id / score / source だけなので、
    `text` と `metadata` は空で埋める。**`_merged_results()` はこの4つしか見ない。**
    """
    by_query: dict[str, list[SubqueryRun]] = {}
    for row in read_jsonl(path):
        results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                score=r["score"],
                text="",
                chunk_type="",
                metadata={},
                source=r.get("source", ""),
            )
            for r in row["results"]
        ]
        by_query.setdefault(row["query_id"], []).append(
            SubqueryRun(step=row["step"], subquery=row["subquery"], results=results)
        )
    return by_query


def rebuild(agent: ReadingAgent, runs: list[SubqueryRun]) -> list[str]:
    """1クエリぶんの runs から候補論文の列を組み直す。

    本走行の `_build_prediction()` と同じ順序を踏む: 深さで切る → マージ →
    剪定 → 論文単位に潰す。`pool_rescore` は GPU が要るので効かない
    （`_rescore_pool` は reranker が無ければ剪定だけを行う）。
    """
    depth = agent.adaptive_depth
    trimmed = []
    for run in runs:
        results = run.results
        if depth is not None:
            results = results[: agent._depth_for(results, depth)]
        else:
            results = results[: agent.retrieve_top_k]
        trimmed.append(SubqueryRun(step=run.step, subquery=run.subquery, results=results))

    # 本走行の chunks（max マージ）を同じ規則で組み直す。
    chunks: dict[str, RetrievalResult] = {}
    for run in trimmed:
        for result in run.results:
            previous = chunks.get(result.chunk_id)
            if previous is None or result.score > previous.score:
                chunks[result.chunk_id] = result

    merged = agent._merged_results(trimmed, chunks)
    merged = agent._rescore_pool(SimpleNamespace(question=""), merged, [])
    return to_gold_papers(merged, max_papers=CANDIDATE_PAPERS_LIMIT)


def recall_at(gold: set[str], ranked: list[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="--dump-runs が出した JSONL")
    parser.add_argument("--pred", required=True, help="土台にする予測 JSONL（同じ走行のもの）")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--output", default=None, help="候補列を差し替えた予測 JSONL")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="agent yaml の params を上書きする（例: --set subquery_merge=rrf）",
    )
    parser.add_argument("--ks", default="10,20,50", help="採点する k（カンマ区切り）")
    args = parser.parse_args()

    params = apply_overrides(load_config(args.agent).get("params", {}), args.overrides)
    # 検索も LLM も使わないので、統合に要る部分だけを持つ ReadingAgent を建てる。
    # reranker を持たせないので _rescore_pool は剪定だけになる。
    agent = ReadingAgent(retriever=SimpleNamespace(reranker=None), llm=None, **params)

    runs_by_query = load_runs(Path(args.runs))
    records = read_jsonl(Path(args.pred))
    gold_rows = {row["query_id"]: row for row in read_jsonl(Path(args.gold))}
    ks = [int(k) for k in args.ks.split(",")]

    changed = 0
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    per_k_ev: dict[int, list[float]] = {k: [] for k in ks}
    for record in records:
        runs = runs_by_query.get(record["query_id"])
        if not runs:
            continue
        candidates = rebuild(agent, runs)
        if candidates != (record.get("candidate_papers") or []):
            changed += 1
        record["candidate_papers"] = candidates

        gold_row = gold_rows.get(record["query_id"])
        if gold_row is None:
            continue
        gold = {p["paper_id"] for p in gold_row.get("gold_papers", [])}
        # 根拠付き gold だけの分母（evidence_candidate_recall）。打ち手の評価はこちら。
        evidence_backed = gold & {e["paper_id"] for e in gold_row.get("evidence", [])}
        for k in ks:
            per_k[k].append(recall_at(gold, candidates, k))
            if evidence_backed:
                per_k_ev[k].append(recall_at(evidence_backed, candidates, k))

    print(f"{len(records)} 件中 {changed} 件の候補列が変わった")
    print(f"\n{'k':>5}{'cr@k':>9}{'ecr@k':>9}")
    for k in ks:
        cr = sum(per_k[k]) / len(per_k[k]) if per_k[k] else float("nan")
        ecr = sum(per_k_ev[k]) / len(per_k_ev[k]) if per_k_ev[k] else float("nan")
        print(f"{k:>5}{cr:>9.3f}{ecr:>9.3f}")
    print(
        f"\n（ecr の分母は根拠付き gold を持つ {len(per_k_ev[ks[0]])} 件。"
        "打ち手の評価は ecr で見る——cr は原理的に取れないピア gold が"
        "常に混ざって天井が張り付く）"
    )

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"候補列を差し替えた予測を {args.output} に書き出しました")


if __name__ == "__main__":
    main()
