#!/usr/bin/env python3
"""検索エージェントを使わずに候補列だけを作り、予測 JSONL に書き出す。

`eval_retrieval.py --agent` と**同じ経路**を通す（生の質問1本で検索 ->
`to_gold_papers(skip_chunk_types)` -> A/B の RRF 統合）が、**採点をしない**ので
gold の無い本番入力（`data/test_inputs.jsonl`）にそのまま使える。
`eval_retrieval.py` は `--queries` に gold_papers 入りの jsonl を要求し、
recall を print するだけで候補列をどこにも書き出さないため、本番の受け渡し物を
作る経路が無かった。ここはその出口だけを足したもの。

**LLM は1回も呼ばない。** サブクエリ分解・反復検索・読解は走らないので、
⚠ `evidence` は出せない（`evidence_chunk_ids` を持つのは読解だけ）。渡せるのは
候補列＝論文の順位まで。

出力は `run_search.py` と同じ予測 JSONL（`candidate_papers` は paper_id の列）なので、
そのまま `build_candidate_handoff.py` に渡せる:

    uv run python scripts/run_agentless.py \\
      --paths configs/paths/nlp02.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \\
      --agent configs/agent_style/agentless/agentless.yaml \\
      --queries data/test_inputs.jsonl \\
      --output predictions_test_agentless.jsonl

    uv run python scripts/build_candidate_handoff.py \\
      --predictions predictions_test_agentless.jsonl \\
      --inputs data/test_inputs.jsonl --no-gold \\
      --output data/test_inputs_with_candidates.jsonl

**ReadingAgent 本体のメソッド（`_combine_rrf`）をそのまま呼ぶ。** ロジックを
書き写すと、ここで出た候補列が本走行の `_build_prediction()` とずれる。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from littraceqa.di_pipeline.agent.reading import CANDIDATE_PAPERS_LIMIT, ReadingAgent
from littraceqa.di_pipeline.config import (
    build_pipeline,
    compose_config,
    load_config,
)
from littraceqa.di_pipeline.contracts import Prediction, Query
from littraceqa.di_pipeline.retrieve.hybrid import paper_scores, to_gold_papers


def load_queries(path: Path) -> list[Query]:
    """入力 jsonl を Query にする。gold は要らない（本番入力をそのまま読む）。"""
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(Query.from_dict(json.loads(line)))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument(
        "--agent",
        required=True,
        help="configs/agent_style/agentless/*.yaml（expansion ブロックが要る）",
    )
    parser.add_argument("--queries", required=True, help="本番入力 jsonl（gold 不要）")
    parser.add_argument("--output", required=True, help="予測 JSONL の書き出し先")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="gold_papers に載せる本数（候補列の先頭から。選定はしない）",
    )
    args = parser.parse_args()

    # agent yaml をそのまま組む。`ReadingAgent.__init__` は `llm` を必須引数に取るので、
    # 組み立てのためだけに fake を渡し、**直後に None へ落とす**（LLM 経路に入ったら
    # 黙って空文字が返るのではなく、必ず TypeError で落ちるようにするため）。
    agent_cfg = load_config(args.agent)
    if agent_cfg.get("llm"):
        raise SystemExit(
            f"{args.agent} に llm ブロックがある。LLM を呼ぶ構成は run_search.py で回す"
        )
    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=load_config(args.search),
        agent={**agent_cfg, "llm": {"name": "fake"}},
    )
    _, retriever, agent = build_pipeline(cfg)
    agent.llm = None
    if not isinstance(agent, ReadingAgent) or agent.paper_expander is None:
        raise SystemExit(f"{args.agent} に expansion ブロックが無い")
    # `anchor_from: verdict` は読解を走らせないと起点が取れないので候補1位に落ちる
    # （`_anchor_papers()` が None を返す）。黙って落ちると設定と実挙動がずれるので出す。
    if getattr(agent.paper_expander, "anchor_from", None) == "verdict":
        print(
            "注意: anchor_from: verdict は読解を走らせないと起点が取れないので、"
            "候補1位起点に落ちる"
        )
    skip_chunk_types = agent.paper_score_skip_chunk_types

    print("既存の索引を読み込み中...")
    started = time.time()
    for indexer in retriever.indexers:
        indexer.load()
    print(f"読み込み完了（{time.time() - started:.1f}秒）")

    # indexer.search() は retrieve() に渡す top_k に関わらず常に per_index_k 件を
    # 返すので、fuse 後にチャンク単位で切り詰められる前に十分な件数を要求しておく
    # （eval_retrieval.py と同じ）。
    request_k = retriever.per_index_k * max(1, len(retriever.indexers))

    queries = load_queries(Path(args.queries))
    predictions: list[dict] = []
    started = time.time()
    for i, query in enumerate(queries, start=1):
        results = retriever.retrieve(query.question, request_k)
        # RRF 統合では**切る前の全長**を A に使う（`_build_prediction()` と同じ）。
        ranked = to_gold_papers(results, skip_chunk_types=skip_chunk_types)
        if ranked:
            ranked = agent._combine_rrf(
                ranked,
                [],
                paper_scores=paper_scores(results, skip_chunk_types=skip_chunk_types),
            )
        ranked = ranked[:CANDIDATE_PAPERS_LIMIT]

        prediction = Prediction.from_query(query)
        prediction.candidate_papers = ranked
        # **選定はしない**（`submit_from: candidates` と同じ）。候補列の順位をそのまま
        # 先頭から max_papers 本渡し、どれを出すかは読解チーム側に任せる。
        prediction.gold_papers = [{"paper_id": pid} for pid in ranked[: args.max_papers]]
        predictions.append(prediction.to_dict())
        elapsed = time.time() - started
        print(
            f"{i}/{len(queries)} 完了 ({query.query_id}, 候補{len(ranked)}本, "
            f"{elapsed:.0f}秒経過)",
            flush=True,
        )

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = [len(p["candidate_papers"]) for p in predictions]
    print(f"\n予測結果を {output} に書き出しました（{len(predictions)}件）")
    print(
        f"候補論文: 1クエリあたり最小{min(counts, default=0)} "
        f"最大{max(counts, default=0)} / 合計{sum(counts)}本"
    )


if __name__ == "__main__":
    main()
