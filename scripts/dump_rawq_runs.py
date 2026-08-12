"""**元の質問そのもの**で1回だけ検索し、`--dump-runs` と同じ形の JSONL に落とす。

`ReadingAgent._decompose()` は LLM が返したサブクエリだけを投げ、元の質問は
**LLM が空を返したときのフォールバックにしか使われない**（`reading.py` の
`return subqueries or [query.question]`）。実測でも `predictions_k100_notable.jsonl`
の step0 は全55件がちょうど4本で、質問文そのものは **0/55** 本だった。

一方、生の質問1本で引くだけの `eval_retrieval.py`（LLM なし）は @1 でフル走行を
上回っている（ecr@1 0.637 vs 0.605 / cr@1 0.557 vs 0.529）。分解した4本には
元の質問が持っていた語の組み合わせが残らず、融合で1位が薄まっている疑いがある。

ここで落とした run を `scripts/replay_rawq.py --extra-runs` に渡すと、
**step0 に5本目のサブクエリとして元の質問を足した場合**の候補列が再現できる。
LLM は1回も呼ばないので、増えるコストは「55クエリ × 検索1回」だけ。

    uv run python scripts/dump_rawq_runs.py \
      --paths configs/paths/default.yaml --process configs/process_style/mineru.yaml \
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
      --queries data/validation_inputs.jsonl --output runs_rawq.jsonl --top-k 20

`--top-k` は agent の `retrieve_top_k` に合わせる（notable は 20）。属性フィルタは
`HybridRetriever` 側のフォールバック抽出が元の質問から取るので、本走行の
`ReadingAgent._extract_attribute_filter()` と同じ制約が効く。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from littraceqa.di_pipeline.config import build_pipeline, compose_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--process", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--queries", default="data/validation_inputs.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=20, help="agent の retrieve_top_k に合わせる")
    args = parser.parse_args()

    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=load_config(args.search),
        agent={"name": "simple", "params": {}},
    )
    _, retriever, _ = build_pipeline(cfg)

    print("既存の索引を読み込み中...", flush=True)
    for indexer in retriever.indexers:
        indexer.load()
    print("読み込み完了", flush=True)

    with Path(args.queries).open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle if line.strip()]

    started = time.time()
    with Path(args.output).open("w", encoding="utf-8") as out:
        for i, query in enumerate(queries, 1):
            question = query["question"]
            results = list(retriever.retrieve(question, args.top_k))
            out.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        # 本走行の step0 に足す前提。`replay_rawq.py` が step0 の
                        # 末尾に差し込む。
                        "step": 0,
                        "subquery": question,
                        "results": [
                            {
                                "chunk_id": r.chunk_id,
                                "paper_id": r.paper_id,
                                "rank": rank,
                                "score": r.score,
                                "source": r.source,
                            }
                            for rank, r in enumerate(results)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            print(
                f"{i}/{len(queries)} 完了 ({time.time() - started:.0f}秒)", flush=True
            )

    print(f"{args.output} に書き出しました", flush=True)


if __name__ == "__main__":
    main()
