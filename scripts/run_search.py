#!/usr/bin/env python3
"""configs/{paths,process_style,search_style,agent_style}/*.yaml を組み合わせて

前処理・索引構築・検索・評価を一気通貫で実行する e2e スクリプト。

使い方:
    # 初回: 前処理 + 索引構築をしてから検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/pypdf.yaml \\
      --search configs/search_style/bm25_qwen3.yaml \\
      --agent configs/agent_style/simple.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # 2回目以降: 既存の索引を読み込んで検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/pypdf.yaml \\
      --search configs/search_style/bm25_qwen3.yaml \\
      --agent configs/agent_style/simple.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from litqa.config import build_pipeline, compose_config, load_config
from litqa.contracts import Chunk, Query


def load_papers(path: Path) -> list[dict]:
    papers = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


# 本番の入力に実際に入っているのはこの4つだけ。
_PRODUCTION_FIELDS = ("query_id", "question", "answer_types", "table_schema")


def load_queries(path: Path, production_input: bool = False) -> list[Query]:
    """クエリを読み込む。

    production_input=True にすると、本番入力に無いフィールド
    （task_family / primary_evidence_type / benchmark）を捨ててから Query を作る。
    手元の validation_inputs.jsonl は55件すべてに task_family が入っているが、
    本番入力には無い。task_family は提出論文数（cutoff）を決めるのに使うので、
    これを与えたまま評価すると「正解を教えてもらった状態」の点数になり、
    本番の点数と乖離する。比較実験ではこちらを使うこと。
    """
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if production_input:
                record = {k: v for k, v in record.items() if k in _PRODUCTION_FIELDS}
            queries.append(Query.from_dict(record))
    return queries


def log_experiment(
    args: argparse.Namespace, metrics: dict, n_queries: int
) -> None:
    """どの組み合わせで何点だったかを results/experiments.jsonl に追記する。"""
    path = Path("results/experiments.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": args.paths,
        "process": args.process,
        "search": args.search,
        "agent": args.agent,
        "queries": args.queries,
        "production_input": args.production_input,
        "n_queries": n_queries,
        "output": args.output,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"実験結果を {path} に追記しました")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--build", action="store_true", help="前処理 + 索引構築をする（初回のみ）"
    )
    parser.add_argument(
        "--production-input",
        action="store_true",
        help="task_family / primary_evidence_type を捨てて本番と同じ4フィールドで走らせる"
        "（比較実験ではこちらを使う）",
    )
    args = parser.parse_args()

    cfg = compose_config(
        paths=load_config(args.paths),
        process=load_config(args.process),
        search=load_config(args.search),
        agent=load_config(args.agent),
    )
    preprocessor, retriever, agent = build_pipeline(cfg)

    if args.build:
        chunks_path = Path(cfg["paths"]["chunks"])

        if preprocessor is not None:
            metadata_path = Path(
                cfg.get("paths", {}).get("paper_metadata", "data/paper_metadata.jsonl")
            )
            papers = load_papers(metadata_path)

            chunks = []
            for paper in tqdm(papers, desc="前処理中"):
                chunks.extend(preprocessor.process(paper))

            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with chunks_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            print(f"{len(chunks)} チャンクを {chunks_path} に保存しました")

        else:
            if not chunks_path.exists():
                print(f"エラー: {chunks_path} が存在しません", file=sys.stderr)
                sys.exit(1)
            chunks = load_chunks(chunks_path)

        for indexer in retriever.indexers:
            print(f"  {indexer.name} を構築中...")
            indexer.build(chunks)
        print("索引構築完了")

    else:
        print("既存の索引を読み込み中...")
        for indexer in retriever.indexers:
            try:
                indexer.load()
            except Exception as exc:
                print(
                    f"エラー: {indexer.name} の索引読み込みに失敗しました: {exc}\n"
                    f"先に --build を付けて索引を構築してください。",
                    file=sys.stderr,
                )
                sys.exit(1)
        print("読み込み完了")

    queries = load_queries(Path(args.queries), production_input=args.production_input)
    if args.production_input:
        print("本番と同じ4フィールド（task_family を捨てて）で走らせます")
    print(f"{len(queries)} 件の質問に対して検索中...")

    predictions = []
    for i, query in enumerate(queries):
        pred = agent.run(query)
        predictions.append(pred.to_dict())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} 完了")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"予測結果を {output_path} に書き出しました")

    print("\n採点中...")
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate.py",
            "--gold", "data/validation.jsonl",
            "--pred", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    try:
        metrics = json.loads(result.stdout)["metrics"]
    except (json.JSONDecodeError, KeyError):
        print("採点結果を解釈できなかったので実験ログには残しません", file=sys.stderr)
        return
    log_experiment(args, metrics, len(queries))


if __name__ == "__main__":
    main()
