#!/usr/bin/env python3
"""configs/{paths,process_style,search_style,agent_style}/*.yaml を組み合わせて

前処理・索引構築・検索・評価を一気通貫で実行する e2e スクリプト。

使い方:
    # 初回: 前処理 + 索引構築をしてから検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/abstract_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # 2回目以降: 既存の索引を読み込んで検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/abstract_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
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

from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.config import build_pipeline, compose_config, load_config
from littraceqa.di_pipeline.contracts import Chunk, Query


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


def _load_matching_experiments(
    process: str, search: str, agent: str, limit: int = 3
) -> list[dict]:
    """results/experiments.jsonl から同じ組み合わせの過去記録を、直近 limit 件取り出す。"""
    path = Path("results/experiments.jsonl")
    if not path.exists():
        return []
    matches = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("process"), record.get("search"), record.get("agent")) == (
                process,
                search,
                agent,
            ):
                matches.append(record)
    return matches[-limit:]


def generate_comment(llm, args: argparse.Namespace, metrics: dict, n_queries: int) -> str:
    """指標を読んで、LLM に簡潔な所感を書かせる。llm が無ければ固定文言を返す。

    log_experiment() が今回の記録を results/experiments.jsonl に追記した後に
    呼ばれる前提なので、直近の一致レコードの末尾(=今回分)を除いて過去分だけを渡す。
    """
    if llm is None:
        return "(LLMコメントなし: このagent_styleはLLMを使用しない設定です)"

    history = _load_matching_experiments(args.process, args.search, args.agent, limit=4)[:-1]
    history_text = "\n".join(
        f"- {record['timestamp']}: {json.dumps(record['metrics'], ensure_ascii=False)}"
        for record in history
    ) or "(同じ組み合わせの過去記録なし)"

    prompt = (
        "あなたは検索システムの実験結果を確認する研究者です。次の実験結果を読み、"
        "指標の良し悪しや気になる点、次に試すとよさそうなことを日本語で簡潔にコメントしてください。\n\n"
        f"設定: process={args.process}, search={args.search}, agent={args.agent}\n"
        f"クエリ数: {n_queries} (production_input={args.production_input})\n"
        f"今回の指標: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        f"同じ組み合わせの過去の実行記録(古い順):\n{history_text}\n\n"
        '出力は JSON のみとし、{"comment": "..."} の形式で3〜5文程度にまとめてください。'
    )
    try:
        parsed = parse_json_object(llm(prompt))
    except Exception as exc:
        return f"(LLMコメントの生成に失敗しました: {exc})"
    if not parsed or not isinstance(parsed.get("comment"), str):
        return "(LLMコメントの生成に失敗しました: 応答をパースできませんでした)"
    return parsed["comment"]


def write_report(
    args: argparse.Namespace, metrics: dict, n_queries: int, comment: str
) -> None:
    """1回の実行につき、設定・指標・LLMコメントをまとめた Markdown を report/ に1枚書く。"""
    process_name = Path(args.process).stem
    search_name = Path(args.search).stem
    agent_name = Path(args.agent).stem
    now = datetime.now()

    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{process_name}_{search_name}_{agent_name}.md"
    path = report_dir / filename

    lines = [
        f"# {process_name} + {search_name} + {agent_name}",
        "",
        f"- 実行日時: {now.isoformat(timespec='seconds')}",
        f"- paths: `{args.paths}`",
        f"- process: `{args.process}`",
        f"- search: `{args.search}`",
        f"- agent: `{args.agent}`",
        f"- queries: `{args.queries}` ({n_queries}件, production_input={args.production_input})",
        f"- output: `{args.output}`",
        "",
        "## 指標",
        "",
        "| 指標 | 値 |",
        "|---|---|",
    ]
    for key, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")
    lines.extend(["", "## コメント", "", comment, ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"レポートを {path} に書き出しました")


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
    comment = generate_comment(getattr(agent, "llm", None), args, metrics, len(queries))
    write_report(args, metrics, len(queries), comment)


if __name__ == "__main__":
    main()
