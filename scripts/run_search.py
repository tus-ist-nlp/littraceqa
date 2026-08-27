#!/usr/bin/env python3
"""検索を回して予測 jsonl を書き出す。

システムの構成そのものは `littraceqa.di_pipeline.pipeline` にある。ここが受け取るのは
**実行環境の場所（paths）と入出力だけ**で、手法のつまみは引数に出てこない。

使い方:
    # 初回: 前処理 + 索引構築をしてから検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # 2回目以降: 既存の索引を読み込んで検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --production-input

指標が要るときは検索と切り離して scripts/evaluate.py を直接呼ぶ:
    uv run python scripts/evaluate.py --gold data/validation.jsonl --pred predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.di_pipeline.contracts import Chunk, Query
from littraceqa.di_pipeline.pipeline import (
    Paths,
    build_agent,
    build_expander_index,
    build_indexers,
    build_preprocessor,
)


def load_papers(path: Path) -> list[dict]:
    papers = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


# 採点の基準になる gold。分割実行(val_a/val_b)の網羅率もこれで測る。
GOLD_PATH = Path("data/validation.jsonl")


def read_predictions(path: Path) -> dict[str, dict]:
    """予測 jsonl を query_id -> レコードで読む。"""
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record.get("query_id", ""))] = record
    return records


def merge_predictions(output_path: Path, others: list[str]) -> tuple[Path, list[str]]:
    """今回の予測に他の分割実行の予測を結合し、結合後のファイルパスを返す。

    val_a(28件)/val_b(27件) のように分けて回すと、片側だけを 55件の gold に対して
    採点することになり、全 macro 指標が網羅率のぶんだけ薄まる（val_a だけだと約半分）。
    別構成との比較でその数字を使うと誤った結論になるので、もう片方が既にあるなら
    結合してから採点する。query_id が衝突したときは今回の実行を優先する。
    """
    merged = read_predictions(output_path)
    for other in others:
        other_path = Path(other)
        if not other_path.exists():
            print(f"エラー: --merge-with に指定した {other_path} が存在しません", file=sys.stderr)
            sys.exit(1)
        records = read_predictions(other_path)
        overlap = sorted(set(records) & set(merged))
        if overlap:
            print(
                f"警告: {other_path} は今回の予測と {len(overlap)} 件重複しています"
                f"（例: {', '.join(overlap[:3])}）。今回の実行の予測を優先します。",
                file=sys.stderr,
            )
        for query_id, record in records.items():
            merged.setdefault(query_id, record)

    merged_path = output_path.with_name(f"{output_path.stem}_merged{output_path.suffix}")
    with merged_path.open("w", encoding="utf-8") as f:
        for _, record in sorted(merged.items()):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(merged)} 件に結合して {merged_path} に書き出しました（採点はこちらを使います）")
    return merged_path, others


def dump_runs(handle, query_id: str, runs: list) -> None:
    """サブクエリ1本 = 1行で、検索が返した順位とスコアを書き出す。

    テキストは載せない（chunk_id から chunk_store で引ける／土台ファイルが
    数百MBになるため）。オフライン再生に要るのは順位とスコアだけ。
    """
    for run in runs:
        handle.write(
            json.dumps(
                {
                    "query_id": query_id,
                    "step": run.step,
                    "subquery": run.subquery,
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "paper_id": r.paper_id,
                            "rank": rank,
                            "score": r.score,
                            "source": r.source,
                        }
                        for rank, r in enumerate(run.results, 1)
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def check_coverage(scored_path: Path) -> dict[str, Any]:
    """採点対象が gold の何件を覆っているかを返し、欠けていれば警告する。"""
    if not GOLD_PATH.exists():
        return {}
    gold_ids = set(read_predictions(GOLD_PATH))
    pred_ids = set(read_predictions(scored_path))
    covered = len(gold_ids & pred_ids)
    total = len(gold_ids)
    if covered < total:
        print(
            f"\n警告: gold {total}件のうち {covered}件しか予測がありません。"
            f"macro 指標は約 {covered / total:.0%} に薄まった値になります。\n"
            f"        残りの分割を回してから --merge-with {scored_path} を付けて実行すると"
            f"{total}件で採点されます。この行の数字を別構成と比べないでください。\n",
            file=sys.stderr,
        )
    return {"covered": covered, "gold_total": total}


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


# 本番の入力に実際に入っているフィールドはこの4つだけ（確定仕様）。
# multiple_choice の options は**本番では与えられない**ので、ここには入れない。
# `multiple_choice_options` は本番入力に実在する（`data/test_inputs.jsonl` 71件中50件）。
# ただし `Query.from_dict` が読むのは `options`（validation の gold から結合する oracle 用）
# なので、残しても Query には載らない——本番入力の定義を正しく保つためだけに並べてある。
_PRODUCTION_FIELDS = (
    "query_id",
    "question",
    "answer_types",
    "multiple_choice_options",
    "table_schema",
)


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




# reranker の実効設定として記録する属性（インスタンス側の名前 -> 記録名）。
# yaml に書かれていない既定値（instruction / compile など）も残さないと、
# 「この数字がどの設定で出たのか」が後から再現できない。














def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", required=True,
        help="configs/paths/*.yaml（実行環境ごとの pdf_dir / chunks_dir / index_dir）",
    )
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
    parser.add_argument(
        "--dump-runs",
        default=None,
        metavar="RUNS.JSONL",
        help="サブクエリ1本ごとの検索結果（step / subquery / 上位チャンクの順位とスコア）を"
        "書き出す。あとから候補列の組み立てだけをオフラインでやり直すための土台。",
    )
    parser.add_argument(
        "--merge-with",
        nargs="+",
        default=[],
        metavar="PREDICTIONS.JSONL",
        help="他の分割実行の予測 jsonl を結合して1本にまとめる。"
        "val_a(28件)/val_b(27件) のように分けて回したとき、片側だけを 55件の gold に"
        "採点すると macro が網羅率のぶん薄まるため、採点前に結合しておく。",
    )
    args = parser.parse_args()

    paths = Paths.load(args.paths)

    if args.build:
        # 前処理と索引構築だけ。ここではまだチャンクが無いので agent は組み立てない。
        papers = load_papers(paths.paper_metadata)
        chunks = []
        for paper in tqdm(papers, desc="前処理中"):
            chunks.extend(build_preprocessor(paths).process(paper))

        paths.chunks.parent.mkdir(parents=True, exist_ok=True)
        with paths.chunks.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        print(f"{len(chunks)} チャンクを {paths.chunks} に保存しました")

        # 検索の索引3本と、ランキングB が読む SPECTER2 索引。後者は融合に渡らないが、
        # ここで作らないと再構築する方法が無くなる。
        for indexer in [*build_indexers(paths), build_expander_index(paths)]:
            print(f"  {indexer.name} を構築中...")
            indexer.build(chunks)
        print("索引構築完了")

    agent = build_agent(paths)
    print("既存の索引を読み込み中...")
    for indexer in agent.retriever.indexers:
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
        print("本番と同じ4フィールド（query_id/question/answer_types/table_schema）で走らせます")
    print(f"{len(queries)} 件の質問に対して検索中...")

    # --dump-runs: サブクエリ1本ごとの検索結果を別ファイルに落とす。
    # Prediction.trace には入れない（提出ファイルが膨らむため）。候補列の組み立てを
    # あとからオフラインでやり直したいとき用の土台。
    runs_file = open(args.dump_runs, "w", encoding="utf-8") if args.dump_runs else None

    predictions = []
    for i, query in enumerate(queries):
        pred = agent.run(query)
        predictions.append(pred.to_dict())
        if runs_file is not None:
            dump_runs(runs_file, query.query_id, getattr(agent, "last_runs", []))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} 完了")

    if runs_file is not None:
        runs_file.close()
        print(f"サブクエリ単位の検索結果を {args.dump_runs} に書き出しました")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"予測結果を {output_path} に書き出しました")

    # 分割実行を単独で採点すると macro が網羅率のぶん薄まるので、結合しておく。
    scored_path = output_path
    if args.merge_with:
        scored_path, _ = merge_predictions(output_path, args.merge_with)
    check_coverage(scored_path)

    print(
        "\n採点するには:\n"
        f"  uv run python scripts/evaluate.py --gold {GOLD_PATH} --pred {scored_path}"
    )


if __name__ == "__main__":
    main()
