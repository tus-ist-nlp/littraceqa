#!/usr/bin/env python3
"""configs/{paths,process_style,search_style,agent_style}/*.yaml を組み合わせて

前処理・索引構築・検索・評価を一気通貫で実行する e2e スクリプト。

使い方:
    # 初回: 前処理 + 索引構築をしてから検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # 2回目以降: 既存の索引を読み込んで検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml \\
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
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.common import config_label
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


def git_sha() -> str | None:
    """実行時のコミットハッシュ。git 管理外なら None（記録は best effort）。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:  # noqa: BLE001 - 由来情報が取れなくても実験は続行する。
        pass
    return None


# reranker の実効設定として記録する属性（インスタンス側の名前 -> 記録名）。
# yaml に書かれていない既定値（instruction / compile など）も残さないと、
# 「この数字がどの設定で出たのか」が後から再現できない。
_RERANKER_EFFECTIVE_ATTRS = {
    "model_name": "model",
    "device": "device",
    "fp16": "fp16",
    "batch_size": "batch_size",
    "max_tokens": "max_tokens",
    "instruction": "instruction",
    "compile": "compile",
}


def _flatten(prefix: str, params: dict | None) -> dict:
    """{"k": 60} -> {"fuser_k": 60}。ネストしたままだと差分比較で読めないため平らにする。"""
    if not isinstance(params, dict):
        return {}
    return {f"{prefix}_{key}": value for key, value in params.items()}


def tuned_params(cfg: dict, retriever_obj: Any = None) -> dict:
    """チューニング対象のパラメータだけを平らな dict にまとめる。

    experiments.jsonl には解決済みの cfg 全体も残すが、そちらは index_dir などの
    環境依存の値も含んで長い。実験を横並び比較するときに見たいのは
    「振ったつまみ」なので、それだけを抜き出す。

    ネストした *_params は平らにする（reranker_max_tokens のように1つずつ列になり、
    1つだけ変えた実験の差分が読めるようにするため）。reranker は yaml に書かない
    既定値も効いてしまうので、組み立て済みインスタンスから実効値を拾う。
    """
    retriever = cfg.get("retriever", {})
    agent = cfg.get("agent", {})
    agent_params = agent.get("params", {})
    fuser = retriever.get("fuser", {})
    reranker = retriever.get("reranker", {})

    # reranker: 実インスタンスの属性を優先し、取れなければ yaml 宣言値にフォールバック。
    reranker_effective = _flatten("reranker", reranker.get("params"))
    obj = getattr(retriever_obj, "reranker", None)
    if obj is not None:
        for attr, label in _RERANKER_EFFECTIVE_ATTRS.items():
            if hasattr(obj, attr):
                reranker_effective[f"reranker_{label}"] = getattr(obj, attr)

    return {
        # 検索側
        "per_index_k": retriever.get("per_index_k"),
        "pool_k": retriever.get("pool_k"),
        "indexers": [ix.get("index_name", ix["name"]) for ix in retriever.get("indexers", [])],
        "fuser": fuser.get("name"),
        **_flatten("fuser", fuser.get("params")),
        "reranker": reranker.get("name"),
        **reranker_effective,
        # エージェント側
        "agent": agent.get("name"),
        "agent_llm": (agent.get("llm") or {}).get("name"),
        **{f"agent_{k}": v for k, v in agent_params.items()},
    }


def log_experiment(
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    cfg: dict,
    retriever_obj: Any = None,
    coverage: dict | None = None,
    merged_from: list[str] | None = None,
) -> None:
    """どの組み合わせで何点だったかを results/experiments.jsonl に追記する。

    設定ファイルの「パス」だけだと、同じ yaml を書き換えて振った実験が
    全部同じ行に見えてしまい、後から「この数字はどのパラメータで出たのか」が
    追えない。compose_config() が解決した実際の値ごと残す。

    coverage は gold の何件を採点できたか。分割実行を単独で採点した行は
    macro が薄まっているので、後から比較対象に選ばないようここに残す。
    """
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
        **({"coverage": coverage} if coverage else {}),
        **({"merged_from": merged_from} if merged_from else {}),
        "git_sha": git_sha(),
        "tuned_params": tuned_params(cfg, retriever_obj),
        "config": cfg,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"実験結果を {path} に追記しました")


def _load_matching_experiments(
    process: str, search: str, agent: str, n_queries: int | None = None, limit: int = 3
) -> list[dict]:
    """results/experiments.jsonl から同じ組み合わせの過去記録を、直近 limit 件取り出す。

    n_queries を渡すと件数が同じ記録だけに絞る。分割実行(val_a 28件 / val_b 27件)は
    同じ組み合わせの別の行として残るので、絞らないと「今回の実行の片割れ」を
    前回として渡してしまい、LLM が存在しない改善(28件の薄まった値 -> 55件の値)を
    書く。件数が違う行はそもそも比較対象にならない。
    """
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
            if (record.get("process"), record.get("search"), record.get("agent")) != (
                process,
                search,
                agent,
            ):
                continue
            if n_queries is not None and record.get("n_queries") != n_queries:
                continue
            matches.append(record)
    return matches[-limit:]


def generate_comment(llm, args: argparse.Namespace, metrics: dict, n_queries: int) -> str:
    """指標を読んで、LLM に簡潔な所感を書かせる。llm が無ければ固定文言を返す。

    log_experiment() が今回の記録を results/experiments.jsonl に追記した後に
    呼ばれる前提なので、直近の一致レコードの末尾(=今回分)を除いて過去分だけを渡す。
    """
    if llm is None:
        return "(LLMコメントなし: このagent_styleはLLMを使用しない設定です)"

    history = _load_matching_experiments(
        args.process, args.search, args.agent, n_queries=n_queries, limit=4
    )[:-1]
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
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    comment: str,
    cfg: dict,
    retriever_obj: Any = None,
    coverage: dict | None = None,
    merged_from: list[str] | None = None,
) -> None:
    """1回の実行につき、設定・指標・LLMコメントをまとめた Markdown を1枚書く。

    クエリ1件ごとの診断は書かない（`scripts/audit_report.py` が作る単一HTMLに集約した）。
    """
    # agent_style はフォルダで分類してあるので、stem だけだと reading_loop/rrf と
    # reading_expand_rrf/rrf が同じ "rrf" になる（config_label がフォルダ名を前に付ける）。
    process_name = config_label(args.process)
    search_name = config_label(args.search)
    agent_name = config_label(args.agent)
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
        f"- queries: `{args.queries}` (採点 {n_queries}件, "
        f"production_input={args.production_input})",
        f"- output: `{args.output}`",
    ]
    sha = git_sha()
    if sha:
        lines.append(f"- git: `{sha[:12]}`")
    if merged_from:
        lines.append(
            "- 分割実行を結合して採点: " + ", ".join(f"`{p}`" for p in merged_from)
        )
    if coverage and coverage["covered"] < coverage["gold_total"]:
        lines.append(
            f"- **警告: gold {coverage['gold_total']}件のうち {coverage['covered']}件しか"
            f"予測が無い。** 下の macro 指標は約 {coverage['covered'] / coverage['gold_total']:.0%} に"
            "薄まった値であり、他の構成と比較してはいけない。残りの分割を回して "
            "`--merge-with` で結合し直すこと。"
        )
    # yaml は後から書き換わるので、レポート単体で「どの値で回したか」が分かるように
    # 解決済みのパラメータをここに焼き込む。
    lines.extend(
        [
            "",
            "## 設定（この実行時の実際の値）",
            "",
            "| パラメータ | 値 |",
            "|---|---|",
        ]
    )
    for key, value in tuned_params(cfg, retriever_obj).items():
        if value is None:
            continue
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)}` |")
    lines.extend(
        [
            "",
            "## 指標",
            "",
            "| 指標 | 値 |",
            "|---|---|",
        ]
    )
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
    parser.add_argument(
        "--dump-runs",
        default=None,
        metavar="RUNS.JSONL",
        help="サブクエリ1本ごとの検索結果（step / subquery / 上位チャンクの順位とスコア）を"
        "書き出す。scripts/replay_merge.py がこれを土台に、サブクエリ間マージ・検索の深さ・"
        "プールの剪定をオフラインで振れる（本走行4〜5時間に対して数十秒）。"
        "土台にするなら retrieve_top_k を大きめ（例: reading_normal/fat.yaml の100）にして"
        "1回だけ回す。",
    )
    parser.add_argument(
        "--merge-with",
        nargs="+",
        default=[],
        metavar="PREDICTIONS.JSONL",
        help="他の分割実行の予測 jsonl を結合してから採点する。"
        "val_a(28件)/val_b(27件) のように分けて回したとき、片側だけを 55件の gold に"
        "採点すると macro が網羅率のぶん薄まる。2本目の実行にこれを付ければ"
        "55件で採点された行が results/ と report/ に残る。",
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
        print("本番と同じ4フィールド（query_id/question/answer_types/table_schema）で走らせます")
    print(f"{len(queries)} 件の質問に対して検索中...")

    # --dump-runs: サブクエリ1本ごとの検索結果を別ファイルに落とす。
    # Prediction.trace には入れない（提出ファイルが膨らむため）。
    # scripts/replay_merge.py がこれを土台に、マージ方法や深さをオフラインで振る。
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

    # 分割実行を単独で採点すると macro が薄まるので、結合できるなら結合してから採点する。
    scored_path = output_path
    merged_from: list[str] = []
    if args.merge_with:
        scored_path, merged_from = merge_predictions(output_path, args.merge_with)
    coverage = check_coverage(scored_path)

    print("\n採点中...")
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate.py",
            "--gold", str(GOLD_PATH),
            "--pred", str(scored_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    try:
        scored = json.loads(result.stdout)
        metrics = scored["metrics"]
    except (json.JSONDecodeError, KeyError):
        print(result.stdout)
        print("採点結果を解釈できなかったので実験ログには残しません", file=sys.stderr)
        return
    print(json.dumps({k: v for k, v in scored.items() if k != "per_query"},
                     ensure_ascii=False, indent=2))
    # 記録に残すのは「採点した件数」。結合したなら今回回した件数ではなく結合後の件数。
    n_scored = coverage.get("covered", len(queries))
    log_experiment(args, metrics, n_scored, cfg, retriever, coverage, merged_from)
    comment = generate_comment(getattr(agent, "llm", None), args, metrics, n_scored)
    write_report(args, metrics, n_scored, comment, cfg, retriever, coverage, merged_from)


if __name__ == "__main__":
    main()
