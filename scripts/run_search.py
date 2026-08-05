#!/usr/bin/env python3
"""configs/{paths,process_style,search_style,agent_style}/*.yaml を組み合わせて

前処理・索引構築・検索・評価を一気通貫で実行する e2e スクリプト。

使い方:
    # 初回: scripts/sync_official_release.py で公式入力を取得し、
    # 前処理 + 索引構築をしてから検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \\
      --output predictions.jsonl \\
      --build

    # 2回目以降: 既存の索引を読み込んで検索
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \\
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

from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.candidate_handoff import production_query_from_record
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


def load_mc_options(path: Path) -> dict[str, dict]:
    """Read public option text from current or legacy JSONL shapes.

    Current official input rows carry ``multiple_choice_options`` directly.
    ``--options-file`` remains only as a migration aid for the repository's old
    validation snapshot, whose public option text lived under
    ``answer.multiple_choice.options``. The gold label is never read.
    """

    options_map: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_options = record.get("multiple_choice_options")
            if raw_options is None:
                raw_options = record.get("options")
            if raw_options is None:
                answer = record.get("answer")
                answer = answer if isinstance(answer, dict) else {}
                mc = answer.get("multiple_choice")
                mc = mc if isinstance(mc, dict) else {}
                raw_options = mc.get("options")
            if raw_options is None:
                continue
            parsed = Query.from_dict(
                {
                    "query_id": str(record.get("query_id") or "options"),
                    "question": "options compatibility record",
                    "answer_types": ["multiple_choice"],
                    "options": raw_options,
                }
            )
            options_map[str(record["query_id"])] = parsed.options or {}
    return options_map


def load_queries(
    path: Path, production_input: bool = True, options_path: Path | None = None
) -> list[Query]:
    """クエリを読み込む。

    既定では本番入力に無いフィールド
    （task_family / primary_evidence_type など）を捨ててから Query を作る。
    手元の validation_inputs.jsonl は55件すべてに task_family が入っているが、
    本番入力には無い。task_family は提出論文数（cutoff）を決めるのに使うので、
    これを与えたまま評価すると「正解を教えてもらった状態」の点数になり、
    本番の点数と乖離する。production_input=False は過去実験の再現専用で、
    その結果を本番相当の比較には使わないこと。

    現行の公式 multiple-choice 入力には ``multiple_choice_options`` が入る。
    options_path は、これを持たない古いvalidationスナップショットへ公開済みの
    選択肢本文だけを補う後方互換用で、gold label は読み込まない。
    """
    options_map = load_mc_options(options_path) if options_path else {}
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record.get("multiple_choice_options") is None
                and record.get("options") is None
                and record["query_id"] in options_map
            ):
                record["options"] = options_map[record["query_id"]]
            query = (
                production_query_from_record(record)
                if production_input
                else Query.from_dict(record)
            )
            queries.append(query)
    return queries


def should_evaluate_against_validation(
    queries: list[Query], validation_gold_path: Path
) -> bool:
    """Return true only for the complete public validation query set.

    The search runner also accepts held-out ``test`` and ``test_extra`` input
    files.  Scoring either of those against validation gold silently produces
    meaningless zero-heavy metrics and writes them to the experiment log.  A
    partial validation run has the same dilution problem, so automatic scoring
    is allowed only when both unique query-id sets match exactly.
    """

    if not validation_gold_path.is_file() or not queries:
        return False
    query_ids = [query.query_id for query in queries]
    gold_ids = [
        str(record.get("query_id") or "")
        for record in load_papers(validation_gold_path)
    ]
    return (
        len(query_ids) == len(set(query_ids))
        and len(gold_ids) == len(set(gold_ids))
        and set(query_ids) == set(gold_ids)
    )


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
    options_joined: bool = False,
) -> None:
    """どの組み合わせで何点だったかを results/experiments.jsonl に追記する。

    設定ファイルの「パス」だけだと、同じ yaml を書き換えて振った実験が
    全部同じ行に見えてしまい、後から「この数字はどのパラメータで出たのか」が
    追えない。compose_config() が解決した実際の値ごと残す。

    options_joined は古い入力へ公開選択肢を互換結合した実行かどうか。
    現行の公式入力は選択肢を直接含むが、由来を再現できるよう記録に残す。
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
        "options_joined": options_joined,
        "n_queries": n_queries,
        "output": args.output,
        "git_sha": git_sha(),
        "tuned_params": tuned_params(cfg, retriever_obj),
        "config": cfg,
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
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    comment: str,
    cfg: dict,
    retriever_obj: Any = None,
    options_joined: bool = False,
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
    ]
    sha = git_sha()
    if sha:
        lines.append(f"- git: `{sha[:12]}`")
    if options_joined:
        lines.append(
            "- **[compat] 古い入力スナップショットへ公開済みの multiple-choice "
            "options を補完**"
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with production-safe input projection by default."""

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
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--production-input",
        dest="production_input",
        action="store_true",
        default=True,
        help="開発専用フィールドを捨てる（既定。本番相当の安全な入力契約）",
    )
    input_mode.add_argument(
        "--include-development-fields",
        dest="production_input",
        action="store_false",
        help="過去実験の再現専用。validation限定フィールドを残す（本番比較不可）",
    )
    parser.add_argument(
        "--options-file",
        default=None,
        help="古い入力に公開multiple-choice optionsだけを補う互換JSONL。"
        "現行公式入力では不要。gold labelは読まない。",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

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

    # 現行入力は multiple_choice_options を直接含む。options_path は古いvalidation
    # スナップショットを移行する場合だけ使う。
    options_path = Path(args.options_file) if args.options_file else None
    queries = load_queries(
        Path(args.queries),
        production_input=args.production_input,
        options_path=options_path,
    )
    if args.production_input:
        print(
            "現行の公式入力契約（benchmark、multiple_choice_options、table_schemaの"
            "条件付きフィールドを含む）で走らせます"
        )
    else:
        print(
            "[development-only] validation限定フィールドを残します。"
            "この実行結果は本番相当の比較や公式提出には使えません。",
            file=sys.stderr,
        )
    if options_path is not None:
        n_opt = sum(1 for q in queries if q.options)
        print(
            f"[compat] multiple-choice options を {options_path} から補完しました"
            f"（optionsを持つ質問: {n_opt}件）"
        )
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

    validation_gold_path = Path("data/validation.jsonl")
    if not should_evaluate_against_validation(queries, validation_gold_path):
        print(
            "\n採点をスキップしました: 入力query_id集合が公開validation 55件と"
            "完全一致しません。test/test_extraや部分実行をvalidation goldで採点・"
            "実験記録しません。"
        )
        return

    print("\n採点中...")
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate.py",
            "--gold", str(validation_gold_path),
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
    options_joined = options_path is not None
    log_experiment(args, metrics, len(queries), cfg, retriever, options_joined)
    comment = generate_comment(getattr(agent, "llm", None), args, metrics, len(queries))
    write_report(args, metrics, len(queries), comment, cfg, retriever, options_joined)


if __name__ == "__main__":
    main()
