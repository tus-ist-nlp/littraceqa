"""`--dump-runs` の土台から、本走行と**同じ順序**で候補列を組み直す再生台。

`replay_merge.py` との違いは2つ。どちらも「元の質問を混ぜたら @1 が戻るか」を
測るために要る:

* **`to_gold_papers()` を 50 で切らない。** RRF 統合は「切る前の全長」を
  ランキングA に使う（`_build_prediction()` と同じ）。`replay_merge.rebuild()` は
  `max_papers=50` で切ってしまうので、統合を含む再生には使えない。
* **展開・統合（`_combine_rrf`）と `paper_score_skip_chunk_types` を通す。**
  `anchor_from: verdict` の起点は土台の予測 trace の `selected`（＝読解 LLM が
  最後に返した paper_ids）から復元する。

`--extra-runs` を渡すと、その JSONL を **step0 の末尾に追加のサブクエリ1本として
差し込む**。本実験（元の質問そのものを検索に混ぜる）はこれで測る。

    # 1) 土台の再現を確認する（--extra-runs なし。予測ファイルと一致するはず）
    uv run python scripts/replay_rawq.py \
      --paths configs/paths/default.yaml --process configs/process_style/mineru.yaml \
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
      --agent configs/agent_style/reading_expand_rrf/notable.yaml \
      --runs runs_notable.jsonl --pred predictions_k100_notable.jsonl --verify

    # 2) 元の質問1本を足した版
    uv run python scripts/replay_rawq.py ... --extra-runs runs_rawq.jsonl

**ReadingAgent 本体のメソッドをそのまま呼ぶ**（`_merged_results` / `_rescore_pool` /
`_combine_rrf`）。ロジックを書き写すとオフラインの結論が本走行に効かなくなる。

**chunk_type はダンプに無いので chunk_id から復元する。** `mineru_chunker.py` の
採番が `#c****`(title_abstract / text_span) / `#tab****`(table) / `#fig****`(figure) /
`#eq****`(equation_algorithm) と1対1なので、`skip_chunk_types: [table]` は正確に再現できる。
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
from littraceqa.di_pipeline.config import build_paper_expander, compose_config, load_config
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers

# chunk_id の接尾 -> chunk_type（preprocess/mineru_chunker.py の採番と1対1）
_SUFFIX_TYPES = (("tab", "table"), ("fig", "figure"), ("eq", "equation_algorithm"))


def apply_overrides(params: dict, overrides: list[str]) -> dict:
    """--set key=value を agent yaml の params に反映する（振るとき用）。"""
    params = dict(params)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set は key=value の形で指定する: {item!r}")
        params[key.strip()] = yaml.safe_load(raw)
    return params


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunk_type_of(chunk_id: str) -> str:
    suffix = chunk_id.split("#", 1)[1] if "#" in chunk_id else ""
    for prefix, chunk_type in _SUFFIX_TYPES:
        if suffix.startswith(prefix):
            return chunk_type
    # `#c0000` は title_abstract、それ以外の `#cNNNN` は text_span。
    # どちらも table ではないので、この再生では区別しなくてよい。
    return "title_abstract" if suffix == "c0000" else "text_span"


def load_runs(path: Path) -> dict[str, list[SubqueryRun]]:
    """query_id -> [SubqueryRun]。ファイルの並び順＝本走行の投げた順を保つ。

    `subquery_merge: max` では `_merged_results()` が `chunks` の**挿入順**を
    そのまま返すので、run の順序と各 run 内の順位が候補列に効く。
    """
    by_query: dict[str, list[SubqueryRun]] = {}
    for row in read_jsonl(path):
        results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                score=r["score"],
                text="",
                chunk_type=chunk_type_of(r["chunk_id"]),
                metadata={},
                source=r.get("source", ""),
            )
            for r in row["results"]
        ]
        by_query.setdefault(row["query_id"], []).append(
            SubqueryRun(step=row["step"], subquery=row["subquery"], results=results)
        )
    return by_query


def merge_runs(
    base: list[SubqueryRun], extra: list[SubqueryRun] | None
) -> list[SubqueryRun]:
    """追加の run を step0 の末尾に差し込む（＝5本目のサブクエリとして投げた形）。"""
    if not extra:
        return base
    head = [run for run in base if run.step == 0]
    tail = [run for run in base if run.step != 0]
    return head + list(extra) + tail


def build_candidates(
    agent: ReadingAgent,
    runs: list[SubqueryRun],
    verdict: list[str],
    question: str = "",
) -> list[str]:
    """`_build_prediction()` と同じ順序で候補列を作る。

    マージ -> 剪定 -> 論文単位（**切らない**）-> 生質問1位の固定 -> A/B 統合 -> 50 で切る。

    `rawq_pin` の固定は `_rawq_ranking()` が **step0 の run をサブクエリ文字列で
    引き当てる**ので、`question` に生質問の run と同じ文字列を渡す必要がある
    （`--extra-runs` の各行の `subquery` がそれにあたる）。
    """
    chunks: dict[str, RetrievalResult] = {}
    for run in runs:
        for result in run.results[: agent.retrieve_top_k]:
            previous = chunks.get(result.chunk_id)
            if previous is None or result.score > previous.score:
                chunks[result.chunk_id] = result

    merged = agent._merged_results(runs, chunks)
    merged = agent._rescore_pool(SimpleNamespace(question=""), merged, [])
    candidates = to_gold_papers(merged, skip_chunk_types=agent.paper_score_skip_chunk_types)
    candidates = agent._pin_front(
        candidates, agent._rawq_ranking(SimpleNamespace(question=question), runs)
    )
    if candidates and getattr(agent.paper_expander, "combine", None) == "rrf":
        candidates = agent._combine_rrf(candidates, [], verdict)
    return candidates[:CANDIDATE_PAPERS_LIMIT]


def final_verdict(record: dict, at_step: int | None = None) -> list[str]:
    """土台の trace から、読解 LLM が返した paper_ids を復元する。

    `run()` は verdict を上書きしながら回すので、`selected` が入っている
    最後のステップが最終の verdict にあたる。

    `at_step` を渡すとそのステップ時点の verdict を使う。**`max_steps` を削った
    走行を厳密に再生するために要る**——step N までしか回さない走行の状態は、
    フル走行の step N 時点の状態と（LLM のばらつきを除いて）同一だから。
    起点（`anchor_from: verdict`）を揃えずに run だけ削ると、「step2 の検索は
    切ったのに step2 の読解の成果は使っている」という有り得ない条件になる。
    """
    selected: list[str] = []
    for step in record.get("trace") or []:
        if not isinstance(step, dict):
            continue
        if at_step is not None and step.get("step", 0) > at_step:
            break
        if step.get("selected"):
            selected = list(step["selected"])
    return selected


def recall_at(gold: set[str], ranked: list[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--process", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--runs", required=True, help="--dump-runs の JSONL（土台）")
    parser.add_argument("--pred", required=True, help="同じ走行の予測 JSONL（verdict の復元に使う）")
    parser.add_argument("--extra-runs", default=None, help="step0 に足す追加 run の JSONL")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--output", default=None, help="候補列を差し替えた予測 JSONL")
    parser.add_argument("--ks", default="1,5,10,20,50")
    parser.add_argument(
        "--verdict-at-step",
        type=int,
        default=None,
        help="ランキングB の起点をこのステップ時点の verdict にする（max_steps を削る再生用）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="土台の candidate_papers と一致するかを確認する（--extra-runs なしで使う）",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="agent yaml の params を上書きする（例: --set subquery_merge=rrf）",
    )
    args = parser.parse_args()

    cfg = compose_config(
        load_config(args.paths),
        load_config(args.process),
        load_config(args.search),
        load_config(args.agent),
    )
    agent_cfg = cfg["agent"]
    agent = ReadingAgent(
        retriever=SimpleNamespace(reranker=None),
        llm=None,
        paper_expander=build_paper_expander(agent_cfg),
        **apply_overrides(agent_cfg.get("params", {}), args.overrides),
    )

    runs_by_query = load_runs(Path(args.runs))
    extra_by_query = load_runs(Path(args.extra_runs)) if args.extra_runs else {}
    records = read_jsonl(Path(args.pred))
    gold_rows = {row["query_id"]: row for row in read_jsonl(Path(args.gold))}
    ks = [int(k) for k in args.ks.split(",")]

    # scenario -> k -> 各クエリの recall
    cr: dict[str, dict[int, list[float]]] = {}
    ecr: dict[str, dict[int, list[float]]] = {}
    matched = 0
    changed = 0
    for record in records:
        query_id = record["query_id"]
        runs = runs_by_query.get(query_id)
        if not runs:
            continue
        extra = extra_by_query.get(query_id)
        candidates = build_candidates(
            agent,
            merge_runs(runs, extra),
            final_verdict(record, args.verdict_at_step),
            question=extra[0].subquery if extra else "",
        )
        if candidates == (record.get("candidate_papers") or []):
            matched += 1
        else:
            changed += 1
        record["candidate_papers"] = candidates

        gold_row = gold_rows.get(query_id)
        if gold_row is None:
            continue
        gold = {p["paper_id"] for p in gold_row.get("gold_papers", [])}
        backed = gold & {e["paper_id"] for e in gold_row.get("evidence", [])}
        family = gold_row.get("task_family") or ""
        scenario = "multi" if "multi" in family else "single"
        for name in (scenario, "total"):
            for k in ks:
                cr.setdefault(name, {}).setdefault(k, []).append(recall_at(gold, candidates, k))
                if backed:
                    ecr.setdefault(name, {}).setdefault(k, []).append(
                        recall_at(backed, candidates, k)
                    )

    if args.verify:
        print(f"土台と一致: {matched} 件 / 不一致: {changed} 件")
    else:
        print(f"{matched + changed} 件中 {changed} 件の候補列が変わった")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    for name in ("total", "single", "multi"):
        if name not in cr:
            continue
        n = len(cr[name][ks[0]])
        print(f"\n[{name}] n={n}")
        print(f"{'k':>5}{'cr@k':>9}{'ecr@k':>9}")
        for k in ks:
            print(f"{k:>5}{mean(cr[name][k]):>9.4f}{mean(ecr.get(name, {}).get(k, [])):>9.4f}")

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n候補列を差し替えた予測を {args.output} に書き出しました")


if __name__ == "__main__":
    main()
