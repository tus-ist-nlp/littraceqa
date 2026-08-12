"""既存の予測ファイルの候補列に、論文→論文展開だけをオフラインでやり直す。

本走行は1構成4〜5時間かかるが、展開・統合は `candidate_papers`（論文IDの列）を
入れ替えるだけの処理で、LLM も検索索引も要らない。予測ファイルを土台にすれば
数秒〜数十秒で回るので、`related_offset` や `neighbors` の調整はここで詰める。

**ReadingAgent 本体のメソッドをそのまま呼ぶ。** ロジックを書き写すとオフラインの
結論が本走行に効かなくなるため、`_combine_rrf` / `_expand_candidates` を再実装しない。

    uv run python scripts/replay_expansion.py \
      --paths configs/paths/default.yaml \
      --process configs/process_style/mineru.yaml \
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter.yaml \
      --agent configs/agent_style/reading_expand_rrf/rrf.yaml \
      --pred predictions_8b_chunk_cand50.jsonl \
      --output predictions_8b_chunk_rrf_offline.jsonl \
      --set related_offset=25

`--set` は agent yaml の `expansion` ブロックを1つだけ上書きする（振るとき用）。

**限界**: 土台の予測に残っているのは候補上位 CANDIDATE_PAPERS_LIMIT(50) 本なので、
ランキングA は50位までしか復元できない。本走行では打ち切り前の全長を使うため、
ここで出る数字は下限として読む。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.agent.reading import CANDIDATE_PAPERS_LIMIT, ReadingAgent
from littraceqa.di_pipeline.config import (
    build_paper_expander,
    compose_config,
    load_config,
)
from littraceqa.di_pipeline.contracts import Query


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_overrides(agent_cfg: dict, overrides: list[str]) -> dict:
    """--set key=value を agent yaml の expansion ブロックに反映する。"""
    if not overrides:
        return agent_cfg
    agent_cfg = dict(agent_cfg)
    expansion = dict(agent_cfg.get("expansion") or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set は key=value の形で指定する: {item!r}")
        expansion[key.strip()] = yaml.safe_load(raw)
    agent_cfg["expansion"] = expansion
    return agent_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument(
        "--search",
        required=True,
        help="configs/search_style/*.yaml（索引パスの導出と、rerank 経路の reranker 構築に使う）",
    )
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--pred", required=True, help="土台にする予測 JSONL")
    parser.add_argument("--output", required=True, help="書き出す予測 JSONL")
    parser.add_argument("--queries", default="data/validation_inputs.jsonl")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="agent yaml の expansion ブロックを上書きする（例: --set related_offset=25）",
    )
    parser.add_argument("--no-eval", action="store_true", help="採点を走らせない")
    args = parser.parse_args()

    agent_yaml = apply_overrides(load_config(args.agent), args.overrides)
    cfg = compose_config(
        load_config(args.paths),
        load_config(args.process),
        load_config(args.search),
        agent_yaml,
    )
    agent_cfg = cfg["agent"]
    expander = build_paper_expander(agent_cfg)
    if expander is None and not (agent_cfg.get("params") or {}).get("title_protect"):
        raise SystemExit(f"{args.agent} に expansion も title_protect もない")

    # rerank 経路（combine を使わない位置挿入）のときだけ reranker を建てる。
    # RRF 統合はランキングB を reranker に晒さないのが眼目なので GPU を触らない。
    reranker = None
    if getattr(expander, "rerank", False) and getattr(expander, "combine", None) != "rrf":
        reranker_cfg = cfg["retriever"]["reranker"]
        reranker = registry.build(
            "reranker", reranker_cfg["name"], **reranker_cfg.get("params", {})
        )

    # 検索も LLM も使わないので、展開に必要な部分だけを持つ ReadingAgent を建てる。
    agent = ReadingAgent(
        retriever=SimpleNamespace(reranker=reranker),
        llm=None,
        paper_expander=expander,
        **agent_cfg.get("params", {}),
    )

    questions = {
        row["query_id"]: row.get("question", "")
        for row in read_jsonl(Path(args.queries))
    }

    records = read_jsonl(Path(args.pred))
    changed = 0
    for record in records:
        candidates = record.get("candidate_papers") or []
        if not candidates:
            continue
        trace = list(record.get("trace") or [])
        # 前回の展開ぶんの trace は捨てる（同じ土台に別設定を重ねるため）。
        trace = [
            t
            for t in trace
            if not ({"paper_expansion", "paper_fusion", "title_protect"} & set(t))
        ]
        query = Query(
            query_id=record["query_id"],
            question=questions.get(record["query_id"], ""),
            answer_types=[],
        )

        if expander is None:
            fused = candidates
        elif getattr(expander, "combine", None) == "rrf":
            fused = agent._combine_rrf(candidates, trace)[:CANDIDATE_PAPERS_LIMIT]
        else:
            fused = agent._expand_candidates(query, candidates, {}, trace)

        # 名指し保護も候補列の並べ替えだけなのでオフラインで再生できる。
        # 本走行と同じく展開・統合のあとに効かせる（_build_prediction と同じ順番）。
        if agent.title_protect is not None and fused:
            fused = agent._apply_title_protect(fused, agent._named_papers(query), trace)

        if fused != candidates:
            changed += 1
        record["candidate_papers"] = fused
        record["trace"] = trace

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} 件中 {changed} 件の候補列を書き換えた -> {output}", file=sys.stderr)

    if args.no_eval:
        return
    subprocess.run(
        ["uv", "run", "python", "scripts/evaluate.py",
         "--gold", args.gold, "--pred", str(output)],
        check=False,
    )


if __name__ == "__main__":
    main()
