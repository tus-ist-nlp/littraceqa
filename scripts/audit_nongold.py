#!/usr/bin/env python3
"""検索上位に入った「gold でない論文」を LLM に判定させ JSONL を書く。

scripts/audit_queries.py が gold 論文の品質を判定するのに対し、こちらは
**候補上位 N 本のうち gold でないもの**を対象にする。知りたいのは2つ:

- そのクエリとどんな関係があるのか（あるいは無いのか）
- なぜ gold 論文より上位に来たのか

判定は audit_queries.py と同じ設計を踏襲する:

- 1クエリ1回の LLM 呼び出しにまとめる（論文を個別に呼ぶと「gold と比べてなぜ上か」
  を判定できない。gold の情報を同じプロンプトに入れる必要がある）
- 論文全文は入れず title + abstract に絞る。gold 側だけ evidence の所在を渡す
  （「gold の根拠が本文の奥にあって abstract に出ない」という理由を判定させるため）
- 順位は予測ファイル（candidate_papers）から機械的に取り、LLM には推測させない

実行例:
    uv run python scripts/audit_nongold.py \
      --pred predictions_8b_chunk_expand_fused_offline.jsonl \
      --output audits/nongold_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_queries import (  # noqa: E402
    ABSTRACT_CHARS,
    ChunkStore,
    answer_text_of,
    rank_map_of,
    read_jsonl,
)

from littraceqa.common import ROOT, compact_text, try_parse_json_object  # noqa: E402

load_dotenv(ROOT / ".env")

from littraceqa.di_pipeline import registry  # noqa: E402
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM  # noqa: F401,E402
from littraceqa.di_pipeline.llm.fake import FakeLLM  # noqa: F401,E402

# クエリとの関係。「無関係」も選べるようにしておく（無いなら無いと言わせる）。
RELATIONS = (
    "same_topic_different_finding",  # 同じ主題だが問われている事実は持たない
    "same_method_different_task",  # 同じ手法を別のタスクに使っている
    "cited_or_baseline",  # gold が引用・比較しているベースライン側
    "peer_of_gold",  # gold と並ぶ同クラスタの論文（質問文は名指ししていない）
    "shared_terminology_only",  # 用語が重なるだけで主題が違う
    "unrelated",  # 関係が見当たらない
    "possible_gold",  # 実は回答の根拠になりうる（gold 付与漏れの疑い）
)

# gold より上位に来た理由。検索側の打ち手に結びつく粒度で切る。
OUTRANK_CAUSES = (
    "query_terms_verbatim",  # 質問文の語がタイトル/abstract にそのまま出る
    "topic_centroid",  # その主題の代表的な論文で埋め込みが近い
    "gold_evidence_is_deep",  # gold の根拠が本文の奥・表・図にあり abstract に出ない
    "gold_is_unnamed_peer",  # gold 側が質問に名指しされないピアなので上がりようがない
    "question_is_underspecified",  # 質問文だけでは gold を一意に指せない
    "not_applicable",  # gold より上位ではない
)

CONFIDENCES = ("high", "medium", "low")


def paper_brief(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """title_abstract チャンクから title と abstract を取り出す。"""
    for chunk in chunks:
        if chunk.get("chunk_type") == "title_abstract":
            meta = chunk.get("metadata") or {}
            return {
                "title": str(meta.get("title") or ""),
                "venue": str(meta.get("venue") or ""),
                "year": str(meta.get("year") or ""),
                "abstract": compact_text(chunk.get("text"), max_chars=ABSTRACT_CHARS),
            }
    return {"title": "", "venue": "", "year": "", "abstract": ""}


def evidence_summary(gold: dict[str, Any], paper_id: str) -> str:
    """その gold 論文に紐づく evidence の所在を1行にまとめる。

    本文の奥・表・図にしか無いのか、abstract に出る話なのかを LLM に判断させる材料。
    """
    items = [e for e in (gold.get("evidence") or []) if str(e.get("paper_id")) == paper_id]
    if not items:
        return "evidence なし（質問文が名指ししていないピア論文）"
    parts = []
    for item in items:
        loc = item.get("locator") or {}
        where = loc.get("table_id") or loc.get("figure_id") or loc.get("section") or ""
        parts.append(f"{item.get('source_type')} p.{loc.get('page')} {where}".strip())
    return " / ".join(parts)


def build_prompt(
    gold: dict[str, Any],
    gold_briefs: list[dict[str, Any]],
    nongold_briefs: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "あなたは文献検索システムの誤りを分析する監査者です。",
        "ある質問に対して検索システムが返した上位の論文のうち、",
        "**gold（正解）として登録されていない論文**について、",
        "(1) その質問とどんな関係があるのか（無いなら無いと言う）",
        "(2) なぜ gold 論文より上位に来たのか",
        "を判定してください。論文の良し悪しの評価ではありません。",
        "",
        "# 質問",
        compact_text(gold.get("question"), max_chars=2000),
        "",
        "# 正解回答",
        compact_text(answer_text_of(gold), max_chars=2000),
        "",
        "# gold 論文（正解。比較対象として渡す）",
    ]
    for b in gold_briefs:
        rank = b["rank"] if b["rank"] is not None else "候補圏外"
        lines += [
            f"## {b['paper_id']}  検索順位: {rank}",
            f"タイトル: {b['title']}  [{b['venue']} {b['year']}]",
            f"この論文の根拠の所在: {b['evidence_where']}",
            f"abstract: {b['abstract']}",
            "",
        ]
    lines += [
        "# 判定対象（gold でないのに上位に来た論文）",
        "",
        "「この論文より下位の gold」は機械的に数えた事実です。1本でもあれば"
        "その論文は gold を追い越しているので、必ず理由を答えてください"
        "（not_applicable にしてよいのは 0本 のときだけ）。",
        "",
    ]
    for b in nongold_briefs:
        lines += [
            f"## {b['paper_id']}  検索順位: {b['rank']}",
            f"タイトル: {b['title']}  [{b['venue']} {b['year']}]",
            f"この論文より下位の gold: {b['golds_below']}本"
            f"（うち候補圏外: {b['golds_missing']}本）",
            f"abstract: {b['abstract']}",
            "",
        ]
    lines += [
        "# 出力",
        "次の形の JSON オブジェクトだけを出力してください（説明文や ```json は付けない）。",
        "papers には判定対象の論文を1件ずつ、渡された全件ぶん並べること。",
        "",
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "対象の paper_id",
                        "relation": f"次のいずれか: {', '.join(RELATIONS)}",
                        "relation_detail": "質問とこの論文の関係を1〜2文で。"
                        "無関係なら何が一致して拾われたのかを書く",
                        "outrank_cause": f"次のいずれか: {', '.join(OUTRANK_CAUSES)}",
                        "outrank_detail": "gold より上位に来た理由を1〜2文で。質問文・タイトル・"
                        "abstract のどの語が効いたと考えられるかを具体的に書く",
                        "could_be_gold": "true/false。この論文だけで質問に答えられるなら true",
                        "confidence": f"次のいずれか: {', '.join(CONFIDENCES)}",
                    }
                ]
            },
            ensure_ascii=False,
            indent=1,
        ),
        "",
        "注意:",
        "- 検索順位と「この論文より下位の gold」の本数は与えられた値をそのまま使う。推測しない。",
        "- 「この論文より下位の gold」が 0本 のときだけ outrank_cause を not_applicable にする。",
        "- could_be_gold は真偽値で答える（文字列 \"false\" ではなく false）。"
        "質問の答えがこの論文の中にあると言える場合だけ true。",
        "- 「同じ分野だから」で済ませず、質問文のどの語・どの概念が効いたのかを書くこと。",
        "- gold の根拠が表・図・本文の奥にしか無い場合、abstract で勝てないのは"
        "  gold_evidence_is_deep に当たる。",
    ]
    return "\n".join(lines)


def as_bool(value: Any) -> bool:
    """"false" / "no" / "" を False に落とす。LLM は真偽値を文字列で返すことがある。"""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("true", "yes", "1")


def normalize(payload: Any, valid_ids: set[str]) -> list[dict[str, Any]]:
    """LLM 返答を正規化する。未知の値は None に落として集計時に弾けるようにする。"""
    rows = payload if isinstance(payload, list) else payload.get("papers") or []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in valid_ids:
            continue
        relation = str(row.get("relation") or "")
        cause = str(row.get("outrank_cause") or "")
        confidence = str(row.get("confidence") or "")
        out.append(
            {
                "paper_id": paper_id,
                "relation": relation if relation in RELATIONS else None,
                "relation_detail": compact_text(row.get("relation_detail"), max_chars=1000),
                "outrank_cause": cause if cause in OUTRANK_CAUSES else None,
                "outrank_detail": compact_text(row.get("outrank_detail"), max_chars=1000),
                # LLM は "false" という文字列を返すことがある。bool() に通すと
                # 非空文字列なので true になってしまうため、明示的に判定する。
                "could_be_gold": as_bool(row.get("could_be_gold")),
                "confidence": confidence if confidence in CONFIDENCES else None,
            }
        )
    return out


def audit_query(
    llm: Any,
    gold: dict[str, Any],
    store: ChunkStore,
    rank_of: dict[str, int],
    top_n: int,
    judge_model: str,
) -> list[dict[str, Any]]:
    query_id = str(gold.get("query_id"))
    gold_ids = {str(g.get("paper_id")) for g in (gold.get("gold_papers") or [])}
    ranked = sorted(rank_of, key=lambda p: rank_of[p])[:top_n]
    nongold_ids = [p for p in ranked if p not in gold_ids]
    if not nongold_ids:
        return []

    gold_briefs = []
    for paper_id in sorted(gold_ids):
        brief = paper_brief(store.paper_chunks(paper_id))
        gold_briefs.append(
            {
                **brief,
                "paper_id": paper_id,
                "rank": rank_of.get(paper_id),
                "evidence_where": evidence_summary(gold, paper_id),
            }
        )
    nongold_briefs = []
    for pid in nongold_ids:
        rank = rank_of[pid]
        # 「この論文より下位の gold」を機械的に数えて渡す。LLM に順位比較をさせない
        # （gold が複数ある multi_paper では「gold より上位か」が一意に決まらないため）。
        below = [g for g in gold_ids if rank_of.get(g) is None or rank_of[g] > rank]
        nongold_briefs.append(
            {
                **paper_brief(store.paper_chunks(pid)),
                "paper_id": pid,
                "rank": rank,
                "golds_below": len(below),
                "golds_missing": sum(1 for g in below if rank_of.get(g) is None),
            }
        )

    prompt = build_prompt(gold, gold_briefs, nongold_briefs)
    payload, ok = try_parse_json_object(llm(prompt))
    judged = normalize(payload, set(nongold_ids)) if ok else []
    by_id = {row["paper_id"]: row for row in judged}

    records = []
    for brief in nongold_briefs:
        paper_id = brief["paper_id"]
        best_gold = min([r for r in (rank_of.get(g) for g in gold_ids) if r], default=None)
        records.append(
            {
                "query_id": query_id,
                "paper_id": paper_id,
                "task_family": gold.get("task_family"),
                "rank": brief["rank"],
                # 「少なくとも1本の gold より上か」。圏外 gold があれば真になる。
                "outranks_some_gold": any(
                    rank_of.get(g) is None or rank_of[g] > brief["rank"] for g in gold_ids
                ),
                "outranks_all_gold": best_gold is None or brief["rank"] < best_gold,
                "paper": {k: brief[k] for k in ("title", "venue", "year")},
                "question": gold.get("question"),
                **{
                    k: by_id.get(paper_id, {}).get(k)
                    for k in (
                        "relation",
                        "relation_detail",
                        "outrank_cause",
                        "outrank_detail",
                        "could_be_gold",
                        "confidence",
                    )
                },
                "judge_model": judge_model,
            }
        )
    return records


def already_judged(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(r.get("query_id")) for r in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="検索上位の非 gold 論文について、クエリとの関係と上位化の理由を判定する。"
    )
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--paths", default="configs/paths/default.yaml")
    parser.add_argument("--process", default="mineru")
    parser.add_argument("--pred", required=True, help="candidate_papers を持つ予測 JSONL")
    parser.add_argument("--top-n", type=int, default=5, help="上位何本までを対象にするか")
    parser.add_argument("--output", default="audits/nongold_audit.jsonl")
    parser.add_argument("--llm", default="azure_openai")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    paths = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    store = ChunkStore(f"{paths['chunks_dir']}/{args.process}_chunks.jsonl")

    llm_params: dict[str, Any] = {}
    if args.llm == "azure_openai":
        llm_params["reasoning_effort"] = args.reasoning_effort
    llm = registry.build("llm", args.llm, **llm_params)
    judge_model = getattr(llm, "deployment", None) or args.llm

    gold_records = read_jsonl(Path(args.gold))
    if args.query_ids:
        wanted = {q.strip() for q in args.query_ids.split(",") if q.strip()}
        gold_records = [g for g in gold_records if str(g.get("query_id")) in wanted]
    if args.limit:
        gold_records = gold_records[: args.limit]

    pred_by_id = {str(r.get("query_id")): r for r in read_jsonl(Path(args.pred))}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    skip = already_judged(output) if args.resume else set()
    mode = "a" if args.resume else "w"

    with output.open(mode, encoding="utf-8") as handle:
        for i, gold in enumerate(gold_records, start=1):
            query_id = str(gold.get("query_id"))
            if query_id in skip:
                continue
            print(f"[{i}/{len(gold_records)}] {query_id}", file=sys.stderr, flush=True)
            rank_of = rank_map_of(pred_by_id.get(query_id, {}))
            for record in audit_query(llm, gold, store, rank_of, args.top_n, judge_model):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
