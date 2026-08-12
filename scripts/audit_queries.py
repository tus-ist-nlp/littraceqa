#!/usr/bin/env python3
"""クエリ品質監査: (query_id, paper_id) 単位で gold の妥当性を LLM に判定させる。

docs/query_audit_spec.md の実装。gold_papers の各論文について、evidence が
回答の根拠としてどう機能するか（relevance）と、不要な場合の混入理由
（noise_type）を判定し、データセットの事実・検索結果と join した JSONL を書く。

集計（クエリラベル・補正指標）と HTML ビューアは scripts/audit_report.py が
担当する。判定の誤りと描画の誤りを切り分けるため、この分離を崩さない（spec 4.4）。

設計上の決定（spec 6 検討事項への回答）:

- **判定コスト**: gold 論文の全文は投入しない。title_abstract チャンク +
  evidence を含むチャンクの周辺抜粋に絞る。evidence が無い論文だけ本文冒頭を足す。
- **no_evidence の扱い**: relevance = no_evidence は「その論文由来の evidence_id が
  存在しない」という**データセットの事実**なので、LLM に判定させず機械的に確定する
  （LLM が supporting と言っても上書きする）。そのうえで「本文には回答を支持する
  記述がありそうか」を LLM に別フィールド body_supports_answer で答えさせ、
  単なるアノテーション漏れの可能性を残す。

実行例:
    uv run python scripts/audit_queries.py \
      --gold data/validation.jsonl \
      --paths configs/paths/default.yaml \
      --pred predictions_8b_chunk_b_merged.jsonl \
      --output audits/query_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from littraceqa.common import ROOT, compact_text, try_parse_json_object

load_dotenv(ROOT / ".env")

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM  # noqa: F401
from littraceqa.di_pipeline.llm.fake import FakeLLM  # noqa: F401

# LLM に選ばせる relevance（no_evidence は機械的に確定するので LLM には選ばせない）。
LLM_RELEVANCE = ("supporting", "partial", "irrelevant", "contradicting")
# noise_type は「回答に不要」な論文（supporting / partial 以外）にだけ付く。
NOISE_TYPES = (
    "same_topic_different_finding",
    "same_method_different_task",
    "citation_neighbor",
    "shared_author_or_venue",
    "distractor_by_design",
    "annotation_error",
)
CONFIDENCES = ("high", "medium", "low")

ABSTRACT_CHARS = 2500
EVIDENCE_EXCERPT_CHARS = 1200
BODY_EXCERPT_CHARS = 2000


class ChunkStore:
    """mineru_chunks.jsonl から論文単位でチャンクを読む。

    3.8GB のファイルなので、隣の offsets.json（paper_id -> [start, length]）が
    あればシークで読む。無い小さなファイル（テスト等）は初回に全走査してキャッシュ。
    """

    def __init__(self, chunks_path: str | Path):
        self.path = Path(chunks_path)
        self.offsets: dict[str, list[int]] | None = None
        self._scan_cache: dict[str, list[dict[str, Any]]] | None = None
        offsets_path = Path(f"{self.path}.offsets.json")
        if offsets_path.exists():
            payload = json.loads(offsets_path.read_text(encoding="utf-8"))
            self.offsets = payload.get("offsets", payload)

    def paper_chunks(self, paper_id: str) -> list[dict[str, Any]]:
        if self.offsets is not None:
            span = self.offsets.get(paper_id)
            if not span:
                return []
            start, length = span
            with open(self.path, "rb") as handle:
                handle.seek(start)
                data = handle.read(length).decode("utf-8")
            return [json.loads(line) for line in data.splitlines() if line.strip()]

        if self._scan_cache is None:
            self._scan_cache = {}
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    self._scan_cache.setdefault(chunk.get("paper_id", ""), []).append(chunk)
        return self._scan_cache.get(paper_id, [])


def strip_title_prefix(text: str) -> str:
    """チャンク本文の先頭に付く `[ACL 2025] タイトル` 行を落とす（mineru_chunker の仕様）。"""
    raw = str(text or "")
    if "\n" in raw:
        return raw.split("\n", 1)[1]
    return raw


def paper_context(
    chunks: list[dict[str, Any]], evidence_items: list[dict[str, Any]]
) -> dict[str, Any]:
    """判定プロンプトに入れる論文抜粋。abstract + evidence 周辺に絞る（spec 6 判定コスト）。"""
    abstract = ""
    for chunk in chunks:
        if chunk.get("chunk_type") == "title_abstract":
            abstract = compact_text(chunk.get("text"), max_chars=ABSTRACT_CHARS)
            break

    excerpts: list[dict[str, Any]] = []
    for item in evidence_items:
        needle = compact_text(item.get("evidence_text_or_value"))
        chunk_excerpt = ""
        if needle:
            for chunk in chunks:
                text = compact_text(chunk.get("text"))
                pos = text.lower().find(needle.lower())
                if pos >= 0:
                    start = max(0, pos - EVIDENCE_EXCERPT_CHARS // 2)
                    chunk_excerpt = text[start : start + EVIDENCE_EXCERPT_CHARS]
                    break
        excerpts.append(
            {
                "evidence_id": item.get("evidence_id"),
                "evidence_text": needle,
                "source_type": item.get("source_type"),
                "locator": item.get("locator"),
                "chunk_excerpt": chunk_excerpt,
            }
        )

    # evidence が無い論文は abstract だけだと noise_type の判定材料が足りないので
    # 本文冒頭を足す。ある論文では evidence 周辺で足りるため付けない（コスト削減）。
    body_excerpt = ""
    if not evidence_items:
        parts: list[str] = []
        total = 0
        for chunk in chunks:
            if chunk.get("chunk_type") != "text_span":
                continue
            body = compact_text(strip_title_prefix(chunk.get("text", "")))
            if not body:
                continue
            parts.append(body)
            total += len(body)
            if total >= BODY_EXCERPT_CHARS:
                break
        body_excerpt = " ".join(parts)[:BODY_EXCERPT_CHARS]

    return {"abstract": abstract, "evidence_excerpts": excerpts, "body_excerpt": body_excerpt}


def answer_text_of(gold: dict[str, Any]) -> str:
    """gold の answer を判定プロンプト用の1テキストにまとめる。"""
    answer = gold.get("answer") or {}
    parts: list[str] = []
    freeform = (answer.get("freeform") or {}).get("text")
    if freeform:
        parts.append(str(freeform))
    mc = answer.get("multiple_choice") or {}
    if isinstance(mc, dict) and mc.get("gold"):
        options = mc.get("options") or {}
        parts.append(f"選択肢の正解: {mc['gold']} = {options.get(mc['gold'], '')}")
    table = answer.get("table") or {}
    rows = table.get("rows") if isinstance(table, dict) else None
    if rows:
        parts.append("表形式の正解行: " + json.dumps(rows, ensure_ascii=False)[:1500])
    return "\n".join(parts)


def build_prompt(
    gold: dict[str, Any], contexts: dict[str, dict[str, Any]]
) -> str:
    """1クエリ分の判定プロンプト。gold 論文全件を1回で判定させる。

    論文を1件ずつ別々に呼ぶと relation_to_gold（supporting な論文との関係）が
    判定できないため、クエリ単位でまとめて渡す。55クエリ = 55呼び出しで済む。
    """
    lines: list[str] = [
        "あなたは文献QAデータセットの品質監査者です。",
        "以下の質問と正解回答に対し、gold として登録された各論文が回答の根拠として",
        "どう機能するかを判定してください。これは評価データの品質判定であり、",
        "論文自体の良し悪しの評価ではありません。",
        "",
        "# 質問",
        compact_text(gold.get("question"), max_chars=2000),
        "",
        "# 正解回答",
        answer_text_of(gold) or "(回答なし)",
        "",
        f"# gold 論文（{len(contexts)}件）",
    ]
    papers_by_id = {
        str(p.get("paper_id")): p
        for p in gold.get("gold_papers") or []
        if isinstance(p, dict)
    }
    for paper_id, context in contexts.items():
        paper = papers_by_id.get(paper_id, {})
        lines += [
            "",
            f"## {paper_id}",
            f"- タイトル: {paper.get('title', '')}",
            f"- 会議/年: {paper.get('venue', '')} {paper.get('year', '')}",
        ]
        excerpts = context["evidence_excerpts"]
        if excerpts:
            lines.append("- この論文に紐づく evidence:")
            for ex in excerpts:
                locator = json.dumps(ex.get("locator") or {}, ensure_ascii=False)
                lines.append(
                    f"  - [{ex.get('evidence_id')}] ({ex.get('source_type')}, {locator}) "
                    f"値: {ex.get('evidence_text')!r}"
                )
                if ex.get("chunk_excerpt"):
                    lines.append(f"    該当箇所の本文抜粋: {ex['chunk_excerpt']}")
        else:
            lines.append("- evidence: なし（この論文由来の evidence_id が1件も登録されていない）")
        if context.get("abstract"):
            lines.append(f"- abstract: {context['abstract']}")
        if context.get("body_excerpt"):
            lines.append(f"- 本文冒頭の抜粋: {context['body_excerpt']}")

    lines += [
        "",
        "# 判定基準",
        "relevance（evidence がある論文のみ。次の4値から選ぶ）:",
        "- supporting: 回答の記述を直接支持する。この evidence を除くと回答が成立しない",
        "- partial: 文脈や前提を与えるが、回答の主張そのものは含まない",
        "- irrelevant: 回答と論理的なつながりがない。テーマが同じなだけ",
        "- contradicting: 回答と矛盾する。誤答の根拠になりうる",
        "",
        "evidence が無い論文では relevance を \"no_evidence\" と書き、かわりに",
        "body_supports_answer（abstract・本文抜粋から見て、本文に回答を直接支持する",
        "記述が存在しそうか）を true/false で判定する。",
        "",
        "noise_type（回答に不要な論文 = irrelevant / contradicting / no_evidence のみ。",
        "supporting / partial では null）:",
        "- same_topic_different_finding: 研究テーマは同じだが、問われている知見を含まない",
        "- same_method_different_task: 手法・モデルが共通。タスクや対象データが異なる",
        "- citation_neighbor: 正解論文の引用元/引用先。文献グラフ上の近傍",
        "- shared_author_or_venue: 著者・会議が共通。内容的な必然性は薄い",
        "- distractor_by_design: 誤答を誘発する設計に見える。表層的に答えに近い記述を含む",
        "- annotation_error: 関連が説明できない。アノテーションミスと考えられる",
        "",
        "各論文について:",
        "- evidence_role: 回答のどの主張をどう支えるか。supporting 以外なら「どうずれているか」",
        "  を具体的に。evidence が無い論文では「その論文に何が書かれているか」を書く",
        "- relation_to_gold: supporting な gold 論文との具体的な関係（比較対象の手法、",
        "  同じベンチマークを使う後続研究、など本文の記述に即して）",
        "- misleading_risk: この論文が誤答の根拠になりうるか。なりうる場合、どの記述が",
        "  どう誤読を誘うか。リスクが無ければ null",
        "- confidence: high / medium / low",
        "",
        "# 出力フォーマット（JSON のみ。自由記述は日本語）",
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "...",
                        "relevance": "supporting",
                        "evidence_role": "...",
                        "noise_type": None,
                        "relation_to_gold": "...",
                        "misleading_risk": None,
                        "body_supports_answer": None,
                        "confidence": "high",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        f"papers には gold 論文 {len(contexts)}件を全件、上と同じ paper_id で含めること。",
    ]
    return "\n".join(lines)


def normalize_judgment(
    item: dict[str, Any], has_evidence: bool
) -> dict[str, Any]:
    """LLM の1論文分の判定を検証し、機械的に確定できる項目は上書きする。

    - evidence の有無はデータセットの事実なので、no_evidence は LLM の返答に
      関わらず機械的に確定する（逆に evidence がある論文が no_evidence を
      名乗ることも許さない）。
    - noise_type は supporting / partial では必ず null（spec 3.2）。
    """
    error: str | None = None
    if has_evidence:
        relevance = item.get("relevance")
        if relevance not in LLM_RELEVANCE:
            error = f"relevance の値が不正: {relevance!r}"
            relevance = None
    else:
        relevance = "no_evidence"

    noise_type = item.get("noise_type")
    if relevance in ("supporting", "partial"):
        noise_type = None
    elif noise_type is not None and noise_type not in NOISE_TYPES:
        error = (error + " / " if error else "") + f"noise_type の値が不正: {noise_type!r}"
        noise_type = None

    confidence = item.get("confidence")
    if confidence not in CONFIDENCES or error:
        confidence = "low"

    # body_supports_answer は no_evidence の論文にだけ意味を持つ（アノテーション漏れの疑い）。
    body_supports = item.get("body_supports_answer")
    if relevance != "no_evidence" or not isinstance(body_supports, bool):
        body_supports = None

    return {
        "relevance": relevance,
        "evidence_role": str(item.get("evidence_role") or ""),
        "noise_type": noise_type,
        "relation_to_gold": str(item.get("relation_to_gold") or ""),
        "misleading_risk": item.get("misleading_risk") or None,
        "body_supports_answer": body_supports,
        "confidence": confidence,
        **({"judge_error": error} if error else {}),
    }


def audit_query(
    llm: Any,
    gold: dict[str, Any],
    store: ChunkStore,
    rank_of: dict[str, int],
    judge_model: str,
    retries: int = 2,
) -> list[dict[str, Any]]:
    """1クエリを判定し、(query_id, paper_id) 単位のレコード列を返す。"""
    gold_papers = [p for p in gold.get("gold_papers") or [] if isinstance(p, dict)]
    evidence_by_paper: dict[str, list[dict[str, Any]]] = {}
    for item in gold.get("evidence") or []:
        if isinstance(item, dict) and item.get("paper_id"):
            evidence_by_paper.setdefault(str(item["paper_id"]), []).append(item)

    contexts: dict[str, dict[str, Any]] = {}
    for paper in gold_papers:
        paper_id = str(paper.get("paper_id"))
        contexts[paper_id] = paper_context(
            store.paper_chunks(paper_id), evidence_by_paper.get(paper_id, [])
        )

    prompt = build_prompt(gold, contexts)
    judged: dict[str, dict[str, Any]] = {}
    for _ in range(retries + 1):
        payload, ok = try_parse_json_object(llm(prompt))
        if ok:
            for item in payload.get("papers") or []:
                paper_id = str(item.get("paper_id") or "").strip()
                if paper_id in contexts:
                    judged[paper_id] = item
        if set(judged) >= set(contexts):
            break

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    for paper in gold_papers:
        paper_id = str(paper.get("paper_id"))
        evidence_items = evidence_by_paper.get(paper_id, [])
        item = judged.get(paper_id)
        if item is None:
            # リトライしても判定が返らなかった論文。evidence の有無だけは事実として残す。
            item = {"evidence_role": "LLM の判定が得られなかった", "confidence": "low"}
        judgment = normalize_judgment(item, has_evidence=bool(evidence_items))
        records.append(
            {
                "query_id": gold.get("query_id"),
                "paper_id": paper_id,
                "task_family": gold.get("task_family"),
                **judgment,
                "evidence_ids": [e.get("evidence_id") for e in evidence_items],
                "retrieval": {"rank": rank_of.get(paper_id), "score": None},
                "judge_model": judge_model,
                "judged_at": now,
                # ビューア（audit_report.py）がコーパスを再読せず自己完結で描画
                # できるよう、データセットの事実と判定時に見せた抜粋を同梱する。
                "paper": {
                    "title": paper.get("title"),
                    "venue": paper.get("venue"),
                    "year": paper.get("year"),
                },
                "question": gold.get("question"),
                "answer_text": answer_text_of(gold),
                "context": contexts[paper_id],
            }
        )
    return records


def rank_map_of(pred: dict[str, Any]) -> dict[str, int]:
    """予測レコードの candidate_papers（関連度順）から paper_id -> 順位(1始まり)。"""
    ranks: dict[str, int] = {}
    for i, item in enumerate(pred.get("candidate_papers") or []):
        paper_id = item.get("paper_id", "") if isinstance(item, dict) else str(item)
        if paper_id and paper_id not in ranks:
            ranks[paper_id] = i + 1
    return ranks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_judged(path: Path) -> set[str]:
    """--resume 用。出力済み JSONL に含まれる query_id の集合。"""
    if not path.exists():
        return set()
    return {str(r.get("query_id")) for r in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="gold 論文の品質を LLM に判定させ JSONL を書く。")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--paths", default="configs/paths/default.yaml")
    parser.add_argument("--process", default="mineru", help="chunks ファイル名の接頭辞（{process}_chunks.jsonl）")
    parser.add_argument("--pred", default=None, help="検索結果を join する予測 JSONL（candidate_papers の順位を使う）")
    parser.add_argument("--output", default="audits/query_audit.jsonl")
    parser.add_argument("--llm", default="azure_openai", help="registry に登録された LLM 名（fake でドライラン）")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--query-ids", default=None, help="カンマ区切りで対象クエリを絞る")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="出力済みの query_id を飛ばして追記する")
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

    pred_by_id: dict[str, dict[str, Any]] = {}
    if args.pred:
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
            print(f"[{i}/{len(gold_records)}] {query_id}", file=sys.stderr)
            rank_of = rank_map_of(pred_by_id.get(query_id, {}))
            for record in audit_query(llm, gold, store, rank_of, judge_model):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
