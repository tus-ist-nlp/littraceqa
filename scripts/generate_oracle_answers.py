#!/usr/bin/env python3
"""gold paper(正解論文)をそのまま渡し、回答生成(answer generation)とevidence選定を
まとめて測る簡易オラクルスクリプト。

retrieval自体は一切せず、「gold paperの全チャンクをLLMに1回読ませたら
multiple_choice/freeform/tableにどれだけ正しく答えられ、根拠(evidence)チャンクを
正しく選べるか」という回答生成の上限(ceiling)を測る。
src/littraceqa/di_pipeline/agent/reading.py の ReadingAgent は Answer
(freeform/multiple_choice/table)をまだ埋めない(gold paper/evidence特定までが仕事)
ので、その先の「回答生成」がどの程度できそうかを、検索精度を切り離して単体で
見るためのもの。

evidenceの選ばせ方はReadingAgentの_read_and_judgeと同じ発想: チャンクに
chunk_idを振って提示し、LLMにはそのchunk_idを選ばせるだけにする(座標や
ページ番号を自由記述させると捏造しやすいため)。選ばれたchunk_idは実在する
ものだけに絞ってから、Chunk.metadataの page/table_id/figure_id 等を機械的に
Evidenceへ変換する(src/littraceqa/di_pipeline/agent/evidence.pyのevidence_from_result
を流用。ChunkとRetrievalResultはpaper_id/text/chunk_type/metadataの
フィールド名が共通なのでダックタイピングでそのまま渡せる)。

src/littraceqa/azure/run_rag.py のような検索・エンティティブースト・
corpus enumeration等は一切行わない、Azure OpenAIをJSONモードで1回呼ぶだけの
最小実装。paper_precision/recall/f1 はgold paperをそのまま渡す構造上ほぼ1.0に
なるが、それはこのスクリプトが測っている対象ではない
(retrievalの精度は既存のsearch_style実験側で見る)。

使い方:
    uv run python scripts/generate_oracle_answers.py \
      --paths configs/paths/default.yaml \
      --output predictions_oracle.jsonl

その後 scripts/evaluate.py を自動で呼び、精度を表示する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.config import load_config
from littraceqa.di_pipeline.contracts import Answer, Chunk, Prediction, Query
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM

# 1論文ぶんのチャンク一覧(メタデータ付き)を渡すには十分すぎる安全マージン
# (念のための上限。通常の論文なら収まる)。
_MAX_CONTEXT_CHARS = 100_000


def load_queries(path: Path) -> list[Query]:
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(Query.from_dict(json.loads(line)))
    return queries


def load_gold_info(path: Path) -> dict[str, dict]:
    """query_id -> {"paper_ids": [...], "options": {...} | None}。

    gold answer本体(freeform.text / multiple_choice.gold / table.rows)は
    プロンプトに漏れたら意味がなくなるので、ここでは一切読み取らない。
    multiple_choiceのoptionsだけは問題文の一部(選択肢そのもの)なので読む。
    """
    info: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mc = (record.get("answer") or {}).get("multiple_choice") or {}
            info[record["query_id"]] = {
                "paper_ids": [p["paper_id"] for p in record.get("gold_papers", [])],
                "options": mc.get("options") if isinstance(mc, dict) else None,
            }
    return info


def load_paper_chunks(chunks_path: Path, paper_ids: set[str]) -> dict[str, list[Chunk]]:
    """必要な論文ぶんだけ、chunks.jsonlを1回streamしてChunkを集める。"""
    by_paper: dict[str, list[Chunk]] = {pid: [] for pid in paper_ids}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            bucket = by_paper.get(record.get("paper_id"))
            if bucket is not None:
                bucket.append(Chunk(**record))
    return by_paper


def format_chunks(chunks: list[Chunk]) -> str:
    """reading.py の _format_paper と同じ形式: chunk_id + メタデータ + 本文。"""
    lines = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        where = [f"type={chunk.chunk_type}"]
        for key in ("page", "section", "table_id", "figure_id", "equation_id"):
            if metadata.get(key) is not None:
                where.append(f"{key}={metadata[key]}")
        lines.append(f"- chunk_id: {chunk.chunk_id} ({', '.join(where)})")
        lines.append(f"  {chunk.text}")
    return "\n".join(lines)


def build_prompt(query: Query, options: dict | None, paper_chunks: list[Chunk]) -> str:
    schema_fields = [
        '"paper_ids": ["対象論文のID"]',
        '"evidence_chunk_ids": ["根拠にしたchunk_idの配列。一覧に無いIDは作らないこと"]',
    ]
    blocks = [
        "以下は根拠となる論文のチャンク一覧です。この内容だけを根拠に質問に答えてください。",
        format_chunks(paper_chunks)[:_MAX_CONTEXT_CHARS],
        f"質問: {query.question}",
    ]
    if "freeform" in query.answer_types:
        schema_fields.append('"freeform": {"text": "原文からの短い逐語的な値・語句"}')
    if "multiple_choice" in query.answer_types and options:
        opt_lines = "\n".join(f"{k}: {v}" for k, v in options.items())
        blocks.append(f"選択肢:\n{opt_lines}")
        schema_fields.append('"multiple_choice": {"gold": "選んだ選択肢のアルファベット1文字"}')
    if "table" in query.answer_types and query.table_schema:
        columns = "\n".join(
            f'- "{c.get("name")}" (type: {c.get("type", "string")}'
            f'{", row key" if c.get("is_row_key") else ""})'
            for c in query.table_schema
        )
        blocks.append(f"必要な表の列:\n{columns}")
        schema_fields.append('"table": {"rows": [{"列名": "値", ...}, ...]}')

    schema = "{ " + ", ".join(schema_fields) + " }"
    blocks.append(f"次のJSON形式だけで答えてください（説明文は不要）:\n{schema}")
    return "\n\n".join(blocks)


def build_answer(payload: dict, query: Query) -> Answer:
    return Answer(
        freeform=payload.get("freeform") if "freeform" in query.answer_types else None,
        multiple_choice=(
            payload.get("multiple_choice") if "multiple_choice" in query.answer_types else None
        ),
        table=payload.get("table") if "table" in query.answer_types else None,
    )


def build_evidence(payload: dict, paper_chunks: list[Chunk]) -> list:
    """LLMが返したevidence_chunk_idsを、実在するchunkだけに絞ってEvidenceへ変換する。"""
    by_id = {chunk.chunk_id: chunk for chunk in paper_chunks}
    evidence = []
    for chunk_id in payload.get("evidence_chunk_ids") or []:
        chunk = by_id.get(str(chunk_id))
        if chunk is not None:
            evidence.append(evidence_from_result(chunk))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths/default.yaml")
    parser.add_argument("--process-name", default="mineru", help="chunksファイル名の接頭辞")
    parser.add_argument("--queries", default="data/validation_inputs.jsonl")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()

    paths = load_config(args.paths)
    chunks_path = Path(paths["chunks_dir"]) / f"{args.process_name}_chunks.jsonl"

    queries = load_queries(Path(args.queries))
    gold_info = load_gold_info(Path(args.gold))

    needed_paper_ids = {
        pid
        for query in queries
        for pid in gold_info.get(query.query_id, {}).get("paper_ids", [])
    }
    print(
        f"{len(queries)} 件の質問、対象論文 {len(needed_paper_ids)} 件ぶんの"
        f"chunkを {chunks_path} から読み込み中..."
    )
    paper_chunks_by_id = load_paper_chunks(chunks_path, needed_paper_ids)

    llm = AzureOpenAILLM(reasoning_effort=args.reasoning_effort)

    predictions = []
    for query in tqdm(queries, desc="oracle answer生成"):
        gold = gold_info.get(query.query_id, {})
        paper_ids = gold.get("paper_ids", [])
        paper_chunks = [
            chunk for pid in paper_ids for chunk in paper_chunks_by_id.get(pid, [])
        ]
        prompt = build_prompt(query, gold.get("options"), paper_chunks)

        pred = Prediction.from_query(query)
        pred.gold_papers = [{"paper_id": pid} for pid in paper_ids]
        try:
            raw = llm(prompt)
            payload = parse_json_object(raw) or {}
            pred.answer = build_answer(payload, query)
            pred.evidence = build_evidence(payload, paper_chunks)
        except Exception as exc:  # noqa: BLE001 - 1件の失敗で全体を止めない
            print(f"WARN {query.query_id}: 回答生成に失敗しました ({exc})", file=sys.stderr)
        predictions.append(pred.to_dict())

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"予測結果を {output_path} に書き出しました")

    print("\n採点中...")
    result = subprocess.run(
        ["uv", "run", "python", "scripts/evaluate.py", "--gold", args.gold, "--pred", str(output_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
