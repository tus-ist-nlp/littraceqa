#!/usr/bin/env python3
"""検索結果の候補論文を質問に貼り合わせ、読解エージェント用の1ファイルにまとめる。

読解エージェントを別実装で作るとき、必要な入力は「質問」と「検索が拾った候補論文」の
2つだが、いまは前者が data/validation_inputs.jsonl、後者が predictions_*.jsonl の
`candidate_papers` に分かれている。両方を突き合わせる手間を毎回かけないよう、
1行1クエリの JSONL に合成する。

出力の形（本番入力の4フィールド + 候補 + メタ + gold）:

    {
      "query_id": "q_001",
      "question": "...",
      "answer_types": ["freeform", "multiple_choice"],
      "table_schema": null,
      "candidate_papers": [
        {"rank": 1, "paper_id": "acl2025_00005", "title": "...", "venue": "ACL", "year": 2025},
        ...
      ],
      "_meta": {"source_predictions": "...", "search": "...", "n_candidates": 50},
      "_gold": {"task_family": ..., "primary_evidence_type": ..., "gold_papers": [...],
                "evidence": [...], "answer": {...}}
    }

**gold を `_gold` の下に隔離してあるのは事故防止のため。** 同じ階層に並べると、
`gold_papers[].title`（どの論文かの答えそのもの）、`answer.multiple_choice.options`
（f53e1da で oracle 実験専用と決めた項目）、`task_family` / `primary_evidence_type`
（本番入力に存在しない項目）を、悪意なく読んでしまう。トップレベルの4フィールドだけが
本番と同じ入力で、`_gold` は採点のときだけ触ってよい。

候補に title/venue/year を付けてあるのは gold ではないので安全（本文を引く前に
候補一覧を眺められるようにするため）。本文と図表画像が要るときは paper_id を
littraceqa.chunk_store.ChunkStore に渡す。

使い方:
    uv run python scripts/build_candidate_handoff.py \\
      --predictions predictions_8b_chunk_b_merged.jsonl \\
      --output data/validation_with_candidates.jsonl

    # 本番入力を模して gold を落とす（配布用）
    uv run python scripts/build_candidate_handoff.py \\
      --predictions predictions_8b_chunk_b_merged.jsonl --no-gold \\
      --output data/validation_with_candidates_blind.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

Record = dict[str, Any]

# 本番入力に実在するフィールド（f53e1da）。ここを増やすと評価が本番と乖離する。
PRODUCTION_FIELDS = ("query_id", "question", "answer_types", "table_schema")

# 採点にしか使ってはいけないフィールド。トップレベルから _gold へ移す。
GOLD_FIELDS = (
    "task_family",
    "primary_evidence_type",
    "gold_papers",
    "evidence",
    "answer",
)


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_paper_titles(path: Path) -> dict[str, Record]:
    """paper_id -> {title, venue, year}。候補一覧の見出しにだけ使う。"""
    titles: dict[str, Record] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            titles[record["paper_id"]] = {
                "title": record.get("title"),
                "venue": record.get("venue"),
                "year": record.get("year"),
            }
    return titles


def find_run_record(predictions_path: Path, experiments_path: Path) -> Record | None:
    """その予測ファイルを作った実行を results/experiments.jsonl から探す。

    候補の並びは検索構成に依存するので、どの構成の候補なのかを出力に残す。
    見つからなくても本題ではないので黙って None を返す。
    """
    if not experiments_path.exists():
        return None
    # 分割実行を結合したファイルは run_search.py:101 が `{stem}_merged` として
    # 作るが、experiments.jsonl の `output` には結合前の名前が入る。両方で照合する。
    names = {predictions_path.name}
    if predictions_path.stem.endswith("_merged"):
        base = predictions_path.stem[: -len("_merged")]
        names.add(f"{base}{predictions_path.suffix}")
    match: Record | None = None
    for record in read_jsonl(experiments_path):
        if Path(str(record.get("output") or "")).name in names:
            match = record  # 同名で複数あれば最後（最新）を採る
    return match


def build_candidates(
    paper_ids: list[str], titles: dict[str, Record], limit: int
) -> list[Record]:
    ranked = paper_ids if limit <= 0 else paper_ids[:limit]
    candidates: list[Record] = []
    for rank, paper_id in enumerate(ranked, start=1):
        info = titles.get(paper_id, {})
        candidates.append(
            {
                "rank": rank,
                "paper_id": paper_id,
                "title": info.get("title"),
                "venue": info.get("venue"),
                "year": info.get("year"),
            }
        )
    return candidates


def build_rows(
    inputs: list[Record],
    gold: dict[str, Record],
    predictions: dict[str, Record],
    titles: dict[str, Record],
    meta_base: Record,
    limit: int,
    include_gold: bool,
) -> tuple[list[Record], list[str]]:
    rows: list[Record] = []
    missing: list[str] = []
    for sample in inputs:
        query_id = sample["query_id"]
        prediction = predictions.get(query_id)
        if prediction is None:
            missing.append(query_id)
            continue
        row: Record = {field: sample.get(field) for field in PRODUCTION_FIELDS}
        candidates = build_candidates(
            prediction.get("candidate_papers") or [], titles, limit
        )
        row["candidate_papers"] = candidates
        row["_meta"] = {**meta_base, "n_candidates": len(candidates)}
        if include_gold:
            source = gold.get(query_id) or sample
            row["_gold"] = {
                field: source.get(field)
                for field in GOLD_FIELDS
                if source.get(field) is not None
            }
        rows.append(row)
    return rows, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions", type=Path, required=True, help="候補論文の供給元 predictions_*.jsonl"
    )
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/validation_inputs.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/validation.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/paper_metadata.jsonl")
    parser.add_argument(
        "--experiments", type=Path, default=ROOT / "results/experiments.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/validation_with_candidates.jsonl"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="候補の上限（0で predictions にあるだけ全部。既定の予測は上限50本）",
    )
    parser.add_argument(
        "--no-gold", action="store_true", help="_gold を出力しない（配布用のブラインド版）"
    )
    args = parser.parse_args(argv)

    inputs = read_jsonl(args.inputs)
    predictions = {record["query_id"]: record for record in read_jsonl(args.predictions)}
    gold = (
        {record["query_id"]: record for record in read_jsonl(args.gold)}
        if args.gold.exists()
        else {}
    )
    titles = load_paper_titles(args.metadata)

    run = find_run_record(args.predictions, args.experiments)
    meta_base: Record = {"source_predictions": args.predictions.name}
    if run is not None:
        meta_base["search"] = run.get("search")
        meta_base["agent"] = run.get("agent")
        meta_base["run_timestamp"] = run.get("timestamp")

    rows, missing = build_rows(
        inputs, gold, predictions, titles, meta_base, args.max_candidates, not args.no_gold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = [row["_meta"]["n_candidates"] for row in rows]
    no_title = sum(
        1 for row in rows for c in row["candidate_papers"] if c["title"] is None
    )
    print(f"{args.output}: {len(rows)}件（入力 {len(inputs)}件）")
    print(
        f"  候補論文: 合計{sum(counts)}本 / 1クエリあたり最小{min(counts, default=0)} "
        f"最大{max(counts, default=0)}"
    )
    if no_title:
        print(f"  警告: metadata に無く title が引けなかった候補 {no_title}件")
    if missing:
        print(
            f"  警告: 予測に無く候補を付けられなかったクエリ {len(missing)}件: "
            f"{', '.join(missing[:10])}",
            file=sys.stderr,
        )
    if not args.no_gold:
        without_gold = sum(1 for row in rows if not row.get("_gold"))
        if without_gold:
            print(f"  警告: gold が見つからなかったクエリ {without_gold}件", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
