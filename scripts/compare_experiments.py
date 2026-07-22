#!/usr/bin/env python3
"""results/experiments.jsonl を「振ったパラメータ × 指標」の表で横並び比較する。

パラメータを少しずつ変えて実験を回すとき、experiments.jsonl を素で読んでも
「どの行がどのつまみを変えたものか」が分からない。このスクリプトは
実験間で**値が違うパラメータだけ**を抜き出して並べるので、
「何を変えたら指標がどう動いたか」が1枚で見える。

run_search.py が各実行で tuned_params（解決済みの実際の値）を記録している前提。
それ以前の古い記録は tuned_params を持たないので `-` と表示される。

使い方:
    # 直近10件を比較
    uv run python scripts/compare_experiments.py

    # search_style で絞り、recall カーブを見る
    uv run python scripts/compare_experiments.py \
      --filter-search bm25_qwen3 --metrics candidate_recall

    # 全パラメータを表示（差分のあるものだけでなく）
    uv run python scripts/compare_experiments.py --all-params
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_DEFAULT_METRICS = (
    "paper_f1_macro",
    "paper_precision_macro",
    "paper_recall_macro",
    "candidate_recall_at20_single_macro",
    "candidate_recall_at20_multi_macro",
    "candidate_recall_at20_total_macro",
)


def load_experiments(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def differing_params(records: list[dict], show_all: bool) -> list[str]:
    """実験間で値が違うパラメータ名だけを返す（show_all なら全部）。

    共通のつまみまで並べると横に長くなって差分が読めなくなるので、既定では
    「実際に振ったもの」だけに絞る。
    """
    keys: list[str] = []
    for record in records:
        for key in record.get("tuned_params", {}):
            if key not in keys:
                keys.append(key)
    if show_all:
        return keys
    varying = []
    for key in keys:
        values = {
            json.dumps(record.get("tuned_params", {}).get(key), sort_keys=True)
            for record in records
        }
        if len(values) > 1:
            varying.append(key)
    return varying


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="results/experiments.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="直近何件を比較するか")
    parser.add_argument(
        "--filter-search", default=None, help="search_style 名に含まれる文字列で絞る"
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="表示する指標名（部分一致、カンマ区切り）。省略時は主要指標",
    )
    parser.add_argument(
        "--all-params", action="store_true", help="差分のないパラメータも表示する"
    )
    args = parser.parse_args()

    path = Path(args.experiments)
    if not path.exists():
        print(f"{path} がありません。まず run_search.py を実行してください。")
        return

    records = load_experiments(path)
    if args.filter_search:
        records = [r for r in records if args.filter_search in r.get("search", "")]
    records = records[-args.limit :]
    if not records:
        print("該当する実験がありません。")
        return

    if args.metrics:
        needles = [s.strip() for s in args.metrics.split(",") if s.strip()]
        metric_keys: list[str] = []
        for record in records:
            for key in record.get("metrics", {}):
                if any(n in key for n in needles) and key not in metric_keys:
                    metric_keys.append(key)
    else:
        metric_keys = list(_DEFAULT_METRICS)

    param_keys = differing_params(records, args.all_params)

    header = ["時刻", "search"] + param_keys + metric_keys
    rows = []
    for record in records:
        tuned = record.get("tuned_params", {})
        row = [
            record["timestamp"][5:16],
            Path(record.get("search", "")).stem[:28],
        ]
        row += [format_value(tuned.get(k)) if tuned else "-" for k in param_keys]
        row += [format_value(record.get("metrics", {}).get(k)) for k in metric_keys]
        rows.append(row)

    widths = [
        max(len(str(header[i])), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if not param_keys:
        print()
        print(
            "（振ったパラメータの差分なし。--all-params で全パラメータ、"
            "または tuned_params を持つ実行が2件以上必要）"
        )


if __name__ == "__main__":
    main()
