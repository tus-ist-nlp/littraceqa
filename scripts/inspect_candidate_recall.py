#!/usr/bin/env python3
"""クエリIDごとに gold paper が取れたか取れなかったかを一覧する診断スクリプト。

scripts/evaluate.py は 55件をマクロ平均した集計値しか出さないので、
「どのクエリで検索が失敗しているか」が分からない。ここでは1クエリ1行で
gold 論文ごとの順位（candidate_papers の何位で拾えたか）を出し、
伸びしろの所在を特定できるようにする。

見るのは予測の `candidate_papers`（打ち切り前に検索が拾えた論文の順位列）。
提出セット `gold_papers` ではなく候補列を見るのは、検索力と LLM の選定力を
分離するため（agent/reading.py の _build_prediction 参照）。

使い方:
    # 1つの予測ファイルを診断（失敗クエリだけ見たいときは --only-missed）
    uv run python scripts/inspect_candidate_recall.py --pred predictions_xxx.jsonl

    # 2つの構成を比較（どのクエリで良化/悪化したか）
    uv run python scripts/inspect_candidate_recall.py \\
      --pred predictions_new.jsonl --baseline predictions_old.jsonl

    # CSV に落として表計算で見る
    uv run python scripts/inspect_candidate_recall.py --pred p.jsonl --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (  # noqa: E402  (scripts/ 直下の evaluate.py を再利用)
    candidate_paper_ids,
    normalize_id,
    paper_id_set,
    read_jsonl,
)

# 「拾えた」とみなす順位のしきい値。看板指標が recall@10/@20 なのでその2つを出す。
RANK_MARKS = (10, 20)


def rank_of(paper_id: str, ranked: list[str]) -> int | None:
    """candidate_papers の何位か（1始まり）。入っていなければ None。"""
    try:
        return ranked.index(paper_id) + 1
    except ValueError:
        return None


def attribute_filter_of(record: dict) -> str:
    """trace から属性フィルタが発火したかを取り出す（reading.py が残している）。"""
    for step in record.get("trace") or []:
        af = step.get("attribute_filter")
        if af:
            venue = af.get("venue") or ""
            year = af.get("year") or ""
            return f"{venue} {year}".strip()
    return ""


def rows_for(pred_path: Path, gold_path: Path) -> list[dict]:
    gold_by_id = {normalize_id(r.get("query_id")): r for r in read_jsonl(gold_path)}
    pred_by_id = {normalize_id(r.get("query_id")): r for r in read_jsonl(pred_path)}

    rows = []
    for query_id, gold in gold_by_id.items():
        pred = pred_by_id.get(query_id, {})
        gold_ids = sorted(paper_id_set(gold))
        ranked = candidate_paper_ids(pred) or []
        ranks = {p: rank_of(p, ranked) for p in gold_ids}
        found = [p for p, r in ranks.items() if r is not None]

        row = {
            "query_id": query_id,
            "task_family": normalize_id(gold.get("task_family")) or "?",
            "n_gold": len(gold_ids),
            "n_found": len(found),
            "attribute_filter": attribute_filter_of(pred),
            "n_candidates": len(ranked),
            "ranks": ranks,
            "question": (gold.get("question") or "").replace("\n", " "),
            "missing": [p for p, r in ranks.items() if r is None],
        }
        for k in RANK_MARKS:
            hit = sum(1 for r in ranks.values() if r is not None and r <= k)
            row[f"recall@{k}"] = hit / len(gold_ids) if gold_ids else 1.0
        rows.append(row)
    return rows


def format_ranks(ranks: dict[str, int | None]) -> str:
    """gold ごとの順位を "pid:順位" で並べる。未発見は '-'。"""
    return " ".join(f"{p.split('_')[-1]}:{r if r is not None else '-'}" for p, r in ranks.items())


def print_table(rows: list[dict], only_missed: bool, verbose: bool) -> None:
    target = [r for r in rows if r["n_found"] < r["n_gold"]] if only_missed else rows
    target.sort(key=lambda r: (r[f"recall@{RANK_MARKS[-1]}"], -r["n_gold"]))

    header = (
        f"{'query_id':<14}{'family':<10}{'gold':>5}{'発見':>5}"
        f"{'r@10':>7}{'r@20':>7}  {'属性':<12}{'gold順位(未発見=-)'}"
    )
    print(header)
    print("-" * (len(header) + 10))
    for r in target:
        fam = r["task_family"].replace("hidden_source_", "")[:9]
        print(
            f"{r['query_id']:<14}{fam:<10}{r['n_gold']:>5}{r['n_found']:>5}"
            f"{r['recall@10']:>7.2f}{r['recall@20']:>7.2f}  {r['attribute_filter'] or '-':<12}"
            f"{format_ranks(r['ranks'])}"
        )
        if verbose:
            print(f"{'':>14}Q: {r['question'][:110]}")

    print()
    print(f"表示 {len(target)} / 全 {len(rows)} クエリ")
    perfect = sum(1 for r in rows if r["n_found"] == r["n_gold"])
    zero = sum(1 for r in rows if r["n_found"] == 0)
    print(f"  gold を全部拾えた: {perfect} / 1本も拾えなかった: {zero}")
    # task_family ごとの内訳（伸びしろの所在）
    for fam in sorted({r["task_family"] for r in rows}):
        sub = [r for r in rows if r["task_family"] == fam]
        for k in RANK_MARKS:
            avg = sum(r[f"recall@{k}"] for r in sub) / len(sub)
            print(f"  {fam:<28} recall@{k:<3} = {avg:.4f}  ({len(sub)}件)", end="")
        print()


def print_diff(rows: list[dict], base_rows: list[dict]) -> None:
    base_by_id = {r["query_id"]: r for r in base_rows}
    k = RANK_MARKS[-1]
    diffs = []
    for r in rows:
        b = base_by_id.get(r["query_id"])
        if b is None:
            continue
        d = r[f"recall@{k}"] - b[f"recall@{k}"]
        if d != 0:
            diffs.append((d, r, b))
    diffs.sort(key=lambda x: x[0])

    print(f"=== recall@{k} が変化したクエリ（悪化順） ===")
    if not diffs:
        print("  変化なし")
        return
    print(f"{'query_id':<14}{'family':<10}{'baseline':>9}{'new':>7}{'差':>8}  gold順位(new)")
    for d, r, b in diffs:
        fam = r["task_family"].replace("hidden_source_", "")[:9]
        print(
            f"{r['query_id']:<14}{fam:<10}{b[f'recall@{k}']:>9.2f}{r[f'recall@{k}']:>7.2f}"
            f"{d:>+8.2f}  {format_ranks(r['ranks'])}"
        )
    worse = sum(1 for d, _, _ in diffs if d < 0)
    better = sum(1 for d, _, _ in diffs if d > 0)
    print(f"\n  悪化 {worse} 件 / 改善 {better} 件")


def write_markdown(rows: list[dict], path: Path, pred_path: Path, gold_path: Path) -> None:
    """report/ に置く一覧を Markdown で書く（run_search.py の report と同じ場所）。"""
    from datetime import datetime

    rows = sorted(rows, key=lambda r: (r[f"recall@{RANK_MARKS[-1]}"], -r["n_gold"]))
    total = len(rows)
    perfect = sum(1 for r in rows if r["n_found"] == r["n_gold"])
    zero = sum(1 for r in rows if r["n_found"] == 0)

    lines = [
        f"# クエリ別 candidate_recall 診断: {pred_path.name}",
        "",
        f"- 生成: {datetime.now().isoformat(timespec='seconds')}",
        f"- 予測: `{pred_path}`",
        f"- 正解: `{gold_path}`",
        f"- 対象: {total} クエリ（gold を全部拾えた {perfect} / 1本も拾えなかった {zero}）",
        "",
        "見ているのは予測の `candidate_papers`（打ち切り前に検索が拾えた論文の順位列）。",
        "提出セット `gold_papers` ではなく候補列なので、検索力と LLM の選定力を分離できる。",
        "",
        "## task_family ごとの集計",
        "",
        "| task_family | 件数 | " + " | ".join(f"recall@{k}" for k in RANK_MARKS) + " |",
        "|---|---:|" + "---:|" * len(RANK_MARKS),
    ]
    for fam in sorted({r["task_family"] for r in rows}):
        sub = [r for r in rows if r["task_family"] == fam]
        cells = " | ".join(
            f"{sum(x[f'recall@{k}'] for x in sub) / len(sub):.4f}" for k in RANK_MARKS
        )
        lines.append(f"| {fam} | {len(sub)} | {cells} |")

    lines += [
        "",
        "## クエリ別一覧（recall が低い順）",
        "",
        "`gold順位` は gold 論文ごとの candidate_papers 内の順位。`-` は候補に入らなかったもの。",
        "",
        "| query_id | task_family | gold | 発見 | "
        + " | ".join(f"r@{k}" for k in RANK_MARKS)
        + " | 属性 | gold順位 | 質問 |",
        "|---|---|---:|---:|" + "---:|" * len(RANK_MARKS) + "---|---|---|",
    ]
    for r in rows:
        marks = " | ".join(f"{r[f'recall@{k}']:.2f}" for k in RANK_MARKS)
        ranks = format_ranks(r["ranks"]).replace("|", "\\|")
        question = r["question"][:90].replace("|", "\\|")
        lines.append(
            f"| {r['query_id']} | {r['task_family']} | {r['n_gold']} | {r['n_found']} | "
            f"{marks} | {r['attribute_filter'] or '-'} | `{ranks}` | {question} |"
        )

    # 取りこぼしの内訳。どこを直せば効くかを見分けられるようにする。
    missed = [r for r in rows if r["n_found"] < r["n_gold"]]
    if missed:
        buried = [r for r in missed if any(v is not None and v > RANK_MARKS[-1] for v in r["ranks"].values())]
        lines += [
            "",
            "## 取りこぼしの内訳",
            "",
            f"- 取りこぼしたクエリ: {len(missed)} / {total}",
            f"- うち **候補には入っているが {RANK_MARKS[-1]}位より下** に埋まっている gold を含む: "
            f"{len(buried)} 件（reranker の改善で取れる余地がある層）",
            "",
        ]
        if buried:
            lines.append("| query_id | " + f"{RANK_MARKS[-1]}位より下の gold（順位） |")
            lines.append("|---|---|")
            for r in buried:
                deep = " ".join(
                    f"{p.split('_')[-1]}:{v}"
                    for p, v in r["ranks"].items()
                    if v is not None and v > RANK_MARKS[-1]
                )
                lines.append(f"| {r['query_id']} | `{deep}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown を {path} に書き出しました（{total}クエリ）")


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["query_id", "task_family", "n_gold", "n_found", "n_candidates", "attribute_filter"]
    fields += [f"recall@{k}" for k in RANK_MARKS]
    fields += ["gold_ranks", "missing_papers", "question"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fields if k in r}
            out["gold_ranks"] = ";".join(
                f"{p}:{v if v is not None else ''}" for p, v in r["ranks"].items()
            )
            out["missing_papers"] = ";".join(r["missing"])
            out["question"] = r["question"]
            writer.writerow(out)
    print(f"CSV を {path} に書き出しました（{len(rows)}行）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="クエリIDごとに gold paper が取れたかを一覧する（検索の失敗箇所を特定する）"
    )
    parser.add_argument("--pred", required=True, help="予測 JSONL（run_search.py の --output）")
    parser.add_argument("--gold", default="data/validation.jsonl", help="正解付き JSONL")
    parser.add_argument(
        "--baseline", default=None, help="比較対象の予測 JSONL（指定すると差分のみ表示）"
    )
    parser.add_argument(
        "--only-missed", action="store_true", help="gold を取りこぼしたクエリだけ表示"
    )
    parser.add_argument("--verbose", action="store_true", help="質問文も表示")
    parser.add_argument("--csv", default=None, help="CSV 出力先")
    parser.add_argument(
        "--report",
        nargs="?",
        const="auto",
        default=None,
        help="Markdown を report/ に書き出す。パス省略で report/{timestamp}_recall_{予測名}.md",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    rows = rows_for(pred_path, gold_path)

    if args.baseline:
        base_rows = rows_for(Path(args.baseline), gold_path)
        print_diff(rows, base_rows)
        print()
    print_table(rows, args.only_missed, args.verbose)

    if args.csv:
        write_csv(rows, Path(args.csv))
    if args.report:
        if args.report == "auto":
            from datetime import datetime

            stem = pred_path.stem.replace("predictions_", "")
            name = f"{datetime.now():%Y%m%d_%H%M%S}_recall_{stem}.md"
            report_path = Path("report") / name
        else:
            report_path = Path(args.report)
        write_markdown(rows, report_path, pred_path, gold_path)


if __name__ == "__main__":
    main()
