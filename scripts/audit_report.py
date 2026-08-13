#!/usr/bin/env python3
"""クエリ品質監査 JSONL の集計と HTML ビューア生成。

scripts/audit_queries.py が書いた (query_id, paper_id) 単位の判定 JSONL から、

- クエリラベル（良問 good / やや良問 fair / 悪問 noisy。spec 3.3 の導出規則）
- 補正指標 paper_recall_macro / paper_recall_macro_clean（spec 4.2）
- noise_type 分布・contradicting フラグ・未検出 supporting の一覧（spec 5）
- 単一ファイルの HTML ビューア（spec 4.4。外部依存なし）

を再生成する。HTML は JSONL のビューアであり判定ロジックを含まない。
ラベル・集計はすべてここで導出するので、JSONL があれば何度でも作り直せる。

実行例:
    uv run python scripts/audit_report.py \
      --audit audits/query_audit.jsonl \
      --pred predictions_8b_chunk_b_merged.jsonl \
      --output report/query_audit.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from littraceqa.common import config_label

CLEAN_RELEVANCES = ("supporting", "partial")

LABEL_JA = {"good": "良問", "fair": "やや良問", "noisy": "悪問"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_query(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("query_id")), []).append(record)
    return grouped


def query_label(records: list[dict[str, Any]]) -> str:
    """spec 3.3 の導出規則。gold_paper 単位の判定から機械的に決める。

    relevance が欠けている（LLM 判定失敗）論文は supporting とは認めない
    = fair 側に倒す。no_evidence はデータセットの事実なので最優先。
    """
    relevances = [r.get("relevance") for r in records]
    if any(rel == "no_evidence" for rel in relevances):
        return "noisy"
    if relevances and all(rel == "supporting" for rel in relevances):
        return "good"
    return "fair"


def has_contradicting(records: list[dict[str, Any]]) -> bool:
    """ラベルとは独立のフラグ（spec 3.3）。誤答の根拠が gold に混入している事例。"""
    return any(r.get("relevance") == "contradicting" for r in records)


def submitted_paper_ids(pred: dict[str, Any]) -> set[str]:
    """予測レコードの提出論文集合（予測ファイルではキー名が gold_papers）。"""
    ids: set[str] = set()
    for item in pred.get("gold_papers") or []:
        paper_id = item.get("paper_id", "") if isinstance(item, dict) else str(item)
        if paper_id:
            ids.add(str(paper_id))
    return ids


def corrected_metrics(
    grouped: dict[str, list[dict[str, Any]]], pred_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """spec 4.2。従来 recall と、分母を supporting/partial に絞った clean recall を並記する。

    両者の差そのものがデータセットのノイズ量。clean の分母が 0 になるクエリは
    マクロ平均から除外する。
    """
    recalls: list[float] = []
    clean_recalls: list[float] = []
    excluded = 0
    for query_id, records in grouped.items():
        submitted = submitted_paper_ids(pred_by_id.get(query_id, {}))
        gold_ids = {str(r.get("paper_id")) for r in records}
        if gold_ids:
            recalls.append(len(gold_ids & submitted) / len(gold_ids))
        clean_ids = {
            str(r.get("paper_id"))
            for r in records
            if r.get("relevance") in CLEAN_RELEVANCES
        }
        if clean_ids:
            clean_recalls.append(len(clean_ids & submitted) / len(clean_ids))
        else:
            excluded += 1
    recall = sum(recalls) / len(recalls) if recalls else None
    clean = sum(clean_recalls) / len(clean_recalls) if clean_recalls else None
    return {
        "paper_recall_macro": recall,
        "paper_recall_macro_clean": clean,
        "dataset_noise_gap": (clean - recall) if recall is not None and clean is not None else None,
        "clean_excluded_queries": excluded,
        "n_queries": len(grouped),
    }


def summarize(
    grouped: dict[str, list[dict[str, Any]]],
    pred_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """レポート冒頭と stdout に出す集計。すべて JSONL から導出する派生物。"""
    label_counts = Counter(query_label(records) for records in grouped.values())
    relevance_counts = Counter(
        str(r.get("relevance")) for records in grouped.values() for r in records
    )
    noise_counts = Counter(
        r["noise_type"]
        for records in grouped.values()
        for r in records
        if r.get("noise_type")
    )
    contradicting = sorted(
        query_id for query_id, records in grouped.items() if has_contradicting(records)
    )
    # 利用フロー3: 「未検出かつ supporting」= retriever の実際の失敗事例。
    unretrieved_supporting = sorted(
        (str(r.get("query_id")), str(r.get("paper_id")))
        for records in grouped.values()
        for r in records
        if r.get("relevance") == "supporting" and (r.get("retrieval") or {}).get("rank") is None
    )
    # no_evidence なのに本文が回答を支持していそうな論文 = アノテーション漏れの疑い。
    missing_annotation_suspects = sorted(
        (str(r.get("query_id")), str(r.get("paper_id")))
        for records in grouped.values()
        for r in records
        if r.get("relevance") == "no_evidence" and r.get("body_supports_answer") is True
    )
    # 棒グラフ用: task_family ごとのラベル内訳（例: 悪問が multi に偏る様子を見る）。
    label_counts_by_family: dict[str, dict[str, int]] = {}
    for records in grouped.values():
        family = str(records[0].get("task_family"))
        counts = label_counts_by_family.setdefault(family, {})
        lab = query_label(records)
        counts[lab] = counts.get(lab, 0) + 1

    summary: dict[str, Any] = {
        "n_queries": len(grouped),
        "n_pairs": sum(len(records) for records in grouped.values()),
        "label_counts": dict(label_counts),
        "label_counts_by_family": label_counts_by_family,
        "relevance_counts": dict(relevance_counts),
        "noise_type_distribution": dict(noise_counts.most_common()),
        "contradicting_queries": contradicting,
        "unretrieved_supporting": [list(pair) for pair in unretrieved_supporting],
        "missing_annotation_suspects": [list(pair) for pair in missing_annotation_suspects],
    }
    if pred_by_id is not None:
        summary["metrics"] = corrected_metrics(grouped, pred_by_id)
    return summary


def experiment_label(row: dict[str, Any]) -> str:
    """experiments.jsonl の1行から report/*.md と同じ形式のラベルを作る。

    report フォルダのファイル名 {timestamp}_{process}_{search}_{agent}.md と
    揃えることで、ビューアの実験セレクタと手元のレポートを突き合わせられる。
    """
    ts = re.sub(r"[-:]", "", str(row.get("timestamp") or "")).replace("T", "_")
    stems = [config_label(str(row.get(k) or "")) for k in ("process", "search", "agent")]
    return "_".join([part for part in [ts, *stems] if part]) or "unknown"


def candidate_rank_map(pred: dict[str, Any]) -> dict[str, int]:
    """candidate_papers（関連度順）から paper_id -> 順位(1始まり)。"""
    ranks: dict[str, int] = {}
    for i, item in enumerate(pred.get("candidate_papers") or []):
        paper_id = item.get("paper_id", "") if isinstance(item, dict) else str(item)
        if paper_id and paper_id not in ranks:
            ranks[paper_id] = i + 1
    return ranks


def load_experiments(
    experiments_path: Path | None,
    pred_paths: list[Path],
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """実験（予測ファイル）の一覧を集める。

    - experiments.jsonl の各行から output（予測 JSONL）が実在するものを拾う
      （同じ output が複数回記録されていたら最新行を採用）
    - --pred で明示されたファイルを追加（experiments.jsonl に無い merged 等のため）

    各実験は監査済みクエリに対する網羅数（coverage）と補正指標を持つ。
    分割実行（val_a のみ等）の予測はクエリの一部しか持たないので、指標は
    「その実験がカバーするクエリだけ」で計算し、必ず件数を並記する（薄まった
    マクロ平均を55件の値と比較させないため）。
    """
    entries: dict[str, dict[str, Any]] = {}  # resolved path -> entry
    if experiments_path and experiments_path.exists():
        for row in read_jsonl(experiments_path):
            output = str(row.get("output") or "")
            path = Path(output)
            if not path.is_absolute():
                path = experiments_path.parent.parent / path
            # scratchpad 等の一時ファイルはデバッグ実行なのでセレクタに並べない
            # （--pred で明示すれば入れられる）。
            if not path.exists() or str(path.resolve()).startswith("/tmp/"):
                continue
            entries[str(path.resolve())] = {
                "name": experiment_label(row),
                "path": path,
                "timestamp": str(row.get("timestamp") or ""),
                # 分割実行（val_a/val_b）の統合判定に使う構成キー。
                "config": "_".join(
                    config_label(str(row.get(k) or ""))
                    for k in ("process", "search", "agent")
                ),
            }
    for path in pred_paths:
        if path.exists():
            entries.setdefault(
                str(path.resolve()),
                {"name": path.stem, "path": path, "timestamp": "~", "config": None},
            )
            entries[str(path.resolve())]["explicit"] = True

    experiments = []
    for entry in sorted(entries.values(), key=lambda e: e["timestamp"]):
        pred_by_id = {str(r.get("query_id")): r for r in read_jsonl(entry["path"])}
        covered = {q for q in grouped if pred_by_id.get(q, {}).get("candidate_papers")}
        # candidate_papers を持たない旧形式の予測は順位を出せないので除外する
        # （--pred で明示された場合は「なぜ表示されない」と迷わないよう残す）。
        if not covered and not entry.get("explicit"):
            continue
        covered_grouped = {q: grouped[q] for q in covered}
        experiments.append(
            {
                "name": entry["name"],
                "explicit": entry.get("explicit", False),
                "timestamp": entry["timestamp"],
                "config": entry.get("config"),
                "pred_by_id": pred_by_id,
                "covered": covered,
                "coverage": len(covered),
                "metrics": corrected_metrics(covered_grouped, pred_by_id)
                if covered
                else None,
            }
        )
    return merge_split_runs(experiments, grouped)


def merge_split_runs(
    experiments: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """同一構成の分割実行（val_a / val_b）をセレクタ上で1実験に統合する。

    - 同一構成でより網羅的な実行（--merge-with 済みの55件ファイル等）に
      包含される片割れは落とす
    - 残った同一構成の部分実行が複数あれば、予測を結合した1エントリにする
      （同じクエリを両方が持つ場合はタイムスタンプの新しい方を採用）。
      指標は結合後のカバー分で計算し直す
    """
    n_queries = len(grouped)
    result = []
    partials_by_config: dict[str, list[dict[str, Any]]] = {}
    for exp in experiments:
        if exp.get("config") and exp["coverage"] < n_queries:
            partials_by_config.setdefault(exp["config"], []).append(exp)
        else:
            result.append(exp)

    for config, parts in partials_by_config.items():
        fuller = [e for e in result if e.get("config") == config]
        keep = [
            p for p in parts if not any(p["covered"] <= f["covered"] for f in fuller)
        ]
        if len(keep) <= 1:
            result.extend(keep)
            continue
        keep.sort(key=lambda e: e["timestamp"])
        pred_by_id: dict[str, dict[str, Any]] = {}
        for part in keep:  # 新しい実行が後から上書きする
            pred_by_id.update(part["pred_by_id"])
        covered = set().union(*(p["covered"] for p in keep))
        result.append(
            {
                "name": f"{keep[-1]['name']}（分割{len(keep)}本を結合）",
                "explicit": any(p["explicit"] for p in keep),
                "timestamp": keep[-1]["timestamp"],
                "config": config,
                "pred_by_id": pred_by_id,
                "covered": covered,
                "coverage": len(covered),
                "metrics": corrected_metrics(
                    {q: grouped[q] for q in covered}, pred_by_id
                ),
            }
        )
    result.sort(key=lambda e: e["timestamp"])
    return result


def default_experiment_index(experiments: list[dict[str, Any]], n_queries: int) -> int:
    """--pred で明示された実験を優先し、無ければ最新のフル網羅実験を選ぶ。"""
    for i, exp in enumerate(experiments):
        if exp["explicit"]:
            return i
    for i in range(len(experiments) - 1, -1, -1):
        if experiments[i]["coverage"] == n_queries:
            return i
    return max(len(experiments) - 1, 0)


TOP_CANDIDATES = 5


def load_paper_titles(path: Path) -> dict[str, list[Any]]:
    """paper_id -> [title, venue, year]。検索上位の論文名を出すためだけに読む。

    27,487行あるので、必要な paper_id だけ引ければよい（呼び出し側で絞る）。
    """
    titles: dict[str, list[Any]] = {}
    if not path.exists():
        return titles
    for row in read_jsonl(path):
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            titles[paper_id] = [row.get("title") or "", row.get("venue") or "", row.get("year")]
    return titles


def top_candidate_ids(pred: dict[str, Any], n: int = TOP_CANDIDATES) -> list[str]:
    """candidate_papers の上位 n 本の paper_id（重複を落として出現順）。"""
    ids: list[str] = []
    for item in pred.get("candidate_papers") or []:
        paper_id = item.get("paper_id", "") if isinstance(item, dict) else str(item)
        if paper_id and paper_id not in ids:
            ids.append(paper_id)
        if len(ids) >= n:
            break
    return ids


def viewer_payload(
    grouped: dict[str, list[dict[str, Any]]],
    pred_by_id: dict[str, dict[str, Any]] | None,
    experiments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """HTML に埋め込むクエリ単位のデータ。描画に必要な導出値だけここで付ける。

    experiments を渡すと、各論文に by_exp（実験ごとの {r: 順位, s: 提出}、
    その実験の対象外クエリは null）を実験リストと同じ順で付ける。
    あわせてクエリごとに top_by_exp（実験ごとの検索上位 TOP_CANDIDATES 本の
    paper_id 列）を持たせる。タイトルは重複するので ID だけ置き、
    実体は DATA.paper_meta に1回だけ入れる。
    """
    queries = []
    for query_id in sorted(grouped):
        records = grouped[query_id]
        first = records[0]
        submitted = (
            submitted_paper_ids(pred_by_id.get(query_id, {})) if pred_by_id else set()
        )
        papers = []
        for r in records:
            paper_id = str(r.get("paper_id"))
            paper: dict[str, Any] = {**r, "submitted": paper_id in submitted}
            if experiments is not None:
                by_exp: list[dict[str, Any] | None] = []
                for exp in experiments:
                    if query_id not in exp["covered"]:
                        by_exp.append(None)
                        continue
                    pred = exp["pred_by_id"].get(query_id, {})
                    by_exp.append(
                        {
                            "r": candidate_rank_map(pred).get(paper_id),
                            "s": int(paper_id in submitted_paper_ids(pred)),
                        }
                    )
                paper["by_exp"] = by_exp
            papers.append(paper)
        entry = {
            "query_id": query_id,
            "task_family": first.get("task_family"),
            "label": query_label(records),
            "contradicting": has_contradicting(records),
            "question": first.get("question"),
            "answer_text": first.get("answer_text"),
            "judge_model": first.get("judge_model"),
            "papers": papers,
        }
        if experiments is not None:
            entry["top_by_exp"] = [
                top_candidate_ids(exp["pred_by_id"].get(query_id, {}))
                if query_id in exp["covered"]
                else None
                for exp in experiments
            ]
        queries.append(entry)
    return queries


# ---------------------------------------------------------------------------
# HTML ビューア（外部依存なし・単一ファイル）
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#667; --line:#ddd; --card:#f6f7f9;
  --good:#1a7f37; --fair:#9a6700; --noisy:#cf222e; --accent:#0969da;
  /* 検索上位表で gold に当たった行の地。--good を薄く敷いた値を明示的に置く。 */
  --hit:#e6f4ea;
  /* チャート2系列（series = task_family）。CVD検証済み: 青×橙は protan/deutan/tritan
     いずれも ΔE>28 で分離する。ラベル(良問/やや良問/悪問)は行テキストで識別するので
     3状態色をチャートの識別に使わない（赤×琥珀は deutan ΔE1.5 で分離不能のため）。 */
  --chart-single:#0969da; --chart-multi:#bc4c00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --card:#161b22;
    --chart-single:#4493f8; --chart-multi:#db6d28; --hit:#12261a; } }
/* claude.ai のテーマトグルは data-theme を打つ。メディアクエリより優先させる。 */
:root[data-theme="dark"] { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --card:#161b22;
  --chart-single:#4493f8; --chart-multi:#db6d28; --hit:#12261a; }
:root[data-theme="light"] { --bg:#fff; --fg:#1a1a1a; --muted:#667; --line:#ddd; --card:#f6f7f9;
  --chart-single:#0969da; --chart-multi:#bc4c00; --hit:#e6f4ea; }
* { box-sizing:border-box; }
body { margin:0; padding:16px 24px; background:var(--bg); color:var(--fg);
  font:14px/1.6 -apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif; }
h1 { font-size:18px; } h2 { font-size:15px; }
.summary { display:flex; flex-wrap:wrap; gap:16px; margin:12px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 14px; }
.stat b { font-size:18px; display:block; font-variant-numeric:tabular-nums; }
.chart { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 18px; margin:12px 0; max-width:640px; }
.chart h2 { margin:0 0 2px; }
.chart .note { font-size:12px; color:var(--muted); margin:0 0 10px; }
.legend { display:flex; gap:16px; font-size:12px; color:var(--muted); margin-bottom:10px; }
.chip { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px;
  vertical-align:-1px; }
.chip.single { background:var(--chart-single); } .chip.multi { background:var(--chart-multi); }
.crow { display:grid; grid-template-columns:70px 1fr; gap:10px; align-items:center;
  padding:6px 0; cursor:pointer; border-radius:6px; }
.crow:hover { background:var(--bg); }
.crow + .crow { border-top:1px solid var(--line); }
.clabel { font-size:13px; text-align:right; }
.cbars { display:flex; flex-direction:column; gap:2px; }
.cbar { display:flex; align-items:center; gap:6px; }
.cbar i { display:block; height:13px; border-radius:0 4px 4px 0; min-width:2px; }
.cbar.single i { background:var(--chart-single); } .cbar.multi i { background:var(--chart-multi); }
.cbar b { font-size:12px; font-variant-numeric:tabular-nums; font-weight:600; }
.filters { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:16px 0;
  padding:10px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
.filters select { padding:4px 6px; }
.query { border:1px solid var(--line); border-radius:8px; margin:8px 0; overflow:hidden; }
.qhead { display:flex; gap:10px; align-items:baseline; padding:8px 12px; cursor:pointer; }
.qhead:hover { background:var(--card); }
.qid { font-family:monospace; color:var(--accent); white-space:nowrap; }
.qtext { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
  color:#fff; white-space:nowrap; }
.badge.good { background:var(--good);} .badge.fair { background:var(--fair);}
.badge.noisy { background:var(--noisy);} .badge.flag { background:#8250df; }
.badge.family { background:var(--muted); }
.qbody { display:none; padding:12px 16px; border-top:1px solid var(--line); }
.query.open .qbody { display:block; }
.paper { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:10px 14px; margin:8px 0; }
.paper h3 { margin:0 0 4px; font-size:14px; }
.rel { font-weight:600; }
.rel.supporting { color:var(--good);} .rel.partial { color:var(--fair);}
.rel.irrelevant,.rel.no_evidence,.rel.contradicting { color:var(--noisy);}
.kv { margin:2px 0; } .kv b { color:var(--muted); font-weight:600; margin-right:6px; }
.excerpt { font-size:13px; color:var(--muted); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:6px 10px; margin:4px 0;
  max-height:180px; overflow:auto; white-space:pre-wrap; }
.report { border-left:3px solid var(--accent); padding:4px 12px; margin:8px 0;
  background:var(--card); border-radius:0 8px 8px 0; }
.evmap { border-collapse:collapse; margin:8px 0; font-size:13px; width:100%; }
.evmap th, .evmap td { border:1px solid var(--line); padding:4px 10px; text-align:left; }
.evmap th { background:var(--card); color:var(--muted); font-weight:600; }
.evmap td { font-variant-numeric:tabular-nums; }
.evmap .noev { color:var(--noisy); font-weight:600; }
/* 検索上位表で gold に当たった行。色だけに頼らず「gold」の文字も列に出す。 */
.evmap tr.hit td { background:var(--hit); }
.mono { font-family:monospace; }
.muted { color:var(--muted); }
"""

_JS = """
const state = { label:'', family:'', noise:'', unretrieved:false, contradicting:false,
  exp: DATA.default_exp||0 };
function esc(s){ const d=document.createElement('div'); d.textContent=s==null?'':String(s); return d.innerHTML; }
function relBadge(r){ return `<span class="rel ${esc(r)}">${esc(r||'unjudged')}</span>`; }
// 選択中の実験での {r: 検索順位, s: 提出} 。その実験の対象外クエリは null。
// by_exp が無い古い JSONL では監査時に join した値へフォールバックする。
function expEntry(p){
  if(p.by_exp && (DATA.experiments||[]).length) return p.by_exp[state.exp];
  return {r:(p.retrieval||{}).rank, s:p.submitted?1:0};
}
function rankText(p){ const e=expEntry(p); if(!e) return '対象外'; return e.r==null?'未検出':`${e.r}位`; }
function subMark(p){ const e=expEntry(p); return (e&&e.s)?'○':'—'; }
function paperCard(p){
  const ctx = p.context||{};
  const evid = (ctx.evidence_excerpts||[]).map(e=>`
    <div class="kv"><b>${esc(e.evidence_id)}</b><span class="mono">${esc(e.evidence_text)}</span>
      <span class="muted">(${esc(e.source_type)} ${esc(JSON.stringify(e.locator||{}))})</span></div>
    ${e.chunk_excerpt?`<div class="excerpt">${esc(e.chunk_excerpt)}</div>`:''}`).join('');
  return `<div class="paper">
    <h3>${esc(p.paper_id)} — ${esc((p.paper||{}).title||'')}
      <span class="muted">[${esc((p.paper||{}).venue||'')} ${esc((p.paper||{}).year||'')}]</span></h3>
    <div class="kv">${relBadge(p.relevance)}
      ／ 検索順位: ${rankText(p)} ／ 提出: ${subMark(p)}
      ／ 確信度: ${esc(p.confidence)}
      ${p.noise_type?`／ noise_type: <b>${esc(p.noise_type)}</b>`:''}
      ${p.body_supports_answer===true?'／ <b>本文に根拠がある可能性（アノテ漏れ疑い）</b>':''}</div>
    <div class="kv"><b>evidence_role</b>${esc(p.evidence_role)}</div>
    <div class="kv"><b>relation_to_gold</b>${esc(p.relation_to_gold)}</div>
    ${p.misleading_risk?`<div class="kv"><b>誤答リスク</b>${esc(p.misleading_risk)}</div>`:''}
    ${evid?`<div class="kv"><b>evidence 本文</b></div>${evid}`:''}
    ${ctx.abstract?`<div class="kv"><b>abstract</b></div><div class="excerpt">${esc(ctx.abstract)}</div>`:''}
    ${ctx.body_excerpt?`<div class="kv"><b>本文冒頭</b></div><div class="excerpt">${esc(ctx.body_excerpt)}</div>`:''}
  </div>`;
}
function evidenceMap(q){
  // gold 論文 × evidence の対応表。「evidence が4件あるから4論文に付いている」
  // という誤読を防ぐため、論文ごとの件数とゼロ本(★)を先頭で見せる。
  const rows = q.papers.map(p=>{
    const ids = p.evidence_ids||[];
    const ev = ids.length
      ? `${ids.map(esc).join(', ')} <span class="muted">(${ids.length}件)</span>`
      : '<span class="noev">★ no_evidence（0件）</span>';
    return `<tr><td class="mono">${esc(p.paper_id)}</td>
      <td>${esc(((p.paper||{}).title||'').slice(0,60))}</td>
      <td>${ev}</td><td>${relBadge(p.relevance)}</td>
      <td>${rankText(p)}</td><td>${subMark(p)}</td></tr>`;
  }).join('');
  return `<div style="overflow-x:auto"><table class="evmap">
    <tr><th>gold 論文</th><th>タイトル</th><th>evidence</th><th>relevance</th><th>検索順位</th><th>提出</th></tr>
    ${rows}</table></div>`;
}
function topCandidates(q){
  // 選択中の実験で実際に検索が返した上位N本。gold と突き合わせられるように
  // gold 行に印を付ける（gold 側の順位表は evidenceMap にある）。
  const ids = (q.top_by_exp||[])[state.exp];
  if(!ids) return '';
  const gold = new Set(q.papers.map(p=>p.paper_id));
  const rows = ids.map((id,i)=>{
    const m = (DATA.paper_meta||{})[id] || ['','',''];
    const isGold = gold.has(id);
    return `<tr${isGold?' class="hit"':''}><td>${i+1}</td><td class="mono">${esc(id)}</td>
      <td>${esc(m[0])}</td><td class="muted">${esc(m[1])} ${esc(m[2]==null?'':m[2])}</td>
      <td>${isGold?'<b>gold</b>':'—'}</td></tr>`;
  }).join('');
  const hits = ids.filter(id=>gold.has(id)).length;
  return `<div style="overflow-x:auto"><table class="evmap">
    <tr><th colspan="5">実際の検索結果 上位${ids.length}本
      <span class="muted">（この実験で gold は ${hits}/${ids.length} 本）</span></th></tr>
    <tr><th>順位</th><th>paper_id</th><th>タイトル</th><th>会議</th><th>gold か</th></tr>
    ${rows}</table></div>`;
}
function reportBlock(q){
  // spec 4.3: ラベルごとの説明。3.2 の判定フィールドをそのまま根拠として描画する。
  if(q.label==='good'){
    const rows = q.papers.map(p=>`<div class="kv"><b>${esc((p.evidence_ids||[]).join(', ')||p.paper_id)}</b>${esc(p.evidence_role)}</div>`).join('');
    return `<div class="report"><b>良問: evidence → 主張の対応</b>${rows}</div>`;
  }
  if(q.label==='fair'){
    const rows = q.papers.filter(p=>p.relevance!=='supporting')
      .map(p=>`<div class="kv"><b>${esc(p.paper_id)} (${esc(p.relevance)})</b>${esc(p.evidence_role)}</div>`).join('');
    return `<div class="report"><b>やや良問: supporting でない evidence のずれ</b>${rows}</div>`;
  }
  const rows = q.papers.filter(p=>p.relevance==='no_evidence')
    .map(p=>`<div class="kv"><b>${esc(p.paper_id)}</b> ${esc(p.evidence_role)}<br>
      <b>混入理由</b>${esc(p.noise_type||'?')} — ${esc(p.relation_to_gold)}<br>
      <b>誤答リスク</b>${esc(p.misleading_risk||'なし')}</div>`).join('');
  return `<div class="report"><b>悪問: evidence を持たない gold 論文</b>${rows}</div>`;
}
function selLabel(v){
  const s=document.getElementById('f_label');
  s.value = (state.label===v) ? '' : v;   // 同じ行をもう一度クリックで解除
  state.label = s.value; render();
}
function matches(q){
  if(state.label && q.label!==state.label) return false;
  if(state.family && q.task_family!==state.family) return false;
  if(state.noise && !q.papers.some(p=>p.noise_type===state.noise)) return false;
  if(state.unretrieved && !q.papers.some(p=>{const e=expEntry(p); return e && e.r==null;})) return false;
  if(state.contradicting && !q.contradicting) return false;
  return true;
}
function render(){
  const root=document.getElementById('queries');
  const shown=DATA.queries.filter(matches);
  document.getElementById('count').textContent=`${shown.length} / ${DATA.queries.length} クエリ`;
  root.innerHTML=shown.map(q=>`
    <div class="query" id="qq_${esc(q.query_id)}">
      <div class="qhead" onclick="this.parentElement.classList.toggle('open')">
        <span class="qid">${esc(q.query_id)}</span>
        <span class="badge ${esc(q.label)}">${esc(DATA.label_ja[q.label]||q.label)}</span>
        ${q.contradicting?'<span class="badge flag">contradicting</span>':''}
        <span class="badge family">${esc(q.task_family)}</span>
        <span class="qtext">${esc(q.question)}</span>
        <span class="muted">${q.papers.length}本</span>
      </div>
      <div class="qbody">
        <div class="kv"><b>質問</b>${esc(q.question)}</div>
        <div class="kv"><b>正解回答</b>${esc(q.answer_text)}</div>
        ${evidenceMap(q)}
        ${topCandidates(q)}
        ${reportBlock(q)}
        ${q.papers.map(paperCard).join('')}
      </div>
    </div>`).join('');
}
function fmt3(v){ return (v==null)?'—':v.toFixed(3); }
function renderExpStats(){
  const x=(DATA.experiments||[])[state.exp];
  const m=(x&&x.metrics)||{};
  document.getElementById('m_recall').textContent=fmt3(m.paper_recall_macro);
  document.getElementById('m_clean').textContent=fmt3(m.paper_recall_macro_clean);
  document.getElementById('m_gap').textContent=fmt3(m.dataset_noise_gap);
  // 未検出 supporting はこの実験の候補列で数え直す（対象外クエリは数えない）
  let unret=0;
  DATA.queries.forEach(q=>q.papers.forEach(p=>{
    const e=expEntry(p); if(p.relevance==='supporting' && e && e.r==null) unret++;
  }));
  document.getElementById('m_unret').textContent=String(unret);
  const cov=document.getElementById('expcov');
  if(x) cov.textContent = `${x.coverage}/${DATA.queries.length}件`
    + (x.coverage<DATA.queries.length
       ? '（部分実行 — 指標はカバー分のみ。55件の実験と比較しないこと）' : '');
}
function init(){
  const eSel=document.getElementById('f_exp');
  (DATA.experiments||[]).forEach((x,i)=>eSel.insertAdjacentHTML('beforeend',
    `<option value="${i}">${esc(x.name)}${x.coverage<DATA.queries.length?` [${x.coverage}/${DATA.queries.length}]`:''}</option>`));
  eSel.value=String(state.exp);
  eSel.onchange=e=>{state.exp=Number(e.target.value); renderExpStats(); render();};
  renderExpStats();
  const families=[...new Set(DATA.queries.map(q=>q.task_family))].sort();
  const fSel=document.getElementById('f_family');
  families.forEach(f=>fSel.insertAdjacentHTML('beforeend',`<option value="${esc(f)}">${esc(f)}</option>`));
  const noises=[...new Set(DATA.queries.flatMap(q=>q.papers.map(p=>p.noise_type).filter(Boolean)))].sort();
  const nSel=document.getElementById('f_noise');
  noises.forEach(n=>nSel.insertAdjacentHTML('beforeend',`<option value="${esc(n)}">${esc(n)}</option>`));
  document.getElementById('f_label').onchange=e=>{state.label=e.target.value;render();};
  fSel.onchange=e=>{state.family=e.target.value;render();};
  nSel.onchange=e=>{state.noise=e.target.value;render();};
  document.getElementById('f_unret').onchange=e=>{state.unretrieved=e.target.checked;render();};
  document.getElementById('f_contra').onchange=e=>{state.contradicting=e.target.checked;render();};
  render();
}
init();
"""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value)) if value is not None else "—"


def label_chart_html(summary: dict[str, Any]) -> str:
    """ラベル分布の task_family 別グループ横棒。

    色は task_family（2系列、青×橙）に割り、ラベル(良問/やや良問/悪問)は行の
    テキストで識別する。3状態色(緑/琥珀/赤)をチャートの識別色にしないのは、
    赤×琥珀が deutan で分離できない(ΔE1.5)ため。行クリックでそのラベルに絞り込む。
    """
    by_family = summary.get("label_counts_by_family") or {}
    if not by_family:
        return ""
    ordered = [f for f in ("hidden_source_single_paper", "multi_paper") if f in by_family]
    ordered += [f for f in sorted(by_family) if f not in ordered]
    ordered = ordered[:2]  # 系列色は2色ぶんだけ定義してある
    css_class = {family: cls for family, cls in zip(ordered, ("single", "multi"))}
    max_count = max(
        (count for family in ordered for count in by_family[family].values()), default=1
    )

    legend = "".join(
        f'<span><span class="chip {css_class[family]}"></span>'
        f"{html.escape(family)} ({sum(by_family[family].values())}件)</span>"
        for family in ordered
    )
    rows = []
    for label in ("good", "fair", "noisy"):
        bars = []
        for family in ordered:
            count = by_family[family].get(label, 0)
            width = count / max_count * 100
            bars.append(
                f'<div class="cbar {css_class[family]}" '
                f'title="{LABEL_JA[label]} / {html.escape(family)}: {count}件">'
                f'<i style="width:{width:.1f}%"></i><b>{count}</b></div>'
            )
        rows.append(
            f'<div class="crow" onclick="selLabel(\'{label}\')">'
            f'<div class="clabel">{LABEL_JA[label]}</div>'
            f'<div class="cbars">{"".join(bars)}</div></div>'
        )
    return (
        '<div class="chart"><h2>クエリラベル分布（task_family 別）</h2>'
        '<p class="note">行をクリックするとそのラベルで絞り込み</p>'
        f'<div class="legend">{legend}</div>{"".join(rows)}</div>'
    )


def build_html(
    queries: list[dict[str, Any]],
    summary: dict[str, Any],
    title: str = "クエリ品質監査",
    experiments: list[dict[str, Any]] | None = None,
    default_exp: int = 0,
    paper_meta: dict[str, list[Any]] | None = None,
) -> str:
    label_counts = summary.get("label_counts") or {}
    # 判定にだけ依存する統計は静的タイル。検索順位・提出に依存するもの
    # （recall/clean/差/未検出supporting）は実験セレクタに連動して JS が書き換える。
    stats = [
        ("クエリ数", summary.get("n_queries")),
        ("(query, paper) ペア", summary.get("n_pairs")),
        ("良問", label_counts.get("good", 0)),
        ("やや良問", label_counts.get("fair", 0)),
        ("悪問", label_counts.get("noisy", 0)),
        ("contradicting", len(summary.get("contradicting_queries") or [])),
        ("アノテ漏れ疑い", len(summary.get("missing_annotation_suspects") or [])),
    ]
    stats_html = "".join(
        f'<div class="stat"><b>{_fmt(v)}</b>{html.escape(str(k))}</div>' for k, v in stats
    )
    stats_html += (
        '<div class="stat"><b id="m_recall">—</b>paper_recall_macro</div>'
        '<div class="stat"><b id="m_clean">—</b>paper_recall_macro_clean</div>'
        '<div class="stat"><b id="m_gap">—</b>ノイズ由来の差</div>'
        '<div class="stat"><b id="m_unret">—</b>未検出 supporting</div>'
    )
    noise_html = " ／ ".join(
        f"{html.escape(str(k))}: {v}"
        for k, v in (summary.get("noise_type_distribution") or {}).items()
    )
    exp_meta = [
        {"name": e["name"], "coverage": e["coverage"], "metrics": e["metrics"]}
        for e in experiments or []
    ]
    payload = {
        "queries": queries,
        "summary": summary,
        "label_ja": LABEL_JA,
        "experiments": exp_meta,
        "default_exp": default_exp,
        # 検索上位に出た論文だけのタイトル辞書。クエリ側は ID しか持たないので
        # 同じ論文が何度出ても実体は1回で済む。
        "paper_meta": paper_meta or {},
    }
    # </script> がデータ内に現れても script 要素が閉じないようにエスケープする。
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="muted" style="margin:4px 0"><a href="experiments.html" id="explink"
  style="color:var(--accent)">→ 実験一覧（結果と設定値）</a></p>
<div class="filters">
  <label>実験（report/ のレポート名と対応） <select id="f_exp"></select></label>
  <span id="expcov" class="muted"></span>
</div>
<div class="summary">{stats_html}</div>
{label_chart_html(summary)}
<div class="muted">noise_type 分布: {noise_html or 'なし'}</div>
<div class="filters">
  <label>ラベル <select id="f_label"><option value="">すべて</option>
    <option value="good">良問</option><option value="fair">やや良問</option>
    <option value="noisy">悪問</option></select></label>
  <label>task_family <select id="f_family"><option value="">すべて</option></select></label>
  <label>noise_type <select id="f_noise"><option value="">すべて</option></select></label>
  <label><input type="checkbox" id="f_unret"> 未検出の gold を持つクエリのみ</label>
  <label><input type="checkbox" id="f_contra"> contradicting を含むクエリのみ</label>
  <span id="count" class="muted"></span>
</div>
<div id="queries"></div>
<script>const DATA = {data_json};</script>
<script>{_JS}</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="監査 JSONL から集計と HTML ビューアを再生成する。")
    parser.add_argument("--audit", default="audits/query_audit.jsonl")
    parser.add_argument(
        "--pred",
        action="append",
        default=None,
        help="予測 JSONL（複数可）。experiments.jsonl に載っていない merged 等を"
        "セレクタに足す。最初に指定したものが既定の表示実験になる",
    )
    parser.add_argument(
        "--experiments",
        default="results/experiments.jsonl",
        help="実験記録。output の予測ファイルが実在する行を実験セレクタに並べる"
        "（report/*.md と同じ命名）。'' で無効化",
    )
    parser.add_argument(
        "--paper-metadata",
        default="data/paper_metadata.jsonl",
        help="検索上位に出た論文のタイトル・会議名の引き当てに使う",
    )
    parser.add_argument("--output", default="report/query_audit.html")
    args = parser.parse_args()

    grouped = group_by_query(read_jsonl(Path(args.audit)))
    experiments = load_experiments(
        Path(args.experiments) if args.experiments else None,
        [Path(p) for p in args.pred or []],
        grouped,
    )
    default_exp = default_experiment_index(experiments, len(grouped))
    primary = experiments[default_exp]["pred_by_id"] if experiments else None

    summary = summarize(grouped, primary)
    # stdout の指標は既定実験のカバー分だけで計算した値に差し替える
    # （部分実行の予測を55件の gold に当てて薄まった値を出さないため）。
    if experiments:
        summary["metrics"] = {
            "experiment": experiments[default_exp]["name"],
            "coverage": experiments[default_exp]["coverage"],
            **(experiments[default_exp]["metrics"] or {}),
        }
    queries = viewer_payload(grouped, primary, experiments)

    # 検索上位に実際に出た論文のタイトルだけを埋め込む（全27,487件は入れない）。
    needed = {pid for q in queries for ids in (q.get("top_by_exp") or []) if ids for pid in ids}
    all_titles = load_paper_titles(Path(args.paper_metadata))
    paper_meta = {pid: all_titles[pid] for pid in needed if pid in all_titles}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(
            queries,
            summary,
            experiments=experiments,
            default_exp=default_exp,
            paper_meta=paper_meta,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
