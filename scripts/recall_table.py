#!/usr/bin/env python3
"""candidate_recall だけを実験横断で並べた HTML 表を作る。

実験を回すほど results/experiments.jsonl と report/*.md が増えて「で、結局
cr は上がったのか下がったのか」が読みにくくなる。この表は縦を指標
（cr@k × single/multi/total）、横を実験（report のファイル名）に固定して、
値の濃淡と1つ前の実験との差だけを見せる。

実行例:
    uv run python scripts/recall_table.py --output report/candidate_recall.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

from audit_report import group_by_query, load_experiments, read_jsonl
from evaluate import evaluate

KS = (1, 5, 10, 20, 50, 70)
SCENARIOS = [("total", "全55件"), ("multi", "multi_paper"), ("single", "single_paper")]
PREFIXES = [
    ("candidate_recall", "candidate_recall"),
    ("evidence_candidate_recall", "evidence_candidate_recall（根拠付き gold のみ）"),
]

_CSS = """
.viz-root{color-scheme:light;
  --bg:#fbfbfa; --surface:#fcfcfb; --card:#fff; --text:#0b0b0b; --muted:#52514e;
  --line:#e2e1dc; --qid:#1c5cab;
  --v1:#1c5cab; --n1:#fff; --v2:#3987e5; --n2:#fff; --v3:#86b6ef; --n3:#0b0b0b;
  --v4:#cde2fb; --n4:#0b0b0b; --v5:#eef4fd; --n5:#52514e;
  --up:#1c5cab; --down:#b3401a;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;
  --bg:#131312; --surface:#1a1a19; --card:#201f1e; --text:#fff; --muted:#c3c2b7;
  --line:#33322e; --qid:#86b6ef;
  --v1:#9ec5f4; --n1:#0b0b0b; --v2:#5598e7; --n2:#0b0b0b; --v3:#256abf; --n3:#fff;
  --v4:#184f95; --n4:#fff; --v5:#123a6e; --n5:#c3c2b7;
  --up:#86b6ef; --down:#e8845c;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --bg:#131312; --surface:#1a1a19; --card:#201f1e; --text:#fff; --muted:#c3c2b7;
  --line:#33322e; --qid:#86b6ef;
  --v1:#9ec5f4; --n1:#0b0b0b; --v2:#5598e7; --n2:#0b0b0b; --v3:#256abf; --n3:#fff;
  --v4:#184f95; --n4:#fff; --v5:#123a6e; --n5:#c3c2b7;
  --up:#86b6ef; --down:#e8845c;}
*{box-sizing:border-box} body{margin:0;background:var(--bg)}
.viz-root{padding:22px 18px 60px;background:var(--bg);color:var(--text);
  font:14px/1.6 -apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif}
.wrap{max-width:1400px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
.lede{color:var(--muted);font-size:13.5px;margin:0 0 14px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:11px 14px;margin:0 0 12px;display:flex;flex-wrap:wrap;gap:16px;
  align-items:center;font-size:13px}
.panel label{display:flex;align-items:center;gap:6px}
.key{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}
.box{width:20px;height:13px;border-radius:3px;border:1px solid var(--line)}
.tblwrap{overflow:auto;max-height:80vh;border:1px solid var(--line);border-radius:10px}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12.5px}
th,td{padding:4px 8px;font-variant-numeric:tabular-nums;white-space:nowrap}
thead th{position:sticky;top:0;z-index:3;background:var(--card);color:var(--muted);
  text-align:left;font-size:10.5px;font-weight:600;line-height:1.35;padding:7px 8px;
  vertical-align:bottom;white-space:normal;word-break:break-all;
  min-width:126px;max-width:146px;font-family:ui-monospace,Menlo,monospace;
  border-bottom:1px solid var(--line)}
thead th.lft{left:0;z-index:4;min-width:150px;font-family:inherit;font-size:12px}
tbody th{position:sticky;left:0;background:var(--surface);text-align:left;
  font-weight:400;font-family:ui-monospace,Menlo,monospace;font-size:12px;
  border-right:1px solid var(--line);z-index:2}
td{text-align:right}
tr.grp th,tr.grp td{background:var(--card);border-top:2px solid var(--qid)}
tr.grp th{color:var(--qid);font-weight:700;font-size:13px;font-family:inherit}
td.v1{background:var(--v1);color:var(--n1);font-weight:700}
td.v2{background:var(--v2);color:var(--n2);font-weight:600}
td.v3{background:var(--v3);color:var(--n3)}
td.v4{background:var(--v4);color:var(--n4)}
td.v5{background:var(--v5);color:var(--n5)}
.d{font-size:10px;margin-left:5px;opacity:.85}
.eff{font-size:9px;margin-left:4px;padding:0 4px;border-radius:7px;
  border:1px solid currentColor;opacity:.65;font-weight:400;vertical-align:1px}
td.na{color:var(--muted);opacity:.5}
.up{color:var(--up)} .down{color:var(--down)}
tbody tr:hover th,tbody tr:hover td{filter:brightness(1.07)}
.best::after{content:"★";font-size:9px;margin-left:3px;opacity:.8}
.note{color:var(--muted);font-size:12.5px;margin:10px 2px 0}
code{font-family:ui-monospace,Menlo,monospace;font-size:.9em}
"""

_JS = """
const state={prefix:'candidate_recall',delta:true};
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function band(v){ if(v==null)return''; if(v>=.9)return'v1'; if(v>=.8)return'v2';
  if(v>=.65)return'v3'; if(v>=.45)return'v4'; return'v5'; }
function render(){
  const E=DATA.experiments, N=E.length;
  let h='<thead><tr><th class="lft">指標</th>'
    +E.map(e=>`<th title="${esc(e)}">${esc(e)}</th>`).join('')+'</tr></thead><tbody>';
  DATA.scenarios.forEach(([sc,label])=>{
    h+=`<tr class="grp"><th>${esc(label)}</th>`+'<td></td>'.repeat(N)+'</tr>';
    DATA.ks.forEach((k,ki)=>{
      const vals=E.map((_,i)=>DATA.values[`${state.prefix}_at${k}_${sc}`]?.[i] ?? null);
      // ★は「その k を実際に測れている列」だけで競わせる（実効深さが違う値を
      // 並べて最良を決めると、単に候補列が長い実験が勝つ）。
      const eligible=vals.filter((v,i)=>v!=null&&DATA.max_rank[i]>=k);
      const best=(eligible.length&&Math.max(...eligible)-Math.min(...eligible)>1e-9)
        ?Math.max(...eligible):null;
      h+=`<tr><th>@${k}</th>`+vals.map((v,i)=>{
        const depth=DATA.max_rank[i], prev=ki>0?DATA.ks[ki-1]:0;
        // 候補列が前の k までしか無い = この k で新しい情報が increase しない -> 空欄
        if(v==null||depth<=prev) return '<td class="na" title="候補列が'+depth+'本しかなく @'+k+' は測れていない">—</td>';
        let d='';
        // 差は「両方とも表示されているセル同士」でしか出さない。空欄の列と
        // 比べた矢印は、測れていない値との差を示すことになる。
        const prevShown=i>0&&vals[i-1]!=null&&DATA.max_rank[i-1]>prev;
        if(state.delta&&prevShown){
          const diff=v-vals[i-1];
          if(Math.abs(diff)>=0.0005)
            d=`<span class="d ${diff>0?'up':'down'}">${diff>0?'▲':'▼'}${Math.abs(diff).toFixed(3).slice(1)}</span>`;
        }
        const b=(best!=null&&Math.abs(v-best)<1e-9)?' best':'';
        // k に届かないが前の k より深い -> 値は出しつつ実効深さを併記する
        const eff=depth<k?`<span class="eff" title="候補列は${depth}本。実効的には @${depth}">実効@${depth}</span>`:'';
        return `<td class="${band(v)}${b}">${v.toFixed(3)}${eff}${d}</td>`;
      }).join('')+'</tr>';
    });
  });
  document.getElementById('tbl').innerHTML=h+'</tbody>';
}
function init(){
  document.getElementById('f_metric').onchange=e=>{state.prefix=e.target.value;render();};
  document.getElementById('f_delta').onchange=e=>{state.delta=e.target.checked;render();};
  render();
}
init();
"""


def build_html(
    experiments: list[str],
    values: dict[str, list[float | None]],
    max_rank: list[int],
) -> str:
    import json

    payload = {
        "experiments": experiments,
        "values": values,
        "ks": list(KS),
        "max_rank": max_rank,
        "scenarios": [[s, label] for s, label in SCENARIOS],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(
        f'<option value="{html.escape(p)}">{html.escape(label)}</option>'
        for p, label in PREFIXES
    )
    return f"""<title>candidate_recall の実験横断比較</title>
<style>{_CSS}</style>
<div class="viz-root"><div class="wrap">
<h1>candidate_recall の実験横断比較</h1>
<p class="lede">縦が指標（@k × シナリオ）、横が実験。値は候補列（LLM 絞り込み前）に gold が
入っていた割合で、大きいほど良い。各セルの小さい矢印は<b>1つ左の実験との差</b>。</p>
<div class="panel">
  <label>指標 <select id="f_metric">{options}</select></label>
  <label><input type="checkbox" id="f_delta" checked> 左隣との差を表示</label>
  <span class="key"><span class="box" style="background:var(--v1)"></span>0.90+</span>
  <span class="key"><span class="box" style="background:var(--v2)"></span>0.80+</span>
  <span class="key"><span class="box" style="background:var(--v3)"></span>0.65+</span>
  <span class="key"><span class="box" style="background:var(--v4)"></span>0.45+</span>
  <span class="key"><span class="box" style="background:var(--v5)"></span>0.45未満</span>
  <span class="key">★ = その行の最良</span>
</div>
<div class="tblwrap"><table id="tbl"></table></div>
<p class="note">列見出しは <code>report/</code> 直下のレポートファイル名。左から古い順。
55件フル網羅の実行のみを載せている（分割実行は結合済み）。<br>
予測に残す候補は <code>reading.py</code> の <code>CANDIDATE_PAPERS_LIMIT</code>（既定50）で
切られるので、記録より深い k は測れない。そこで:<br>
<b>空欄（—）</b>= 候補列が1つ前の k までしか無く、この k で新しい情報が無い（例: 50本しか
記録の無い実験の @70 は @50 と同値になるだけ）。<br>
<b>「実効@N」</b>= 候補列が N 本で k に届かないが、1つ前の k よりは深く測れている。値は本物だが
実質 @N なので、<b>実効深さの違う列と大小を比べないこと</b>（★も k を満たす列だけで比較している）。</p>
</div></div>
<script>const DATA = {data_json};</script>
<script>{_JS}</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="candidate_recall の実験横断表を作る。")
    parser.add_argument("--audit", default="audits/query_audit.jsonl")
    parser.add_argument("--gold", default="data/validation.jsonl")
    parser.add_argument("--experiments", default="results/experiments.jsonl")
    parser.add_argument("--output", default="report/candidate_recall.html")
    args = parser.parse_args()

    grouped = group_by_query(read_jsonl(Path(args.audit)))
    exps = [
        e
        for e in load_experiments(Path(args.experiments), [], grouped)
        if e["coverage"] == len(grouped)
    ]
    # 列見出しは report/ の実ファイル名に合わせる（無ければ実験名のまま）。
    reports = {p.stem for p in Path("report").glob("*.md")}
    names = []
    for e in exps:
        hit = sorted(r for r in reports if r.startswith(e["name"]))
        names.append(hit[0] if hit else e["name"])

    # 指標は各実験の予測から測り直す。experiments.jsonl の記録値を引くと
    # 分割実行を結合した行に対応する記録が無く、取りこぼす。
    gold_records = read_jsonl(Path(args.gold))
    metrics = [
        evaluate(gold_records, list(e["pred_by_id"].values()))["metrics"] for e in exps
    ]

    # 各実験の「候補列に実際に記録されている最大順位」。recall_at_k は ranked[:k]
    # を見るだけなので、50本しか記録の無い予測の @100 は @50 と同値になる。
    # 数式は正しいが「100位まで見た結果」ではないので、表側で区別する
    # （前の k から情報が増えないセルは空欄、深いが k に届かないセルは実効深さを併記）。
    # CANDIDATE_PAPERS_LIMIT は記録上限であって検索の上限ではない。
    max_rank = []
    for e in exps:
        lengths = [
            len(p.get("candidate_papers") or []) for p in e["pred_by_id"].values()
        ]
        max_rank.append(max(lengths) if lengths else 0)

    values: dict[str, list[Any]] = {}
    for prefix, _ in PREFIXES:
        for scenario, _label in SCENARIOS:
            for k in KS:
                key = f"{prefix}_at{k}_{scenario}"
                values[key] = [m.get(f"{key}_macro") for m in metrics]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(names, values, max_rank), encoding="utf-8")
    print(f"wrote {output} ({len(exps)} 実験)", file=sys.stderr)


if __name__ == "__main__":
    main()
