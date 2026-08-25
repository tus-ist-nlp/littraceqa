#!/usr/bin/env bash
# runs_fat.jsonl（--dump-runs の出力）を土台に、仕様4・6・7 をまとめて振る。
#
# 全部 candidate_papers を組み直すだけの処理で、LLM も検索索引も GPU も要らない。
# 1条件あたり数十秒。本走行（4〜5時間）に上げるのはここで勝った設定だけにする。
#
#   bash scripts/sweep_merge.sh [runs.jsonl] [predictions.jsonl]
#
# 読む数字は **ecr@20**（打ち手の評価は ecr で見る——cr は原理的に取れない
# ピア gold が常に混ざって天井が張り付く）。cr@20 も並べて出るので、
# 「ecr が上がって cr が下がる」向きが出ていないかを必ず確認する。
set -euo pipefail

RUNS="${1:-runs_fat.jsonl}"
PRED="${2:-predictions_fat.jsonl}"
AGENT=configs/agent_style/reading.yaml   # 既定値の土台。--set で1つずつ振る
KS=5,10,20,50

run() {
  local label="$1"; shift
  echo "=============================================================="
  echo "== ${label}"
  echo "=============================================================="
  uv run python scripts/replay_merge.py \
    --runs "$RUNS" --pred "$PRED" --agent "$AGENT" --ks "$KS" "$@" 2>&1 |
    grep -vE "^\s*$"
  echo
}

# ---- 基準線: 本走行と同じ設定（再生の忠実性の確認も兼ねる） ----------------
# ここが「0 件の候補列が変わった」にならないときは、再生の前提が崩れている。
run "基準線 (retrieve_top_k=100, max マージ)" --set retrieve_top_k=100

# ---- 仕様4: サブクエリ間マージ ---------------------------------------------
run "4. subquery_merge=rrf" --set retrieve_top_k=100 --set subquery_merge=rrf
for k in 10 30 60 120; do
  run "4. subquery_merge=rrf, rrf_k=${k}" \
    --set retrieve_top_k=100 --set subquery_merge=rrf --set subquery_rrf_k="$k"
done

# ---- retrieve_top_k 単体の効き（土台が太いので今回だけ振れる） -------------
# 深さの自動決定（7）が「浅くしても落ちない」ことの前提を確認する。
for k in 20 40 60 100; do
  run "retrieve_top_k=${k} (max マージ)" --set retrieve_top_k="$k"
done

# ---- 仕様7: 深さをスコアの落差で決める -------------------------------------
# shallow_k/deep_k は上の単体スイープで妥当だった範囲から取る。
for gap in 0.05 0.15 0.30; do
  run "7. adaptive_depth gap=${gap} (10/40)" \
    --set "adaptive_depth={enabled: true, probe_rank: 4, gap_threshold: ${gap}, shallow_k: 10, deep_k: 40}"
done
run "7. adaptive_depth gap=0.15 (20/100)" \
  --set "adaptive_depth={enabled: true, probe_rank: 4, gap_threshold: 0.15, shallow_k: 20, deep_k: 100}"

# ---- 仕様6: プールの剪定 ----------------------------------------------------
# pool_rescore は reranker(GPU) が要るので再生できない。ここで振れるのは剪定だけ。
for n in 200 500 1000; do
  run "6. pool_prune_to=${n}" --set retrieve_top_k=100 --set pool_prune_to="$n"
done

# ---- 4 と 7 の組み合わせ ----------------------------------------------------
run "4+7. rrf + adaptive_depth" \
  --set subquery_merge=rrf \
  --set "adaptive_depth={enabled: true, probe_rank: 4, gap_threshold: 0.15, shallow_k: 10, deep_k: 40}"
