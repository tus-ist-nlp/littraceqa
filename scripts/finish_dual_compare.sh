#!/usr/bin/env bash
# nlp-02 の bm25_dual ダンプ完了を待って、回収 -> 統合 -> 採点まで通す。
#
# nlp-02 側は「検索して runs を落とす」だけ（展開索引 specter2/bib/mlt が
# nlp-01 にしか無いため）。統合と採点は必ずこちらで行い、**比較相手の
# runs_rawq.jsonl（k100・同じ生の質問・同じ top_k=20）と完全に同条件**にする。
set -u
cd /home/iseakira/littraceqa

COMMON=(--paths configs/paths/default.yaml
        --process configs/process_style/mineru.yaml
        --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml
        --agent configs/agent_style/reading_expand_rrf/notable.yaml
        --pred predictions_k100_notable.jsonl
        --verdict-at-step 0
        --ks 1,5,10,20,50)

echo "=== $(date) ダンプ完了を待つ ==="
for i in $(seq 1 180); do
  n=$(ssh -o BatchMode=yes nlp-02 'wc -l < ~/littraceqa_rawq/runs_rawq_dual.jsonl 2>/dev/null || echo 0' 2>/dev/null)
  echo "$(date +%H:%M:%S) dual dump: ${n:-?} / 55"
  [ "${n:-0}" -ge 55 ] && break
  sleep 60
done

if [ "${n:-0}" -lt 55 ]; then
  echo "!! 55件に届かないまま打ち切り（$n 件）。nlp-02 の tmux dual を確認すること"
  exit 1
fi

rsync -a nlp-02:~/littraceqa_rawq/runs_rawq_dual.jsonl ./ || exit 1
echo "回収完了: $(wc -l < runs_rawq_dual.jsonl) 件"

echo
echo "########## 基準: bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100 ##########"
uv run python scripts/replay_rawq.py "${COMMON[@]}" --runs runs_rawq.jsonl 2>/dev/null

echo
echo "########## 実験1: bm25_dual（+ bm25s_paper） ##########"
uv run python scripts/replay_rawq.py "${COMMON[@]}" --runs runs_rawq_dual.jsonl \
  --output predictions_rawq_dual_offline.jsonl 2>/dev/null

echo
echo "=== $(date) 完了 ==="
