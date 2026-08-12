#!/usr/bin/env bash
# faiss_qwen3 の分散ビルドを確実に止める。
#
# **なぜ専用スクリプトが要るか**: `pkill -f build_faiss_qwen3_shard.py` では止まらない。
# torch.multiprocessing の spawn ワーカーはコマンドラインが
#   python -c 'from multiprocessing.spawn import spawn_main; ...'
# になるためパターンに一致せず、親だけが死んで**子が孤児として走り続ける**
# (実際に2回踏んだ)。GPUを掴んでいるプロセスから辿るのが確実。
#
# 単一GPU指定のときは親プロセス自身が埋め込むので子が居らず、pkill が効いてしまう。
# この非対称性が「一部のシャードだけ生き残る」状況を生む。
#
# 停止後は同じコマンドで再実行すれば _embeddings.done から再開できる。
#
# 使い方:
#   bash scripts/stop_faiss_build.sh              # このマシン
#   ssh nlp02 'bash ~/littraceqa/scripts/stop_faiss_build.sh'
set -uo pipefail

me=$(id -un)

echo "=== 自分($me)のGPUプロセスを特定 ==="
pids=""
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    pid=$(echo "$pid" | tr -d ' ')
    [ -z "$pid" ] && continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    # **他人のジョブを巻き込まない**。同じGPUを共有していることがある。
    [ "$owner" = "$me" ] || { echo "  PID $pid は $owner のジョブ。触らない"; continue; }
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-70)
    echo "  PID $pid を停止対象にする: $cmd"
    pids="$pids $pid"
done

# 親プロセス(単一GPU構成では親自身が埋め込む)も対象に入れる
for pid in $(pgrep -u "$me" -f build_faiss_qwen3_shard.py 2>/dev/null); do
    case " $pids " in *" $pid "*) ;; *) pids="$pids $pid"; echo "  PID $pid を停止対象にする(親)";; esac
done

[ -z "$pids" ] && { echo "停止対象なし"; exit 0; }

echo "=== SIGTERM 送信 ==="
# SIGKILL は使わない。終了時に embeddings.flush() が走るので、書き戻しを待つ。
kill $pids 2>/dev/null

echo "=== 終了待ち(HDDだと最終flushに数分かかる) ==="
for i in $(seq 1 60); do
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
        p=$(echo "$p" | tr -d ' ')
        [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$me" ] && echo "$p"
    done | wc -l)
    [ "$remaining" = "0" ] && { echo "全解放(${i}0秒)"; exit 0; }
    sleep 10
done

echo "警告: 10分待っても解放されないプロセスがあります"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
exit 1
