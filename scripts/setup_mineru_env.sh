#!/usr/bin/env bash
# MinerU 用の隔離 venv (.venv-mineru) を構築し、pipeline バックエンドの
# モデル一式をダウンロードする。
# 本体の .venv とは torch / transformers のバージョンが両立しないため分ける。
# 詳細は requirements-mineru.txt の先頭コメントを参照。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# MinerU は requires-python <3.14。本体の 3.13 とは別に立てる。
uv venv .venv-mineru --python 3.12
uv pip install --python .venv-mineru -r requirements-mineru.txt

# pipeline バックエンドのモデル（layout / OCR / 数式 / 表）を取得する。
# 未取得なら初回実行時に落ちてくるが、4シャードが同時に取りに行くと競合するので
# ここで1度だけ取っておく。
.venv-mineru/bin/mineru-models-download -s huggingface -m pipeline

echo
echo "完了: .venv-mineru"
echo "実行例:"
echo "  .venv-mineru/bin/python scripts/run_mineru.py \\"
echo "    --paths configs/paths/default.yaml --gpus 0,1,2,3"
