#!/usr/bin/env bash
# Build the isolated venv for MinerU (.venv-mineru) and download the pipeline
# backend's models.
# It is separate from the main .venv because their torch and transformers versions
# cannot coexist; see the comment at the top of requirements-mineru.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# MinerU requires-python is <3.14, so this stands apart from the main venv's 3.13.
uv venv .venv-mineru --python 3.12
uv pip install --python .venv-mineru -r requirements-mineru.txt

# Fetch the pipeline backend's models (layout / OCR / equations / tables). They
# would download on first use anyway, but four shards fetching at once collide, so
# it happens once, here.
.venv-mineru/bin/mineru-models-download -s huggingface -m pipeline

echo
echo "done: .venv-mineru"
echo "for example:"
echo "  .venv-mineru/bin/python scripts/run_mineru.py \\"
echo "    --paths configs/paths/default.yaml --gpus 0,1,2,3"
