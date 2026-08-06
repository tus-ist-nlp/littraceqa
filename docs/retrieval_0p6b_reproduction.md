# Reproduce the two-lane retrieval evaluation

This guide runs the retrieval-only experiment with prebuilt MinerU indexes and
`Qwen/Qwen3-Reranker-0.6B`. It does not run MinerU, read the original PDFs, call
Azure, or run an answer-generation agent.

The search configuration is pinned in
`configs/search_style/bm25_two_lane_qwen3_0p6b_reranker.yaml`. It performs:

1. Chunk BM25 and Paper BM25 retrieval.
2. A base lane and a bounded candidate-expansion lane.
3. Qwen3 0.6B reranking within each 50-paper lane.
4. Weighted RRF with lane weights `1.0` and `1.15`.
5. A final Qwen3 rerank over at most 100 unique papers.

## Requirements

- Linux or WSL2. The BM25 loader uses Linux file-locking APIs.
- Python 3.13.
- An NVIDIA GPU with bfloat16 support and a driver compatible with CUDA 12.8.
- At least 32 GiB system RAM and 8 GiB GPU memory.
- About 12 GiB free disk space for the same-server path, or 30 GiB when the
  indexes must also be copied to another computer.
- `uv` and Git LFS.

The CUDA toolkit does not need to be installed separately. The locked PyTorch
wheel supplies the CUDA runtime; the host still needs a working NVIDIA driver.

## Fast path on the existing server

Use this path when the computer already mounts `/data2`. It reads the prebuilt
indexes in place, so each user does not need another 8 GiB copy. Each user
should still use their own clone, virtual environment, model cache, and output
directory. Do not switch branches inside another user's working tree.

### 1. Check out the branch in a user-owned clone

For a new clone:

```bash
git lfs install
git clone --branch KumagaiKotaro/di_rag_improvement \
  git@github.com:tus-ist-nlp/littraceqa.git \
  "$HOME/projects/littraceqa-di-rag"
cd "$HOME/projects/littraceqa-di-rag"
git lfs pull
```

For an existing clean clone:

```bash
git fetch origin
git switch KumagaiKotaro/di_rag_improvement
git pull --ff-only origin KumagaiKotaro/di_rag_improvement
git lfs pull
```

Commit `3b92388` or a later commit contains the portable 0.6B environment.
Uncommitted or unpushed changes from another checkout are never visible here.

### 2. Create a private Python environment

Do not reuse another user's `.venv`; its interpreter paths and package cache
belong to that account.

```bash
uv python install 3.13
uv sync --locked --python 3.13 --extra retrieval
```

### 3. Check shared-index access

The server path config points directly at the completed full-corpus indexes:

```text
/data2/kumagai/littraceqa_data/mineru_eval/
  accuracy_ladder_c27487/accuracy_27487/sparse/index/mineru/
```

Check access before loading several gigabytes into memory:

```bash
SHARED_INDEX=/data2/kumagai/littraceqa_data/mineru_eval/accuracy_ladder_c27487/accuracy_27487/sparse/index/mineru

test -r "$SHARED_INDEX/bm25s/CURRENT.json" && \
test -r "$SHARED_INDEX/paper_bm25/method_alias_graph.json" && \
test -r "$SHARED_INDEX/specter2_paper_embeddings/index_config.json" && \
echo "Shared indexes are readable"
```

As of 2026-08-06, several files are private to the `kumagai` Unix account, so
the check fails for other accounts even though the data is under `/data2`.
After confirming the laboratory sharing policy, the owner can grant read-only
access once; this does not grant write access:

```bash
SHARED_INDEX=/data2/kumagai/littraceqa_data/mineru_eval/accuracy_ladder_c27487/accuracy_27487/sparse/index/mineru

chmod -R o+rX \
  "$SHARED_INDEX/bm25s" \
  "$SHARED_INDEX/paper_bm25" \
  "$SHARED_INDEX/specter2_paper_embeddings"
```

The evaluation loader memory-maps or reads these indexes and does not modify
them. Never use `configs/paths/server_shared_retrieval.yaml` with
`run_search.py --build`.

### 4. Select the Qwen3 0.6B cache

The existing 0.6B cache is currently below `/home/kumagai`, which other Unix
accounts cannot traverse. The `kumagai` account can reuse it, while other
accounts must download the pinned 1.2 GiB snapshot once or wait for a shared
copy.

When running as the `kumagai` Unix account, reuse the existing cache directly
and skip the download command:

```bash
export HF_HUB_CACHE=/home/kumagai/littraceqa_data/cache/qwen3-reranker-0.6b/hub
```

For every other account, use a private cache until the shared copy exists:

```bash
export HF_HUB_CACHE="$HOME/littraceqa_data/cache/qwen3-reranker-0.6b/hub"

uv run --locked --no-sync python scripts/download_qwen3_reranker.py \
  --cache-dir "$HF_HUB_CACHE"
```

The owner can optionally prepare the server-wide read-only cache once:

```bash
SHARED_HF=/data2/kumagai/models/huggingface/hub
MODEL_CACHE=models--Qwen--Qwen3-Reranker-0.6B

rsync -a \
  "/home/kumagai/littraceqa_data/cache/qwen3-reranker-0.6b/hub/$MODEL_CACHE/" \
  "$SHARED_HF/$MODEL_CACHE/"
chmod -R o+rX "$SHARED_HF/$MODEL_CACHE"
```

After that copy exists and is readable, every server user can replace the
user-owned value with:

```bash
export HF_HUB_CACHE=/data2/kumagai/models/huggingface/hub
```

### 5. Run the two-query smoke test

Check current GPU use first. Select one idle physical GPU; after filtering with
`CUDA_VISIBLE_DEVICES`, the config's `cuda:0` means that selected GPU.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=2  # Replace 2 with an idle physical GPU ID.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p "$HOME/littraceqa_data/mineru_eval/runs"

uv run --locked --no-sync python scripts/eval_retrieval.py \
  --paths configs/paths/server_shared_retrieval.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_two_lane_qwen3_0p6b_reranker.yaml \
  --queries data/validation_answer_bearing_gold_draft.jsonl \
  --query-id q_020 \
  --query-id q_023 \
  --ks 1,3,5,8,10,20,50,100 \
  --read-only-root /data2 \
  --allow-shared-index-load \
  --output "$HOME/littraceqa_data/mineru_eval/runs/two_lane_0p6b_smoke.json"
```

`--allow-shared-index-load` acknowledges the memory cost of loading the shared
indexes; it does not permit writes. The output remains in the current user's
home directory. For all 55 queries, remove the two `--query-id` options and use
a different output file. Add `--resume` when restarting the exact same command.

The original PDFs, MinerU output, and merged chunk file are not read during
this retrieval-only evaluation.

## Portable path for another computer

### 1. Install the locked retrieval environment

Run all commands from the repository root.

```bash
git lfs install
git lfs pull
uv python install 3.13
uv sync --locked --python 3.13 --extra retrieval
```

Plain `uv sync` installs only the base dataset tools. The `retrieval` extra is
required for PyTorch, Transformers, BM25, and YAML support. It intentionally
does not install Marker, Azure SDKs, FAISS, or the MinerU preprocessing stack.

Verify the resolved environment without changing it:

```bash
uv run --locked --no-sync python -c \
  'import bm25s, numpy, torch, transformers, yaml; print({"python_torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.is_available(), "transformers": transformers.__version__, "bm25s": bm25s.__version__, "numpy": numpy.__version__, "yaml": yaml.__version__})'
nvidia-smi
```

The pinned Linux environment should report PyTorch `2.11.0+cu128`,
Transformers `5.13.1`, BM25S `0.3.9`, and NumPy `2.5.1`.

### 2. Download the pinned reranker once

Choose a user-owned cache and keep the same `HF_HUB_CACHE` value for download
and evaluation.

```bash
export HF_HUB_CACHE="$HOME/littraceqa_data/cache/huggingface/hub"

uv run --locked --no-sync python scripts/download_qwen3_reranker.py \
  --cache-dir "$HF_HUB_CACHE"
```

The script reads both the model name and immutable revision from the search
YAML. The evaluation itself uses `local_files_only: true`, so it cannot start a
network download unexpectedly.

### 3. Transfer the prebuilt indexes

The evaluation needs these directories, not the MinerU source data:

```text
artifacts/retrieval/index/mineru/
├── bm25s/
├── paper_bm25/
└── specter2_paper_embeddings/
```

They occupy about 8 GiB after build-only checkpoint shards are excluded. From
the destination computer, replace `USER@SERVER` with an authorized SSH source:

```bash
mkdir -p artifacts/retrieval/index/mineru

SOURCE=/data2/kumagai/littraceqa_data/mineru_eval/accuracy_ladder_c27487/accuracy_27487/sparse/index/mineru

rsync -a --partial --info=progress2 \
  --exclude='.resumable-bm25-parts/' \
  --exclude='.resumable-bm25-scores/' \
  "USER@SERVER:$SOURCE/bm25s/" \
  artifacts/retrieval/index/mineru/bm25s/

rsync -a --partial --info=progress2 \
  "USER@SERVER:$SOURCE/paper_bm25/" \
  artifacts/retrieval/index/mineru/paper_bm25/

rsync -a --partial --info=progress2 \
  "USER@SERVER:$SOURCE/specter2_paper_embeddings/" \
  artifacts/retrieval/index/mineru/specter2_paper_embeddings/
```

Do not copy the shared MinerU directory. Do not put indexes or model weights in
Git. The `paper_bm25` copy must include `method_alias_graph.json`, and the
SPECTER2 copy must include `index_config.json`, `embeddings.npy`, and
`papers.jsonl`. A missing SPECTER2 sidecar can cause a silent sparse-only
fallback and lower accuracy.

### 4. Run the two-query smoke test

The smoke test loads the full indexes but evaluates only two multi-paper
questions. It does not call an LLM API.

```bash
mkdir -p runs
export HF_HUB_CACHE="$HOME/littraceqa_data/cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

uv run --locked --no-sync python scripts/eval_retrieval.py \
  --paths configs/paths/local_retrieval.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_two_lane_qwen3_0p6b_reranker.yaml \
  --queries data/validation_answer_bearing_gold_draft.jsonl \
  --query-id q_020 \
  --query-id q_023 \
  --ks 1,3,5,8,10,20,50,100 \
  --output runs/two_lane_0p6b_smoke.json
```

The recorded reference smoke test completed two queries with no failures and
reached Recall@100 of `1.0`. This is a connectivity and reproducibility check,
not a statistically useful accuracy result.

### 5. Run or resume all 55 validation questions

Remove the two `--query-id` options and use a new output path:

```bash
uv run --locked --no-sync python scripts/eval_retrieval.py \
  --paths configs/paths/local_retrieval.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_two_lane_qwen3_0p6b_reranker.yaml \
  --queries data/validation_answer_bearing_gold_draft.jsonl \
  --ks 1,3,5,8,10,20,50,100 \
  --output runs/two_lane_0p6b_validation.json
```

The evaluator writes a checkpoint after each question. If interrupted, run the
same command with `--resume`. The checkpoint rejects changed configs, query
data, or cutoffs instead of mixing incompatible results.

## CPU-only fallback

The checked-in search YAML is the GPU benchmark and uses `cuda:0`, bfloat16,
and batch size 2. For a CPU-only functional check, copy the YAML locally and
change only these values:

```yaml
device: cpu
dtype: float32
batch_size: 1
```

Do not compare CPU fallback timings with the recorded GPU run. Native Windows
is not supported; use WSL2.

## What is and is not reproduced

This setup reproduces the selected two-lane 0.6B retrieval configuration and
its software versions. The two-query run has been measured, but a full 55-query
score for this exact two-lane configuration has not yet been established. Do
not report the older one-lane 55-query metrics as results from this config.
