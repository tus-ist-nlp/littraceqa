# LitTraceQA

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).
Install Git LFS **before** cloning — `data/paper_metadata.jsonl` is stored via
LFS and will otherwise be fetched as a broken pointer file.

```bash
git lfs install
git clone git@github.com:tus-ist-nlp/littraceqa.git
cd littraceqa
uv sync
```

## Azure RAG pipeline

This repository now includes a resumable Azure pipeline for the competition:

1. Extract the downloaded PDF zip files.
2. Analyze PDFs with Azure AI Document Intelligence.
3. Chunk the extracted paper text/tables/figures.
4. Embed chunks with Azure OpenAI and upload them to Azure AI Search.
5. Run hybrid RAG over the validation inputs and write a submission JSONL.

### 1. Configure secrets

Fill in `.env` with your deployed Azure resources:

```bash
cp .env.example .env
```

Required values:

```text
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1536
```

If your Azure OpenAI deployment uses the newer `/openai/v1` endpoint style, set
`AZURE_OPENAI_USE_V1=true`. Otherwise keep it `false` and set
`AZURE_OPENAI_API_VERSION`.

### 2. Prepare PDFs

Download the eight zip files from the shared Box folder and put them under
`archives/`.

```bash
uv run python scripts/extract_pdf_archives.py --archives archives --output pdfs
```

The script tries to map PDFs to `paper_id` using `paper_metadata.jsonl` and writes
canonical files such as `pdfs/acl2025_00001.pdf`.

If you already used `scripts/download_pdfs.py`, you can skip this step as long
as `pdfs/{paper_id}.pdf` exists.

### 3. Document Intelligence

Smoke test one PDF first:

```bash
uv run python scripts/process_document_intelligence.py --limit 1
```

Process the full cache:

```bash
uv run python scripts/process_document_intelligence.py
```

Main outputs:

```text
artifacts/docint/raw/{paper_id}.json
artifacts/docint/chunks/{paper_id}.jsonl
artifacts/docint/chunks.jsonl
```

The raw JSON is kept so chunking can be rebuilt without another paid
Document Intelligence call:

```bash
uv run python scripts/process_document_intelligence.py --merge-only
```

To process the Box zip archives without keeping individual PDFs on disk, use:

```bash
uv run python scripts/process_box_archives_document_intelligence.py --list-archives
uv run python scripts/process_box_archives_document_intelligence.py --workers 2
```

This downloads one zip file at a time into `artifacts/box_tmp/`, sends each PDF
member to Document Intelligence from memory, writes chunks under
`artifacts/docint/chunks/`, and deletes the temporary zip before the next one.
Increase `--workers` only if your Document Intelligence quota allows it.

### 4. Azure AI Search index

Create the index and upload all chunks:

```bash
uv run python scripts/build_search_index.py --recreate
```

For a cheap first test:

```bash
uv run python scripts/build_search_index.py --recreate --limit 100
```

`AZURE_OPENAI_EMBEDDING_DIMENSIONS` must match the actual output dimension of
your embedding deployment.

You can also upload lightweight metadata chunks for every candidate paper. This
helps paper retrieval before all PDFs have been processed:

```bash
uv run python scripts/build_metadata_chunks.py
uv run python scripts/build_search_index.py \
  --chunks artifacts/metadata/chunks.jsonl \
  --embedding-batch-size 64 \
  --skip-index-create
```

### 5. Run RAG

```bash
uv run python scripts/run_rag.py \
  --input data/validation_inputs.jsonl \
  --output runs/validation_submission.jsonl
```

Evaluate against the public validation labels:

```bash
uv run python scripts/evaluate.py \
  --gold data/validation.jsonl \
  --pred runs/validation_submission.jsonl
```

Useful smoke-test commands:

```bash
uv run python scripts/run_rag.py --limit 2 --retrieval-only
uv run python scripts/run_rag.py --limit 2
```

### 6. Semantic grading

For development analysis, compare a prediction file against
`data/validation.jsonl` with Azure OpenAI:

```bash
uv run python scripts/grade_with_azure_openai.py \
  --pred runs/validation_submission.jsonl \
  --output runs/azure_openai_grades.jsonl \
  --report runs/azure_openai_grading_report.md
```

This judge is not the official metric; it is useful for seeing whether failures
come from paper retrieval, evidence retrieval, or final answer generation.
