# LitTraceQA

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).
Install Git LFS **before** cloning; `data/paper_metadata.jsonl` is stored via
LFS and will otherwise be fetched as a broken pointer file.

```bash
git lfs install
git clone git@github.com:tus-ist-nlp/littraceqa.git
cd littraceqa
uv sync
```

The base install covers the dataset scripts (`scripts/`) and the
provider-agnostic tools (`littraceqa.extract_pdf_archives`,
`littraceqa.fix_chunk_locators`, `littraceqa.validate_submission`,
`littraceqa.compare_runs`). The Azure baseline additionally needs the
optional `azure` extra:

```bash
uv sync --extra azure
```

## Azure RAG pipeline (baseline)

The Azure-based baseline lives under `src/littraceqa/azure/` and is invoked
with `uv run --extra azure python -m littraceqa.azure.<module>`. (It is a
subpackage rather than a top-level `src/azure/` because a local package named
`azure` would shadow the Azure SDK's `azure.*` namespace packages.)

The pipeline is resumable and covers:

1. Extracting downloaded PDF zip files.
2. Analyzing PDFs with Azure AI Document Intelligence.
3. Chunking extracted paper text/tables/figures.
4. Embedding chunks with Azure OpenAI and uploading them to Azure AI Search.
5. Running hybrid RAG over validation inputs and writing a submission JSONL.

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
```

Optional: `AZURE_OPENAI_EMBEDDING_DIMENSIONS` defaults to 1536 and only needs
setting when your embedding deployment outputs a different dimension.

If your Azure OpenAI deployment uses the newer `/openai/v1` endpoint style, set
`AZURE_OPENAI_USE_V1=true`. Otherwise keep it `false` and set
`AZURE_OPENAI_API_VERSION`.

Validate the finished `.env` (missing required values, malformed endpoints,
unknown variable names):

```bash
uv run --extra azure python -m littraceqa.azure.check_azure_env
```

### 2. Prepare PDFs

Download the eight zip files from the shared Box folder and put them under
`archives/`.

```bash
uv run python -m littraceqa.extract_pdf_archives --archives archives --output pdfs
```

The command tries to map PDFs to `paper_id` using `paper_metadata.jsonl` and
writes canonical files such as `pdfs/acl2025_00001.pdf`.

If you already used `scripts/download_pdfs.py`, you can skip this step as long
as `pdfs/{paper_id}.pdf` exists.

### 3. Document Intelligence

Smoke test one PDF first:

```bash
uv run --extra azure python -m littraceqa.azure.process_document_intelligence --limit 1
```

Process the full cache:

```bash
uv run --extra azure python -m littraceqa.azure.process_document_intelligence
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
uv run --extra azure python -m littraceqa.azure.process_document_intelligence --merge-only
```

To process the Box zip archives without keeping individual PDFs on disk, use:

```bash
uv run --extra azure python -m littraceqa.azure.process_box_archives_document_intelligence --list-archives
uv run --extra azure python -m littraceqa.azure.process_box_archives_document_intelligence --workers 2
```

This downloads one zip file at a time into `artifacts/box_tmp/`, sends each PDF
member to Document Intelligence from memory, writes chunks under
`artifacts/docint/chunks/`, and deletes the temporary zip before the next one.
Increase `--workers` only if your Document Intelligence quota allows it.

### 4. Azure AI Search index

Create the index and upload all chunks:

```bash
uv run --extra azure python -m littraceqa.azure.build_azure_search_index --recreate
```

For a cheap first test:

```bash
uv run --extra azure python -m littraceqa.azure.build_azure_search_index --recreate --limit 100
```

`AZURE_OPENAI_EMBEDDING_DIMENSIONS` must match the actual output dimension of
your embedding deployment.

### 5. Run RAG

```bash
uv run --extra azure python -m littraceqa.azure.run_rag \
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
uv run --extra azure python -m littraceqa.azure.run_rag --limit 2 --retrieval-only
uv run --extra azure python -m littraceqa.azure.run_rag --limit 2
```

### 6. Semantic grading

For development analysis, compare a prediction file against
`data/validation.jsonl` with Azure OpenAI:

```bash
uv run --extra azure python -m littraceqa.azure.grade_with_azure_openai \
  --pred runs/validation_submission.jsonl \
  --output runs/azure_openai_grades.jsonl \
  --report runs/azure_openai_grading_report.md
```

This judge is not the official metric; it is useful for seeing whether failures
come from paper retrieval, evidence retrieval, or final answer generation.

## Further tools

Operational detail for these lives in `RUNBOOK.md`; one line each here:

- `littraceqa.azure.reanalyze_papers` — re-run Document Intelligence on an
  explicit set of papers via direct `pdf_url` downloads (raw JSON always
  saved), instead of reprocessing whole Box archives.
- `littraceqa.azure.extract_pdfs_from_box` — stage the PDFs referenced by a
  prediction file (or `--paper-id` list) from the Box archives into
  `artifacts/pdf_cache/`.
- `littraceqa.azure.figure_answer` — vision second pass for figure-primary
  questions: renders figure pages from cached PDFs and revises
  answers/evidence.
- `littraceqa.validate_submission` — gold-free lint of a prediction file; the
  mandatory final gate before any submission.
- `littraceqa.compare_runs` — per-question metric diff of two prediction files
  using the official `scripts/evaluate.py` logic.

Note: most modules accept the metadata path as `--metadata-file`, but
`process_document_intelligence`, `process_box_archives_document_intelligence`,
`reanalyze_papers`, and `extract_pdf_archives` call the same flag `--metadata`.
