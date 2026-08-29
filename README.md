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
`littraceqa.compare_runs`). The two RAG pipelines each need their own
optional extra, and **the two are mutually exclusive in one environment**
(`di_pipeline` pins `pypdfium2==4.30.0` transitively via `marker-pdf`, which
conflicts with `azure`'s `pypdfium2>=5.11.0`; `uv` will refuse to resolve
both extras together):

```bash
uv sync --extra azure         # Azure RAG pipeline (baseline)
uv sync --extra di_pipeline   # DI-based hybrid retrieval pipeline
```

## Hybrid retrieval pipeline

The whole system is wired up in `src/littraceqa/di_pipeline/pipeline.py` — read that
file to see every stage and every tuned value in one place (`CLAUDE.md` explains why
each value was chosen). Only machine-dependent paths live in `configs/paths/*.yaml`.
It is run via `scripts/run_search.py`, e.g.:

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl
```

### Handing candidates off to a separate reading agent

Retrieval and reading can be developed independently: retrieval decides *which
papers*, reading decides *which passages answer the question*. Two helpers exist
so a reading agent can be built against a frozen set of retrieval candidates,
without re-running search (which costs hours of GPU time per configuration).

**1. Build the handoff file.** `scripts/build_candidate_handoff.py` joins the
questions with the `candidate_papers` of an existing prediction file:

```bash
uv run python scripts/build_candidate_handoff.py \
  --predictions predictions_{run_id}.jsonl \
  --output data/validation_with_candidates.jsonl
```

Each line carries the four production input fields, the ranked candidates, run
provenance, and the gold — the last one quarantined under `_gold`:

```jsonc
{
  "query_id": "q_001",
  "question": "Among the two prompt compression methods, ...",
  "answer_types": ["freeform", "multiple_choice"],
  "table_schema": null,
  "candidate_papers": [
    {"rank": 1, "paper_id": "acl2025_00005", "title": "500x Compressor: ...",
     "venue": "ACL", "year": 2025}
  ],
  "_meta": {"source_predictions": "...",
            "agent": "...", "run_timestamp": "...", "n_candidates": 50},
  "_gold": {"task_family": "...", "primary_evidence_type": "...",
            "gold_papers": [...], "evidence": [...], "answer": {...}}
}
```

**Only the four top-level input fields may be read at inference time.** `_gold`
is nested precisely so that leakage has to be deliberate: `gold_papers[].title`
*is* the answer to "which paper", `answer.multiple_choice.options` is oracle-only
(see commit `f53e1da`), and `task_family` / `primary_evidence_type` do not exist
in the competition's real input. Pass `--no-gold` to emit a blind copy for
distribution. Point `--predictions` at a different run to regenerate the file
against another search configuration.

**2. Read the papers.** `littraceqa.chunk_store.ChunkStore` resolves a
`paper_id` to its full MinerU chunks. The corpus stores each paper's lines
contiguously, so a `paper_id -> (offset, length)` index is enough to seek
straight to it:

```python
from littraceqa.chunk_store import ChunkStore

store = ChunkStore("/data2/iseakira/pdfs/chunks/mineru_chunks.jsonl")
chunks = store.load_paper("acl2025_00005")   # every chunk, in body order
text = store.paper_text("acl2025_00005")     # the same, joined into one string
figures = store.figures("acl2025_00005")     # table/figure chunks whose image exists
```

The 1.0 MB index is built on first use (~25 s over the 3.8 GB corpus) and cached
next to it as `mineru_chunks.jsonl.offsets.json`; it is rebuilt automatically if
the corpus size or mtime changes. Subsequent startups take 0.03 s and a single
paper loads in 0.7 ms. An unknown `paper_id` yields an empty list rather than
raising, since IDs arrive from retrieval output.

Chunk `metadata` carries everything an evidence locator needs (`page`,
`section`, `table_id`, `figure_id`, `equation_id`) plus `image_path` for tables
and figures. Those image paths are absolute, so if the corpus is copied to
another machine pass `ChunkStore(..., image_root="/new/path/to/mineru")` to
rebase them.

Budget note: a paper averages ~24k tokens, so loading all 50 candidates of one
query is ~1.1M tokens. Feeding whole papers requires either filtering to a
handful of papers first, or one model call per paper.

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
- `littraceqa.chunk_store` — `paper_id` to full MinerU chunks (text plus figure
  image paths) via a cached byte-offset index; see the handoff section above.
- `littraceqa.validate_submission` — gold-free lint of a prediction file; the
  mandatory final gate before any submission.
- `littraceqa.compare_runs` — per-question metric diff of two prediction files
  using the official `scripts/evaluate.py` logic.

Note: most modules accept the metadata path as `--metadata-file`, but
`process_document_intelligence`, `process_box_archives_document_intelligence`,
`reanalyze_papers`, and `extract_pdf_archives` call the same flag `--metadata`.
