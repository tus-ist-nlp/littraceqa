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
`littraceqa.compare_runs`). The heavier legacy pipelines have their own
optional extras, and **`di_pipeline` and `azure` are mutually exclusive**
(`di_pipeline` pins `pypdfium2==4.30.0` transitively via `marker-pdf`, which
conflicts with `azure`'s `pypdfium2>=5.11.0`; `uv` will refuse to resolve
both extras together):

```bash
uv sync --extra azure         # Azure RAG pipeline (baseline)
uv sync --extra di_pipeline   # DI-based hybrid retrieval pipeline
uv sync --extra corpus_qa     # MinerU + pairwise AOAI reader (current work)
```

## DI-based hybrid retrieval pipeline

This pipeline lives under `src/littraceqa/di_pipeline/` (preprocessors,
indexers, fusers/rerankers, and agents wired up via dependency injection —
see `CLAUDE.md` and `configs/README.md` for the full design and usage) and
is run via `scripts/run_search.py`, e.g.:

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/abstract_specter2_body_qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl
```

## Pairwise AOAI reader (primary reading path)

PR #7 contains a query-specific ranked list of papers, not chunk text. Each
candidate has only `rank`, `paper_id`, `title`, `venue`, and `year`. Retrieval
and reranking are therefore already finished. The primary reader does not run
DI, search, reranking, or re-search:

The sanitized validation sidecar contains every candidate supplied by PR #7:
3--50 papers per query (2,227 total; 30 of 55 queries have all 50). The reader
judges every supplied candidate rather than assuming that every query has 50.

1. hydrate one candidate paper from the student-built MinerU corpus;
2. send one `query x candidate paper` pair to Azure OpenAI and require exact
   evidence chunk IDs;
3. repeat for every ranked candidate (up to all 50);
4. send only the accepted original evidence chunks to Azure OpenAI again to
   construct the answer.

Long papers are deterministically split inside step 2. API errors, invalid JSON,
and invented IDs stop the run and are never silently treated as irrelevant.
Every paper judgment is checkpointed separately.

Use the small reading-only environment for this path (the generic `openai`
client supplies Azure OpenAI support; Azure Search/DI SDKs are not installed):

```bash
uv sync --extra corpus_qa --group dev
```

PR #7 originally colocates `_gold` with the candidate ranking. Never pass that
file directly to an agent. Create a physical gold-free sidecar first (the
checked-in `data/validation_candidates.jsonl` was generated this way):

```bash
git show a020604:data/validation_with_candidates.jsonl \
  | uv run python scripts/export_candidate_handoff.py \
      --input - \
      --output data/validation_candidates.jsonl
```

Configure Azure OpenAI in the repository-root `.env` (never commit real values):

```bash
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>
```

The candidate sidecar contains no question, gold, answer, evidence, task family,
or primary evidence type. The reader projects the query onto exactly the four
organizer-confirmed fields: `query_id`, `question`, `answer_types`, and
`table_schema`. Never use PR #7's combined file directly as inference input.

First run the gold-free corpus check. `--image-root` rebases the absolute image
paths embedded by MinerU after copying the corpus to another machine:

```bash
uv run python scripts/preflight_corpus_qa.py \
  --queries data/validation_inputs.jsonl \
  --candidates data/validation_candidates.jsonl \
  --paper-metadata data/paper_metadata.jsonl \
  --chunks /path/to/mineru_chunks.jsonl \
  --chunk-index runs/mineru_chunks.offsets.json \
  --image-root /path/to/mineru/output
```

Run one complete validation question first. Do not reduce `--max-candidates`
for a scored run: q_022 needs rank 44 and q_045 needs rank 37.

```bash
uv run python scripts/run_aoai_pairwise_reader.py \
  --queries data/validation_inputs.jsonl \
  --candidates data/validation_candidates.jsonl \
  --paper-metadata data/paper_metadata.jsonl \
  --chunks /path/to/mineru_chunks.jsonl \
  --chunk-index runs/mineru_chunks.offsets.json \
  --image-root /path/to/mineru/output \
  --reader configs/agent_style/aoai_pairwise_reader.yaml \
  --run-dir runs/aoai_validation \
  --query-id q_001
```

Resume the same question after interruption by adding `--resume`. To inspect or
re-run one pair without touching other checkpoints:

```bash
uv run python scripts/run_aoai_pairwise_reader.py <same arguments> \
  --query-id q_001 --paper-id acl2025_00005 --stage judge --resume --force
```

After q_001 is satisfactory, omit `--query-id`, keep the same run directory,
and add `--resume` to reuse its 50 pair judgments while completing all 55
questions. The run directory contains:

```text
manifest.json
preflight.json
q_001/candidate_judgments.jsonl
q_001/answer.json
q_001/submission.json
reading_traces.jsonl
submission.jsonl
```

Run the gold-free submission gate before upload:

```bash
uv run python -m littraceqa.validate_submission \
  --inputs data/validation_inputs.jsonl \
  --predictions runs/aoai_validation/submission.jsonl \
  --paper-metadata data/paper_metadata.jsonl \
  --strict-official-shape
```

Only after inference is finished, load validation gold and generate one error
report per question:

```bash
uv run python scripts/analyze_aoai_reading.py \
  --gold data/validation.jsonl \
  --candidates data/validation_candidates.jsonl \
  --traces runs/aoai_validation/reading_traces.jsonl \
  --output-dir runs/aoai_validation/error_analysis
```

This separates candidate misses, relevance false negatives/positives, evidence
localization, table/figure/citation/equation reading, multi-paper integration,
answer reasoning, serialization, and dataset inconsistencies. It also evaluates
papers that own gold evidence separately from official `gold_papers`, which may
contain multiple-choice distractor papers.

The four-field contract omits multiple-choice option text. The reader extracts
a meaning-level answer into the trace, but no system can derive an A-D mapping
that was never supplied. The submission letter is therefore only a deterministic
structural placeholder; the post-hoc validation report marks this separately as
`multiple_choice_protocol_blocker`.

The current
[public Hugging Face format page](https://huggingface.co/datasets/LitTraceQA/LitTraceQA/blob/main/docs/format.md)
still describes `benchmark` and `multiple_choice_options`. That page conflicts
with the organizer's later direct clarification that held-out input contains
only `query_id`, `question`, `answer_types`, and `table_schema`. This production
path deliberately follows the direct clarification and physically discards
every other input field.

## Azure RAG pipeline (legacy baseline)

The Azure-based baseline lives under `src/littraceqa/azure/` and is invoked
with `uv run --extra azure python -m littraceqa.azure.<module>`. (It is a
subpackage rather than a top-level `src/azure/` because a local package named
`azure` would shadow the Azure SDK's `azure.*` namespace packages.)

This baseline is retained only for historical reproducibility. It is not the
submission path: parts of it depend on validation-only metadata that the
organizer has confirmed will not be present in held-out test inputs.

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
- `littraceqa.validate_submission` — gold-free lint of a prediction file; the
  mandatory final gate before any submission.
- `littraceqa.compare_runs` — per-question metric diff of two prediction files
  using the official `scripts/evaluate.py` logic.

Note: most modules accept the metadata path as `--metadata-file`, but
`process_document_intelligence`, `process_box_archives_document_intelligence`,
`reanalyze_papers`, and `extract_pdf_archives` call the same flag `--metadata`.
