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
`littraceqa.compare_runs`). The heavier pipelines have their own optional
extras. **Do not combine `di_pipeline` with either `azure` or
`pairwise_reader`**: its transitive pins conflict with the newer PDF/image
dependencies in those reading environments, and `uv` intentionally refuses
those combinations. `azure` and `pairwise_reader` may be combined:

```bash
uv sync --extra azure         # Azure RAG pipeline (baseline)
uv sync --extra di_pipeline   # DI-based hybrid retrieval pipeline
uv sync --extra pairwise_reader # MinerU + pairwise AOAI reader (current work)
```

## DI-based hybrid retrieval pipeline

This pipeline lives under `src/littraceqa/di_pipeline/` (preprocessors,
indexers, fusers/rerankers, and agents wired up via dependency injection —
see `CLAUDE.md` and `configs/README.md` for the full design and usage) and
is run via `scripts/run_search.py`, e.g.:

```bash
uv run python scripts/sync_official_release.py

uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/abstract_specter2_body_qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
  --output predictions.jsonl
```

`run_search.py` now strips validation-only fields and constructs every query
from the organizer's production input contract by default. The explicit
`--include-development-fields` escape hatch exists only to reproduce old
experiments; results from that mode are not comparable to the held-out test.
This legacy runner writes rich development traces, not a strict official
submission. Use the pairwise reader and its submission validator below for an
upload file.

## Pairwise AOAI reader (primary reading path)

This path reads a separate, fixed candidate-paper ranking and the student's
MinerU corpus. It does not run DI, retrieval, reranking, or re-search. Use the
small reading-only environment:

The design rationale, RAG reading abstraction, validation/test distribution
audit, reproducibility protocol, and paper-ready method draft are documented in
[`docs/littraceqa_rag_reader_method.md`](docs/littraceqa_rag_reader_method.md).

```bash
uv sync --extra pairwise_reader --group dev
```

### Pin the current organizer release

The production contract is pinned in
`configs/official_release_manifest.json`. Download the exact organizer inputs,
schemas, metadata, validator, and reference evaluator into the ignored
`artifacts/` tree; every file is SHA-256 checked:

```bash
uv run python scripts/sync_official_release.py
```

The pinned input has four common fields—`query_id`, `benchmark`, `question`,
and `answer_types`—plus `multiple_choice_options` for multiple-choice questions
or `table_schema` for table questions. It never contains `task_family` or
`primary_evidence_type`. Option counts are variable and may include label `E`.

The released splits at the pinned revision are:

- `validation_inputs.jsonl`: 55 questions;
- `test.jsonl`: 71 required leaderboard questions;
- `test_extra.jsonl`: 4,901 optional diagnostic questions.

The required production target is the 71-question `test.jsonl`, not all 5,027
rows obtained by adding validation, test, and test_extra. Do not concatenate the
three splits. Run the optional 4,901-question `test_extra` only when the team has
an explicit diagnostic objective and has approved its separate AOAI budget.

`test` scores papers, evidence, and answers. `test_extra` scores papers and
answers, so its uploaded evidence field is optional. The reader still requires
source-grounded papers/chunks internally for both splits; `--evidence-policy`
controls only whether those validated locators are serialized into the final
submission, never whether the model may answer without grounding.

### Candidate-sidecar prerequisite

PR #7 contains rankings for the 55 validation questions only. It does **not**
contain rankings for the 71 `test` or 4,901 `test_extra` questions. The reader
therefore cannot run a challenge split until the retrieval owner creates a
gold-free sidecar with exactly this shape:

```json
{"query_id":"ltqa_...","candidate_papers":[{"paper_id":"acl2025_00001","rank":1,"title":"...","venue":"ACL","year":2025}]}
```

The loader requires exact query coverage and rejects gold answers, evidence,
development hints, copied options, extra query IDs, duplicate papers, and
non-consecutive ranks. Never pass PR #7's combined development file directly to
the model. The checked-in validation sidecar was sanitized with:

```bash
git show a020604:data/validation_with_candidates.jsonl \
  | uv run python scripts/export_candidate_handoff.py \
      --input - \
      --output data/validation_candidates.jsonl
```

### Reading and validation flow

For every question, the reader:

1. resolves only a unique, literal canonical title that grammatically owns a
   named paper-local figure/table/equation/reference; mismatching candidates are
   checkpointed as deterministic zero-call distractors, while every other pair
   is hydrated from MinerU and receives one base AOAI three-field judgment call;
   fuzzy title matches and titles that merely co-occur in citations or comparisons
   never activate this destructive gate;
2. asks Stage 1 for two separate Boolean judgments—whether the paper belongs in
   the answer-relevant paper set and whether the supplied context contains usable
   answer evidence—plus the smallest exact chunk-ID set; Python validates the
   three-field response and rejects invented or cross-paper IDs;
3. builds the submitted relevant-paper set from the first judgment, but hands a
   paper to Stage 2 only when both judgments are true and at least one cited chunk
   is valid; visual handoff additionally requires a cited image that was actually
   attached; the Stage-2 context starts from those validated IDs, and any
   deterministic same-paper visual sibling or official-locator rescue chunk is
   logged separately as Python-supplied context;
4. binds every calculation input to named source facts, then deterministically
   recomputes arithmetic, rounded/exact division, counts, argmax/argmin, Yes/No
   comparisons, option label/text mapping, table columns/types, and evidence
   support; every operation is also bound to its final answer fragment, so one
   of several counts cannot silently disagree with the selected option;
5. keeps the broader Stage-1 relevant-paper set, the narrower Stage-2 handoff
   pool, and the minimal chunks directly supporting the selected answer separate.

Stage 1 never splits a paper into several AOAI requests. A long paper is
deterministically compacted into one bounded paper context, with at most 10
selected images, and every pair not eliminated by the narrow named-owner gate
receives one base judgment call.
The final rendered Stage-1 prompt normally contains three examples selected by
the rule-based four-way question type: one common negative plus one usable and
one not-usable example for `visual`, `citation`, `calculation`, or `other`.
Queries that require a literal term inside a primary/main Figure receive one
additional hard-negative example that separates Figure pixels/caption from a
paper title, surrounding prose, and merely related tree-search terminology.
Including those examples and query metadata, the prompt is guarded at 240,000
characters; the paper context is reduced again locally if necessary, without
another AOAI request.
Every completed paper judgment is checkpointed. JSON repair, an image-policy
text-only fallback, and provider retry can add requests only on failure; they
are not normal paper partitions. API errors,
invalid JSON, invented IDs, absent required images, inconsistent calculations,
and invalid official locators never become silent guesses.

Stage-1 checkpoints retain both the model's raw Boolean decisions (`model_*`)
and Python's effective routing decisions. Python forces a uniquely resolved
literal owner to A=true, suppresses B when effective A is false or no valid ID
remains, and exposes the resulting `send_to_answer_agent` flag. This makes model
errors distinguishable from deterministic ownership and safety corrections.

### Reading an externally selected paper set

When retrieval is owned by another component and its paper IDs are already the
final paper set, use `configs/agent_style/aoai_selected_paper_reader.yaml`.  In
this mode a sidecar row may contain canonical paper IDs only:

```json
{"query_id":"ltqa_...","candidate_papers":["acl2025_00001","neurips2025_00123"]}
```

`--paper-metadata` rehydrates title, venue, and year.  The loader still requires
exact query coverage and rejects answer, evidence, and development-only fields.
The reader passes the supplied paper IDs to `gold_papers` unchanged; neither
AOAI extraction nor the final answer call may add, remove, rank, or reject one.
For each selected paper, the first call extracts minimal source-linked facts and
chunk IDs.  The second call receives those facts plus deterministic same-paper
fallback context, constructs the answer, and chooses minimal submission
evidence.  Multi-paper open-ended tables must emit a grounded row for every
selected paper, while explicit row-inventory questions may report a truly
ungrounded named row in `completeness.missing` rather than inventing it.

This policy separates retrieval quality from reading quality: paper F1 is the
upstream system's result, while this reader is responsible for evidence
localization, table/figure coordinates, calculations, multiple-choice mapping,
and answer serialization.

Every provider adapter invocation also has a durable two-phase audit in
`<run-dir>/<query_id>/provider_attempts.jsonl`. The coordinator fsyncs a unique
`PREPARE` event before allowing the worker to enter the adapter, then fsyncs a
`FINALIZE` response or structured provider-error event before validation or a
whole-job retry continues. A response followed by a failed repair therefore
cannot disappear from billing totals. A process crash between the two events is
retained as an explicitly uncertain, potentially billable attempt instead of
being counted as zero. `provider_usage_summary.json` and each row of
`reading_traces.jsonl` materialize unique-attempt totals, request-ID history,
safe rate-limit metadata, and token usage without prompts, API keys, endpoints,
or exception bodies.

Configure Azure OpenAI in the repository-root `.env` (never commit real values):

```bash
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>
```

### Inspect the exact AOAI prompts without an API call

The pairwise prompts and their synthetic few-shot examples can be rendered for
review before running Azure OpenAI. With only an organizer input JSONL, the
command uses conspicuously synthetic paper/evidence placeholders:

```bash
uv run python scripts/render_aoai_prompts.py \
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/test.jsonl \
  --query-id ltqa_03af2c583a696a04 \
  --stage all \
  --format markdown \
  --output runs/prompt_previews/test_001.md
```

Add a sanitized candidate sidecar and real, already-formatted context to inspect
the corresponding live-task prompt body. This command still makes no API call:

```bash
uv run python scripts/render_aoai_prompts.py \
  --queries /path/to/test.jsonl \
  --query-id ltqa_03af2c583a696a04 \
  --candidates /path/to/test_candidates.jsonl \
  --paper-id acl2025_00001 \
  --paper-text-file /path/to/formatted_single_paper_context.txt \
  --accepted-summary-file /path/to/accepted_summary.json \
  --evidence-file /path/to/formatted_answer_evidence.txt \
  --stage all \
  --format json \
  --output runs/prompt_previews/test_001.json
```

The preview records the system and user messages, prompt version, SHA-256,
selected few-shot IDs, official query projection, and whether synthetic context
was used. Do not treat a synthetic preview as an executable scientific answer.

First run the gold-free corpus check. `--image-root` is the explicit trust
boundary for visual input: corpus-supplied paths are never opened directly;
only a validated `paper_id/auto/images/filename` suffix is rebased below this
root. Omitting the root disables every declared image and makes preflight fail
until the path is configured. Traversal and root-external symlinks are fatal.
Corrupt, oversized, or undecodable files are never sent to AOAI; preflight is
fatal when that leaves the corpus globally unavailable or an explicitly visual
query without a candidate figure, and otherwise reports the isolated file as a
warning. AOAI calls are capped at 10 validated JPEG/PNG/GIF/WebP images and
20 MiB per image. `answer_types=["table"]` is an output shape and does not by
itself force table evidence:

```bash
uv run python scripts/preflight_candidate_corpus.py \
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
  --candidates data/validation_candidates.jsonl \
  --paper-metadata artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/paper_metadata.jsonl \
  --chunks /path/to/mineru_chunks.jsonl \
  --chunk-index runs/mineru_chunks.offsets.json \
  --image-root artifacts/student_corpus/mineru_images_candidates
```

Run one complete validation question first. The runner judges every candidate
in a sidecar row by default. Do not set `--max-candidates` for a scored run:
the validation sidecar has up to 50 papers per query, q_022 needs rank 44, and
q_045 needs rank 37; a future challenge sidecar may contain a different count.

```bash
uv run python scripts/run_aoai_pairwise_reader.py \
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
  --candidates data/validation_candidates.jsonl \
  --paper-metadata artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/paper_metadata.jsonl \
  --chunks /path/to/mineru_chunks.jsonl \
  --chunk-index runs/mineru_chunks.offsets.json \
  --reader configs/agent_style/aoai_pairwise_reader.yaml \
  --run-dir runs/aoai_validation \
  --query-id q_001
```

`configs/agent_style/aoai_pairwise_reader_hybrid.yaml` remains only as a
configuration-name compatibility alias. It now uses the same one-call
text-plus-selected-images judgment as the primary config; it no longer performs
a text call followed by a second visual-refinement call.

`max_answer_papers` limits how many validated Stage-1 handoff papers Stage 2 may
review; these are papers for which both Boolean judgments are true and usable
chunk IDs survived validation. `max_evidence` separately limits the distinct
evidence chunks in the final submission. The broader Stage-1 relevant-paper set
is not truncated by this Stage-2 review limit.

The production reader also separates output-token reservations by semantic
stage. Candidate judgments return three-field JSON and reserve
`judgment_max_completion_tokens: 1024`; the final answer explicitly keeps
12,000 tokens for structured-table repairs. The
Azure rate limiter includes the requested output ceiling when it estimates TPM,
even when the actual Stage-1 JSON is much shorter, so one shared 12,000-token
ceiling creates avoidable throttling. The client timeout is 60 seconds: normal
validation calls completed within 22 seconds in the measured run, while a dead
connection otherwise occupied a worker for the former 180-second timeout.

Resume the same question after interruption by adding `--resume`. To inspect or
re-run one pair without touching other checkpoints:

```bash
uv run python scripts/run_aoai_pairwise_reader.py <same arguments> \
  --query-id q_001 --paper-id acl2025_00005 --stage judge --resume --force
```

After q_001 is satisfactory, omit `--query-id`, keep the same run directory,
and add `--resume` to reuse its 50 pair judgments while completing all 55
questions. An unfiltered run prints the query/paper/minimum-call scope and
refuses to start unless `--confirm-full-run` is present:

```bash
uv run python scripts/run_aoai_pairwise_reader.py <same arguments> \
  --workers 100 --resume --confirm-full-run
```

### Table-only regeneration and source adjudication

Once a complete run has trustworthy multiple-choice answers, papers, and
evidence, isolate later answer experiments to released table queries with
`--answer-type table`. Create the first candidate in a fresh run directory with
`--stage all`; under the fixed-selected config this processes only released
table queries and emits a valid 21-row table-only candidate:

```bash
uv run python scripts/run_aoai_pairwise_reader.py <same arguments> \
  --reader configs/agent_style/aoai_selected_paper_reader.yaml \
  --run-dir runs/table_sample_01 \
  --answer-type table --stage all
```

To reuse those current-code Stage-1 judgments, copy the entire completed
directory to each new sample directory, including its manifest and per-query
checkpoints, then rerun only Stage 2 there with
`--stage answer --resume --force`. Never make two samples write to the same run
directory. Generate independent candidates in distinct directories, then
compare full 71-row submissions (or table-only 21-row JSONL files) without
permitting a candidate to change papers, evidence, multiple-choice answers,
freeform answers, non-table records, or official query order:

```bash
uv run python scripts/adjudicate_table_answers.py review \
  --inputs artifacts/official_release/<revision>/data/test.jsonl \
  --base runs/frozen_base/submission.jsonl \
  --candidate sample_01=runs/table_sample_01/submission.jsonl \
  --candidate sample_02=runs/table_sample_02/submission.jsonl \
  --output-dir runs/table_review
```

Inspect every candidate table against the cited PDF or extracted source. Copy
`table_decisions.template.json` to a decision file and, for all 21 table
queries, record the selected candidate, `source_checked=true`, a non-empty
reason, and at least one official-shaped locator already present in the frozen
evidence set. Promotion is fail-closed and replaces only `answer.table`:

```bash
uv run python scripts/adjudicate_table_answers.py compose \
  --inputs artifacts/official_release/<revision>/data/test.jsonl \
  --base runs/frozen_base/submission.jsonl \
  --candidate sample_01=runs/table_sample_01/submission.jsonl \
  --candidate sample_02=runs/table_sample_02/submission.jsonl \
  --decisions runs/table_review/table_decisions.json \
  --output runs/table_review/submission.jsonl
```

The review seals the inputs, base, candidates, query order, and table schemas
with SHA-256. Composition rejects changed artifacts and writes an audit manifest
proving that all papers/evidence and all non-table answer components remained
unchanged. This workflow records human source adjudication; it does not claim
that setting `source_checked=true` automatically verifies scientific content.

`--workers 100` is the initial concurrency ceiling, not 100 independent input
files. The runner keeps the original JSONL intact, round-robins every
query-paper pair through one global pool, and checkpoints each success before
submitting more work. All workers share one token-aware launch pacer. The
production configs set `target_tpm: 2400000`, leaving 131,000 TPM below the
2,531,000 limit observed in the validation response headers. Before each launch
the client estimates the system-plus-user text at 3.2 characters/token, adds
the effective per-call `max_completion_tokens`, and adds 512 tokens per
high-detail image. These values are calibrated to the complete v18 call sample,
not presented as a worst-case tokenizer guarantee. The interval is
`max(0.075, estimated_tokens * 60 / target_tpm)` seconds, so the 0.075-second
value remains a microburst floor rather than the primary rate control.
Reservations are ordered across every thread on the shared client and paid
before launch; therefore a long prompt cannot immediately follow a short
prompt's smaller slot. Provider calls themselves remain concurrent. Omitting
`target_tpm` preserves the original fixed-interval behavior; rare estimation
outliers are handled by the deployment margin plus the outer 429/AIMD scheduler.

If Azure returns HTTP 429, only the rate-limited jobs are requeued; the pool
drains, cools down, and reduces its effective concurrency until it reaches the
deployment's real RPM/TPM capacity. A clean success window then raises the cap
additively toward 100 again, and the learned cap is shared with Stage 2 instead
of being reset. Provider `retry-after-ms` / `Retry-After` advice takes
precedence over the 60-second fallback; successful calls retain only a safe
allowlist of RPM/TPM limit headers for later throughput analysis. The Azure SDK
retry count is zero in this configuration so a hidden per-thread Retry-After
cannot fight the global scheduler.

Local startup work is parallel too: unique MinerU images are validated and
hashed with up to 64 workers, and uncached papers are not hashed once on the
coordinator and again in their AOAI worker. A Stage-1 response that remains
structurally invalid after its repair is logged with both raw calls and isolated
to that pair. Every other pair finishes; rerunning with `--resume` retries only
the missing checkpoint. Authentication, configuration, corpus, and unexpected
provider failures remain fatal.

For a cache-empty run, the exact minimum is the candidate-pair count minus
deterministic named-owner rejections, plus one Stage-2 call per question. With
71 questions and 50 candidates each, `3,621` is the no-rejection upper baseline;
the CLI prints both the actual zero-call rejection count and the reduced minimum
before any JSON repair, image-policy fallback, or provider retry. The optional
4,901-question test_extra would be vastly
larger and is not part of the normal leaderboard run. The CLI prints the actual
pair count from the supplied sidecar before it asks for confirmation.
Any run selecting more than 71 questions is rejected unless the separate
`--confirm-optional-test-extra` cost gate is also supplied.

Use `--evidence-policy optional` only for an explicitly selected
`test_extra.jsonl` run; filenames never weaken the safe `required` default.
The missing-image override is diagnostic only and is accepted solely with
`--stage judge`, never while constructing a submission answer.

The run directory contains:

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
  --inputs artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
  --predictions runs/aoai_validation/submission.jsonl \
  --paper-metadata artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/paper_metadata.jsonl \
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

This separates candidate misses, Stage-1 A relevance errors, Stage-1 B handoff
errors, evidence localization, table/figure/citation/equation reading,
multi-paper integration, answer reasoning, serialization, and dataset
inconsistencies. It also evaluates
papers that own gold evidence separately from official `gold_papers`, which may
contain multiple-choice distractor papers.

For multiple choice, the reader now solves the semantic answer first, returns
both the released label and exact option text, validates the pair, and writes
that real label to the submission. There is no production placeholder.

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
  --input artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
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
