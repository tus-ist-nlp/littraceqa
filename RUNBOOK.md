# LitTraceQA Operational Runbook (post-Document-Intelligence)

Sequence to run once the currently active Document Intelligence (DI) job
finishes. All commands run from the repo root on the pipeline machine with
`uv run python ...` (PowerShell examples below). Do not run any step while a
previous step's writer is still active.

Key dates: test set release **8/3**, submission deadline **8/19** (EMNLP 2026).

Baseline to beat (`runs/validation_submission_metadata_plus_partial_pdf.jsonl`,
official `scripts/evaluate.py`): paper_f1 **0.338**, evidence_f1 **0.029**,
multiple_choice **0.0**, freeform **0.0**, table metrics **0.0**.

---

## 1. Confirm the DI job finished

The Box-archive DI pipeline (`littraceqa.azure.process_box_archives_document_intelligence`)
must be fully stopped before anything below touches `artifacts\docint\`.

```powershell
# Manifest tail: the last lines show the final archive/paper statuses.
Get-Content artifacts\docint\_box_archive_manifest.jsonl -Tail 5

# No python process of this repo may still be running.
Get-Process python* | Format-Table Id, ProcessName, StartTime
```

Do not proceed while the process is alive — steps 2 and 4 write the same
per-paper files and the consolidated `chunks.jsonl` (the merge is not
concurrency-safe).

## 2. Retry failed papers, with raw DI output enabled

The manifest shows roughly **721 failed** entries, of which about **109 papers
never succeeded**. The module is resumable: rerunning it skips papers that
already have chunks and retries the failures. Enable `--save-raw` this time —
raw DI JSON (captions, bounding regions, cell indices) is what every future
chunking fix needs, and it was discarded on the first pass.

```powershell
uv run --extra azure python -m littraceqa.azure.process_box_archives_document_intelligence --save-raw
```

Note on the 3 oversized PDFs: they exceed `--max-pdf-bytes` (default 300 MB).
Raising the cap (e.g. `--max-pdf-bytes 500000000`) makes them processable but
costs extra DI money and time for what are usually media-heavy appendix
monsters. Decide by checking whether any of them is a gold/validation paper;
otherwise skipping them is fine.

Gate for this step: coverage of the 70 gold-referenced validation papers
(5 had no chunk file at review time: neurips2025_03461, 03561, 04875, 04876,
05262).

## 3. Fix table/figure caption numbers in the chunk locators

The old chunker numbered `table_id`/`figure_id` by DI detection order, not by
the papers' printed caption numbers; the evaluator matches on the caption
number, so shifted ids score zero. First measure, then apply:

```powershell
# Dry-run report + before/after gold-evidence match counts (read-only).
uv run python -m littraceqa.fix_chunk_locators --check-gold data\validation.jsonl

# Apply (atomic per-file rewrite). Only run after step 1 confirmed no writer is active.
uv run python -m littraceqa.fix_chunk_locators --in-place
```

Reference numbers on the 2026-07 corpus (measured, not projected): gold table
evidence coarse-key matches **40/64 -> 42/64**, figures **10/18 -> 11/18**.
Most remaining misses are page mismatches between the corpus PDFs and the
organizers' annotation (unfixable by renumbering); papers reported as
"conflicts" keep their original ids on the unalignable pages. Add
`--details <path.csv>` to `--check-gold` to get one row per gold table/figure
evidence item (query_id, paper_id, key, matched before/after) and spot
per-item regressions hidden inside the aggregate.

These gains are small by construction — post-hoc caption recovery from text
chunks is limited. The bigger evidence-recall lever is step 2's targeted DI
re-analysis with `--save-raw` of the retrieved/gold papers plus the new
caption-based chunker, which reads `table_id`/`figure_id` straight from the
DI captions. Chunks produced that way carry a `caption` entry in
`locator_json` and are skipped by this fixer.

## 4. Re-consolidate chunks.jsonl (single writer only)

The consolidated file only refreshes at archive boundaries and was ~1,300
papers stale during the run. Rebuild it once, with nothing else writing:

```powershell
uv run --extra azure python -m littraceqa.azure.process_document_intelligence --merge-only
```

Sanity check: distinct `paper_id` count in `artifacts\docint\chunks.jsonl`
should match the per-paper file count in `artifacts\docint\chunks\`.

## 5. Rebuild the Azure AI Search index (the expensive step)

Schema changes require `--recreate` (Azure rejects in-place field changes).
The new schema adds filterable `table_id`/`figure_id` string fields (parsed
from `locator_json`, empty when absent) and the semantic configuration
`littraceqa-semantic` (title_field=title, content_fields=[content],
keywords_fields=[section]).

```powershell
uv run --extra azure python -m littraceqa.azure.build_azure_search_index --recreate `
    --embedding-batch-size 256 --upload-batch-size 256
```

Use the streaming input and checkpoint/resume flags added alongside the new
schema (see `uv run --extra azure python -m littraceqa.azure.build_azure_search_index --help` for
the exact names) — never load the ~6 GB `chunks.jsonl` into memory, and make
sure a mid-run failure can resume instead of re-embedding from chunk 0.

Failure handling during the run:

- **Poison chunks are skipped, not fatal.** An embedding batch rejected with
  a permanent 4xx (token-dense tables, OCR mojibake) is retried once with
  halved content, then skipped so one bad chunk never aborts the multi-hour
  run. Embed-skips do not block the paper's checkpoint (`--resume` will not
  re-hit them); persistent upload failures do (`--resume` retries those
  papers).
- **Failed keys are recorded.** All skipped-embed and upload-failed document
  keys land in `<checkpoint stem>_failed_keys.txt` next to `--checkpoint`
  (default: `artifacts\docint\index_upload_checkpoint_failed_keys.txt`).
  Review it after the run and re-upload if any key matters.

Note on the OLD live index: appending new-schema documents into the still
deployed pre-`table_id`/`figure_id` index (e.g. with `--skip-index-create`)
does not fail — the uploader fetches the live index definition and
automatically omits `table_id`/`figure_id` from the uploaded documents, so
those chunks simply stay unfilterable by object id. That mode is for
emergency top-ups only; the full `--recreate` rebuild is the intended path.

Budget and duration for ~1.94M chunks (~650-700M tokens):

- Wall clock: roughly **1-2 h** at embedding batch 256 with 4 concurrent
  workers (versus tens of hours at the old sequential batch-16 default).
- Embedding cost: about **$14** on text-embedding-3-small ($0.02/1M tokens),
  or **$85-90** on -large. A failed run without checkpointing repays this.

## 6. Full validation run + official scoring + per-question diff

Run RAG over the validation inputs with the answer-side fixes enabled
(multiple-choice options file, query decomposition for multi_paper questions,
run-metadata sidecar):

```powershell
uv run --extra azure python -m littraceqa.azure.run_rag `
    --input data\validation_inputs.jsonl `
    --output runs\validation_full_index.jsonl `
    --options-file data\validation.jsonl
```

(Decomposition/sidecar flags: see `uv run --extra azure python -m littraceqa.azure.run_rag --help`;
the sidecar `.meta.json` records args, deployments, and index state for the
mandatory code submission.)

If a run is interrupted, re-run the same command with `--resume`: completed
rows are kept, placeholder rows from per-query failures (empty answer and no
papers) are retried, and the output file is rewritten atomically before
appending. LLM responses that fail JSON parsing even after one retry are
captured in `<output>_raw_failures.jsonl` next to the output — check it when
`failed=` in the summary line is non-zero.

`--options-file` behavior in `run_rag`: the flag defaults to none. When it is
omitted, the validation options `data\validation.jsonl` are auto-joined ONLY
if `--input` resolves to the default `data\validation_inputs.jsonl`; any
other input — in particular the hidden test inputs — runs **without** options
unless `--options-file` is passed explicitly. A WARN is printed when the
options file shares no query_id with the input, and per-question options are
dropped (with a WARN) when the options file's question text does not match
the input's. See the "Test-day options" policy below.

Score officially, then diff against the baseline per question:

```powershell
uv run python scripts\evaluate.py --gold data\validation.jsonl --pred runs\validation_full_index.jsonl

uv run python -m littraceqa.compare_runs `
    --gold data\validation.jsonl `
    --pred-a runs\validation_submission_metadata_plus_partial_pdf.jsonl `
    --pred-b runs\validation_full_index.jsonl
```

## 6b. Test-day PDF staging + figure vision pass

After the first RAG pass, stage the PDFs of every predicted paper so the
vision pass (`littraceqa.azure.figure_answer`) can render figure pages.
Do NOT rely on direct downloads: OpenReview (and several publisher sites)
block scripted PDF fetches — the Box shared folder is the reliable source.

```powershell
# Collect all paper_ids referenced by the predictions (gold_papers + evidence),
# locate each paper's zip via artifacts\docint\_box_archive_manifest.jsonl,
# download each needed archive once into artifacts\box_tmp, extract only the
# matching PDFs into artifacts\pdf_cache, and delete the zip. Cached PDFs are
# skipped, so re-runs are cheap.
uv run --extra azure python -m littraceqa.azure.extract_pdfs_from_box `
    --from-predictions runs\test_full_index.jsonl

# Preview first (no downloads, no writes): reports cache state and which
# archives would be fetched. Exit 1 means some paper is not in the manifest.
uv run --extra azure python -m littraceqa.azure.extract_pdfs_from_box `
    --from-predictions runs\test_full_index.jsonl --dry-run
```

A full archive is ~5-15 GiB, so budget disk (`--reserve-gb`, default 20) and
time; the download resumes a partial `.part` file across retries. Individual
paper_ids can also be staged with `--paper-id` / `--paper-id-file`.

Then run the vision second pass over the figure-primary questions (it reads
PDFs from `artifacts\pdf_cache` first and only falls back to `pdf_url`
downloads, which is why the Box staging above comes first):

```powershell
uv run --extra azure python -m littraceqa.azure.figure_answer `
    --predictions runs\test_full_index.jsonl `
    --inputs data\test_inputs.jsonl `
    --output runs\test_full_index_vision.jsonl
```

On test day pass the organizers' options file explicitly (see "Test-day
options"); the auto-join only fires for the default validation inputs.

The vision pass also tops up short `multi_paper` figure-question paper lists
(`--compare-expand`, on by default): the vision model reports the comparison
methods it sees on the rendered pages, those names plus citation-dense
baseline excerpts from the submitted papers' local chunk files are fed through
`run_rag`'s P3 resolution ladder (bibliography-title matching, search-score
floors), and any corpus-resolved comparison-group papers are appended to
`gold_papers` (never removed, capped at 4, enumeration questions skipped).
This recovers comparison-group papers whose names never appeared in the
retrieved text context that run_rag's own compare-expand pass saw (e.g. the
TCM/ECM cluster on q_031). It needs the `AZURE_SEARCH_*` env vars; if the
search client cannot be built the pass logs a warning and skips expansion.
Disable with `--no-compare-expand`.

## 7. Mandatory final gate: gold-free submission lint

Before every submission (dev or hidden test), lint the prediction file. It
needs only the released inputs file, so it works on the test set too:

```powershell
uv run python -m littraceqa.validate_submission `
    --inputs data\validation_inputs.jsonl `
    --predictions runs\validation_full_index.jsonl `
    --options-file data\validation.jsonl
```

Exit code 0 is required. It catches the silent zero-score failure modes:
missing/duplicate query_ids, answer objects not matching `answer_types`,
letters outside the option keys (without known options any single A-Z letter
passes, but empty/multi-character answers fail), empty freeform, empty table
rows on table questions, and evidence items missing
`paper_id`/`source_type`/`locator.page` (the evaluator drops those silently).
For the test set, pass the test inputs file and the test options file (if the
organizers release options separately) — see "Test-day options" below.

---

## Policies

### Test-day options

On 8/3, multiple-choice options must be provided **explicitly**: pass the
organizers' test options file via `--options-file` to both `run_rag` and
`validate_submission`. Never reuse the validation options
(`data\validation.jsonl`) for the test inputs — a query_id collision would
silently attach the wrong options to the wrong questions. `run_rag`
deliberately auto-joins the validation options only for the default
validation input and never for any other `--input`. If the organizers release
no options file, the prompts run without options (pass-through: the model's
single-letter answer is submitted as-is) and `validate_submission` accepts
any single A-Z letter for those questions.

### Index freeze

Freeze the index about **one week before the 8/3 test release (i.e. by ~7/27)**:
steps 2-5 (chunk fixes, re-merge, re-embed, re-index) must all be done by
then, because each chunk-schema change forces the expensive step 5 again.
After the freeze, only prompt/answer-side changes are allowed (options
injection, extractive freeform prompting, table_schema injection, evidence
selection) — none of these require reindexing, so test day on 8/3 is a
parameter change, not a first attempt.

### Overfitting guard

All tuning targets the same 55 dev questions (41 MC, 26 freeform, 11 table).
One MC question is worth ~2.4 points of accuracy; one gold-paper swing moves
paper_f1 by ~1.8 points. Treat any aggregate move smaller than **2 flipped
questions** as noise, and always review `compare_runs` per-query flips instead
of aggregates. Prefer structurally justified changes (options in prompt,
schema injection, K by task_family) over prompt wording tuned until the
55-question score moves — the latter is dev-set overfitting that will not
transfer to the hidden test set.

### Cost

The Azure AI Search service bills continuously while it exists (per partition,
including ~12 GB of vector storage), independent of query volume. After the
8/19 submission, **stop and delete the Search service** (export the index
definition first if reproducibility requires it). Also decommission the DI
resource; the AOAI deployments are pay-per-token and can stay.
