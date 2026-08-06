# LitTraceQA Operational Runbook (post-Document-Intelligence)

Sequence to run once the currently active Document Intelligence (DI) job
finishes. All commands run from the repo root on the pipeline machine with
`uv run python ...` (PowerShell examples below). Do not run any step while a
previous step's writer is still active.

Key dates: test set release **8/3**, submission deadline **8/19** (EMNLP 2026).

## Required challenge scope and AOAI budget

Treat the organizer splits as separate jobs:

- `validation_inputs.jsonl`: 55 development questions;
- `test.jsonl`: 71 required leaderboard questions and the default production job;
- `test_extra.jsonl`: 4,901 optional diagnostic questions.

Do not concatenate them into a 5,027-question AOAI run. `test_extra` is not
required for the main challenge submission and should be run only after a
separate diagnostic goal and budget are approved. The pairwise reader uses one
Stage-1 base AOAI call per query-paper pair, without splitting long papers into
additional calls, followed by one Stage-2 base answer call per question. Thus a
cache-empty 71-question run with 50 candidates per question has an exact minimum
of 3,621 calls; JSON repairs, image-policy text fallbacks, and provider retries
can increase that number. Always
review the CLI's actual query, candidate-pair, and minimum-call counts before
passing `--confirm-full-run`. A selection larger than 71 questions additionally
requires `--confirm-optional-test-extra`; this prevents an ordinary full-run
confirmation from accidentally authorizing the 4,901 optional questions.

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

Run RAG over the pinned current validation inputs. Multiple-choice options are
already present in each applicable input row; do not join validation gold:

```powershell
uv run --extra azure python -m littraceqa.azure.run_rag `
    --input artifacts\official_release\bd35dc14cf0483e0ffa51fa2a54d2689c13f9845\data\validation_inputs.jsonl `
    --output runs\validation_full_index.jsonl
```

(Decomposition/sidecar flags: see `uv run --extra azure python -m littraceqa.azure.run_rag --help`;
the sidecar `.meta.json` records args, deployments, and index state for the
mandatory code submission.)

If a run is interrupted, re-run the same command with `--resume`: completed
rows are kept, placeholder rows from per-query failures (empty answer and no
papers) are retried, and the output file is rewritten atomically before
appending. LLM responses that fail JSON parsing even after one retry are
captured next to the output in a file named with the `.jsonl` suffix replaced
by `_raw_failures.jsonl` (e.g. `runs\x.jsonl` -> `runs\x_raw_failures.jsonl`)
— check it when `failed=` in the summary line is non-zero.

`--options-file` is a migration aid for an obsolete local snapshot only. Current
official validation/test/test_extra rows carry `multiple_choice_options`
directly. Never pass a gold record as an options sidecar for a current run.

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
    --inputs artifacts\official_release\bd35dc14cf0483e0ffa51fa2a54d2689c13f9845\data\test.jsonl `
    --output runs\test_full_index_vision.jsonl
```

The pinned `test.jsonl` already carries `multiple_choice_options` for every
multiple-choice row. Do not attach a validation-gold or separately maintained
options sidecar.

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
    --inputs artifacts\official_release\bd35dc14cf0483e0ffa51fa2a54d2689c13f9845\data\validation_inputs.jsonl `
    --predictions runs\validation_full_index.jsonl
```

Exit code 0 is required. It catches the silent zero-score failure modes:
missing/duplicate query_ids, answer objects not matching `answer_types`,
letters outside each input row's option keys, empty freeform, empty table rows
on table questions, and evidence items missing `paper_id`, `source_type`, or an
official page/section/object locator (the evaluator drops those silently).
For the test set, pass the pinned official test input itself; its applicable
rows already contain the valid option mapping.

---

## Policies

### Test-day options

The current official input schema includes `multiple_choice_options` on every
multiple-choice row. Use that inline mapping as authoritative for validation,
test, and test_extra. Do not join `data\validation.jsonl` or any separately
maintained options file into a challenge input.

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
schema injection, ownership checks, and evidence validation) over prompt wording
tuned until the 55-question score moves — the latter is dev-set overfitting that
will not transfer to the hidden test set. `task_family` and
`primary_evidence_type` are unavailable at test time and must not drive the
production path.

### Query-rewrite experimentation (query_lab)

`littraceqa.azure.query_lab` is a DEV-ONLY retrieval lab for iterating on the
query-rewrite prompt in `prompts\query_rewrite.txt`. It reads gold papers from
`data\validation.jsonl` to compute paper recall@K (same gold-access policy as
`compare_runs`); it is never part of the submission path.

One-time reference baseline (original question, single hybrid search — later
runs auto-compare against it):

```powershell
uv run --extra azure python -m littraceqa.azure.query_lab --name _baseline --baseline-only
```

The loop — edit the prompt, run with a fresh name, read the report:

```powershell
# 1. edit prompts\query_rewrite.txt ('#' lines are stripped; {question} required)
# 2. run (one AOAI rewrite call + one hybrid search per query, per question)
uv run --extra azure python -m littraceqa.azure.query_lab --name exp01
# 3. read runs\query_lab\exp01\results.md (aggregate recall@K vs the _baseline
#    column, then a per-question table with missing gold papers and the
#    rewritten queries; the console prints the 5 worst questions)
```

Each run writes `rewrites.jsonl` (replayable with `--rewrites-file` to re-score
without new AOAI calls), `results.json`, `results.md` and `meta.json` (prompt
text + sha256) under `runs\query_lab\<name>\`. Default scope is
`--family multi_paper`; use `--family all` / `--query-id` / `--limit` to
narrow or widen.

Handoff into the full pipeline: once a prompt wins in the lab, replay it in an
end-to-end run — with the flag unset, `run_rag` behavior is unchanged:

```powershell
uv run --extra azure python -m littraceqa.azure.run_rag `
    --input artifacts\official_release\bd35dc14cf0483e0ffa51fa2a54d2689c13f9845\data\validation_inputs.jsonl `
    --output runs\validation_rewrite_exp01.jsonl `
    --rewrite-prompt-file prompts\query_rewrite.txt
```

Then score with `scripts\evaluate.py` / `compare_runs` as in step 6 — this
re-scoring is **mandatory**, because the lab and the pipeline merge hits
differently: the lab interleaves per-query hits by best rank, while
`--rewrite-prompt-file` appends extra-query hits *after* the organic hits
(precision-conservative), so a paper found only by a rewritten query gets a
much weaker RRF position end-to-end. Lab recall deltas are an optimistic
upper bound, not a prediction of the pipeline score. Mind the
overfitting guard above: judge prompts by per-query flips, not aggregates.

### Cost

The Azure AI Search service bills continuously while it exists (per partition,
including ~12 GB of vector storage), independent of query volume. After the
8/19 submission, **stop and delete the Search service** (export the index
definition first if reproducibility requires it). Also decommission the DI
resource; the AOAI deployments are pay-per-token and can stay.
