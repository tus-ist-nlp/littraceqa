# Data Directory

LitTraceQA public development files:

```text
validation.jsonl          # gold development set
validation_inputs.jsonl   # input-only copy for participant dry runs
sample_submission.jsonl   # empty submission template
paper_metadata.jsonl      # searchable candidate paper metadata pool
```

The current public split is a small development set for workshop challenge participants. Each file is JSON Lines, with one complete sample per line. Do not place PDF caches, raw paper PDFs, or annotation-workflow scratch files in this directory unless they are part of the intended public release.

Questions are scoped to the papers listed in `paper_metadata.jsonl`. Use the
canonical `paper_id` values from that file when submitting retrieved papers.

Each sample should follow:

```text
../schema/littraceqa.schema.json
```
