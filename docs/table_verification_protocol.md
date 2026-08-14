# Table-only verification protocol

This protocol is an optional post-processing layer for table answers. It keeps
candidate generation separate from promotion and makes any source review
explicit. It is not part of fully automatic reader inference.

## Generate independent table candidates

The reader can select queries by the declared `answer_types` field. The
selection contains no query-ID allowlist. Configure the normal reader inputs,
then run each sample in a distinct directory:

```bash
uv run python scripts/run_aoai_pairwise_reader.py \
  --queries "$QUERIES" \
  --candidates "$CANDIDATES" \
  --paper-metadata "$PAPER_METADATA" \
  --chunks "$CHUNKS" \
  --reader configs/agent_style/aoai_pairwise_reader.yaml \
  --run-dir runs/table_sample_01 \
  --answer-type table \
  --stage all
```

Explicit query IDs may be combined with `--answer-type table` for a smaller
diagnostic. Separate run directories prevent samples from overwriting one
another's checkpoints and manifests.

## Create a sealed review bundle

Candidate files may contain the complete query set or only all declared table
queries. The frozen base must be complete.

```bash
uv run python scripts/adjudicate_table_answers.py review \
  --inputs "$QUERIES" \
  --base runs/frozen_base/submission.jsonl \
  --candidate sample_01=runs/table_sample_01/submission.jsonl \
  --candidate sample_02=runs/table_sample_02/submission.jsonl \
  --output-dir runs/table_review
```

The review phase validates coverage, the provided input order and schemas, row
types, candidate identities, and duplicate row keys. It emits a JSON
comparison, a Markdown review document, and a decision template. The review
document includes each question, schema, frozen paper/evidence set, and
candidate table.

Copy the generated template before editing it. For every table query, select
the base or one candidate after checking the cited PDF or extracted source.
Each decision must:

- set `source_checked=true`;
- contain non-empty notes;
- cite at least one locator already present in the frozen evidence.

## Compose the reviewed result

```bash
uv run python scripts/adjudicate_table_answers.py compose \
  --inputs "$QUERIES" \
  --base runs/frozen_base/submission.jsonl \
  --candidate sample_01=runs/table_sample_01/submission.jsonl \
  --candidate sample_02=runs/table_sample_02/submission.jsonl \
  --decisions runs/table_review/table_decisions.json \
  --output runs/table_review/predictions.jsonl \
  --audit runs/table_review/predictions.audit.json
```

Composition may replace only `answer.table` for table queries. It verifies
deep equality of paper lists, evidence lists, multiple-choice and freeform
answers, every non-table record, the provided query order, and table schemas.
Non-table JSONL lines are reused byte for byte.

The review seals the inputs, frozen base, candidates, order, and schemas with
SHA-256. Composition rechecks those hashes and records a SHA-256 hash of the
exact decision bytes in an audit manifest. Existing paths are not silently
overwritten, and the audited output is published only after the audit artifact
is durable.

## Reporting

When this protocol contributes to reported results, distinguish it from fully
automatic inference. Record the number of generated candidates, the source
materials inspected, the decision artifact hash, and the final audit hash.
