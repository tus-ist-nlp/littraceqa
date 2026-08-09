# Paper selection error audit (2026-08-09)

## Purpose

This note records the review of paper-selection errors after applying
`f1_method_owner` to the completed 55-query retrieval run. The answer-bearing
gold is used to diagnose retrieval and selection behavior; it is not a
replacement for the official validation gold.

Inputs:

- Repository commit before this audit: `5c349262e6d94802f00cbe0aa993f7ca63804798`
- Retrieval output SHA-256: `729c86d8d4442ab6ca7f21f490204ad417b5295188eaf8add447c3b439f917c9`
- Answer-bearing gold SHA-256: `0a5d950d98861340c50a97ad80ede7492ecee769d5d55c8b07cd82153bbff451`
- Evidence gold SHA-256: `2e71668f7e72305b44834766c13ce187f627ecd75bb663c93d696e09d184003d`
- Official gold SHA-256: `daa5ed246c00a5e4bb571843baa985b6256700da8a7ae5695bd642dfd4298e41`

## Reviewed errors

Before the accepted guard, an external review classified 90 paper-level
records from the original error reports. The corrected table had no duplicate
`(query_id, paper_id, record_type)` rows and contained:

- 22 missed papers required to answer their questions;
- 59 missed official/evidence papers that were useful context but not required
  for the reference answer; and
- 9 unnecessary submitted papers.

The 22 answer-required misses comprised 19 cardinality errors and 3 ranking
errors. None was outside the saved top-50 candidate pool. The 9 false
positives were all judged unnecessary for the answer.

The cardinality errors have two different causes and should not be treated as
one problem:

| Error group | Queries | Missed answer-bearing papers |
|---|---|---:|
| Open or implicit enumeration | q_020, q_022, q_023, q_025 | 16 |
| Explicit multi-method comparison | q_042, q_043, q_045 | 3 |

The first group does not name every paper or method in the question. Method
alias matching cannot determine its result size; each candidate needs to be
checked against the question's venue, year, modality, and evidence conditions.

## Accepted single-source guard

The existing enumeration parser interpreted models, datasets, and experimental
conditions as separate papers in two single-source questions. A conservative
guard now treats these forms as having no explicit multi-paper count:

- `In the ... experiments of X, ...`
- `What is the ... performance of X on/with ...`

Explicit paper counts and true comma-separated method lists still take
precedence. The guard removed three false positives without changing recall.

| Gold definition | F1 before | F1 after |
|---|---:|---:|
| Answer-bearing | 0.8830 | 0.8982 |
| Evidence-backed | 0.7636 | 0.7788 |
| Official | 0.6879 | 0.7030 |

The selected paper sets for the 71 test questions and 4,901 `test_extra`
questions did not change. This is a regression check, not evidence that the
validation improvement transfers to held-out gold.

The retrieval entity parser also had a case-normalization bug: generic names
such as `ACL`, `LoRA`, and `RAG` were compared in lower case against upper-case
sets and were not removed. After fixing the comparison, all 17 affected
validation questions were searched again with the 4B configuration. Their
complete top-50 rankings and all gold ranks were unchanged, so the fix restores
the intended behavior without changing the established validation result.

## Evidence coverage experiment

An optional post-ranking step now handles two cases where the question gives
enough table structure to override the global paper order safely:

- an explicitly named paper, Table ID, and at least two equations all match
  one table in the top-20 papers; or
- a two-column output schema names non-method rows and one value column, and
  one table uniquely contains every enumerated row plus that value column.

The second rule excludes method, model, system, and paper rows. Both rules
fall back to `f1_method_owner` when no unique table is found or MinerU output
is missing or malformed. They read only the top-20 papers and never add a
paper outside the retrieval result.

On validation, only q_033 and q_052 changed. q_033 moved from an unrelated
rank-1 paper to the rank-6 ECM paper whose Table 3 contains both requested
weighting equations. q_052 replaced four submitted papers with the rank-3
paper whose Table 5 contains all six benchmark rows and the Kitchen column.

| Gold definition | F1 before | F1 after evidence coverage |
|---|---:|---:|
| Answer-bearing | 0.8982 | 0.9273 |
| Evidence-backed | 0.7788 | 0.8079 |
| Official | 0.7030 | 0.7212 |

The 71 test and 4,901 `test_extra` selections did not change. This establishes
that the current gates are fail-closed on those inputs, but not that the gain
transfers to held-out gold. The feature is therefore opt-in rather than part
of the default select style.

```bash
PYTHONPATH=. uv run python scripts/eval_paper_selection.py \
  --retrieval /path/to/post_lane_removal.json \
  --gold data/validation.jsonl \
  --questions data/validation_inputs.jsonl \
  --evidence-coverage-mineru-dir /path/to/mineru
```

`--questions` must point to production-style inputs because the reconstructed
gold files do not retain `table_schema`.

## Rejected experiments

Two experiments were intentionally not kept:

1. Counting parallel `what ... does A ..., and what ... does B ...` clauses as
   two papers improved two validation questions, but changed 33 `test_extra`
   records. Manual inspection found 12 clear same-paper comparisons and 3
   ambiguous cases, so the rule was not reliable enough for F1-oriented use.
2. Sending the query-matched chunk instead of the paper head to Qwen improved
   some individual ranks, but reduced eight-query selector F1 from 0.5917 to
   0.5292. The implementation change was reverted.

## Reproduce the review files

`scripts/report_paper_selection_errors.py` writes only queries whose selected
set differs from the chosen gold definition. It includes the reference answer,
gold evidence, pre- and post-rerank positions, candidate provenance, abstracts,
and separate labels for missed and unnecessary papers.

```bash
PYTHONPATH=.:src uv run python scripts/report_paper_selection_errors.py \
  --retrieval /path/to/post_lane_removal.json \
  --gold data/validation_answer_bearing_gold_draft.jsonl \
  --questions data/validation_inputs.jsonl \
  --paper-metadata data/paper_metadata.jsonl \
  --select configs/select_style/f1_method_owner.yaml \
  --evidence-coverage-mineru-dir /path/to/mineru \
  --analysis-cutoff 20 \
  --read-only-root /data2 \
  --output ~/littraceqa_data/mineru_eval/analysis/answer_bearing_errors.json
```

Run the same command with `data/validation_evidence_gold.jsonl` and
`data/validation.jsonl` to separate semantic answer errors from official-gold
differences.

## Remaining work

After evidence coverage, the answer-bearing result has eight imperfect
queries, 21 missed papers, and two false positives. The most useful next steps
are:

1. Resolve explicitly named methods or variants to distinct supporting papers
   for q_042, q_043, and q_045. This targets three misses.
2. Select open-set papers from verified venue/year/modality and evidence
   constraints instead of a fixed count. This targets 16 of the 21 misses.
3. Match each requested answer slot to a paper that actually reports the
   value before cutting the final list. This is different from method
   ownership: in q_029, two required papers contain comparison-table rows for
   other methods but are not the papers that introduced those methods.

These changes require evidence coverage or reliable owner resolution. Adding
more question-only regular expressions would be simpler, but the rejected
ablation shows that it would overfit the validation wording.

Count-only open-set ablations confirmed this risk. A fixed count of ten was
helpful mainly for q_023 and reduced q_022 F1. A fused-score elbow over-selected
clear two-paper questions in `test_extra`. Existing Qwen scores, structured
filter provenance, and the consensus slot are therefore not reliable binary
evidence checks. The safe fallback remains one paper when no candidate-level
condition verifier is available.
