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

An optional post-ranking step now handles four cases where the question gives
enough local evidence to override the global paper order safely:

- an explicitly named paper, Table ID, and at least two equations all match
  one table in the top-20 papers; or
- a two-column output schema names non-method rows and one value column, and
  one table uniquely contains every enumerated row plus that value column; or
- two explicit answer slots resolve to distinct top-20 papers, each paper
  claims the named method and contains the requested table or text evidence.
- an open-set venue/year question identifies one cited paper and requires it
  as a baseline: candidate metadata must match the outer venue and year, the
  MinerU references must contain the cited title and year, and one of the
  first two comparison tables must contain the cited alias in a row with
  numeric measurements.

The second rule excludes method, model, system, and paper rows. The fourth rule
accepts only two to ten verified papers, requires the original selected seed
to be among them, and preserves their retrieval order. All four rules fall
back to `f1_method_owner` when their evidence conditions cannot be verified
conservatively. They inspect only the top-20 papers and never add a paper
outside the retrieval result.

On validation, q_023, q_033, q_042, q_043, q_045, and q_052 changed. q_023
expanded one selected gold paper to all nine required CVPR 2025 papers at
ranks 1, 2, 4, 5, 6, 7, 8, 9, and 10. Each selected paper cites
"Planning-oriented Autonomous Driving" from 2023 and reports UniAD as a
measured baseline in an early comparison table; the non-gold rank-3 candidate
and ranks 11-20 failed at least one condition. q_033 moved from an unrelated
rank-1 paper to the rank-6 ECM paper whose Table 3 contains both requested
weighting equations. q_052 replaced four submitted papers with the rank-3
paper whose Table 5 contains all six benchmark rows and the Kitchen column.
q_042, q_043, and q_045 added the uniquely supported second paper for sCM/IMM,
DEIM/Mr. DETR, and VTI/MoD respectively.

| Gold definition | F1 before | F1 after evidence coverage |
|---|---:|---:|
| Answer-bearing | 0.8982 | 0.9600 |
| Evidence-backed | 0.7788 | 0.8382 |
| Official | 0.7030 | 0.7503 |

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
   two papers without checking evidence improved two validation questions, but
   changed 33 `test_extra` records. Manual inspection found 12 clear same-paper
   comparisons and 3 ambiguous cases. The accepted implementation therefore
   requires unique self-owned targets and direct local evidence before adding
   a second paper.
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

After evidence coverage, the answer-bearing result has four imperfect queries,
ten missed papers, and two false positives. Eight misses belong to the
unresolved open-set questions q_020, q_022, and q_025. The remaining two misses
and both false positives belong to q_029.

All four remaining queries need data review before another selector rule is
accepted:

1. q_020 has non-gold papers whose primary framework figures explicitly use
   MCTS, including SAPIENT and THOUGHTSCULPT. A caption-based verifier cannot
   separate them from the gold papers on the stated condition.
2. q_022 has a non-gold ICML paper, ConfPO, that proposes a preference
   objective without a reference policy in its equation. Excluding it would
   require wording specific to the annotated gold set.
3. q_025 has non-gold papers, including ReflectionFlow and UniGen, that satisfy
   the stated year, scaling, text-to-image, GenEval, and base-model conditions.
   Conversely, the gold ScaleKV paper describes cache compression rather than
   identifying its method as test-time scaling.
4. q_029 contains inconsistent reporter evidence. Multiple top-20 papers
   reproduce the requested values, while the question, datasets, values, and
   gold evidence do not provide a unique paper assignment. A weighted slot
   matcher can reinforce the current false positives instead of resolving
   them.

No additional validation rule was accepted for these cases. The next useful
step is to adjudicate the gold sets or evaluate a general evidence verifier on
independent labels. Adding more question-only patterns would overfit the
validation wording.

Count-only open-set ablations confirmed this risk. A fixed count of ten was
helpful mainly for q_023 and reduced q_022 F1. A fused-score elbow over-selected
clear two-paper questions in `test_extra`. Existing Qwen scores, structured
filter provenance, and the consensus slot are therefore not reliable binary
evidence checks. The safe fallback remains one paper when no candidate-level
condition verifier is available.
