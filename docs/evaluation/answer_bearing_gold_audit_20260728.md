# Answer-bearing gold paper audit (2026-07-28)

## Purpose

This note preserves a manual review of whether each annotated gold paper is
actually required to answer its query. It is separate from
`data/validation_evidence_gold.jsonl`, which applies the weaker rule that a gold
paper is retained when its `paper_id` appears in `evidence[].paper_id`.

Audit provenance:

- Recorded on: `2026-07-28`
- Repository commit before this note: `0493d2573a22ed0b0782fde7060d1a642983a026`
- `data/validation.jsonl` SHA-256:
  `daa5ed246c00a5e4bb571843baa985b6256700da8a7ae5695bd642dfd4298e41`
- `data/validation_evidence_gold.jsonl` SHA-256:
  `2e71668f7e72305b44834766c13ce187f627ecd75bb663c93d696e09d184003d`

The distinction is:

- **Evidence-backed paper**: at least one evidence annotation refers to the
  paper.
- **Answer-bearing paper**: the paper supplies information needed to derive the
  requested answer.

The evidence-backed condition is necessary for the current audit, but it is not
sufficient. A paper can be cited only as background, contrast, or related work.

## Manually reviewed retrieval misses

### q_041

- Question target: the proposal-distribution parameters `(P_mean, P_std)` used
  for CIFAR-10 training in the sCM paper.
- Answer: `P_mean = -1.0`, `P_std = 1.4`.
- Answer-bearing paper: `iclr2025_03031` (sCM). Its evidence directly states
  the two values.
- Unnecessary paper: `iclr2025_03463` (Truncated Consistency Models / TCM). Its
  evidence describes TCM's log-Student-t time weighting and does not supply the
  requested sCM parameters.
- `icml2025_01371` (IMM) is likewise unrelated to the requested values.
- `iclr2025_00615` has no evidence.

Conclusion: `iclr2025_03463` is not required to answer q_041.

### q_044

- Question target: the AlpacaEval 2 LC win rates of D²PO and AlphaDPO on
  Gemma2-9B-Instruct.
- Answer: `59.7%` and `73.4%`.
- Answer-bearing papers:
  - `iclr2025_00978` supplies `59.7%`.
  - `icml2025_00188` supplies `73.4%`.
- Unnecessary paper: `icml2025_00192` (AMPO). Its evidence supplies AMPO's
  unrelated `52.4%` result.
- `acl2025_02365` supplies an unrelated γ-SimPO MT-Bench result.

Conclusion: `icml2025_00192` and `acl2025_02365` are not required to answer
q_044.

### q_051

- Question target: the perturbation budget epsilon, learning rate alpha, and
  sampling variance beta used for visual-noise optimization in the VAP paper.
- Answer: `epsilon = 2`, `alpha = 1/255`, `beta = 8/255`.
- Answer-bearing paper: `neurips2025_03461` (VAP). Its evidence directly states
  all three values.
- Unnecessary paper: `eccv2024_01567` (PAI). Its evidence contrasts PAI with
  perturbation-based methods and does not supply the requested VAP
  hyperparameters.
- `iclr2025_02963` (SID) is another contrast paper.
- `iclr2025_02715` has no evidence.

Conclusion: `eccv2024_01567` is not required to answer q_051.

## Exploratory numeric-answer audit

The following results were reported from matching numeric answer values against
gold-paper evidence for 23 queries that have at least two gold papers and a
numeric answer:

| Classification | Paper count | Fraction |
|---|---:|---:|
| Evidence contains the answer value | 28 | 30.4% |
| Evidence exists but does not contain the answer value | 35 | 38.0% |
| No evidence | 29 | 31.5% |

The reported query-level reduction was:

| Original gold count | Answer-bearing count | Query count |
|---:|---:|---:|
| 4 | 1 | 18 |
| 4 | 2 | 3 |
| 4 | 4 | 1 |
| 4 | 0 | 1 |

Interpretation:

- The three `4 -> 2` queries are q_043, q_044, and q_045. They request two
  values that genuinely require two papers.
- q_025 is the `4 -> 4` case and appears to be a genuine multi-paper question.
- q_020 is the `4 -> 0` case. Its answer lists paper titles, so numeric matching
  cannot detect the answer-bearing papers.
- q_041 and q_051 are representative `4 -> 1` cases.

These aggregate figures were supplied with the manual review. The exact
analysis script and intermediate output were not included, so the counts in
this section are a preserved audit record rather than a result independently
reproduced by the repository.

## Completed draft audit (2026-07-29)

The subsequent paper-by-paper audit is preserved in:

- `data/validation_answer_bearing_gold_draft.jsonl`
- `data/validation_answer_bearing_gold_audit.json`

The source evidence-backed gold file was not modified. File hashes are:

| File | SHA-256 |
|---|---|
| `data/validation_evidence_gold.jsonl` | `2e71668f7e72305b44834766c13ce187f627ecd75bb663c93d696e09d184003d` |
| `data/validation_answer_bearing_gold_draft.jsonl` | `0a5d950d98861340c50a97ad80ede7492ecee769d5d55c8b07cd82153bbff451` |
| `data/validation_answer_bearing_gold_audit.json` | `949c351255977c08886fa95468ba417c91a80a388cdd052377a4873534350cde` |

The audit retains 87 of the 117 evidence-backed paper records:

| Classification | Paper count |
|---|---:|
| Required | 87 |
| Supportive | 18 |
| Irrelevant | 12 |
| Uncertain | 0 |

Fourteen query gold sets changed. The resulting draft contains 43 single-paper
queries and 12 genuine multi-paper queries. The multi-paper query IDs are
`q_020`, `q_022`, `q_023`, `q_025`, `q_028`, `q_029`, `q_030`, `q_042`,
`q_043`, `q_044`, `q_045`, and `q_056`.

Nine queries retain data-quality notes because of question/evidence
inconsistency or because the completeness of a requested enumeration cannot be
established from the supplied gold data alone: `q_001`, `q_020`, `q_021`,
`q_022`, `q_023`, `q_025`, `q_029`, `q_054`, and `q_055`. These notes are not
blockers for the scoped gold-paper retrieval experiment. The experiment only
evaluates retrieval of the supplied answer-bearing paper IDs and does not try
to repair benchmark coverage or add unannotated papers.

The generated files were independently checked for query order, schema
preservation, source-subset constraints, duplicate and unknown paper IDs,
classification totals, evidence references, anchor cases, and hashes. No
inconsistency was found. The file is the fixed working gold for the scoped
retrieval experiment. It remains named **draft answer-bearing gold** because it
is an experimental derivative rather than an official benchmark revision, not
because the nine data-quality notes block its use.

## Full-corpus retrieval result on the draft

The current baseline evaluates all 55 queries against the 27,487-paper
MinerU/BM25 index and reranks 50 paper candidates with
Qwen3-Reranker-0.6B. It uses a base-rank fusion weight of 0.59 and lets Qwen
reorder the original top 20 without allowing papers below that boundary to
displace them. All 55 queries completed with Qwen applied and zero failures.
The output is:

`/data2/kumagai/littraceqa_data/mineru_eval/accuracy_ladder_c27487/accuracy_27487/evaluations/answer_bearing_baseline_pool50_w059_protect20_clean_gpu55.json`

SHA-256:
`fc8b686015c40e843dbcdd1dcd98805563c6df32be75577cb00235030e2f5174`

| Scenario | Recall@3 | Recall@5 | Recall@8 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| Total (55 queries) | 0.8121 | 0.9551 | 0.9636 | 0.9697 | 0.9843 | 0.9955 |
| Single (43 queries) | 0.8837 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Multi (12 queries) | 0.5556 | 0.7940 | 0.8333 | 0.8611 | 0.9282 | 0.9792 |

Compared with the previous 0.56 / unprotected baseline, Recall@3 improved
from 0.7919 to 0.8121 and Recall@20 improved from 0.9798 to 0.9843.
Recall@1, @5, @8, @10, and @50 were unchanged. Stored-score replay found no
per-query regression at any of these cutoffs, and the full GPU execution
matched that replay exactly. The recorded sum of per-query elapsed time was
329.9 seconds (6.00 seconds per query on average).

The all-gold@8 rate is 0.9273 overall and 0.6667 for multi-paper queries.
The hit rate@8 is 1.0: every query has at least one answer-bearing paper in the
top eight. This is the most direct success criterion when one supporting paper
is sufficient. Paper Recall@8 remains useful as the stricter diagnostic for
how much of the supplied multi-paper gold set is recovered.
Among the 46 queries not marked for human review, both Recall@5 and
all-gold@5 are 1.0. This is a sensitivity analysis, not a replacement headline
metric: the nine review queries contain most of the difficult enumeration
cases, so silently excluding them would introduce selection bias.

Of the 87 required papers, 86 occur in the 50-candidate pool. The only pure
candidate-generation miss is the ScaleKV paper for `q_025`. Remaining
top-cutoff misses are concentrated in `q_020`, `q_022`, `q_023`, and `q_025`;
all four are multi-paper queries with non-blocking data-quality notes.

## Known limitations

Numeric answer matching is not reliable as a universal gold-construction rule.
The reported false-negative patterns include:

- q_001: evidence provides a value used in a calculation rather than the final
  answer.
- q_017 and q_019: the answer is obtained by counting citations.
- q_004, q_013, q_018, and q_026: the answer requires counting or reading a
  figure/equation, while evidence text can be empty.
- q_010: the answer is a row label selected by comparing numeric table values.

Therefore, numeric matching can support the manual review of multi-gold
queries, but it must not automatically delete gold papers across the full
validation set.

## Evaluation implications

The current evidence-backed transformation reduced 146 annotated gold papers
to 117, but this audit indicates that some retained papers are still only
context or contrast papers.

Two evaluation policies remain possible:

1. **Evidence-backed retrieval**: retain every paper referenced by evidence and
   treat contextual/contrast papers as retrieval targets.
2. **Answer-bearing retrieval**: retain only papers necessary to derive the
   answer.

Policy 2 is the selected working policy for the current gold-paper retrieval
experiment. The nine data-quality notes do not need to be resolved for this
scope, and enumeration completeness is treated as benchmark-owner
responsibility. Publishing the derivative as an official benchmark correction
would still be a separate decision requiring team agreement.

For current experiments:

- keep `data/validation_evidence_gold.jsonl` unchanged;
- label its metrics as **evidence-backed gold** metrics;
- label the new 87-paper file as **draft answer-bearing gold**;
- report single/multi groups using the actual evaluated gold count;
- use all 55 queries as the primary result without excluding the nine queries
  that have data-quality notes;
- prioritize hit rate when at least one supporting paper is sufficient, while
  retaining Paper Recall as the stricter multi-paper diagnostic;
- preserve q_041, q_044, and q_051 as concrete examples when discussing the
  difference between contextual and answer-bearing papers.
