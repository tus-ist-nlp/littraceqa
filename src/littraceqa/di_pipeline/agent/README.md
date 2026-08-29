# src/littraceqa/di_pipeline/agent/

The agent layer: takes a `Query` and returns a `Prediction`. The retriever settles
*which paper*; the agent settles *where in it the evidence is*. **No answer is
generated** — freeform / multiple_choice / table, and choosing which papers to
submit, all belong to the reading team. What this hands over is the candidate list
and the evidence.

## Files

- `reading.py` — `ReadingAgent`: retrieve, read, re-search for what is missing, and
  repeat. Its settings are `ReadingConfig` and `CombineConfig`.
- `evidence.py` — turn a `RetrievalResult` into the submitted `Evidence`, locator
  and all
- `json_utils.py` — pulling JSON out of an LLM response

---

# How ReadingAgent works

## In one sentence

**An iterative search agent: retrieve, have an LLM read the candidates and settle
the evidence, and when that is not enough, ask what is missing and search again.**

## The whole flow (`run()`)

```
run(query):
  0. _extract_attribute_filter()  take the venue/year constraint ONCE, from the question
  1. _decompose()                 split the question into search subqueries
  ┌─ for step in range(max_steps=3):              ← the loop
  │  2. retriever.retrieve(subquery, retrieve_top_k) for each subquery
  │     → accumulate into chunks: dict[chunk_id -> RetrievalResult]
  │  3. _candidate_papers()        group the chunks per paper, keep the top ones
  │  4. _read_and_judge()          the LLM reads the candidates and judges
  │     → {paper_ids, evidence_chunk_ids, sufficient, missing}
  │  5. break when verdict["sufficient"]
  │  6. _refine(missing)           ask what is missing, get new subqueries → 2
  └─
  7. _build_prediction()          assemble the Prediction
                                  (answer stays empty; generating it is the
                                   reading team's job)
```

**At most two LLM calls per pass**: one for `_decompose` / `_refine`, one for
`_read_and_judge`. With answer generation gone, the extra call per query is gone
with it.

## The decisions behind each step

### 0. Extracting the attribute filter (`_extract_attribute_filter`)

When a question names the venue it is searching in ("Which NAACL 2025 papers ..."),
that constraint is taken **once, from `query.question`**, and reused across every
step of the loop.

- **Never from a subquery.** `_decompose()` sometimes drops terms like "NAACL 2025",
  so extracting from a subquery would simply never fire.
- It is passed as `retrieve(..., attribute_filter=...)` only when something was
  found. Otherwise the argument is not passed at all, and **a question with no
  constraint behaves exactly as it did before**.
- The extractor belongs to the retriever (`pipeline.build_retriever()` hands it an
  `AttributeExtractor`). Without one this returns None. See
  `retrieve/attribute_filter.py`.

### 1. Splitting (`_decompose`)

**Always splits**, single-paper and multi-paper alike: even for a single paper,
"which paper" and "which table inside it" can want different search terms.

**The count is fixed at `SUBQUERY_COUNT` (4) and never branches on task_family.** It
used to ask for 1-3 on single-paper questions and 3-6 on multi-paper ones, and that
branch alone cost **an LLM call per query** to guess task_family, which production
input does not carry. Measured, it bought 0.58 extra subqueries:

| gold | subqueries at step 0 | distribution |
|---|---|---|
| single, 26 queries | 3.08 on average | 25 of 26 at the cap of 3 |
| multi, 29 queries | 3.66 on average | 17 of 29 at the floor of 3 |

The two estimators were also indistinguishable in accuracy (LLM 0.670, heuristic
0.673 over 55 queries), so there was no basis for branching at all. Fixing the count
at 4 sits between the measured averages and removes one LLM call.

The wording is one sentence covering both cases too — "the evidence may live in a
single paper or be spread across several; write both paraphrases that reliably
retrieve the paper(s) in focus and separate subqueries per distinct fact" —
because committing to either loses the other outright when the guess is wrong.

An empty response falls back to the original question as the only subquery.

**`CORPUS_NOTE` ("this goes to a local index, not a web search engine") must lead
the prompt.** Without it the LLM writes Google queries with `site:arxiv.org` and
`filetype:pdf`. Measured, 29-41% of `_refine()`'s output carried them; they match
nothing in the local BM25 and faiss indexes, so rounds 2 and 3 of the loop were
retrieving pure noise. It sits at the head of both `_decompose` and `_refine`.

**When a constraint was extracted, `_constraint_note()` asks for `[NAACL 2025]` at
the front of each subquery.** The narrowing itself is attribute_filter's job, so
this is the constraint **as a search term**: a title_abstract chunk's text really
does begin `[ACL 2025] Title...`, so the same tag works as a BM25 term.

### 2. Retrieving and accumulating

Each subquery's results go into the `chunks` dict. When several subqueries hit the
same chunk_id, **the higher score wins** — last-write-wins would let subquery 3's
low score overwrite subquery 1's top hit, dragging the paper ranking towards
whichever subquery ran last.

### 3. Assembling the candidates (`_candidate_papers`)

Group the chunks by `paper_id` and order the papers by their best chunk. The top
`max_candidates` (20) papers become candidates, with `chunks_per_paper` (2) chunks
each. **That count is the ceiling on what the LLM can read** — a chunk not retrieved
in step 2 is not a candidate at all.

### 4. Reading and judging (`_read_and_judge`) — the heart of it

The candidate chunks are shown to the LLM **in full** (`snippet_chars`, 1800
characters), and one call returns everything at once:

```json
{"papers": [{"paper_id": "...", "evidence_chunk_ids": ["..."]}],
 "sufficient": true, "missing": ""}
```

| field | what it is for |
|---|---|
| `paper_ids` | the papers the LLM confirmed. **Not used for the submission** (choosing is the reading team's job); used only as the anchors of ranking B (`anchor_from: verdict`) |
| `evidence_chunk_ids` | the chunks the submitted evidence comes from |
| `sufficient` | **the loop's stopping condition**; true breaks |
| `missing` | what to search for next (handed to `_refine`) |

**Guards against invention**: a paper_id absent from the candidate list, a chunk_id
that does not exist, and a chunk_id belonging to another paper are all rejected.
`_format_paper()` presents each candidate as `[paper_id] title (venue year)` plus,
per chunk, its chunk_id, type, page / table_id / figure_id / equation_id and text,
so the LLM can choose evidence down to the locator.

**Nothing is truncated here.** The submission is the top `max_papers` of the
candidate list, decided in `_build_prediction`.

### 5. Stopping (inside `run()`)

```python
if verdict is not None and verdict["sufficient"]:   # the LLM says it has enough
    break
if step == self.config.max_steps - 1:               # the iteration cap
    break
subqueries = self._refine(query, missing, tried)
if not subqueries:                                  # the LLM says more searching is futile
    break
```

**It stops on whether the papers the LLM could confirm are enough, never on how many
results came back.** That is the whole difference from an iteration that counts.

### 6. Re-splitting (`_refine`)

Given `missing` (what is absent) and `tried` (the subqueries already issued), the
LLM writes new, **non-overlapping** subqueries. If it judges that further searching
will find nothing, it returns an empty list and the loop ends.

### 7. Assembling the prediction (`_build_prediction`)

The Prediction carries **two lists of papers**, and the difference matters for
evaluation.

```python
candidate_papers = to_gold_papers(every chunk)[:50]   # what retrieval found
gold_papers      = candidate_papers[:max_papers]      # what is submitted
```

- `candidate_papers` is what retrieval found, **before the cut**.
  `candidate_recall@k` is measured on it, which is what **separates retrieval (did
  it find the paper?) from selection (how well was it narrowed?)**. Improvements to
  an indexer, the reranker or the attribute filter show up here.
- `gold_papers` is after the cut. **No selection happens** — which papers to submit
  belongs to the reading team, so the candidate ranking is passed straight through.
  A query where no verdict ever came back takes the same path.
- The per-step `trace` (subqueries / attribute_filter / n_chunks / selected /
  sufficient / missing) is kept, so a run can be followed afterwards.

Evidence becomes a located `Evidence` through `evidence_from_result()`, except for
papers the cut removed.

## The papers to submit are not chosen

The submission is the candidate ranking as it stands, capped at `max_papers`.
**Choosing belongs to the reading team**, so the search agent stops at handing over
the ranking.

**`_read_and_judge()` is still called.** Of the three things one LLM call returns,
the two that are not the selection have jobs of their own: `sufficient` **is the
loop's stopping condition** (without it the loop is fixed at `max_steps`), and
`evidence_chunk_ids` is the evidence (`evidence_f1`). `paper_ids` does reach the
ranking, as the anchors of ranking B (`anchor_from: verdict`).

## How many papers are submitted

The first `max_papers` (10) of the candidate list. **That is all.**

There used to be a path that guessed `task_family` (single/multi) with an LLM and
varied the count by it. Production input has no task_family, and guessing it from
the question is right about 0.67 of the time — not enough to rely on. It cost an
extra LLM call per query for nothing, so the path was removed.

## The settings (`ReadingConfig`)

The knobs live in `ReadingConfig` (`agent/reading.py`) and reach `ReadingAgent` as
one object, `config=ReadingConfig(...)`. **This dataclass is the single definition
of which params exist**, and a misspelling stops at the constructor with a
TypeError. `ReadingAgent.__init__` names only the collaborators (`retriever` /
`llm` / `paper_expander`) and the two settings objects (`config` / `combine`).

| param | default | meaning |
|---|---|---|
| `max_steps` | 3 | cap on iterations |
| `retrieve_top_k` | 20 | how many **chunks** one subquery returns (not papers) |
| `max_candidates` | 20 | how many **papers** the LLM sees; matches `candidate_recall@20` |
| `chunks_per_paper` | 2 | chunks shown per paper |
| `snippet_chars` | 1800 | how much of a chunk is shown |
| `max_papers` | 10 | cap on the submitted papers |
| `paper_score_skip_chunk_types` | none | chunk types excluded from a paper's representative score (`["table"]` measured best) |

`max_candidates: 20` is deliberately equal to the metric's k. At 15, papers ranked
16-20 could be retrieved and still never be seen by the LLM, so a gain in the metric
would not reach the real score.

## The relationship with retrieval

The agent receives nothing but the retriever's output — a list of chunk-level
`RetrievalResult`s — so strengthening retrieval lands directly on
`candidate_papers`, and therefore on `candidate_recall`. What happens inside the
retriever is described below, in **"The retrieval pipeline (`HybridRetriever`)"**.
The agent's own code (`_decompose`, `_read_and_judge`, ...) does not change when an
indexer, the fuser, the reranker or the attribute filter does.

## When evaluating (see CLAUDE.md as well)

- **Read `candidate_recall` and `evidence_candidate_recall`, nothing else.**
  `evaluate.py` leaves the submission-side metrics (paper_* / evidence_* / the
  answer ones) out by default; `--metrics all` adds them.
- A validation record and a production record give the same `Query`. The local
  `validation_inputs.jsonl` carries task_family and production does not, but
  task_family is not a `Query` field, so it reaches nothing. (`--production-input`
  used to strip it, back when the estimator and the cutoff read it; with both gone
  the flag could no longer change a run, and it was removed.)
- The LLM is non-deterministic (the deployment does not accept a temperature) and
  there are only 55 queries, so a difference of a few points may be noise. Run it
  more than once before concluding anything.

---

# The retrieval pipeline (`HybridRetriever`)

The search that **step 2** above
(`self.retriever.retrieve(subquery, retrieve_top_k, ...)`) actually calls. It lives
in `src/littraceqa/di_pipeline/retrieve/` rather than in the agent layer, but it is
described here because it is part of the loop. In goes **one subquery**; out comes
a list of **chunk-level `RetrievalResult`s** — collapsing those to papers is the
agent's job (`_candidate_papers`, `to_gold_papers`).

## agent → retriever → agent

```
ReadingAgent.run(query)
  step0 _extract_attribute_filter(query.question)  → attribute_filter | None
  step1 _decompose(query)                          → [subquery, ...]
  ┌ for step in range(max_steps):
  │   for subquery in subqueries:
  │     results = HybridRetriever.retrieve(subquery, top_k, attribute_filter)  ← here
  │       │
  │       │  === inside retrieve() (retrieve/hybrid.py) ===
  │       │  a. with no attribute_filter but an extractor, derive one from the
  │       │     subquery (a fallback for callers that send a raw question; the
  │       │     agent always passes step 0's value)
  │       │  b. _run_indexers(): query each index (below)      → runs: list[list]
  │       │  c. fuser.fuse(runs, top_k=fuse_k): RRF into one ranking
  │       │       fuse_k = with a reranker ? (pool_k or top_k*3) : top_k
  │       │  d. seed expansion, then the reranker, blended by rank
  │       ▼
  │     accumulate results into the chunks dict (same chunk_id, higher score wins)
  │   _candidate_papers() → _read_and_judge() → sufficient? → _refine()
  └
  _build_prediction()
```

The retriever always returns **chunks**. Collapsing to papers is the agent's
(`candidate_papers` via `to_gold_papers`). That division is how the project's rule —
retrieval identifies the gold paper, the agent identifies the evidence — appears in
the code.

## b. Querying each index (`_run_indexers`)

The indexes come from `pipeline.build_indexers()` — `bm25s`, `bm25s_paper` and
`faiss_qwen3` — and each has `search(query, k) -> list[RetrievalResult]`. **There are
two code paths, depending on whether a constraint was found.**

**No constraint** (the vast majority of questions):
```
runs = [indexer.search(query, per_index_k) for indexer in indexers]   # 100 each
```
Byte for byte what it was before. Adding the attribute filter costs a question with
no constraint nothing at all.

**With a constraint** ("Which NAACL 2025 papers ...", where exactly one venue was
extracted):
```
fetch_k = _fetch_k(filter)                 # work back from selectivity so per_index_k survives
for indexer in indexers:
    raw  = indexer.search(query, fetch_k)  # over-fetch
    kept = filter_results(raw, filter)     # drop by venue/year in the metadata
    if len(kept) < min_filtered_results:   # drained: fail open for this run only
        kept = raw
    runs.append(kept[:per_index_k])
```
- **No index changes.** venue and year are already in `RetrievalResult.metadata`, so
  this is over-fetching and dropping, and every index benefits alike.
- `fetch_k = per_index_k / selectivity * fetch_safety`, capped at `max_fetch_k`.
  **Do not raise that cap**: matching it to per_index_k blew faiss search up from
  1.5s to 91.1s on NAACL.
- It fires only when exactly one venue can be extracted. A bare year, "all venues",
  or two venue names all skip extraction and take the unconstrained path
  (`attribute_filter.py`).

## c. Fusing (`PaperRRFFuser`, `retrieve/paper_rrf.py`)

Several indexes' rankings become one, through **paper-level** Reciprocal Rank
Fusion at `k=60`. Mixing on **rank** rather than absolute score is what lets BM25
(vocabulary) and an embedding (meaning) combine at all, their scales being
unrelated. **Within one run a paper gets one vote however many of its chunks were
hit** — fused per chunk, long papers and table-heavy papers occupy the top on chunk
count alone, and since the metric is per paper that distortion lands straight on the
score.

## d. Seed expansion and the reranker

Before the reranker, `_seed_expand()` appends the top paper's title+abstract to the
question and queries again, fusing the two rounds. **A question does not know what a
paper calls itself**, and this borrows the corpus's own vocabulary; no LLM is
involved. It runs before the reranker so that reranker inference stays at one pass.

`Qwen3Reranker` then scores query x chunk with a yes/no model over the `pool_k`
candidates. With `rerank_blend`, its ranking does not replace the fused one: the two
are blended with RRF and the pre-fusion top `protect_top` are kept at the front.
The reranker judges "does this answer the question", so it always demotes the peer
gold papers a question never names — blending is what keeps ranking A from being
exposed to that.

## This layer's settings (`HybridRetriever.__init__`)

| param | production | meaning |
|---|---|---|
| `per_index_k` | 100 | chunks one index returns, before fusion |
| `pool_k` | 200 | candidates handed to the reranker (None means `top_k*3`) |
| `fetch_safety` | 1.5 | the over-fetch factor when a constraint applies |
| `max_fetch_k` | 3000 | cap on one index's fetch when a constraint applies |
| `min_filtered_results` | 10 | fewer than this after filtering fails open |
| `seed_expansion` | `SeedExpansion(512)` | how much of the top paper's text is borrowed |
| `rerank_blend` | `RerankBlend(0.6, 0.4, 60, 20)` | blend the reranker's rank instead of replacing the order |

All of them are passed in `pipeline.build_retriever()`. **Do not confuse
`per_index_k` and `pool_k` with the agent's `retrieve_top_k`** (how many chunks one
subquery hands back); they are different things.
