# src/littraceqa/di_pipeline/retrieve/

Fusing several indexers' results into one ranking. `HybridRetriever` ties it
together.

- `hybrid.py` — `HybridRetriever`: query each index, fuse with `PaperRRFFuser`,
  re-rank with `Qwen3Reranker`. `to_gold_papers()` collapses a chunk ranking into a
  paper ranking. With `rerank_blend`, the reranker no longer replaces the order:
  its ranking is blended into the pre-rerank one with RRF, and the top
  `protect_top` are kept as a set at the front. **The blended rank is written back
  into `score`**, because everything downstream re-sorts by score.
- `paper_expander.py` — paper-to-paper expansion (`specter2` / `bib_coupling` /
  `bm25_mlt`, fused). `rank(anchors)` returns the neighbours of the anchors it is
  handed, in relevance order. **Choosing the anchors, and the settings for the A/B
  fusion, belong to `ReadingAgent`** (`CombineConfig`).
- `attribute_filter.py` — `AttributeExtractor` / `AttributeFilter`: take the venue
  and year a question states ("Which NAACL 2025 papers ...") and narrow the results
  by them. **The indexes need no changes**, since venue and year are already in
  `RetrievalResult.metadata`. It fires only when exactly one venue can be
  extracted; otherwise the code path is identical to having no filter at all.
- `paper_rrf.py` — `PaperRRFFuser`: **paper-level** Reciprocal Rank Fusion. Within
  one run, a paper gets one vote however many of its chunks were hit.
