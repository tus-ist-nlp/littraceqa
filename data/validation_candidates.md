# Validation candidate handoff

`validation_candidates.jsonl` is a physically sanitized copy of the ranked
paper search results from PR #7, commit
`a0206042354cd872e2cc3ab56c2434bd4c451009`.

- 55 query records
- 2,227 candidates in total; 3--50 per query (30 queries have 50)
- top-level schema: `query_id`, `candidate_papers`
- candidate schema: `rank`, `paper_id`, `title`, `venue`, `year`
- no `_gold`, answers, options, evidence, task family or primary evidence type
- SHA-256: `25298490f84c3180beee77b28d4f2fbda684589e6e4879e6d5c70f13c22e1cad`

It was generated with:

```bash
git show a020604:data/validation_with_candidates.jsonl \
  | uv run python scripts/export_candidate_handoff.py \
      --input - \
      --output data/validation_candidates.jsonl
```

For the current validation reading assignment this ranking is fixed. Do not
re-run DI, retrieval, reranking, or re-search. The downstream reader needs only:

- this sanitized per-query ranking;
- the student MinerU `mineru_chunks.jsonl`, used to hydrate each `paper_id`;
- the corresponding table/figure images; and
- the four organizer-confirmed query fields.

How a future hidden-test ranking is produced is a separate upstream concern and
is intentionally outside the reading/error-analysis workflow.
