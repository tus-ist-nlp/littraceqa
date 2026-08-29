# src/littraceqa/di_pipeline/index/

The indexers: each builds an index from a stream of Chunks and answers a query with
`RetrievalResult`s.

- `bm25_index.py` — `BM25Index` ("bm25s"): sparse retrieval over chunks, via the
  bm25s library.
- `bm25_paper_index.py` — `BM25PaperIndex` ("bm25s_paper"): the same, over whole
  papers. The chunks of a paper are joined so one paper is one document, which
  catches questions whose terms are scattered and never land in a single chunk. A
  hit here has the pseudo id `"{paper_id}#paper"` and is never used as evidence.
- `faiss_qwen3.py` — `Qwen3FAISSIndex` ("faiss_qwen3"): dense retrieval with
  Qwen3-Embedding over FAISS.
- `faiss_specter2.py` — `Specter2FAISSIndex` ("faiss_specter2"): dense retrieval
  with SPECTER2 embeddings. **Not a search index** — it is built only so that
  ranking B (`retrieve/paper_expander.py`) has neighbours to look up.
