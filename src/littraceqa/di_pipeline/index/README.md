# src/littraceqa/di_pipeline/index/

Chunk列から索引を構築し、クエリに対して `RetrievalResult` を返すIndexer実装。

- `bm25_index.py` — `BM25Index`（"bm25s"）: bm25sライブラリによる疎検索（chunk単位）
- `bm25_paper_index.py` — `BM25PaperIndex`（"bm25s_paper"）: bm25sライブラリによる疎検索（論文単位、ablation）。chunkをpaper_idごとに連結して1論文=1ドキュメントとして扱う。chunk_idは擬似ID（`"{paper_id}#paper"`）でevidence用途には使わない
- `faiss_qwen3.py` — `Qwen3FAISSIndex`（"faiss_qwen3"）: Qwen3-Embeddingでベクトル化しFAISSで検索するdense検索
- `faiss_specter2.py` — `Specter2FAISSIndex`（"faiss_specter2"）: SPECTER2埋め込みによるdense検索
