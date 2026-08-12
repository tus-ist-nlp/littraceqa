# src/littraceqa/di_pipeline/index/

Chunk列から索引を構築し、クエリに対して `RetrievalResult` を返すIndexer実装。

- `bm25_index.py` — `BM25Index`（"bm25s"）: bm25sライブラリによる疎検索（chunk単位）
- `bm25_paper_index.py` — `BM25PaperIndex`（"bm25s_paper"）: bm25sライブラリによる疎検索（論文単位、ablation）。chunkをpaper_idごとに連結して1論文=1ドキュメントとして扱う。chunk_idは擬似ID（`"{paper_id}#paper"`）でevidence用途には使わない
- `faiss_qwen3.py` — `Qwen3FAISSIndex`（"faiss_qwen3"）: Qwen3-Embeddingでベクトル化しFAISSで検索するdense検索
- `faiss_azure_openai.py` — `AzureOpenAIFAISSIndex`（"faiss_azure_openai"）: Azure OpenAIの埋め込みAPI（text-embedding-3-large）でベクトル化しFAISSで検索するdense検索。ローカルGPU不要な代わりにAPI課金が発生する（全chunk埋め込みで概算$75〜100）
- `faiss_specter2.py` — `Specter2FAISSIndex`（"faiss_specter2"）: SPECTER2埋め込みによるdense検索
- `colbert_index.py` — `ColBERTIndex`（"colbert"）: PyLate/ColBERTによるlate interaction検索（トークン単位で精密だが重い）
- `siglip_image.py` — `SiglipImageIndex`（"siglip_image"）: figure_vlmが保存した図表画像（metadata["image_path"]）をSigLIPのvision encoderでベクトル化するdense検索。クエリはSigLIPのtext encoderでベクトル化するため、bm25s/faiss_qwen3とは別の埋め込み空間になる（RRFFuserで順位ベースに統合するため併用可能）
