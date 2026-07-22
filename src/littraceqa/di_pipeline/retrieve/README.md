# src/littraceqa/di_pipeline/retrieve/

複数Indexerの検索結果を統合する層。`HybridRetriever` が全体を束ね、`Fuser`/`Reranker`は差し替え可能。

- `base.py` — `Fuser`/`Reranker`/`Retriever` Protocol
- `hybrid.py` — `HybridRetriever`: 各Indexerで検索→`Fuser`で統合→（あれば）`Reranker`で再ランク。`to_gold_papers()`でchunk単位の結果をpaper_id単位のランキングに集約
- `rrf.py` — `RRFFuser`（"rrf"）: Reciprocal Rank Fusionで複数ランキングを1つに統合
- `reranker.py` — `NoneReranker`（"none"）: rerankerを使わない場合のplaceholder
