# src/littraceqa/di_pipeline/retrieve/

複数Indexerの検索結果を統合する層。`HybridRetriever` が全体を束ね、`Fuser`/`Reranker`は差し替え可能。

- `base.py` — `Fuser`/`Reranker`/`Retriever` Protocol
- `hybrid.py` — `HybridRetriever`: 各Indexerで検索→`Fuser`で統合→（あれば）`Reranker`で再ランク。`to_gold_papers()`でchunk単位の結果をpaper_id単位のランキングに集約。`rerank_blend`（既定オフ）を渡すと reranker に順位を置き換えさせず、融合前の順位と RRF で混ぜ、`protect_top` 件の集合を保護する。融合結果は `score` に書き戻す（下流が score で並べ直すため）
- `paper_expander.py` — 論文→論文展開（`specter2` / `bib_coupling` / `bm25_mlt` / `fused`）。`rank(anchors)` が「渡された起点の近傍を関連度順に返す」。**起点を決めるのも A/B 統合の設定を持つのも `ReadingAgent` 側**（`CombineConfig`）
- `attribute_filter.py` — `AttributeExtractor` / `AttributeFilter`: 質問が明示した会議名・年（「Which NAACL 2025 papers ...」）を取り出して検索結果を絞る。索引は無改修（`RetrievalResult.metadata`にvenue/yearが既に載っているため）。会議名が一意に取れたときだけ発火し、取れなければ従来と完全に同一の動作に戻る。`LLMAttributeExtractor` は正規表現が空のときだけ LLM に判定させる後段（`llm_extract: true` で有効、既定オフ）。返答はコーパスに実在する (会議名, 年) の組しか採用しない
- `rrf.py` — `RRFFuser`（"rrf"）: Reciprocal Rank Fusionで複数ランキングを1つに統合
- `reranker.py` — `NoneReranker`（"none"）: rerankerを使わない場合のplaceholder
