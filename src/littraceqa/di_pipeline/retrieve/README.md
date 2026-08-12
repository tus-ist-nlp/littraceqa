# src/littraceqa/di_pipeline/retrieve/

複数Indexerの検索結果を統合する層。`HybridRetriever` が全体を束ね、`Fuser`/`Reranker`は差し替え可能。

- `base.py` — `Fuser`/`Reranker`/`Retriever` Protocol
- `hybrid.py` — `HybridRetriever`: 各Indexerで検索→`Fuser`で統合→（あれば）`Reranker`で再ランク。`to_gold_papers()`でchunk単位の結果をpaper_id単位のランキングに集約。`rerank_blend`（既定オフ）を渡すと reranker に順位を置き換えさせず、融合前の順位と RRF で混ぜ、`protect_top` 件の集合を保護する。融合結果は `score` に書き戻す（下流が score で並べ直すため）
- `paper_expander.py` — 論文→論文展開（`specter2` / `bib_coupling` / `bm25_mlt` / `fused`）。`rank()` が「既存候補を落とさない関連度順」、`rank_pools()` が「anchor ごとに潰さない版」（`consensus`）、`expand()` が「追記すべき論文だけ」を返す
- `relation_graph.py` — コーパス内論文どうしの**明示的な関係**による展開。`title_mention`（A の本文に B の名前が出る）と `method_comention`（A と B が同じ論文の名前を挙げる）。索引は `scripts/build_relation_graphs.py` がコーパス1走査で作り、名指し保護（`agent/reading.py` の `title_protect`）とキャッシュを共有する
- `paper_titles.py` — 論文の「名前」の辞書（`TitleIndex`）。MinerU の分かち書き崩れ（`M o RE`）を英数字連結で吸収し、**大文字小文字を含めて一致**させる（小文字化すると `MoRE` が本文の `more` に当たって壊れる）。曖昧な名前はコーパスの一意性で、ハブ名は出現本数で落とす
- `attribute_filter.py` — `AttributeExtractor` / `AttributeFilter`: 質問が明示した会議名・年（「Which NAACL 2025 papers ...」）を取り出して検索結果を絞る。索引は無改修（`RetrievalResult.metadata`にvenue/yearが既に載っているため）。会議名が一意に取れたときだけ発火し、取れなければ従来と完全に同一の動作に戻る。`LLMAttributeExtractor` は正規表現が空のときだけ LLM に判定させる後段（`llm_extract: true` で有効、既定オフ）。返答はコーパスに実在する (会議名, 年) の組しか採用しない
- `rrf.py` — `RRFFuser`（"rrf"）: Reciprocal Rank Fusionで複数ランキングを1つに統合
- `reranker.py` — `NoneReranker`（"none"）: rerankerを使わない場合のplaceholder
