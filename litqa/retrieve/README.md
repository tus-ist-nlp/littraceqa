# litqa/retrieve/

複数Indexerの結果を統合し、paper rankingを作る層です。

- `base.py` — `Fuser`、`Reranker`、`Retriever`のProtocol
- `hybrid.py` — `HybridRetriever`: 各Indexerで検索し、Fuserで統合し、任意の
  Rerankerを適用する。`to_gold_papers()`でChunk結果をpaper ID単位へ集約
- `rrf.py` — `RRFFuser`（`rrf`）: Reciprocal Rank Fusion
- `reranker.py` — `NoneReranker`（`none`）: rerankingを行わないplaceholder
- `paper_rank_rrf.py` — `PaperRankRRFFuser`（`paper_rank_rrf`）: Chunk BM25、
  論文単位BM25、BGE-M3などのrunをpaper ID単位の順位へ変換してRRF融合

`PaperRankRRFFuser`は既定ではChunk BM25が取得したpaper数を出力予算として保持します。
`fill_to_top_k: true`では融合順位を変えず、他の入力runにだけ存在するpaperを使って
指定件数まで補完します。これにより、同じFuserで疎検索baselineとBGE-M3 hybridを
比較できます。

検索ロジックは`task_family`、`primary_evidence_type`、正解paper数を参照しません。
単一gold／複数goldの区別は検索後の評価だけで行います。
