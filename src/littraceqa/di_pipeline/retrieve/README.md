# src/littraceqa/di_pipeline/retrieve/

複数Indexerの検索結果を統合する層。`HybridRetriever` が全体を束ね、`Fuser`/`Reranker`は差し替え可能。

- `base.py` — `Fuser`/`Reranker`/`Retriever` Protocol
- `hybrid.py` — `HybridRetriever`: 各Indexerで検索→`Fuser`で統合→（あれば）`Reranker`で再ランク。`to_gold_papers()`でchunk単位の結果をpaper_id単位のランキングに集約
- `rrf.py` — `RRFFuser`（"rrf"）: Reciprocal Rank Fusionで複数ランキングを1つに統合
- `reranker.py` — `NoneReranker`（"none"）: rerankerを使わない場合のplaceholder
- `seed_expansion/` — `SeedExpansionRetriever`（"seed_expansion"）: 別のRetrieverを包んで、
  上位1件を種にした再検索と各種の関係展開を重ねるラッパー

## seed_expansion/ の構成

検索1回が通る処理段階ごとにモジュールを分けている。`retriever.py` は引数検証と
呼び出し順序だけを持ち、実際の処理は各段階に委譲する。

- `retriever.py` — `SeedExpansionRetriever`: 引数検証・設定属性の保持・段階の組み立てと呼び出し順序
- `query.py` — クエリ/ヒント抽出。拡張クエリの生成と、method hint を検索レーンから外す `without_method_hints()`
- `candidates.py` — 初期候補生成。各レーンの検索と論文単位RRF融合、出力順とスコアを整合させる整列ヘルパ
- `relations.py` — 関係展開。論文間の明示的な言及（`PaperNeighborhoodExpansion`）、
  手法の所有者と手法間エッジ（`MethodRelationExpansion`）、最終1枠を使う `MethodBridgeExploration`
- `dense.py` — dense近傍展開。安定prefixの後ろを埋める `DenseTailFusion` と、
  最終1枠だけ使う `DenseReciprocalExploration` / `DenseConsensusExploration`
- `protection.py` — 候補保護。クエリが名指しした論文題名（`ExplicitTitleGuard`）と、
  method関係リランクが約束した通常検索結果を後段に落とさせない
- `final_rerank.py` — 最終リランク。確定した論文集合の**順序だけ**を入れ替える。
  検証に1つでも失敗したら元の順序へフォールバックするので、論文の増減は起きない
- `paper_index.py` — 各段階が使う `paper_bm25` indexer の探索を1本化した共通ヘルパ
