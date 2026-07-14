# 検索方式と評価条件

現在の比較は、ビルド済みMinerU v1の共通Chunkを使い、gold annotationを見ずに
rankingを作成します。goldは`scripts/evaluate_retrieval_rankings.py`だけが読みます。

## 比較する設定

| 設定 | 検索単位 | 用途 |
| --- | --- | --- |
| `bm25.yaml` | Chunk | 最小の疎検索 |
| `bm25_paper_rank_rrf_fill_to_top_k.yaml` | Chunk + 論文全文 | PaperRank baseline |
| `bge_m3_title_abstract.yaml` | 1論文1件のtitle/abstract | Dense検索の単独確認 |
| `bm25_paper_bge_m3_rrf.yaml` | 上記3系統 | 現在の最良hybrid |

PaperRank RRFは各runをpaper ID単位の順位へ変換して融合します。
`fill_to_top_k: true`は、融合順位を保ちながら、入力runに存在する未採用paperで
指定cutoffまで補完します。

BGE-M3は共通Chunkの`text`をそのまま使う`common_chunk`形式だけを比較対象とします。
対象は各論文で観測された1件の`title_abstract` Chunk、固定revision
`5617a9f61b028005a4858fdac845db406aefb181`、最大512 token、batch size 1、CPU、
L2正規化、exact NumPy内積検索です。モデルは`local_files_only: true`で読み込み、
実行中にはダウンロードしません。

## Controlled 100-paper diagnostic

次の値は、正解論文をすべて含む100論文と55 validation質問による限定診断です。
本番の27,487論文検索より容易であり、同じ改善幅を本番へ一般化できません。

| 方法 | 単一gold R@5 / R@10 / R@20 | 複数gold R@5 / R@10 / R@20 |
| --- | --- | --- |
| PaperRank baseline | 1.000 / 1.000 / 1.000 | 0.778 / 0.879 / 0.914 |
| BM25 + Paper BM25 + BGE-M3 | 1.000 / 1.000 / 1.000 | 0.778 / 0.905 / 0.931 |

hybrid全体のRecall@5/10/20は`0.883 / 0.950 / 0.964`でした。この結果から、
BGE-M3はこの限定コーパスでは特に複数gold問題の10〜20位の回収を補っています。
単一goldでは既に飽和しているため、改善は確認できません。

## Controlled 200-paper diagnostic

100論文の集合を保ったまま、メタデータ順の非gold論文100件を追加した
200論文条件でも、同じ55問を評価しました。これはコーパスが難しくなったときの
傾向を見るための限定診断です。追加したdistractorはACL論文へ偏っているため、
会議比率を揃えたコーパスの代用にはなりません。

| 方法 | 単一gold R@5 / R@10 / R@20 | 複数gold R@5 / R@10 / R@20 |
| --- | --- | --- |
| PaperRank baseline | 1.000 / 1.000 / 1.000 | 0.723 / 0.845 / 0.888 |
| BM25 + Paper BM25 + BGE-M3 | 0.962 / 1.000 / 1.000 | 0.735 / 0.853 / 0.914 |

hybridは複数goldの平均Recallをすべてのcutoffで改善しましたが、すべてのgoldを
回収できた問題の割合は、@10で`0.690`から`0.655`、@20で`0.759`から
`0.724`へ低下しました。そのため、現時点でhybridを常に優先するのではなく、
平均Recallと全gold回収率を両方報告します。

## BGE-M3の限定再現

事前に、同じ100論文から構築したChunk BM25索引、論文単位BM25索引、MinerU Chunk、
固定revisionのBGE-M3 snapshotを準備します。これらは外部入力です。

```bash
export MINERU_CHUNKS=/path/to/frozen-mineru-v1-chunks.jsonl
export CHUNK_INDEX=/path/to/matching-chunk-bm25-index
export PAPER_INDEX=/path/to/matching-paper-bm25-index
export BGE_MODEL_SNAPSHOT=/path/to/bge-m3-snapshot/5617a9f61b028005a4858fdac845db406aefb181
export OUTPUT_ROOT="$HOME/littraceqa_data/bge_m3_eval"

env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run python scripts/run_bge_m3_retrieval.py \
  --chunks "$MINERU_CHUNKS" \
  --chunk-index-dir "$CHUNK_INDEX" \
  --paper-index-dir "$PAPER_INDEX" \
  --model-path "$BGE_MODEL_SNAPSHOT" \
  --queries data/validation_inputs.jsonl \
  --output-dir "$OUTPUT_ROOT/rankings" \
  --dense-work-dir "$OUTPUT_ROOT/index" \
  --query-limit 2
```

2問のsmoke testで時間とメモリを確認してから、同じ引数で`--query-limit`を外し、
新しい出力ディレクトリまたは整合する`--resume`を使います。runnerは事前宣言した
100論文または200論文のみを受け付け、55問に上限を設け、gold入力を受け取りません。
200論文条件では`--paper-count 200`を追加します。

## 単一gold／複数gold評価

```bash
uv run python scripts/evaluate_retrieval_rankings.py \
  --gold data/validation.jsonl \
  --rankings-dir "$OUTPUT_ROOT/rankings/rankings" \
  --output-dir "$OUTPUT_ROOT/evaluation" \
  --baseline mineru_v1_paper_rank_rrf_fill20_d100
```

グループは各質問のgold paper数が1件か複数件かで決め、validation専用の
`task_family`には依存しません。Recall@5/10/20、Paper F1、全gold回収率、
改善・悪化した質問、正解paperの順位変化を同じcutoffで比較します。検索方式の比較では
corpus checksum、query checksum、候補深さ、出力件数、評価コードを固定してください。
