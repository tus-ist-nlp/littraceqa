# configs/ の使い方

## 構成

前処理・検索・エージェント・環境パスを独立したYAMLとして管理します。

```text
configs/
├── paths/          実行環境の入力・出力root
├── process_style/  Preprocessor
├── search_style/   Indexer、Fuser、Reranker
└── agent_style/    Agentと任意のLLM
```

`litqa/config.py`の`compose_config()`が4つの設定を合成します。同じ検索方式を別の
前処理へ適用しても、Chunkと索引はprocess名ごとのnamespaceへ分離されます。
環境や利用者ごとに変わる絶対パスはprocess/search YAMLへ書かず、paths YAML、
環境変数、またはCLI引数で渡してください。

## 現在の比較に使う設定

前処理:

- `pypdf.yaml`: PDF本文をページ単位で抽出
- `marker.yaml`: Markerで本文・図・表・数式を抽出
- `figure_vlm.yaml`: 図表画像と説明を抽出
- `mineru.yaml`: 既存MinerU `content_list.json`を読み込むv1
- `mineru_v2.yaml`: ページ単位の`content_list_v2.json`を読み込むv2

検索:

- `bm25.yaml`: 共通ChunkのBM25
- `bm25_paper_rank_rrf_fill_to_top_k.yaml`: Chunk BM25と論文単位BM25を
  paper ID単位で融合する、モデル不要のbaseline
- `bge_m3_title_abstract.yaml`: 1論文1件の`title_abstract`をBGE-M3で検索
- `bm25_paper_bge_m3_rrf.yaml`: BM25、論文単位BM25、BGE-M3をPaperRank RRFで融合

BGE-M3は固定revisionを`local_files_only: true`で読みます。cloneや`uv sync`では
snapshotを取得しないため、利用者が外部パスまたはローカルmodel cacheへ準備する
必要があります。

## MinerUを1〜3論文で確認する

```bash
export MINERU_ROOT=/path/to/read-only/mineru
export READ_ONLY_ROOT=/path/to/read-only-data
export ARTIFACT_ROOT="$HOME/littraceqa_data/mineru_eval/smoke"

uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_paper_rank_rrf_fill_to_top_k.yaml \
  --agent configs/agent_style/simple.yaml \
  --queries data/validation_inputs.jsonl \
  --output "$ARTIFACT_ROOT/predictions.jsonl" \
  --gold data/validation.jsonl \
  --build \
  --mineru-root "$MINERU_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --read-only-root "$READ_ONLY_ROOT" \
  --paper-id <paper-id-1> \
  --paper-id <paper-id-2> \
  --limit 2 \
  --workers 1 \
  --batch-size 1 \
  --resume
```

v2を比較する場合は、他の条件を変えず`--process`だけを
`configs/process_style/mineru_v2.yaml`へ変更し、別の`ARTIFACT_ROOT`を使います。
少数論文のsmoke testは変換と接続の確認であり、検索精度の主張には使いません。

## Bounded buildとresume

現在のrunnerは次の制約を持ちます。

- build時は正の`--limit`が必須
- 最大200論文まで。全件実行は拒否
- workerとbatch sizeの既定値は1
- `--artifact-root`は読み取り専用入力の外側に置く
- 論文ごとにChunk shardとstateを保存
- 設定、コード、依存版、入力checksumが一致するときだけresume
- 失敗したpaper IDを記録し、他の論文は続行

`--resume`でも入力checksum確認の読み取りI/Oは発生します。全コーパスへ広げる前に、
増分索引、shard、容量制限、CPU・メモリ制限を設計してください。

## 公平な比較

前処理または検索方式以外は固定します。少なくとも、paper集合、質問、BGE revision、
Chunk長、候補深さ、RRFの`k`、最終cutoff、評価コードを揃えます。gold annotationは
検索終了後の評価段階だけで読み、`task_family`と`primary_evidence_type`を検索に
使用しません。評価方法は[search_style/README.md](search_style/README.md)を
参照してください。
