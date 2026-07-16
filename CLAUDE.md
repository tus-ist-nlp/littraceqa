# 開発ルール

## 言語
- 返答は必ず日本語で行うこと

## プロジェクト概要
- LitTraceQA コンペ（EMNLP 2026）の検索システム
- DI（依存性注入）設計で手法を差し替え可能にする
- contracts.py / registry.py / config.py が骨格

## コーディング規約
- Python 3.11+
- uv でパッケージ管理
- 型アノテーション必須

## 検索手法を追加するときのルール

新しい Indexer / Preprocessor / Agent を実装したときは、
必ず以下も合わせて作成・更新すること。

### 1. configs/ は4フォルダに分離されている
前処理・検索手法・エージェント・共有パスはそれぞれ独立したyamlファイルで、
実行時に4つから1ファイルずつ選んで組み合わせる（`litqa/config.py` の
`compose_config()` が合成する）。1ファイルに全部詰め込まない。

- `configs/paths/{名前}.yaml`: 実行環境ごとの共有パス（pdf_dir, index_dirのルート等）
- `configs/process_style/{preprocessor名}.yaml`: 前処理（`{name, params}`）
- `configs/search_style/{組み合わせ名}.yaml`: 検索手法（indexer群 + fuser + reranker）
- `configs/agent_style/{agent名}.yaml`: エージェント（`{name, llm?, params}`）

`process_style`/`search_style` のファイルには `pdf_dir`/`index_dir` を
**書かない**。`compose_config()` が `paths` から
`{index_dir}/{process名}/{indexer名}` のように自動導出する
（同じ `search_style` を別の `process_style` と組み合わせても
索引パスが衝突しないようにするため）。

**同じ indexer の別バリアントを使うときは `index_name` を必ず付ける。**
索引パスの末尾は既定で indexer 名なので、たとえば `faiss_specter2` を
「全チャンク版」と「abstractのみ版」で並べると、**同じパスを奪い合って
先に作った索引を上書きしてしまう**（数時間かけたビルドが消える）。
indexer エントリに `index_name: faiss_specter2_abstract` のように書けば
末尾だけが変わる。絶対パスを書かない方針はそのまま保てる。

```yaml
indexers:
  - name: faiss_specter2                      # -> {index_dir}/{process}/faiss_specter2
    params: {}
  - name: faiss_specter2
    index_name: faiss_specter2_abstract       # -> {index_dir}/{process}/faiss_specter2_abstract
    params: { chunk_types: [title_abstract] }
```

**`chunk_types` で indexer ごとに索引する粒度を変えられる。**
モデルには設計上の想定粒度がある。SPECTER2 の `proximity` アダプタは
title+abstract で学習された**論文単位**のモデルなので、本文の断片・表・数式を
個別に埋め込むのは学習時の入力分布から外れる。`chunk_types: [title_abstract]`
にすると設計どおりの使い方になる。省略すると全チャンクが対象。

実行例:
```
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/abstract_specter2_body_qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build
```

### 2. 推奨デフォルトの組み合わせ
新しい手法をデフォルト（推奨組み合わせ）にする場合は、この節の記載を更新する。
ablation 用なら触らない。

現在のデフォルト: `process_style/mineru.yaml` + `search_style/abstract_specter2_body_qwen3.yaml`
+ `agent_style/reading.yaml`（MinerU + BM25 + SPECTER2(title_abstract) + Qwen3-Embedding-0.6B(本文) + ReadingAgent）。
27,489件分の chunks・索引（`bm25s` / `faiss_specter2_abstract` / `faiss_qwen3_0p6b`）が
構築済みで、`--build` なしですぐ検索できる。

### 3. configs/ のディレクトリ構成
現在の構成:
```
configs/
├── paths/
│   └── default.yaml
├── process_style/
│   ├── marker.yaml
│   ├── mineru.yaml          : MinerU。事前に scripts/run_mineru.py で変換が必要（デフォルト、構築済み）
│   └── figure_vlm.yaml
├── search_style/
│   ├── bm25.yaml            : BM25 単体
│   ├── bm25_qwen3.yaml      : BM25 + Qwen3-Embedding-8B
│   ├── bm25_colbert.yaml    : BM25 + ColBERT
│   ├── bm25_specter2.yaml   : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-8B + SigLIP（図表画像を直接embedding、ablation用）
│   └── abstract_specter2_body_qwen3.yaml : BM25 + SPECTER2(title_abstractのみ) + Qwen3-Embedding-0.6B(本文のみ)。
│         各モデルを設計どおりの粒度で使う3索引構成（デフォルト、構築済み）
└── agent_style/
    ├── iterative.yaml    : multi_paper のときだけクエリ分解。※反復ループは空回りする（下記）
    ├── reading.yaml      : 分解→読解→不足分の再検索を繰り返す本命。evidence も埋める（デフォルト）
    └── reading_llmcount.yaml : reading から paper_cutoff だけ変えた ablation
```

`iterative` の反復ループは事実上回らない。停止条件が「見つかった論文の**本数**」で、
検索が返した論文が無条件に found に入るため、top_k=20 で引いた時点で初回から
条件を満たして打ち切られる（`_refine` は一度も呼ばれない）。中身を読んで根拠を確認し、
足りなければ本当に検索し直すのは `reading`。`iterative` は「LLM分解あり・検証なし」の
ベースラインとして残してある。

### 3.1 比較実験の作法

**共有ノブを揃えること。** `top_k` / `effort` / `paper_cutoff` / `max_papers` が
エージェント間でズレていると、「エージェントの賢さ」ではなく予算の差を測ってしまう。
現在は全 agent_style で `top_k: 20` / `effort: medium` / `paper_cutoff: task_family` /
`max_papers: 10` に揃えてある。新しい agent_style を足すときもここは揃える。

**提出本数は `paper_cutoff` で制御する。** 論文集合は F1 採点なので本数がスコアを支配する。
`task_family`（single=2/multi=5 で機械的に切る）に固定して比べれば、本数を揃えた上で
選定の質だけを比較できる。`llm`（LLMが選んだ本数をそのまま出す）はその効果を
単独で測るための ablation。

**評価は `--production-input` を付けて回す。** `data/validation_inputs.jsonl` は55件
すべてに `task_family` が入っているが、本番入力には無い（`query_id` / `question` /
`answer_types` / `table_schema` の4つだけ）。`task_family` は提出本数を決めるのに使うので、
与えたまま評価すると「正解を教えてもらった状態」の点数になり本番と乖離する。

差分を取ると効果が分解できる（`simple` / `verifying` は config を削除済みなので、
再現するには git history から復元するか同等の agent_style を作り直す）:

| 比較 | 分かること |
|---|---|
| simple → iterative | クエリ**分解**の効果 |
| simple → verifying | **読解・選定**の効果 |
| verifying → reading | **反復**の効果 |
| reading → reading_llmcount | 提出**本数をLLMに決めさせる**効果 |

結果は `results/experiments.jsonl` に自動で追記される（config名 + metrics + timestamp）。
LLM は非決定的（Opus 4.8 は temperature を受け付けない）でクエリは55件しかないので、
数ポイントの差はノイズの可能性がある。結論を出す前に複数回まわすこと。

### 4. 隔離 venv が必要な前処理
MinerU は本体と依存が両立しない（transformers / torch / requires-python が衝突）。
PDF → content_list.json の変換だけを隔離 venv で先に済ませ、本体の
`MinerUChunker` はその成果物を読むだけにする。

```
bash scripts/setup_mineru_env.sh   # 初回のみ（.venv-mineru を作りモデルを取得）
.venv-mineru/bin/python scripts/run_mineru.py \
  --paths configs/paths/default.yaml --gpus 0,1,2,3
```

出力先は `pdf_dir` の兄弟 `mineru/`（`process_style` yaml にはパスを書かない方針に従い、
`MinerUChunker` が自動導出する）。27,489件で 4GPU 約25時間。変換済みの論文は
飛ばすので、中断しても同じコマンドで再開できる。

### 5. registry への登録確認
@register("indexer", "xxx") のデコレータが付いているか確認する。
付いていないと config から呼び出せない。
