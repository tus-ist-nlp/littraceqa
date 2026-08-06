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
実行時に4つから1ファイルずつ選んで組み合わせる（`src/littraceqa/di_pipeline/config.py` の
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
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
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
    └── reading.yaml      : 分解→読解→不足分の再検索を繰り返す唯一の本命。evidence も埋める（デフォルト）
```

`iterative.yaml` / `reading_llmcount.yaml` / `simple.yaml` / `verifying.yaml` は削除済み
（`iterative` は停止条件が「見つかった論文の本数」で top_k=20 の時点で初回から満たされ、
反復ループが事実上空回りしていた）。以後 agent_style は `reading` 一本で運用する。

**`reading` の打ち切りは task_family に依存しない。** 反復検索の停止条件は
`_read_and_judge()` が返す LLM の `sufficient` 判定のみ（`src/littraceqa/di_pipeline/agent/reading.py`）。
提出本数も `paper_cutoff: llm` にしてあるので、LLM が「これで十分」と判断した時点の
選定をそのまま出す（`max_papers: 10` で頭打ち）。本番入力に `task_family` が無く、
推定しても正解率0.67程度で当てにならないため、本数決定の経路から task_family を外した。

### 3.1 評価の作法

**評価には `scripts/sync_official_release.py` が取得・検証した
`artifacts/official_release/<revision>/data/validation_inputs.jsonl` を使う。** 現行入力は
`query_id` / `benchmark` / `question` / `answer_types` と、回答形式に応じた
`multiple_choice_options` / `table_schema` から成る。`run_search.py` は既定でこの本番契約へ
投影するため `--production-input` の明示は不要。`--include-development-fields` は古い実験の
再現専用で、`task_family` / `primary_evidence_type` を含む結果を本番相当として比較しない。

結果は `results/experiments.jsonl` に自動で追記される（config名 + metrics + timestamp）。
加えて実行1回につき、設定と指標とLLMコメントをまとめた Markdown が
`report/{timestamp}_{process名}_{search名}_{agent名}.md` として1枚書き出される
（`scripts/run_search.py` の `write_report()`）。`results/` `report/` はどちらも
各自のローカルな実行記録で `.gitignore` 対象（チーム共有はしない、生成物なので消えても
再実行すれば復元できる）。
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
