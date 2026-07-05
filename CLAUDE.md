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

実行例:
```
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/pypdf.yaml \
  --search configs/search_style/bm25_qwen3.yaml \
  --agent configs/agent_style/simple.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build
```

### 2. 推奨デフォルトの組み合わせ
新しい手法をデフォルト（推奨組み合わせ）にする場合は、この節の記載を更新する。
ablation 用なら触らない。

現在のデフォルト: `process_style/pypdf.yaml` + `search_style/bm25_qwen3.yaml`
+ `agent_style/simple.yaml`（pypdf + BM25 + Qwen3-Embedding-8B + SimpleAgent）

### 3. configs/ のディレクトリ構成
現在の構成:
```
configs/
├── paths/
│   └── default.yaml
├── process_style/
│   ├── pypdf.yaml
│   └── figure_vlm.yaml
├── search_style/
│   ├── bm25.yaml            : BM25 単体
│   ├── bm25_qwen3.yaml      : BM25 + Qwen3-Embedding-8B（デフォルト）
│   ├── bm25_colbert.yaml    : BM25 + ColBERT
│   ├── bm25_specter2.yaml   : BM25 + SPECTER2
│   └── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-8B + SigLIP（図表画像を直接embedding、ablation用）
└── agent_style/
    ├── simple.yaml
    ├── iterative.yaml
    └── verifying.yaml    : 上位候補をLLMに判定させ、順位カットオフでなく内容ベースで最終提出論文を選ぶ
```

### 4. registry への登録確認
@register("indexer", "xxx") のデコレータが付いているか確認する。
付いていないと config から呼び出せない。
