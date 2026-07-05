# configs/ の使い方

## コンセプト

前処理・検索手法・エージェント・共有パスは、それぞれ独立に差し替え可能な4つの軸として分離されている。
1つのyamlに全部まとめず、**4フォルダから1ファイルずつ選んで組み合わせて使う**。

```
configs/
├── paths/           共有パス（pdf_dir, index_dirのルート等）
├── process_style/    前処理（Preprocessor）
├── search_style/     検索手法（Indexer群 + Fuser + Reranker）
└── agent_style/       エージェント（Agent）
```

これは `litqa/` 側のDI設計（`registry.py` で `@register(kind, name)` したクラスを
`registry.build(kind, name, **params)` で組み立てる仕組み）をそのままconfigの
ファイル単位に反映したもの。`litqa/config.py` の `compose_config()` が4つの
dictを合成し、`build_pipeline()` に渡す。

## なぜ分けているか

- **本文チャンク(pypdf)を図表チャンク(figure_vlm)に差し替えても、検索手法やエージェントの設定を書き直さなくていい**
- **同じ検索手法(search_style)を別の前処理(process_style)と組み合わせても、索引の保存先が衝突しない**
  - `process_style`/`search_style` のファイルには `pdf_dir`/`index_dir` を書かない
  - `compose_config()` が `paths` から `{index_dir}/{process名}/{indexer名}` のように自動導出する
  - 例: `pypdf + bm25s` → `index/pypdf/bm25s`、`figure_vlm + bm25s` → `index/figure_vlm/bm25s`（別物として保存される）
- 新しい手法を1つ追加したいだけなのに、既存の組み合わせファイルを全部複製・修正する必要がない

## 使い方

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/pypdf.yaml \
  --search configs/search_style/bm25_qwen3.yaml \
  --agent configs/agent_style/simple.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build   # 初回のみ（前処理+索引構築）。2回目以降は外す
```

組み合わせを変えたいときは、該当する引数だけ差し替える。他の3つはそのままでよい。

```bash
# 検索手法だけColBERTに変える
  --search configs/search_style/bm25_colbert.yaml

# 前処理を図表チャンク(figure_vlm)に変える
  --process configs/process_style/figure_vlm.yaml
```

4フォルダのファイルはどう組み合わせても壊れない設計なので、新しいyamlを
書く必要があるのは「まだ存在しない前処理・検索手法・エージェント自体」を
追加するときだけ。

## 現在のファイル一覧

```
configs/
├── paths/
│   └── default.yaml
├── process_style/
│   ├── pypdf.yaml            : PDFをページ単位でチャンク化
│   └── figure_vlm.yaml       : Docling+Qwen2-VLで図表をチャンク化
├── search_style/
│   ├── bm25.yaml             : BM25 単体
│   ├── bm25_qwen3.yaml       : BM25 + Qwen3-Embedding-8B（デフォルト）
│   ├── bm25_colbert.yaml     : BM25 + ColBERT
│   ├── bm25_specter2.yaml    : BM25 + SPECTER2
│   └── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-8B + SigLIP（図表画像を直接embedding）
└── agent_style/
    ├── simple.yaml           : 1回検索して終わり（LLM不使用）
    └── iterative.yaml        : 見てから次を決める反復検索（LLM使用）
```

推奨デフォルトの組み合わせ: `process_style/pypdf.yaml` + `search_style/bm25_qwen3.yaml` + `agent_style/simple.yaml`

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
