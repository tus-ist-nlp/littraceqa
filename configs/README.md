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

これは `src/littraceqa/di_pipeline/` 側のDI設計（`registry.py` で `@register(kind, name)` したクラスを
`registry.build(kind, name, **params)` で組み立てる仕組み）をそのままconfigの
ファイル単位に反映したもの。`src/littraceqa/di_pipeline/config.py` の `compose_config()` が4つの
dictを合成し、`build_pipeline()` に渡す。

## なぜ分けているか

- **本文チャンク(mineru)を図表チャンク(figure_vlm)に差し替えても、検索手法やエージェントの設定を書き直さなくていい**
- **同じ検索手法(search_style)を別の前処理(process_style)と組み合わせても、索引の保存先が衝突しない**
  - `process_style`/`search_style` のファイルには `pdf_dir`/`index_dir` を書かない
  - `compose_config()` が `paths` から `{index_dir}/{process名}/{indexer名}` のように自動導出する
  - 例: `mineru + bm25s` → `index/mineru/bm25s`、`figure_vlm + bm25s` → `index/figure_vlm/bm25s`（別物として保存される）
- 新しい手法を1つ追加したいだけなのに、既存の組み合わせファイルを全部複製・修正する必要がない

## 使い方

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/abstract_specter2_body_qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries artifacts/official_release/bd35dc14cf0483e0ffa51fa2a54d2689c13f9845/data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build   # 初回のみ（前処理+索引構築）。2回目以降は外す（mineruは構築済みなので通常不要）
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
│   ├── marker.yaml           : PDFをブロック単位でチャンク化
│   ├── mineru.yaml           : MinerU。事前に scripts/run_mineru.py で変換が必要（デフォルト、構築済み）
│   └── figure_vlm.yaml       : Docling+Qwen2-VLで図表をチャンク化
├── search_style/
│   ├── bm25.yaml             : BM25 単体
│   ├── bm25_qwen3.yaml       : BM25 + Qwen3-Embedding-8B
│   ├── bm25_colbert.yaml     : BM25 + ColBERT
│   ├── bm25_specter2.yaml    : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-8B + SigLIP（図表画像を直接embedding）
│   └── abstract_specter2_body_qwen3.yaml : BM25 + SPECTER2(title_abstractのみ) +
│         Qwen3-Embedding-0.6B(本文のみ)。各モデルを設計どおりの粒度で使う（デフォルト、構築済み）
└── agent_style/
    ├── reading.yaml          : 検索器と一体の旧reader
    ├── aoai_pairwise_reader.yaml : 各query-paper pairを1回のbase AOAI callで判定→根拠回答（読解本命）
    └── aoai_pairwise_reader_hybrid.yaml : 旧ファイル名互換（現在は上と同じ1回判定）
```

読解本命は `scripts/run_aoai_pairwise_reader.py` と
`agent_style/aoai_pairwise_reader.yaml` を使う。PR #7の固定候補を読むだけで、
この経路はDI・検索・rerank・再検索を行わない。論文を複数ファイルや
複数promptへ分割せず、長い論文は単一コンテキストに圧縮して、1つの
query-paper pairを1回のbase AOAI callで判定する。JSON修復・画像policy拒否時の
text-only fallback・provider retryは失敗時のみで、論文分割のための追加callは行わない。
最終prompt全体も `max_judgment_prompt_chars` で検査し、超える場合はAOAIを
呼ぶ前に同じ単一contextをさらに決定的に圧縮する。

公式の実行範囲は、開発用 `validation_inputs.jsonl` が55問、必須の本番
`test.jsonl` が71問である。`test_extra.jsonl` の4,901問は任意の診断用であり、
通常の本番実行に結合しない。5,027問を1つの必須データセットとして
AOAIに送る運用はしない。
検索比較だけを行う場合は従来どおり `scripts/run_search.py` を使う。

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
