# configs/ の使い方

## コンセプト

前処理・検索手法・エージェント・共有パスは、それぞれ独立に差し替え可能な4つの軸として分離されている。
1つのyamlに全部まとめず、**4フォルダから1ファイルずつ選んで組み合わせて使う**。

```
configs/
├── paths/            共有パス（pdf_dir, index_dirのルート等）
├── process_style/    前処理（Preprocessor）
├── search_style/     検索手法（Indexer群 + Fuser + Reranker）
└── agent_style/      エージェント（Agent）
```

これは `src/littraceqa/di_pipeline/` 側のDI設計（`registry.py` で `@register(kind, name)` したクラスを
`registry.build(kind, name, **params)` で組み立てる仕組み）をそのままconfigの
ファイル単位に反映したもの。`src/littraceqa/di_pipeline/config.py` の `compose_config()` が4つの
dictを合成し、`build_pipeline()` に渡す。

## なぜ分けているか

- **前処理を差し替えても、検索手法やエージェントの設定を書き直さなくていい**
- **同じ検索手法(search_style)を別の前処理(process_style)と組み合わせても、索引の保存先が衝突しない**
  - `process_style`/`search_style` のファイルには `pdf_dir`/`index_dir` を書かない
  - `compose_config()` が `paths` から `{index_dir}/{process名}/{indexer名}` のように自動導出する
  - 例: `mineru + bm25s` → `index/mineru/bm25s`（前処理ごとに別物として保存される）
- 新しい手法を1つ追加したいだけなのに、既存の組み合わせファイルを全部複製・修正する必要がない

## 使い方

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_external_all.yaml \
  --agent configs/agent_style/reading_expand_rrf/notable.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --production-input
  # --build は初回のみ（前処理+索引構築）。索引は構築済みなので通常不要
```

組み合わせを変えたいときは、該当する引数だけ差し替える。他の3つはそのままでよい。

```bash
# 検索手法だけ論文の (c) Base A+B に変える（エージェントは同じ）
  --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml
```

4フォルダのファイルはどう組み合わせても壊れない設計なので、新しいyamlを
書く必要があるのは「まだ存在しない前処理・検索手法・エージェント自体」を
追加するときだけ。

## 現在のファイル一覧

```
configs/
├── paths/
│   ├── default.yaml
│   └── nlp02.yaml
├── process_style/
│   └── mineru.yaml           : MinerU。事前に scripts/run_mineru.py で変換が必要（構築済み）
├── search_style/
│   └── bm25_qwen3_8b_rerank_qwen3_8b/ : 埋め込みと reranker を両方 8B にした主力系列
│       ├── chunk_attrfilter_k100.yaml : **論文の (c) Base A+B。** 2索引
│       │                       （bm25s + faiss_qwen3_8b）、chunk 単位 RRF、reranker は
│       │                       順位を完全に置換。per_index_k=100 / pool_k=200 + 属性フィルタ
│       └── k100_external_all.yaml : **論文の (d) Final system（推奨デフォルト）。**
│                               上記との差は4点だけ——bm25s_paper の追加 / paper_rrf /
│                               seed_expansion / rerank_blend。モデル・k・属性フィルタは同一
└── agent_style/
    ├── reading_expand_rrf/
    │   └── notable.yaml      : **(c)(d) 共通のエージェント。** 分解→読解→再検索を繰り返し、
    │                           最後に質問起点のランキングA と論文間展開のランキングB を
    │                           RRF 統合する（combine_rrf_k=10）。anchor_from: verdict と
    │                           paper_score_skip_chunk_types: [table] を含む
    └── agentless/
        └── agentless.yaml    : **論文の (b) No search agent。** LLM を1回も呼ばない。
                                scripts/eval_retrieval.py --agent に渡して使う
```

`agent_style` は agent が `reading` 一本なので、分けているのは**どの任意キーを使うか**。
フォルダ名がその軸で、フォルダの中では `reading_` 接頭辞を付けない。各ファイルの実測値と
選定理由は yaml 冒頭のコメントと `CLAUDE.md` を参照。

**このブランチには論文の4構成に要る yaml だけを置いてある。** 選定の過程で試して不採用に
なった構成（幅を振っただけの版、位置挿入、consensus、クエリ書き換えなど）は
`iseakira/paper-ablation` にある。

`search_style` のファイル名は `bm25_{埋め込み}_{サイズ}[_rerank_{reranker}_{サイズ}]`。
reranker を使う構成は末尾に reranker のモデルとサイズを書いて、埋め込みと reranker の
どちらを何Bにした版かをファイル名だけで見分けられるようにしている
（各構成のモデル名と主要パラメータの一覧は `CLAUDE.md` の「3. configs/ のディレクトリ構成」、
選定理由は各 yaml 冒頭のコメントを参照）。

**同じ土台から派生した系列だけフォルダに畳んである。** `agent_style` と同じく
`config_label()`（`src/littraceqa/common.py`）が `{フォルダ名}_{stem}` に畳むので、
**フォルダ名には既存ファイル名の接頭辞をそのまま使い**、その系列の素の構成には
フォルダ名の末尾の語をファイル名として付ける（命名規則の例:
`reading_expand_rrf/rrf.yaml` -> ラベル `reading_expand_rrf`）。こうすると畳む前と実験ラベルが1文字も変わらないので、
過去の `report/*.md` や `results/experiments.jsonl` と同じ名前で並べて読める。

推奨デフォルトの組み合わせ（= 論文の (d)）: `process_style/mineru.yaml` + `search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_external_all.yaml` + `agent_style/reading_expand_rrf/notable.yaml`

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
