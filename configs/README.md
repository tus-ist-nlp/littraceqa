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
  --search configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build   # 初回のみ（前処理+索引構築）。2回目以降は外す（mineruは構築済みなので通常不要）
```

組み合わせを変えたいときは、該当する引数だけ差し替える。他の3つはそのままでよい。

```bash
# 検索手法だけ論文単位BM25の併用に変える
  --search configs/search_style/bm25_dual.yaml
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
├── search_style/             （全ファイルが indexer に bm25s / bm25s_paper を含むので
│   │                          全ファイル名が bm25_ で始まる。後ろは「土台に何を足すか」）
│   ├── bm25.yaml             : BM25 単体（chunk単位）
│   ├── bm25_paper.yaml       : BM25 単体（論文単位、ablation）
│   ├── bm25_dual.yaml        : chunk BM25 + paper BM25 の RRF（ablation、GPU不要）
│   ├── bm25_dual_qwen3_8b_rerank_qwen3_8b.yaml : 8B 構成に bm25s_paper を足した版
│   ├── bm25_qwen3.yaml       : BM25 + Qwen3-Embedding-0.6B
│   ├── bm25_qwen3_0.6b_rerank_qwen3_0.6b.yaml : 上記 + Qwen3-Reranker-0.6B（2索引ablation）
│   ├── bm25_specter2.yaml    : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_specter2_body_qwen3/ : 各モデルを設計どおりの粒度で使う3索引（0.6B 系列）
│   │   ├── qwen3.yaml        : BM25 + SPECTER2(title_abstractのみ) + Qwen3-0.6B(本文のみ)
│   │   ├── attrfilter.yaml   : 上記 + 会議名・年による属性フィルタ（追加構築なし）
│   │   └── rerank_qwen3_0.6b.yaml : 上記 + Qwen3-Reranker-0.6B（ablation）
│   └── bm25_qwen3_8b_rerank_qwen3_8b/ : 主力系列。埋め込みと reranker を両方 8B に
│       │                       （索引構築が4GPUで約31時間と重い）
│       ├── 8b.yaml           : 素の8B構成（per_index_k=pool_k=100、属性フィルタなし）
│       ├── chunk_attrfilter.yaml : 上記 + 属性フィルタ。per_index_k=pool_k=1000
│       ├── chunk_attrfilter_blend.yaml : 上記 + rerank_blend（索引・reranker は同一）
│       ├── chunk_attrfilter_k100.yaml : **論文の (c) Base A+B。** per_index_k=100 / pool_k=200
│       ├── chunk_attrfilter_k100_blend.yaml : 上記 + rerank_blend（幅違いの対照）
│       ├── paper_attrfilter.yaml : 論文単位 rerank（qwen3_paper, 3GPU）にした版
│       ├── k100_paperbm25.yaml : k100 + bm25s_paper 索引（融合はチャンク単位のまま）
│       ├── k100_paperrrf.yaml : 上記の融合を論文単位RRF にした版
│       ├── k100_seed.yaml    : k100 + Seed Expansion
│       ├── k100_paperrrf_seed.yaml : 上記2つを両方入れた版
│       └── k100_external_all.yaml : **論文の (d) Final system。** 上記3機構 + rerank_blend
└── agent_style/              （agent は reading 一本なので、分けているのは
    │                          どの任意キーを使うか。フォルダ内は reading_ 接頭辞を付けない）
    ├── reading.yaml          : 任意キーを1つも使わない素の構成
    ├── agentless/
    │   └── agentless.yaml    : **論文の (b) No search agent。** LLM を1回も呼ばない
    └── reading_expand_rrf/   : 論文→論文展開を**順位融合**する系列（ランキングA と B を
        │                      RRF 統合して候補列を作り直す）
        ├── rrf.yaml          : 3ソース（SPECTER2 / 書誌結合 / 全文MLT）統合
        ├── rrfk10.yaml       : 上記 + combine_rrf_k 60->10 / neighbors 50->100（セット）
        ├── verdict_anchor.yaml : 上記 + anchor_from: verdict
        ├── notable.yaml      : **論文の (c)(d) が使うエージェント。**
        │                       上記 + paper_score_skip_chunk_types: [table]
        ├── steps2_notable.yaml : 上記 + max_steps: 2（フル走行で不採用）
        ├── rawq_pin.yaml     : notable + 生質問1位のピン留め
        ├── cand50.yaml       : 候補幅を 50 に広げた版
        ├── grounded.yaml     : 反復ループに候補上位を渡す版
        ├── rewrite.yaml      : 検索結果に接地したクエリ書き換え（実測で要改修）
        ├── stacked.yaml      : rrf + 反復ループの3キー（いま足せるものを全部足した構成）
        └── consensus.yaml    : anchor ごとに B を分ける版（実測で不採用）
```

`agent_style` は agent が `reading` 一本なので、分けているのは**どの任意キーを使うか**。
フォルダ名がその軸（幅だけ / expansion / 反復ループ拡張）で、フォルダの中では
`reading_` 接頭辞を付けない。各ファイルの実測値と選定理由は yaml 冒頭のコメントと
`CLAUDE.md` を参照。

`search_style` のファイル名は `bm25_{埋め込み}_{サイズ}[_rerank_{reranker}_{サイズ}]`。
reranker を使う構成は末尾に reranker のモデルとサイズを書いて、埋め込みと reranker の
どちらを何Bにした版かをファイル名だけで見分けられるようにしている
（各構成のモデル名と主要パラメータの一覧は `CLAUDE.md` の「3. configs/ のディレクトリ構成」、
選定理由は各 yaml 冒頭のコメントを参照）。

**同じ土台から派生した系列だけフォルダに畳んである。** `agent_style` と同じく
`config_label()`（`src/littraceqa/common.py`）が `{フォルダ名}_{stem}` に畳むので、
**フォルダ名には既存ファイル名の接頭辞をそのまま使い**、その系列の素の構成には
フォルダ名の末尾の語をファイル名として付ける（`reading_expand_rrf/rrf.yaml` ->
ラベル `reading_expand_rrf`）。こうすると畳む前と実験ラベルが1文字も変わらないので、
過去の `report/*.md` や `results/experiments.jsonl` と同じ名前で並べて読める。

推奨デフォルトの組み合わせ（= 論文の (d) Final system）: `process_style/mineru.yaml` + `search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_external_all.yaml` + `agent_style/reading_expand_rrf/notable.yaml`

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
