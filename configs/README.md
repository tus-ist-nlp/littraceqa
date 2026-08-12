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
  --search configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build   # 初回のみ（前処理+索引構築）。2回目以降は外す（mineruは構築済みなので通常不要）
```

組み合わせを変えたいときは、該当する引数だけ差し替える。他の3つはそのままでよい。

```bash
# 検索手法だけColBERTに変える
  --search configs/search_style/bm25_colbert/colbert.yaml

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
│   ├── bm25.yaml             : BM25 単体（chunk単位）
│   ├── bm25_paper.yaml       : BM25 単体（論文単位、ablation）
│   ├── bm25_dual.yaml        : chunk BM25 + paper BM25 の RRF（ablation、GPU不要）
│   ├── bm25_dual_qwen3_8b_rerank_qwen3_8b.yaml : 8B 主力構成に bm25s_paper を足した版
│   ├── bm25_qwen3.yaml       : BM25 + Qwen3-Embedding-0.6B
│   ├── bm25_qwen3_0.6b_rerank_qwen3_0.6b.yaml : 上記 + Qwen3-Reranker-0.6B（2索引ablation）
│   ├── bm25_specter2.yaml    : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_azure_openai.yaml : BM25 + Azure OpenAI text-embedding-3-large（全chunk、ablation。API課金あり）
│   ├── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-0.6B + SigLIP（図表画像を直接embedding）
│   ├── bm25_qwen3_vl_8b_rerank_qwen3vl_8b.yaml : BM25 + Qwen3-Embedding-0.6B +
│   │     Qwen3-VL-Embedding-8B(図表画像) + Qwen3-VL-Reranker-8B。
│   │     図表を画像のまま扱う唯一の構成（隔離venv .venv-vl が必要）
│   ├── bm25_colbert/         : ColBERT 系列（遅延相互作用）
│   │   ├── colbert.yaml      : colbertv2.0 ベースライン
│   │   └── gte_modern.yaml   : GTE-ModernColBERT-v1（長コンテキスト版）
│   ├── bm25_specter2_body_qwen3/ : デフォルト系列。各モデルを設計どおりの粒度で使う3索引
│   │   ├── qwen3.yaml        : BM25 + SPECTER2(title_abstractのみ) +
│   │   │                       Qwen3-Embedding-0.6B(本文のみ)（デフォルト、構築済み）
│   │   ├── attrfilter.yaml   : 上記 + 会議名・年による属性フィルタ。
│   │   │                       「Which NAACL 2025 papers ...」のような質問で範囲を絞る（追加構築なし）
│   │   └── rerank_qwen3_0.6b.yaml : 上記 + Qwen3-Reranker-0.6B で候補プールを
│   │                           再ランキングしてから絞り込む版（ablation）
│   └── bm25_qwen3_8b_rerank_qwen3_8b/ : 主力系列。埋め込みと reranker を両方8Bに
│       │                       スケールアップ（索引構築が4GPUで約31時間と重い）
│       ├── 8b.yaml           : 素の8B構成（per_index_k=pool_k=100、属性フィルタなし）
│       ├── chunk_attrfilter.yaml : 上記 + 属性フィルタ。per_index_k=pool_k=1000
│       ├── chunk_attrfilter_blend.yaml : 上記 + rerank_blend（索引・reranker は同一）
│       ├── chunk_attrfilter_k100.yaml : chunk_attrfilter の per_index_k を100・pool_k を200に
│       ├── chunk_attrfilter_k100_blend.yaml : 上記 + rerank_blend（幅違いの対照）
│       └── paper_attrfilter.yaml : 論文単位 rerank（qwen3_paper, 3GPU）にした版
└── agent_style/
    ├── reading.yaml          : 分解→読解→不足分の再検索を繰り返す唯一の本命（デフォルト）
    ├── reading_normal/       : 任意キーなし。幅（retrieve_top_k / max_candidates）だけ振った版
    │   ├── topk50.yaml       : retrieve_top_k 50（LLM が読む本数は 20 のまま）
    │   ├── cand50.yaml       : retrieve_top_k / max_candidates とも 50
    │   └── fat.yaml          : retrieve_top_k 100。--dump-runs の土台も兼ねる
    ├── reading_expand_rrf/   : 論文→論文展開・**順位融合**（検索ランキングと関連
    │   │                     ランキングを RRF 統合して候補列を作り直す）
    │   ├── rrf.yaml          : 3ソース（SPECTER2 / 書誌結合 / 全文MLT）統合（現状のベスト）
    │   ├── cand50.yaml       : 上記の候補幅を 50 に広げた版
    │   ├── stacked.yaml      : rrf + 反復ループの3キー + title_protect
    │   │                     （いま足せるものを全部足した構成。実測はまだ無い）
    │   └── rel.yaml / consensus.yaml / protect.yaml : 実測で不採用（再評価用に保存）
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

**同じ土台から派生した3系列だけフォルダに畳んである。** `agent_style` と同じく
`config_label()`（`src/littraceqa/common.py`）が `{フォルダ名}_{stem}` に畳むので、
**フォルダ名には既存ファイル名の接頭辞をそのまま使い**、その系列の素の構成には
フォルダ名の末尾の語をファイル名として付ける（`bm25_colbert/colbert.yaml` ->
ラベル `bm25_colbert`）。こうすると畳む前と実験ラベルが1文字も変わらないので、
過去の `report/*.md` や `results/experiments.jsonl` と同じ名前で並べて読める。

推奨デフォルトの組み合わせ: `process_style/mineru.yaml` + `search_style/bm25_specter2_body_qwen3/qwen3.yaml` + `agent_style/reading.yaml`

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
