# 開発ルール

## 言語
- 返答は必ず日本語で行うこと

## プロジェクト概要
- LitTraceQA コンペ（EMNLP 2026）の検索システム
- DI（依存性注入）設計で手法を差し替え可能にする
- contracts.py / registry.py / config.py が骨格
- **retrieval（indexer/search_style）の目的は gold paper（正解論文ID）の特定であり、
  根拠チャンクの特定（evidence）は ReadingAgent が別途担当する。** indexer を設計・
  追加するときは chunk 単位の粒度を保つことにこだわらなくてよく、論文単位の識別精度を
  優先してよい（例: `bm25s_paper` は論文全体を1ドキュメントとして扱い、
  `chunk_id` は `"{paper_id}#paper"` という擬似IDで evidence 用途には使わない）。

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

**`search_style` / `agent_style` は1階層だけサブフォルダを掘ってよい**（系列が増えて
一覧が読めなくなるため）。実験ラベルとレポート名は `config_label()`
（`src/littraceqa/common.py`）が `{フォルダ名}_{stem}` に畳む。畳む前のファイル名と
同じラベルになるよう、**フォルダ名は既存ファイル名の接頭辞をそのまま使い、
その系列の素の構成にはフォルダ名の末尾の語をファイル名として付ける**
（`bm25_colbert/colbert.yaml` -> `bm25_colbert`、
`bm25_colbert/gte_modern.yaml` -> `bm25_colbert_gte_modern`）。
新しい語を挟むとラベルが変わり、過去の `results/experiments.jsonl` や
監査HTMLの実験セレクタと名前が繋がらなくなる。

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
  --search configs/search_style/bm25_specter2_body_qwen3/qwen3.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build
```

### 2. 推奨デフォルトの組み合わせ
新しい手法をデフォルト（推奨組み合わせ）にする場合は、この節の記載を更新する。
ablation 用なら触らない。

現在のデフォルト: `process_style/mineru.yaml` + `search_style/bm25_specter2_body_qwen3/qwen3.yaml`
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
│   │   （全ファイルが indexer に bm25s/bm25s_paper を含むので全ファイル名が bm25_ で
│   │    始まる。bm25_ の後ろは「土台(BM25)に何を足すか」を表す。
│   │    reranker を使う構成は
│   │      bm25_{埋め込み}_{サイズ}_rerank_{reranker}_{サイズ}
│   │    という形にして、埋め込みと reranker のどちらを何Bにした版かを
│   │    ファイル名だけで見分けられるようにしている。
│   │    **同じ土台から派生した3系列だけフォルダに畳んである。** フォルダ名は
│   │    畳む前のファイル名の**接頭辞そのまま**にしてあるので、`config_label()` が
│   │    作る実験ラベルは畳む前と1文字も変わらない（過去の report/*.md や
│   │    results/experiments.jsonl と同じ名前で並べて読める）。
│   │    フォルダ内でその系列の素の構成は**フォルダ名の末尾の語**を名前にする
│   │    ——`bm25_colbert/colbert.yaml` のラベルは `bm25_colbert`）
│   ├── bm25.yaml            : BM25 単体（chunk単位）
│   ├── bm25_paper.yaml      : BM25 単体（論文単位、ablation。bm25s_paper indexer）
│   ├── bm25_dual.yaml       : chunk BM25 + paper BM25 の RRF（ablation、GPU不要。下記）
│   ├── bm25_dual_qwen3_8b_rerank_qwen3_8b.yaml : 現行ベストの構成に bm25s_paper を
│   │     足した版。索引は両方構築済みなので追加構築ゼロ
│   ├── bm25_qwen3.yaml      : BM25 + Qwen3-Embedding-0.6B
│   ├── bm25_qwen3_0.6b_rerank_qwen3_0.6b.yaml : bm25_qwen3.yaml + Qwen3-Reranker-0.6B（2索引ablation）
│   ├── bm25_specter2.yaml   : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_azure_openai.yaml : BM25 + Azure OpenAI text-embedding-3-large（全chunk、ablation。
│   │     API課金あり、全chunk埋め込みで概算$75〜100）
│   ├── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-0.6B + SigLIP（図表画像を直接embedding、ablation用）
│   ├── bm25_qwen3_vl_8b_rerank_qwen3vl_8b.yaml : BM25 + Qwen3-Embedding-0.6B +
│   │     Qwen3-VL-Embedding-8B(図表画像) + Qwen3-VL-Reranker-8B。
│   │     図表を画像のまま扱う唯一の構成（隔離venv .venv-vl が必要）
│   ├── bm25_colbert/        : ColBERT 系列（遅延相互作用）
│   │   ├── colbert.yaml     : colbertv2.0 ベースライン
│   │   └── gte_modern.yaml  : GTE-ModernColBERT-v1（長コンテキスト版）
│   ├── bm25_specter2_body_qwen3/ : **デフォルト系列。** 各モデルを設計どおりの粒度で使う3索引
│   │   ├── qwen3.yaml       : BM25 + SPECTER2(title_abstractのみ) +
│   │   │                      Qwen3-Embedding-0.6B(本文のみ)（デフォルト、構築済み）
│   │   ├── attrfilter.yaml  : 上記 + 会議名・年による属性フィルタ。
│   │   │                      索引は同じものを使い回すので追加構築は不要
│   │   └── rerank_qwen3_0.6b.yaml : 上記 + Qwen3-Reranker-0.6B（ablation）
│   └── bm25_qwen3_8b_rerank_qwen3_8b/ : **主力系列。** 埋め込みと reranker を両方 8B に
│       │                    した版（索引構築が4GPUで約31時間と重い。0.6b版と対比用）
│       ├── 8b.yaml          : 素の 8B 構成（per_index_k=pool_k=100、属性フィルタなし）
│       ├── chunk_attrfilter.yaml : 上記 + 属性フィルタ。per_index_k=pool_k=1000
│       ├── chunk_attrfilter_blend.yaml : 上記 + `rerank_blend`（reranker に順位を
│       │                      置き換えさせず順位融合する。索引・reranker は完全に同一）
│       ├── chunk_attrfilter_k100.yaml : chunk_attrfilter の per_index_k を 100・
│       │                      pool_k を 200 に絞った版
│       ├── chunk_attrfilter_k100_blend.yaml : 上記 + `rerank_blend`
│       │                      （blend 版どうしの幅違い対照が chunk_attrfilter_blend）
│       └── paper_attrfilter.yaml : chunk_attrfilter を「論文単位rerank
│                              (qwen3_paper, 3GPU)」にした版。per_index_k/pool_k=1000
└── agent_style/
    ├── agentless/        : **検索エージェントを使わない構成（最優先）。**
    │   │                   `eval_retrieval.py --agent` に渡す前提で、LLM を1回も呼ばない。
    │   │                   詳細は docs/agentless_spec.md
    │   ├── agentless.yaml    : 素の版（notable から `anchor_from: verdict` を外しただけ）
    │   └── score_anchor.yaml : + `anchor_from: score`（verdict の LLM 不要版）
    │   （agent は `reading` 一本なので、分けているのは**どの任意キーを使うか**。
    │    reading.yaml は任意キーを1つも使わない素の構成で、フォルダが
    │    「幅を振っただけ」「expansion ブロックを足した（取り込み方で2つ）」
    │    「params の反復ループ拡張キーを足した」に対応する。
    │    **フォルダ内のファイル名は reading_ 接頭辞を持たない**）
    ├── reading.yaml      : 分解→読解→不足分の再検索を繰り返す唯一の本命。evidence も埋める（デフォルト）
    ├── reading_normal/   : 任意キーなし。retrieve_top_k / max_candidates を振っただけの版
    │   ├── topk50.yaml   : retrieve_top_k 50（LLM が読む本数は 20 のまま）
    │   ├── cand50.yaml   : retrieve_top_k / max_candidates とも 50。
    │   │                   reading_expand_rrf/cand50.yaml の「展開なし」対照
    │   └── fat.yaml      : retrieve_top_k 100。`--dump-runs` の土台（replay_merge.py 用）を
    │                       兼ねる。旧 reading_topk100.yaml と中身が同一だったので統合した
    └── reading_expand_rrf/ : 論文→論文展開（順位融合）。ランキングを A/B に分けて
                            RRF 統合し、候補列そのものを作り直す。

    **`reading_loop/` は削除した。** 反復ループの拡張キー（`subquery_merge` /
    `grounded_refine` / `pool_rescore` / `adaptive_depth`）は `agent/reading.py` に
    そのまま残っているので、必要なら任意の agent yaml の `params` に直接書けばよい
    （下記の表を参照）。プリセットの yaml を置くのをやめただけ。

    **`reading_expand_insert/`（位置挿入）は削除した。** 展開した論文を候補列の
    決まった位置に差し込む方式で、順位融合に**全指標で負けた**（方式内のベストが
    cr@20 0.822 / multi@20 0.663 に対し、順位融合は 0.879 / 0.770）。上位を汚さない
    代わりに上位を良くもできず、`cr@20` が定義上動かないのが効かない理由。
    **実装も削除した**（`_expand_candidates()` / expander の `expand()` /
    `expand_results()` / `rerank` / `rerank_top_k` / `insert_at`）。`expansion` ブロックを
    書けば必ず A/B の RRF 統合になる。`combine: rrf` は歴史的なキーで、**もう読まれない**
    （既存 yaml のために受け付けるだけ）。

    reading_expand_rrf/ の中身（順位融合）:
    └── reading_expand_rrf/rrf.yaml : **ランキングを2本に分けて RRF 統合する版（現状のベスト）。**
          位置挿入をやめ、A（質問→論文＝検索）と B（論文→論文＝関連）を順位融合する。
          B は reranker に通さない——reranker は「質問に答えるか」で判定するので、
          質問文が名指ししないピア gold を必ず下げるため。詳細は下記。
    └── reading_expand_rrf/notable.yaml : **現状のベスト（フル走行で確定）。**
          verdict_anchor.yaml + `paper_score_skip_chunk_types: [table]` の1行だけ。
          multi cr@20 0.839 -> **0.859** / multi ecr@20 0.908 -> **0.925** /
          ecr@20 0.952 -> **0.961** / ecr@50 0.980 -> **0.989**、single は完全に不変。
          **`max_steps: 2` と抱き合わせてはいけない**（下記）。
    └── reading_expand_rrf/rawq_pin.yaml : notable + `rawq_pin: 1`。**元の質問を
          step0 に5本目として足し、その1位をランキングA の先頭に固定する。**
          **ecr@5 が土台7本すべてで +0.009〜+0.026**（@5 は外部チームに離されている位置）。
          @1 は土台次第で符号が変わる（下記）。フル走行の実測はまだ無い。
    └── reading_expand_rrf/verdict_anchor.yaml : notable の1つ前。rrfk10 +
          `anchor_from: verdict`。ecr@10 0.868 -> 0.921 / multi@10 0.750 -> 0.850。
    └── reading_expand_rrf/steps2_notable.yaml : 上記 + `max_steps: 2`。**不採用**
          （フル走行で notable より multi ecr@20 が -0.026）。土台1本を信じた失敗の記録。
    └── reading_expand_rrf/rrfk10.yaml : 上記から `combine_rrf_k: 60 -> 10` と
          `neighbors: 50 -> 100` の**2行だけ**変えた版（2行はセット。片方では効かない）。
          **オフラインで測れる範囲では現状の最良で、multi recall が最強**
          （全長A で multi ecr@5 0.620->0.683 / @10 0.771->0.814 / @20 0.871->0.910 /
          @50 0.940->0.951、全体 ecr@20 0.932->0.953）。土台4本×4つのk = 16セルで
          悪化ゼロ。フル走行の実測はまだ無い。数字と理屈は下記。
    └── reading_expand_rrf/verdict_anchor.yaml : 上記 + `anchor_from: verdict`（差分1行）。
          ランキングB の起点を「候補1位」から「候補1位 ∪ **読解 LLM が根拠を確認した
          論文**」に広げる。**multi@10 が5条件すべてで +0.030〜+0.082、single の cr@1 は
          完全に不変**。オフラインで測れる範囲での現状ベスト。フル走行の実測はまだ無い。
    └── reading_expand_rrf/cand50.yaml : 上記の候補幅を 20 -> 50 に広げただけの版。
          統合ロジックは同一で、差分は「LLM が読む本数」と「候補列の長さ」だけ。
          per_index_k / pool_k を 1000 にした search_style と組む。
    └── reading_expand_rrf/rel.yaml / reading_expand_rrf/consensus.yaml /
          reading_expand_rrf/protect.yaml : 外部チームの構成から移植した3案
          （明示的な関係 / Consensus / 名指し保護）。**3つとも実測で不採用**。
          残してあるのは再評価のため。理由と数字は下記の各節。

    └── reading_expand_rrf/stacked.yaml : **いま足せるものを全部足した構成**。
          同フォルダの rrf.yaml + 反復ループの3キー（subquery_merge / grounded_refine /
          adaptive_depth）+ title_protect。レイヤが違う（expansion はトップレベル、
          残りは params）ので同居できる。**この組み合わせの実測はまだ無い。**
          `reading_loop/` を消したので、**拡張キーを持つ唯一の yaml がこれ**になった。

    └── reading_expand_rrf/rewrite.yaml : **検索結果に接地したクエリ書き換え**
          （仕様 `docs/search_agent2_spec.md`、実装 `agent/rewrite.py`）。rrf.yaml との
          差分は `rewrite` / `subquery_dedup` ブロックと `anchors: 1 -> 3` の3つ。
          **`anchors: 3` は外すべき**（土台4本のうち3本で悪化。下記）。フル走行では
          ecr@20 0.941 と rrf.yaml(0.924) を上回ったが、**書き換えは55件中26件でしか
          発火しておらず**（step0 で `sufficient` が出ると `_refine()` に到達しない）、
          @20 に入った gold 7本のうち4本は未発火クエリだった。**書き換えが走った24件
          だけで見ると ecr@5 が -0.094 / @10 が -0.031 で、@20 の +0.021 と釣り合わない。**
          原因は材料の使われ方で、LLM は「B の論文が自分を何と呼ぶか（語彙）」ではなく
          **B の論文のタイトルをそのままクエリにしていた**——候補列にいる論文の
          タイトル語を8割以上含むサブクエリが rrf の 1%(7/651) から 16%(31/191) に増え、
          すでに持っている論文を引き直すだけになっている。`missing`（どの事実が
          足りないか）を材料から外したので、表のセルを狙い撃つ力も落ちた。
          直すなら (1) 材料B のタイトル転記を禁止 (2) `missing` を併用 (3) `anchors` を
          1 に戻す、の3点。**サブクエリ本数は 651 -> 191（-71%）に減っており、
          `subquery_dedup` は効いている。**

    **`subquery_merge` と `pool_rescore` は同時に有効にしない。** `_rescore_pool()` は
    `_merged_results()` の後に走り、reranker がプール全件を再スコアして並びを
    置き換えるので、`subquery_merge` が作った順位が candidate_papers では消える
    （ループ内で LLM が読む候補には効いたまま残る）。2つは同じ問題（サブクエリ間の
    スコア非可換性）への別々の答えなので、足すのではなく選ぶ。

### 検索結果に接地したクエリ書き換え（`rewrite`、既定オフ）

仕様は `docs/search_agent2_spec.md`、実装は `agent/rewrite.py`。
**サブクエリを、質問文からではなく「コーパスが返した本文」から作る。**

`_refine()` の材料は読解 LLM の `missing` だけで、コーパスが何を返したかを一度も
見ない。その結果サブクエリが**ひとつの語彙ファミリーの言い換え**に収束する。
q_022（「reference-free な preference optimization を提案した ICML 2025 論文を列挙」）
では3ステップ23本が全部 `reference-free …` 系で、gold の AlphaPO は28位だった。
AlphaPO は**自分を一度も reference-free と呼ばない**（本文の該当は参考文献の1件だけ）
——自称は `Direct Alignment Algorithm` / `reward shape` で、どれも質問文に無い語。
BM25 単体でもその語彙で投げれば1位で取れる。

`at_step` 以降、`_refine()` を書き換えに置き換える。材料は2系統:

| | 本数 | 見せるもの | 答える問い |
|---|---|---|---|
| A 詳細 | 5 | title + abstract + **ヒットしたチャンク** | いま何を引き当てているか |
| A 俯瞰 | 20 | `[venue year] title` のみ | **軸がズレていないか** |
| B | 20 | title + abstract のみ（`ChunkStore` から） | この論文は自分を何と呼ぶか |

**A を2段にするのは、5本の窓では分布の異常が分布に見えないから。** q_022 の候補列は
上位5本に長文脈論文が1本しか無い（ノイズに見える）が、20本まで広げると10本が長文脈で
軸ズレが一目で分かる。俯瞰はタイトルだけなので追加コストはほぼゼロ。

**B は本文チャンクを見せず本数を稼ぐ。** 欲しいのは語彙だけなので本文は要らない。
`anchors: 3` が前提——ランキングA の top20 に入っていない multi の gold 26本のうち、
B の上位5本に入るのは **anchors=1 で0本 / anchors=3 で5本**（上位50本まで見ても 5 vs 13）。

**`bm25s_paper` 由来のヒットは材料A に使わない**（`PAPER_LEVEL_SOURCES`）。
論文単位索引の text は論文全文なので、snippet で切ると abstract と重複するうえ
「質問のどこに当たったか」を持たない。

**ランキングB は今後も reranker に通さない。** ここで作るのは「B の語彙で reranker に
問い直す別の検索」で、その結果は普通のサブクエリとして `chunks` に積まれる
——だから**展開由来の論文もチャンクを持ち、`evidence` に出られる**ようになる
（現行では展開は読解の後に走るので、multi の候補 top20 の gold 94本中18本が
「順位には効いたが evidence を出せない」状態）。`_combine_rrf()` は現行のまま残す。

`_refine()` を置き換え先に選んだのは実測から。**step1 の305本は leave-one-out で
1本抜いても ecr@50 の gold が減らない（0/305）**（step2 は 1/327、step0 は 4/180）。

### サブクエリの重複除去（`subquery_dedup`、既定オフ）

**本数を先に決めない。** LLM に N 本と指定しても返るのは言い回し違いの同じクエリに
なりがちで、そのぶん検索と reranker が空回りする。**何本作らせるかではなく、
何本残すかを中身で決める。**

判定は**引いてくる論文が重なるか**（Jaccard > `max_overlap` なら捨てる）。文字列の
重複では捕まえられない——別語彙で別の論文を引くクエリは残すべきで、言い回し違いで
同じ論文しか引かないクエリは捨てるべき。

**篩いは BM25 だけで行う。** 本番の検索1本は reranker が `pool_k` 件を推論するが、
BM25 の引き当ては索引を1回叩くだけ。**重複したクエリに reranker を1回も走らせない。**
索引は `retriever.indexers` から `bm25s` を名前で探す（見つからない構成では
`max_queries` の上限だけが効く）。

**論文→論文展開の近さは5種類あり、registry の "expander" で差し替え・併用する**
（`retrieve/paper_expander.py` と `retrieve/relation_graph.py`）:

- `specter2`: SPECTER2(proximity) 埋め込みの近傍。構築済み索引を再利用（追加構築なし）。
- `bib_coupling`: 書誌結合。参考文献の arXiv ID 集合の Jaccard。初回のみコーパス
  1走査で索引を作りキャッシュ（29〜47秒、25,012論文/68,418 ID）。GPU 不要。
- `bm25_mlt`: 論文全文の more-like-this。anchor の title+abstract をクエリにして
  構築済みの `bm25s_paper` 索引を引く。LLM 呼び出しゼロ・追加構築ゼロ。
  `papers.jsonl`(2.5GB) は**クエリ時には読まない**——BM25 本体は `mmap=True` で開き、
  行番号→paper_id と anchor 用テキストは初回1回だけ流し読みして pickle に落とす。
- `title_mention`: **A の本文に B の名前が出てくるか**（正式タイトル / コロン前の
  見出し）。上3つが「内容が近い」なのに対し、これは「名指ししているか」を測る。
  直接リンク（双方向）+ ハブを避けた2ホップ（`max_hub_degree: 4`）。**不採用**（下記）。
- `method_comention`: **A と B が同じ論文の名前を挙げているか**（共言及の Jaccard）。
  「A が B を挙げた」ではなく「A と B が同じ手法群を論じている仲間か」。**不採用**（下記）。
- `fused`: 上記を RRF 融合（agent yaml の `expansion.sources` に並べると自動でこれになる）。

**併用の根拠は「違う gold を拾う」こと**——候補圏外 gold 37本の回収は
SPECTER2 15本 / 書誌結合 11本 / 全文MLT 16本で、**MLT だけが拾えた gold が2本**、
既存2つだけが拾えたのが6本、重複14本。

#### `title_mention` / `method_comention` は**採用しない**（同じ基準で落ちた）

同じ集計を5ソースでやり直した実測（`predictions_8b_chunk_k100_cand50.jsonl` 土台、
候補圏外 gold 35本・うちピア gold 18本）:

| ソース | 回収 | **うち独自** | ピア回収 |
|---|---|---|---|
| specter2 | 21 | 2 | 10 |
| bib_coupling | 18 | 1 | 8 |
| bm25_mlt | 21 | 1 | 10 |
| title_mention | 2 | **0** | 2 |
| method_comention | 2 | **0** | 0 |

**新規2ソースが拾う4本は、既存3ソースが全部すでに拾えている**（既存の和26本、
5ソースの和も26本）。ランキングに足すと multi の ecr@20 が 0.852 -> 0.840 と
わずかに下がる（雑音のぶん）。

**これは CLAUDE.md の既存の結論を覆さず、別経路で追認した。** 「引用グラフは
ほぼ張れない」は arXiv ID 解決ベースの話だったので、本文のタイトル文字列なら
繋がるかを試した——実際にコーパス全体で 16,075本の辺が張れた（1論文あたり中央1本）
が、**その辺が指す論文はトピック類似ですでに届く範囲**だった。
同時期の論文どうしは互いを名指しする機会そのものが少ない。

コードは registry に残してある（既定では `sources` に書かなければ一切動かない）。
コーパスが広がって明示的な言及が増えたときに再評価できる。

#### 名前の照合（`title_mention` / `method_comention` / 名指し保護の共通土台）

実装は `retrieve/paper_titles.py`。索引は `scripts/build_relation_graphs.py` が
コーパス1走査で作る（GPU 不要、キャッシュ後は即ロード。`title_mention` と
`method_comention` と名指し保護が**同じキャッシュを共有する**ので走査は1回）。

**MinerU のタイトルは分かち書きが壊れている。** `M o RE : A Mixture of Low-Rank
Experts` / `T oken S hapley:` / `D e F ine:` / `AIMSC heck:` のように大文字の前で
スペースが入る。本文中では `MoRE` と正しく書かれるので、**英数字以外を全部落として
連結した文字列**をキーにして吸収する。

**小文字化して照合すると壊れる。** 実測（コーパスの21%・5,738論文）で、
`MoRE` / `MoST` / `DeFine` / `DIFFER` / `RANGE` / `CLEAR` / `MUST` が本文の普通の
英単語 more / most / define … に当たり、**96,662本の偽の辺**が張られた
（5,738本中5,719本が `MoRE` を「名指し」していた）。短い識別子は
**大文字小文字も含めて完全一致**させる（長い正式タイトルは引用時に変わりうるので
小文字化して照合する）。

**それでもハブは残るので出現本数で落とす。** 大文字小文字を一致させても
`HTML`(123本) / `CoLA`(42) / `FLAME`(38) / `MUST`(35) / `FLARE`(33) / `MLVU`(27) /
`GRACE`(21) / `LIME`(19) のような「ALLCAPS の普通の英単語・有名略語」が残る。
`max_key_degree` 本を超える論文から名指しされた名前は**弁別に使えない**ので捨てる
（外部チームの「同じ手法名に繋がる論文が10本超なら曖昧」と同じ考え方を、
固定リストではなくコーパスの分布に任せた形）。この2段で 96,662 -> 987 辺になった。

**曖昧な名前はコーパスの一意性で落とす。** `BERT` や `RAG` が危ないのは
「汎用語だから」ではなく「どの論文を指すか決まらないから」なので、
同じ識別子を複数論文が持てば機械的に捨てる（固定リストは補助）。
コロン前の見出しは **CamelCase / ALLCAPS / 数字 / ハイフン**のどれかを持つものだけ
採る——先頭だけ大文字の普通の単語（`Harmony:`）は名前と区別が付かない。

### ランキング統合（`combine: rrf`）— 位置挿入をやめて順位融合する

関連論文を「上位K本だけ決まった位置に差し込む」位置挿入の代わりに、
**ランキングそのものを RRF で統合する**（`agent/reading.py` の `_combine_rrf`）。

- **A（質問→論文）**: 検索。BM25 + 埋め込み → RRF → reranker
- **B（論文→論文）**: 上記3ソースの RRF 融合。**reranker には通さない**

      score(p) = w_A / (k + rank_A) + w_B / (k + related_offset + rank_B)

位置挿入では作れなかった「**A にも B にも居る論文の加点**」が効く。スコアで混ぜて
壊れた（cr@20 0.822 -> 0.773）のは reranker の絶対スコアと展開の仮スコアを足していた
からで、RRF は順位しか見ないのでその問題が起きない。

**素の RRF（`related_weight: 1.0` / `related_offset: 0`）が最良。** 重みを下げると
B 単独の論文が A の裾より下に落ちて統合の意味が消え（w=0.5 で cr@20 0.817）、
上げると B が候補列を占領する（w=2.0 で 0.830）。offset も 0 が最良（15 で 0.839）。

**anchor 自身をランキングB の先頭に置くこと。** 各 expander は anchor を自分の近傍から
外すので、そのままだと anchor は A の `1/(k+1)` しか持てず、「A にも B にも居る」論文
（2項ぶん）に軒並み抜かれる。実測で **single_paper 2件、gold そのものだった候補1位が
top20 から消えた**（single の cr@20 が 1.000 -> 0.923）。論文は自分自身に最も近いので、
B の1位に置くのが定義どおりでもある。

オフライン実測（`predictions_8b_chunk_cand50.jsonl` 土台・55件フル）:

| | cr@20 | ecr@20 | single@20 | multi@20 | cr@50 | ecr@50 | multi@50 |
|---|---|---|---|---|---|---|---|
| 展開なし | 0.789 | 0.868 | 1.000 | 0.601 | 0.832 | 0.889 | 0.681 |
| specter2 のみ | 0.816 | 0.870 | 1.000 | 0.650 | 0.908 | 0.945 | 0.825 |
| 書誌結合 のみ | 0.829 | 0.880 | 1.000 | 0.675 | 0.876 | 0.921 | 0.764 |
| 全文MLT のみ | 0.857 | 0.910 | 1.000 | 0.728 | 0.898 | 0.950 | 0.807 |
| specter2 + 書誌結合 | 0.859 | 0.901 | 1.000 | 0.732 | 0.912 | 0.950 | 0.833 |
| **3ソース（`reading_expand_rrf/rrf`）** | **0.879** | **0.926** | **1.000** | **0.770** | **0.917** | **0.956** | **0.842** |

統合するときは**50本で切る前の全長**をランキングA に使う（`_build_prediction`）。
51位の論文を B が強く推していても、先に切ると押し上げようがないため。

**`expansion` ブロックを書かなければ統合は走らない**（`paper_expander` が None なら
候補列は検索の順位そのまま）。逆に書けば必ず統合される——位置挿入を削除したので
`combine` による分岐は無くなった。

#### 展開まわりの調整は**土台を4本使う**（1本だと結論が反転する）

上の表は `predictions_8b_chunk_cand50.jsonl` 1本を土台にした数字で、**土台を替えると
2〜4pt 平気で動く**。`anchors: 1` の同じ設定でも ecr@20 は土台ごとに
0.904 / 0.926 / 0.944 / 0.929 と **4pt の幅**に散る。55件・根拠付き gold 117本しか
無いので、@20 の1〜2pt は打ち手の効果ではなく土台の揺れである可能性が高い。

展開前の予測が55件フルで残っている土台は4本ある。**打ち手を採るかどうかは
「4本すべてで悪化しないか」で決める**（`replay_expansion.py` は土台を1本しか取らないので、
振るときは expander を1回だけ建てて土台×設定を回すスクリプトを書く。索引ロードが
設定の数だけ走るのを避けるため）:

| 土台 | 由来 |
|---|---|
| `predictions_8b_chunk_k100_cand50.jsonl` | k100 + `reading_normal/cand50` |
| `predictions_8b_chunk_cand50.jsonl` | chunk_attrfilter(k1000) + cand50 |
| `predictions_fat.jsonl` | k100 + `reading_normal/fat` |
| `predictions_8b_chunk_b_merged.jsonl` | chunk_attrfilter の val_a/val_b 結合 |

この基準で**現行値が全軸で最良だと4土台で追認された**（`neighbors` は 20/100/200 とも
悪化、`related_offset` は 10/25/50 とも 4/4 悪化、`related_weight` は 0.5/0.75/1.5 とも
4/4 悪化、単体ソース・2ソースはどれも 4/4 悪化）。

**ただし土台からの再生は候補列が50本で切られた後なので、A の長さが本走行と違う。**
予測ファイルに残るのは `CANDIDATE_PAPERS_LIMIT(50)` 適用後の列で、そこから作る
ランキングA は `|A|=38` 程度にしかならない。本走行の `_build_prediction()` は
**切る前の全長**を A に使う（`retrieve_top_k: 20` で `|A|=72`、100 なら 312）。
A の長さに感度がある打ち手では**再生値が過小評価になる**（`combine_rrf_k` の
ecr@5 の差分は |A|=38 で +0.012、72 で +0.024、312 で +0.050）。

**`runs_fat.jsonl` があれば全長A を復元できる。** `replay_merge.rebuild()` は
`to_gold_papers(max_papers=50)` で切ってしまうので使わず、`_build_prediction()` と
同じ順序を自分で踏む（`_merged_results` -> `_rescore_pool` -> `to_gold_papers()`
**引数なし** -> `_combine_rrf` -> 50で切る）。

#### 統合まわりは**オフラインが本評価**（フル走行を使わない）

`_combine_rrf` は `_build_prediction()` でしか呼ばれない（`reading.py:807`）。
つまり `combine`/`combine_rrf_k`/`related_*`/`neighbors`/`anchors` は
**サブクエリ生成・検索・reranker・読解ループ・evidence のどれにも触れず**、
最後に候補列を並べ替えるだけ。同じランキングA を与えればオフライン再生は本走行と
**一致する**（近似ではない）。

したがって**統合のつまみを測るためだけにフル走行を回さない。** 4〜5時間かけて
追加で分かるのは LLM の実行間ばらつきだけで、それは @20 で数pt あり、
この種の打ち手の +0.02 を覆い隠す側に効く。**打ち手が複数たまってから1回だけ走らせる。**

#### `combine_rrf_k: 60 -> 10`（`reading_expand_rrf/rrfk10.yaml`）

**A/B 統合の RRF の k を下げると @5 / @10 が4土台すべてで上がる。** k は「リスト内の
順位」と「両方に載っていること」のどちらを重く見るかを決めるつまみで、
**k=60 は後者を過大評価していた**。A の1位（B に無い）は `1/61` だが、A でも B でも
r 位（0起点）の論文は `2/(61+r)` を得るので:

    2/(61+r) > 1/61  <=>  r < 61

リスト長が50本の現状では、**A と B の両方に載っていればどれだけ深くても A の1位に勝つ**。
reranker が1位に置いた論文が、A の40位 × B の40位の論文に抜かれていた。k=10 なら
閾値が `r < 11` になり、「両方の**上位**に居るときだけ勝つ」という本来の意味になる。

融合後 top5 の出所を数えると実際にそう動いている（4土台）:

| k | A の上位5位 | A の6位以下 | B にしか無い | top5 の gold 率 |
|---|---|---|---|---|
| 60 | 49〜54% | 38〜45% | 5.5〜8.0% | 25.8〜27.6% |
| **10** | **56〜62%** | **26〜35%** | **9.1〜12.0%** | **26.9〜29.1%** |

**減るのは「A の中位 × B の中位の合わせ技で浮上した論文」だけ**（-12pt）で、
A の上位（reranker が本気で推した論文）と B 単独の上位（近傍1位）は**どちらも増える**。

| 土台 | ecr@5 | ecr@10 | ecr@20 | ecr@50 | multi@20 | single cr@1 |
|---|---|---|---|---|---|---|
| k100_cand50 | 0.773 -> **0.808** | 0.843 -> **0.859** | 0.904 -> 0.904 | 不変 | 0.852 -> 0.852 | 不変 |
| chunk_cand50 | 0.786 -> **0.796** | 0.844 -> **0.865** | 0.926 -> **0.941** | 不変 | 0.859 -> **0.888** | 不変 |
| fat | 0.778 -> **0.806** | 0.874 -> **0.896** | 0.944 -> **0.948** | 不変 | 0.894 -> **0.902** | 不変 |
| b_merged | 0.801 -> **0.822** | 0.864 -> **0.891** | 0.929 -> **0.935** | 不変 | 0.865 -> **0.876** | 不変 |

**悪化がひとつも無い。**@5 / @10 が 4/4 改善、@20 は 3改善0悪化、@50 と single の
cr@1 は完全不変。効くのが @5 なのが重要で、そこは外部チームに 0.145pt 離されている位置。

**この4土台は過小評価。** 上節のとおり土台の A は50本で切られた後（`|A|=38`）なので、
`runs_fat.jsonl` から全長A（`|A|=72`、本走行と同一条件）を復元して測り直した:

| `retrieve_top_k=20`, \|A\|=72 | cr@5 | ecr@5 | ecr@10 | ecr@20 | ecr@50 | multi@5 | multi@10 | multi@20 | single@5 |
|---|---|---|---|---|---|---|---|---|---|
| k=60 | 0.707 | 0.781 | 0.879 | 0.932 | 0.968 | 0.620 | 0.771 | 0.871 | 0.962 |
| **k=10** | **0.736** | **0.806** | **0.896** | **0.948** | 0.968 | **0.666** | **0.803** | **0.902** | 0.962 |
| 差分 | +0.029 | **+0.024** | +0.017 | +0.017 | ±0 | **+0.046** | +0.032 | +0.032 | ±0 |

**伸びは全部 multi_paper に乗る**（single は @5 0.962 / @20 1.000 で飽和済み）。
@50 が完全不変なのは並べ替えているだけで集合が動かないため。

**A が長いほど効果が大きい**（ecr@5 の差分: |A|=38 で +0.012 / 72 で +0.024 /
312（`retrieve_top_k: 100`）で +0.050、multi@5 は +0.095）。機序どおりで、
**k=60 は A が長くなるほど悪化し（ecr@5 0.781 -> 0.762）、k=10 は改善する
（0.806 -> 0.812）**。`per_index_k`/`pool_k` を広げる方向の構成ほど k=60 の害が大きい。

**`related_offset` / `related_weight` は動かさない。** k=10 のもとで `related_offset`(3/10)
や `related_weight`(0.75/1.5) を足すと全部崩れる。**B 内部の `rrf_k` も 60 のまま**
——そちらを下げると4土台とも悪化する（k=10 で 0.897/0.920/0.932/0.929）。

#### `neighbors` の最適値は `combine_rrf_k` に依存する（50 -> 100）

**k=60 のもとでは `neighbors: 100` は悪化していた**（4土台で2改善2悪化）。k を下げると
符号が反転する。理屈は上と同じで、k=60 では B の深い順位でも A の1位に勝ててしまうので
B を100本に増やしたぶんだけ雑音が上位に入る。k=10 なら深い順位は勝てないので、
**B を増やしても入ってくるのは「A でも上位」の論文だけ**になる。

全長A での multi ecr（`retrieve_top_k: 20`、本走行と同一条件）:

| 設定 | multi@5 | multi@10 | **multi@20** | multi@50 | multi cr@20 | 全体 ecr@20 |
|---|---|---|---|---|---|---|
| k=60 nb=50（`rrf.yaml`） | 0.620 | 0.771 | 0.871 | 0.940 | 0.793 | 0.932 |
| k=10 nb=50 | 0.666 | 0.803 | 0.902 | 0.940 | 0.819 | 0.948 |
| **k=10 nb=100（`rrfk10.yaml`）** | **0.683** | **0.814** | **0.910** | **0.951** | 0.815 | 0.953 |
| k=15 nb=100 | 0.651 | 0.806 | 0.914 | 0.951 | 0.819 | 0.955 |
| k=10 nb=200 | 0.674 | 0.814 | 0.910 | 0.940 | 0.815 | 0.953 |

土台4本 × 4つの k（計16セル）の multi ecr で**悪化ゼロ**。k=15 は multi@20 だけ僅かに
上（0.914）だが @5 が 0.651 に落ちるので、@5 が弱点である以上 k=10 を採る。
`neighbors: 200` は 100 を超えず multi@50 が下がる（0.940）ので 100 が打ち止め。
**減らす向き（`neighbors: 20`）は k=60 でも k=10 でも悪化する。**

**この2行はセットで入れる。** 片方だけでは効かない（k=60 のまま nb だけ増やすと悪化、
nb=50 のまま k だけ下げると multi@20 が 0.902 で止まる）。

#### `anchor_from: verdict` — 起点に読解 LLM の確認済み論文を足す

**`_read_and_judge()` が返す `paper_ids` は本文を読んだうえでの判定なのに、順位付けに
一度も使っていなかった**（`submit_from: candidates` なので提出にも使わない）。
`anchor_from: verdict` を書くと、ランキングB の起点を「候補1位」から
「候補1位 ∪ LLM 確認済み」に広げる（`agent/reading.py` の `_anchor_papers()`）。

**何を直しているのか。** multi の根拠あり gold を**クエリ内で順位順**に並べると、
1本目は解けているのに3本目以降が沈んでいる（全長A・`rrfk10` 土台）:

| クエリ内 | n | 中央順位 | @1 | @5 | @20 |
|---|---|---|---|---|---|
| 1本目 | 29 | 1位 | 79% | **100%** | 100% |
| 2本目 | 23 | 4位 | 0% | 83% | 91% |
| 3本目 | 22 | 8位 | 0% | **27%** | 86% |
| 4本目 | 12 | **14位** | 0% | **8%** | 67% |

anchor が1本だと展開できるトピッククラスタも1つなので、そこが埋まらない。

| 土台 | single cr@1 | multi@5 | **multi@10** | multi@20 |
|---|---|---|---|---|
| k100_cand50 | 0.923（不変） | 0.682 → 0.717 | 0.767 → **0.830** | 0.869 → 0.885 |
| chunk_cand50 | 1.000（不変） | 0.610 → 0.686 | 0.762 → **0.842** | 0.899 → 0.908 |
| fat | 0.923（不変） | 0.666 → 0.669 | 0.809 → **0.848** | 0.914 → 0.914 |
| b_merged | 0.962（不変） | 0.680 → 0.726 | 0.786 → **0.868** | 0.876 → 0.899 |
| 全長A | 0.923（不変） | 0.683 → 0.677 | 0.814 → **0.844** | 0.910 → 0.914 |

**伸びは @10 に集中する**（+0.030〜+0.082）。上の表のとおり2本目・3本目がそこに居る。
悪化は全長A の multi@5 が −0.006（gold 1本ぶん）だけ。

**候補1位を必ず起点に残すこと。** LLM 確認済みだけにすると候補1位が B の先頭から
外れ、「A にも B にも居る」論文に抜かれて **single の cr@1 が 0.923 → 0.885 に落ちる**
（`anchors: 3` と同じ事故）。和集合なら single は5条件とも完全に不変。

**効いているのは精度ではなく本数。** LLM 確認済みの gold 率は 68本中52本 = **76%** で、
候補1位の 85%（47/55）より**低い**。それでも効くのは1本の anchor では1クラスタしか
展開できないため。anchor が2本以上になるのは55件中16件で、**うち14件が multi_paper**
（1本 39件 / 2本 9件 / 3本 2件 / 4本以上 5件）——single に副作用が出ないのはこの偏りによる。

**起点の本数は呼び出しの前後で元に戻す**（`_rank_related()`）。各 expander は渡された
リストの先頭 `anchors` 本しか使わないので差し替え時は本数も合わせる必要があるが、
`anchors` は yaml の設定値なのでクエリごとの都合で書き換えたままにしない。

#### `anchors` は上げない（`rewrite.yaml` の `anchors: 3` は**外すべき**）

`expansion.anchors` はランキングB の起点にする候補論文の本数で、各 expander が
anchor ごとに近傍を取り `_interleave()` が交互配置する。`rewrite.yaml` だけが 3 に
しているが、**土台4本で測ると3本で悪化する**（@20 の根拠付き gold の増減:
k100_cand50 +3本 / chunk_cand50 -4 / fat -5 / b_merged -3）。土台をまたいで一貫して
いるのは損失だけ（q_039 は4土台すべてで脱落、q_049 の gold 2本が3土台で脱落）。

**「候補1位が外れやすいから起点を増やす」という機序では説明できない。** 4土台の質は
ほぼ同一（cr@1 0.538〜0.575、top3 の gold 率 0.376〜0.394）なのに符号が割れる。

**`config.py` が `FusedPaperExpander` から `anchors` を除いているのはバグではない。**
そのため `_combine_rrf` の「anchor 自身をランキングB の先頭に置く」保護は
`candidate_papers[:1]` のままになるが、保護をN本に広げた版を測ると**悪化する**
（anchors=3 で ecr@20 0.915 -> 0.907、multi 0.873 -> 0.858）——2位・3位を B の先頭に
据えると、それ自体が「A にも B にも居る」2項ぶんを得て本来上位に来る論文を押し下げる。

#### ランキングB の内部には伸びしろが残っていない（2026-08-08 に一巡した）

`verdict_anchor.yaml`（k=10 / nb=100 / anchor_from: verdict）を基準に、B のつまみを
一通り振った結果。**全部が現行値以下**だった（全長A・55件、詳細は
`docs/offline_findings_spec.md`）:

| 打ち手 | 結果 |
|---|---|
| ソース別重み（specter2 / bib / mlt を 0.5〜2.0） | 均等 1:1:1 が最良 |
| ソース別 `neighbors`（50 / 200） | 3つとも100が最良 |
| A∩B への明示ボーナス | なしが最良（RRF の2項ぶんで足りている） |
| B の pool 先頭に A の上位も置く | 悪化（A[:3] で multi@5 0.677 -> 0.591） |
| **2ホップ展開**（B の上位からもう1ホップ） | 全条件で @50 が 0.960 -> 0.945 |
| verdict を3本目のランキングとして加点 | **no-op**（確認済み論文はすでに最上位に居る） |
| `anchors` を A の上位N本に | 新スタックでも悪化（A[:2] で multi@20 0.914 -> 0.871） |
| `related_weight` / `related_offset` / w_A | 1.0 / 0 / 1.0 が最良 |

**B をいじるのはもう止めてよい。** 伸びしろは読解側にある（下記）。

#### 伸びしろは検索ではなく**読解の evidence 判定**にある

検索プールの天井を測ると、**根拠あり gold 117本のうち @50 に届いていないのは4本だけ**
（プール全体に居るのが105本、最終 @20 が108本、@50 が113本）。
**最終 @20 が検索プールを超える**のは論文→論文展開が引けなかった論文を拾っているため。
つまり**サブクエリ・クエリ書き換えで拾える層はもう薄い**（残るはピア gold の13本）。

一方、読解 LLM の evidence 判定は **recall 60%**（可視域の根拠あり gold 86本中52本）。
**測るときは分母を「展開前の候補列」にすること**——`_read_and_judge()` はループの中で
呼ばれ、展開は `_build_prediction()` の中（読解のあと）なので、融合後の候補列で測ると
LLM が一度も見ていない論文が分母に入る（それで 48% と誤って出した）。

融合後 top20 の根拠あり gold 108本の内訳は **見て確認52 / 見たが未確認34 /
LLM は見ていない22**。ピア gold も top20 の19本中13本が展開由来で、
**どちらも evidence を出せない**。

可視域の gold を全部確認できたとしたときの上限（gold を見ているので到達不能）:

| | multi@5 | multi@10 | 全体@5 | 全体@10 |
|---|---|---|---|---|
| 現行（recall 60%） | 0.677 | 0.844 | 0.812 | 0.918 |
| オラクル（100%） | **0.827** | **0.920** | **0.909** | **0.958** |

**multi@5 で +0.150** ——検索側のどの打ち手より桁が大きい。
**展開を読解の前に動かす**と、順位への効果を保ったまま、いま evidence を出せない
35本（根拠あり22 + ピア13）が出せるようになる。

#### `consensus`（anchor ごとに B を分ける）は**採用しない**

`consensus: true` にすると、ランキングB を `_interleave()` で1本に潰さず
anchor × ソースごとに分けて RRF へ渡す（複数の pool に居る論文が項の数だけ加点される
＝「揃って推された」が信号になる）。B の重みは pool 数で正規化する——ソース3 ×
anchor3 で9本になるので、割らないと B が A（1項）を圧倒する。

**実測では @20 と @50 が明確に悪化した**（`predictions_8b_chunk_k100_cand50.jsonl`
土台・55件フル・`replay_expansion.py`、multi の ecr）:

| anchors | ecr@5 | ecr@10 | **ecr@20** | ecr@50 |
|---|---|---|---|---|
| 潰す（既定） | 0.603 | 0.738 | **0.852** | 0.905 |
| consensus 1 | 0.633 | 0.711 | 0.850 | 0.891 |
| consensus 2 | 0.610 | 0.711 | 0.787 | 0.876 |
| consensus 3 | 0.600 | 0.732 | 0.796 | 0.868 |
| consensus 5 | **0.654** | 0.736 | 0.784 | 0.848 |

**@5 だけは伸びる**（anchors=5 で +0.051）が、LLM が読むのは上位
`max_candidates`(20) 本なので @20 を落とす取引は割に合わない。pool を分けると
RRF の和が広がって、融合ランキングなら上位に来た論文が「どれか1つの pool の
深い順位」に負ける——合意の加点より、融合した順序そのもののほうが情報が多い。

コードは既定オフのまま残してある（`consensus` を書かなければ従来経路）。
@5 を上げたい別の文脈が出たときに再検討できる。

**展開の rerank は 0.6B で代用してよい**（実測23秒 vs 8B の147秒）。8B に替えても
結論は変わらず、差は1〜2本ぶん。ただし向きが逆で、**8B は ecr が上がり cr が下がる**
（ecr@20 0.881->0.887 / cr@20 0.822->0.818）。top20 に押し上げた300本の内訳を数えると
8B は evidence 持ち gold を1本多く拾い no_evidence gold を2本減らしており、
「質問に答える論文を選べている」ほど no_evidence gold を落とすので cr が下がる。
**打ち手の評価は ecr で見る**という方針がここでも当てはまる。

**書誌結合は引用グラフ（A が B を引く）ではない。** このコーパスは2024〜2025年しか
無く同時期の論文は互いに引用できないので、引用リンクはほぼ張れない（anchor から
解決できたコーパス内引用は実測1本）。共有している**古い文献**で繋ぐのが要点で、
TCM とピア3本の Jaccard 0.19〜0.24 に対し無作為30本は中央値 0.000・最大 0.054。
`min_shared: 2` は「共有1本だけ」を切るため（Adam や ResNet のような汎用引用で
繋がってしまう）。
```

**各 search_style のモデルと主要パラメータ**（yaml 冒頭のコメントに選定理由を書いてある。
共通: `per_index_k: 100`、`fuser: rrf (k=60, 全索引 weight 1.0)`）:

| ファイル | indexer（モデル / 主要params） | reranker（モデル / 主要params） |
|---|---|---|
| `bm25.yaml` | `bm25s`（params なし） | `none` |
| `bm25_paper.yaml` | `bm25s_paper`（1論文=1ドキュメント） | `none` |
| `bm25_qwen3.yaml` | `bm25s` + `faiss_qwen3`: `Qwen/Qwen3-Embedding-0.6B`, devices=`cuda:0-3`, batch_size=16, fp16, max_tokens=8192, doc/query_prefix=`passage: `/`query: ` | `none` |
| `bm25_azure_openai.yaml` | `bm25s` + `faiss_azure_openai`: `text-embedding-3-large`（.env のデプロイ名/次元）, batch_size=256, workers=4 | `none` |
| `bm25_colbert/colbert.yaml` | `bm25s` + `colbert`: `colbert-ir/colbertv2.0` | `none` |
| `bm25_colbert/gte_modern.yaml` | `bm25s` + `colbert`(index_name=`colbert_gte_modern`): `lightonai/GTE-ModernColBERT-v1`, device=cuda, batch_size=32, document_length=2048, build_batch_size=50000 | `none` |
| `bm25_specter2.yaml` | `bm25s` + `faiss_specter2`: `allenai/specter2_base`, batch_size=128, fp16, doc_adapter=`proximity`, query_adapter=`adhoc_query` | `none` |
| `bm25_qwen3_siglip.yaml` | 上記 qwen3 + `siglip_image`: `google/siglip-base-patch16-224` | `none` |
| `bm25_qwen3_vl_8b_rerank_qwen3vl_8b.yaml` | `bm25s` + `faiss_qwen3`(0.6B, 同上) + `qwen3_vl_image`: `Qwen/Qwen3-VL-Embedding-8B`, device=cuda:0(16.3GB), batch_size=8, fp16 | `qwen3_vl`: `Qwen/Qwen3-VL-Reranker-8B`, device=cuda:1(**索引と別GPU必須**), fp16, batch_size=4, use_images=true, max_image_docs=0（0は全件）／pool_k=100 |
| `bm25_specter2_body_qwen3/qwen3.yaml` | `bm25s` + `faiss_specter2`(index_name=`faiss_specter2_abstract`, chunk_types=`[title_abstract]`, batch_size=128, fp16) + `faiss_qwen3`(index_name=`faiss_qwen3_0p6b`, 0.6B, chunk_types=本文系4種, batch_size=32, max_tokens=8192) | `none` |
| `bm25_specter2_body_qwen3/attrfilter.yaml` | 同上（索引は完全に同一で使い回す） | `none`／`attribute_filter`: enabled=true, safety=1.5, max_fetch_k=5000, min_results=10 |
| `bm25_specter2_body_qwen3/rerank_qwen3_0.6b.yaml` | 同上（ただし faiss_qwen3 の max_tokens=1024） | `qwen3`: `Qwen/Qwen3-Reranker-0.6B`, device=cuda:3, fp16, batch_size=16, max_tokens=2048／pool_k=100 |
| `bm25_qwen3_0.6b_rerank_qwen3_0.6b.yaml` | `bm25_qwen3.yaml` と同一（索引を使い回すため index_name なし） | `qwen3`: `Qwen/Qwen3-Reranker-0.6B`, device=cuda:1, fp16, batch_size=16, max_tokens=2048／pool_k=100 |
| `bm25_qwen3_8b_rerank_qwen3_8b/8b.yaml` | `bm25s` + `faiss_qwen3`(index_name=`faiss_qwen3_8b`): `Qwen/Qwen3-Embedding-8B`, devices=`cuda:0-3`, batch_size=8, fp16, max_tokens=8192 | `qwen3`: `Qwen/Qwen3-Reranker-8B`, device=cuda:3, fp16（8Bは必須）, batch_size=4, max_tokens=2048／pool_k=100 |
| `bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter.yaml` | 同上（索引は共有）。検索時は `devices: cuda:0` の1枚のみ（`_embed_query` は devices[0] しか使わないので残りを reranker に空ける） | `qwen3`: 同上, **devices=`cuda:1,cuda:2,cuda:3`**, max_batch_tokens=2048／per_index_k=pool_k=**1000**／`attribute_filter`: enabled=true, max_fetch_k=**3000**（上げると faiss が61倍遅くなる） |
| `bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_blend.yaml` | 同上（索引は共有） | `qwen3`: 同上だが devices=`cuda:1,cuda:2`／per_index_k=pool_k=**1000** ＋ `rerank_blend`（順位融合。下記） |
| `bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml` | 同上 | 同上だが devices=`cuda:1,cuda:2`／per_index_k=**100**, pool_k=**200** |
| `bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100_blend.yaml` | 同上（索引・reranker とも k100 版と完全に同一） | 同上 ＋ `rerank_blend`（順位融合。下記） |
| `bm25_qwen3_8b_rerank_qwen3_8b/paper_attrfilter.yaml` | 同上（索引は共有）。ただし検索時は `devices: cuda:0` の1枚のみ（`_embed_query` は devices[0] しか使わないので残りを reranker に空ける） | `qwen3_paper`: `Qwen/Qwen3-Reranker-8B`, **devices=`cuda:1,cuda:2,cuda:3`**, fp16, max_batch_tokens=2048, chunks_per_paper=3／per_index_k=pool_k=**1000**／`attribute_filter`: max_fetch_k=**3000**（上げると faiss が61倍遅くなる） |

### chunk BM25 と paper BM25 を併用する（`bm25s` + `bm25s_paper`）

**`bm25s_paper` 索引は構築済みなのに、長らく単体 ablation でしか使っていなかった。**
両方を RRF で融合すると「chunk 側と paper 側の両方から支持された論文」が上がる。
質問語が論文内の離れた場所に分散していると chunk 側は弱くなり（1チャンクに語が
揃わない）、逆に paper 側は表の1セルや数式のような局所的な根拠が埋もれる。

`eval_retrieval.py` の実測（55件、埋め込みも reranker も無しの純 BM25 ablation）:

| 構成 | recall@5 | **ecr@5** | ecr@10 | ecr@20 | ecr@50 |
|---|---|---|---|---|---|
| `bm25`（chunk のみ） | 0.606 | 0.697 | 0.776 | 0.797 | 0.847 |
| `bm25_paper`（論文のみ） | 0.634 | 0.723 | 0.775 | **0.825** | **0.865** |
| `bm25_dual`（両方 RRF） | **0.663** | **0.748** | **0.791** | 0.815 | 0.843 |

**併用は @5 / @10 に効き、@20 以降は paper 単体が最良**という分かれ方をする。
我々の弱点は @5（外部チームに 0.145pt 離されている位置）なので、そこに
**ecr@5 で +5.1pt** は大きい。ついでに分かったのは **`bm25s_paper` 単体が
`bm25s` 単体より強い**ことで、これは既定構成が chunk BM25 しか使っていない現状の
見直しを示唆する。

上の数字は BM25 だけの ablation なので、埋め込み + reranker を載せた
`bm25_dual_qwen3_8b_rerank_qwen3_8b.yaml` でも同じ向きに出るかは別途確認する。

**注意: `bm25s_paper` の `chunk_id` は `"{paper_id}#paper"` という擬似 ID で
evidence に使えない。** `candidate_recall` は論文単位なので影響を受けないが、
この索引由来のチャンクが ReadingAgent の可視域に入ると evidence の材料にならない。
本走行では **`evidence_f1` が下がらないかを必ず確認する**。

### 属性フィルタ（会議名・年）

`search_style` に `attribute_filter: {enabled: true}` を書くと、質問が明示した
会議名で検索結果を絞り込む（`src/littraceqa/di_pipeline/retrieve/attribute_filter.py`）。
**索引の改修も再構築も不要**。`RetrievalResult.metadata` に既に venue/year が
入っているので、各索引から多めに取ってから落とすだけで、どの indexer にも同じように効く。

```yaml
attribute_filter:
  enabled: true
  safety: 1.5        # 取得件数 = per_index_k / 選択率 * safety
  max_fetch_k: 3000  # 1索引あたりの取得上限。**上げてはいけない**（下記）
  min_results: 10    # 絞り込み後にこれ未満なら絞り込みなしに戻す（fail-open）
  llm_extract: true  # 正規表現が空のときだけ LLM に判定させる（既定 false、下記）
```

**`max_fetch_k` を上げると faiss 検索が桁で遅くなる。** 「絞り込み後に per_index_k 件
残るように」と選択率から逆算するのは自然な発想だが、`per_index_k: 1000` に合わせて
`max_fetch_k: 40000` にしたところ、NAACL(選択率4.3%)で 34,560件の要求になり
**faiss search が 1.5秒 -> 91.1秒（61倍）に膨らんだ**（実測）。`IndexFlatIP` は
全走査して top-k を選ぶので k が効き、しかも取った34,560件はフィルタ後に
per_index_k へ切られるので大半が無駄になる。件数が足りなければ `min_results` の
fail-open で「絞り込みなし」に戻るだけなので、**小さく抑えるのが正しい**。

**発火条件は「会議名が一意に取れたとき」だけ。** 次の場合は抽出せず、
従来と完全に同一のコードパスを通る（本番の質問が会議名を書かなくても損失ゼロ）:

- 年しか書かれていない（コーパスは 2025 が91.3%・2024 が8.7%の2値で、絞る意味が薄い）
- `all venues` を含む（実例の gold は iccv/neurips/icml にまたがっていた）
- 会議名が2種類以上見つかった（引用先の会議名に引きずられないため）

**制約は元の質問から1回だけ取り、サブクエリには渡さない。** `ReadingAgent._decompose()`
が作るサブクエリは「NAACL 2025」を落とすことがあるため、`run()` で
`query.question` から抽出して反復ステップ全体で使い回す（`agent/reading.py`）。
`scripts/eval_retrieval.py` は生の質問を渡すので、`HybridRetriever` 側の
フォールバック抽出でそのまま動く（スクリプトは無改修）。

検証55件での実測: 発火5件、**gold がその制約を満たす率 18/18 = 100%**。
うち3件が multi_paper で伸びしろがある（recall@20 が 0.33〜0.67）。
残り2件は single_paper で既に 1.00。

**年で絞る意味が薄いのは、コーパスで year が venue から一意に決まるから。**
(venue, year) の組は9通りしかなく、**ECCV だけが 2024 で残り8会議は全部 2025**。
つまり「2024 か 2025 か」は「ECCV か否か」と等価で、会議名が取れていれば年は
追加情報を持たない。年だけの制約に意味があるのは 2024（ECCV に絞れる = 91.3%削れる）
のときだけで、2025 では 8.7% しか削れない。

#### LLM 抽出（`llm_extract: true`、既定オフ）

正規表現が空を返した質問だけ、エージェントと同じ LLM に会議名・年を判定させる
（`LLMAttributeExtractor`）。**正規表現を置き換えるのではなく後ろに足す**——検証55件で
正規表現の取り逃がしは無く（会議名に言及する7件を発火5件／`all venues` の意図的除外2件で
すべて正しく処理）、LLM に置き換える利得は実測ゼロだったので、効いている経路には触らない。
LLM が担うのは正規表現が構造上拾えない書き方だけ:

- 別名・正式名称（`NIPS` / `Neural Information Processing Systems` → NeurIPS）
- 語順違い（`a 2025 NeurIPS method` は `_adjacent_year` の「会議名→年」順に合わない）
- 年だけの指定（正規表現版は年単独の制約を作らない）

**誤抽出の被害は3段で抑える**（誤って絞ると gold を候補から落として recall が壊れるため）:

1. プロンプトで「引用先・ベースラインの会議名は対象外」「迷ったら何も返すな」と明示
2. `_validate()` がコーパスに実在する (会議名, 年) の組しか通さない。存在しない会議名を
   答えた時点で**年も含めて返答ごと捨てる**（`AAAI 2025` への返答から year=2025 だけが
   生き残ると、会議名の誤りが ECCV を落とす絞り込みに化ける）
3. 絞り込み後の件数が `min_results` 未満なら従来どおり fail-open

**API 呼び出しは1クエリにつき最大1回。** `extract()`（`HybridRetriever` が制約を
渡されなかったときサブクエリ1本ごとに呼ぶ経路）は正規表現のままで、LLM を通すのは
`ReadingAgent` が元の質問に対して1回だけ呼ぶ `extract_with_llm()` に限定してある。
正規表現で取れた質問では LLM を呼ばないので、増えるのは「会議名が書かれていない質問」
1件につき1回だけ。応答は質問文でキャッシュする。

**reranker を有効にするときは `pool_k` をトップレベルに書く。** RRF 融合後に
チャンクを `pool_k` 件プールし、reranker で並べ替えてから ReadingAgent の
`top_k`（既定20）に絞る。`pool_k` を書かないと reranker に渡る候補が増えない。

### `rerank_blend` — reranker に順位を置き換えさせない（既定オフ）

既定では reranker が RRF 融合後の順位を**完全に置き換える**（`retrieve/hybrid.py`）。
だが reranker は「質問に答えるか」で判定するので、質問文が名指ししないピア gold を
必ず下げる。`reading_expand_rrf/rrf` がランキングB を reranker に通さないのはそのためだが、
**ランキングA の内部では無防備なまま**だった。`rerank_blend` を書くと順位融合になる。

```yaml
rerank_blend:
  original_weight: 0.6   # 融合前（RRF 直後）の順位
  rerank_weight: 0.4     # reranker の順位
  rrf_k: 60
  protect_top: 20        # 融合前の上位N件の「集合」を先頭に残す
```

    score(c) = w_orig / (k + rank_fused) + w_rerank / (k + rank_reranked)

**スコアではなく順位だけを見る**（RRF スコアと yes 確率はスケールが違って足せない。
`_combine_rrf` と同じ理由）。

実装で外せない2点:

- **融合順位を `score` に書き戻す。** 下流はどこも score で並べ直す
  （`agent/reading.py` の貯め込み・`_candidate_papers`・`to_gold_papers`）ので、
  返り値の並び順にしか順位が無いと100%捨てられる。**`protect_top` も同じ理由で
  score に載せる**（最大スコア + 1 を足す）——並び順だけで先頭に寄せても、
  下流が並べ直した瞬間に元に戻る。
- **`rerank(query, fused, len(fused))` に変えても推論コストは増えない。**
  `Qwen3Reranker.rerank` は候補を全件スコアしてから `top_k` で切っているだけ。

**重みは外部チームの 0.59 : 0.41 をそのまま借りない**（本人たちが同じ validation で
選んだ値なので過学習の可能性を認めている）。`original_weight` を 0.0（= 現行の純置換）
/ 0.4 / 0.6 / 1.0（= reranker 無視）で振って選び直す。**cr と ecr を並べて読む**——
reranker を弱めると選別が緩むので、ecr が下がって cr が上がる向きに出ることがある。

reranker が受け付ける params（実装は `src/littraceqa/di_pipeline/retrieve/`）:
- `none`: params なし。
- `qwen3`（`reranker.py`）: `model`(既定 `Qwen/Qwen3-Reranker-0.6B`), `device`, `fp16`,
  `batch_size`, `max_tokens`, `instruction`（判定プロンプト）, `compile`（既定 true、
  可変長入力なので dynamic compile）。
- `qwen3_paper`（`reranker.py`）: `qwen3` の全paramsに加えて `chunks_per_paper`（1論文の
  代表テキストに含める本文チャンク数、既定3）, `devices`（"cuda:1,cuda:2,cuda:3" の形で
  **マルチGPU**）, `max_batch_tokens`（件数ではなくパディング後トークン量でバッチを切る）。
  **チャンク単位ではなく論文単位で並べ替える**。詳細は下の節。
- `qwen3_vl`（`vl_reranker.py`）: `model`(既定 `Qwen/Qwen3-VL-Reranker-8B`), `device`,
  `fp16`, `batch_size`, `prompt`, `use_images`, `max_image_docs`。

**`device` は実行時に空いているGPUへ書き換える前提の値**。yaml に書いてあるのは
書いた時点で空いていたGPUなので、埋め込み索引の構築と同居させないこと
（8B系は fp16 でも約15〜18GB占有し、RTX3090(24GB) では KV cache 分の余裕が薄い）。

### 論文単位 reranker（`qwen3_paper`）

チャンク単位（`qwen3`）との違いは、候補を paper_id でまとめ、
`[venue year] title` + 上位 `chunks_per_paper` 件の本文を1つの代表テキストにして
**論文につき1回だけ推論する**こと。推論回数が「候補チャンク数」から「候補論文数」に
落ちるので `pool_k` を大きくできる。`candidate_recall` は論文単位の指標で
`to_gold_papers()` が論文ごとに max を取るため、reranker の判断がそのまま論文順位になる。

実測コスト（Qwen3-Reranker-8B, fp16, `max_batch_tokens: 2048`, pool_k=1000）:

| 条件 | 1 rerank |
|---|---|
| 3GPU / 86論文 | 8.5秒 |
| 3GPU / 255論文 | 63.7秒 |
| 3GPU / 509論文 | 22.9秒 |
| 1GPU / 998論文 | 152.6秒（3GPUなら56.6秒 = 2.7倍） |

**`devices` で複数GPUを指定するとスレッド並列になる**（CUDA forward が GIL を解放する。
クエリのたびに走るのでプロセス起動方式は使わない）。このとき **`torch.compile` は
自動で無効化される**——compile 済みモデルを複数スレッドから呼ぶと dynamo が
「FX symbolic trace of a dynamo-optimized function」で落ちるため。compile は
実測でほぼ無効果（188 vs 212ms）なので損失はない。

**`batch_size` ではなく `max_batch_tokens` で制御する。** 論文代表テキストは長さが
ばらつく（実測 中央313〜661tok・max 2116）ため、件数固定だと長い外れ値で
`batch_size × 最長` がVRAMを食い、**batch_size=4 でピーク22GB、8/16 は即OOM**した。
トークン量で切れば短い論文を多数詰めつつVRAMを一定に保てる（索引構築と同じ対策）。

`iterative.yaml` / `reading_llmcount.yaml` / `simple.yaml` / `verifying.yaml` は削除済み
（`iterative` は停止条件が「見つかった論文の本数」で top_k=20 の時点で初回から満たされ、
反復ループが事実上空回りしていた）。以後 agent_style は `reading` 一本で運用する。

### 反復ループの拡張（`agent_style` の任意キー、既定オフ）

仕様は `docs/search_agent_spec.md`。反復ステップ間で変えられるのが
**サブクエリの集合だけ**だった状態を4方向に広げる。**キーを書かなければ
現行と1ビットも変わらない**（`retrieve/` は無改修、`Retriever` Protocol も不変）。

土台は `SubqueryRun`（step / subquery / 検索が返した順の results）の列。
`chunks` は chunk_id からの引き当て用に残し、**ランキングの出所だけを移した**
（捏造チェックと evidence 引きが chunk_id で引いているため）。

**プリセットの yaml は置いていない**（`reading_loop/` は削除した）。使うときは
任意の agent yaml の `params` に直接書く。

| params | 既定 | 何をするか |
|---|---|---|
| `subquery_merge` | `max` | `rrf` でサブクエリ間マージを順位融合にする |
| `grounded_refine` | false | `_refine()` に候補上位と「効かなかったサブクエリ」を渡す |
| `pool_rescore` / `pool_prune_to` | false / null | プール全体を元の質問で1回リランク・剪定 |
| `adaptive_depth` | なし | スコアの落差で採る件数を決める |
| `max_steps: 2` | 3 | **反復を1周減らす。全指標で改善し検索が40%減る**（下記） |
| `title_protect` | なし | 質問が名指しした論文を候補列の上位へ引き上げる | `reading_expand_rrf/protect` |

**`title_protect` は実測で効果ゼロだった（既定オフのまま）。** 質問文の論文識別子から
その論文を `promote_to`（既定10位）へ引き上げる仕組みで、**名前の特定自体は完璧に
動く**（55件中26件で発火、名指しされた28本が **28/28 = 100% gold**、誤検出0本）。
それでも指標が1桁も動かないのは、**28本すべてが既に候補1〜4位に居る**から
（順位の内訳: 1位25本 / 3位2本 / 4位1本）。引き上げる対象が存在しない。

これは `single_paper` の cr@1 が 1.000 で飽和していることの言い換えでもある——
**質問が名前で指した論文を検索が取り逃すことは、このコーパスでは起きていない。**
外部チームが「質問に論文名があるのに14位」を救う仕組みを入れているのは、
向こうの @1 が 0.767 だからで、我々には要らない。

抽出そのもの（`paper_titles.TitleIndex.lookup_text`）は精度100%で使える資産なので、
「質問が論文を名指ししているか」を別の目的（分岐・診断）で使いたくなったら再利用できる。

**サブクエリ間の `max` マージは比較可能でない値を突き合わせている。** 同じチャンクが
複数のサブクエリで当たったときスコアが高いほうを残すが、そのスコアは
**異なるサブクエリに対する reranker の yes 確率**で、「サブクエリBにとっての0.9」と
「サブクエリAにとっての0.7」に共通の意味は無い。`rrf` は順位しか見ないので
この問題が起きない（融合は indexer 間と同じ `RRFFuser` にサブクエリ1本を1 run として渡す）。
multi の gold は「1本のサブクエリだけが見つける論文」が多いので、そこに効く。

**`_refine()` はコーパスの反応を一度も見ていない。** 材料が読解 LLM の `missing`
だけなので、実測では `EasySpec` を分光解析ソフトと誤解したまま2ステップ暴走したり
（step0 では gold を1位で引けていた）、`Ours500→1` / `500:1` / `500 to 1` と
表記ゆれの総当たりに流れたりしていた。`grounded_refine` は候補上位 N 本の
`[venue year] title` と、上位に1本も残らなかったサブクエリの名指しを渡す
（**追加の LLM 呼び出しはゼロ**、プロンプトが太るだけ）。

**`adaptive_depth` は task_family の推定に依存しない。** 1位と `probe_rank` 位の
落差を見て、大きければ `shallow_k` 件・平坦なら `deep_k` 件を採る。single は1位が
飛び抜けるので自動的に浅く、multi は平坦なので深くなる。**retriever には常に
`deep_k` を渡して取り、切るのはエージェント側**——reranker の推論件数は
search_style の `pool_k` で決まるので**推論コストは1件も増えない**。

**`pool_rescore` は既定オフのまま出す。** reranker は「質問に答える論文」を
選べているほど no_evidence gold を落とすので、**ecr が上がって cr が下がる**向きに
出やすい。評価は必ず cr と ecr を並べて読む。

#### 表チャンクを「論文の代表スコア」に使わない（未実装。**いちばん確かな打ち手**）

`to_gold_papers(agg="max")` は論文の最高スコアのチャンク1つで論文を代表させる。
**そこに表チャンクが選ばれると論文の順位が壊れる。** 表チャンクは数値と短いラベルが
密なので、BM25 も reranker も語の重なりだけで高いスコアを出しやすく、
論文が質問の主題でなくても表1枚で代表スコアが跳ね上がる。

    代表スコア = max(表以外のチャンクの最高スコア,  w × 表チャンクの最高スコア)

`w` を振ると **0.85 以下は完全に同値**（0.8 / 0.5 / 0.1 / 0.0 が1桁まで一致）なので、
**閾値ではなく規則そのものが効いている**。`w = 0` ＝「表チャンクは代表にしない
（表しか無ければ表を使う）」と書けて、**自由パラメータが無い**。

**`figure` / `equation_algorithm` を一緒に下げると悪化する。** 落とすのは `table` だけ。
`agg="sum"` にすると効果がほぼ消えるのが傍証で、**これは max 集約に固有の歪み**。

**「表しか無い論文には表スコアを使う」というフォールバックを入れてはいけない。**
親切に見えるが実測で負ける（multi@5 0.758 -> 0.720、4分割で悪化3セル）。表しか
手掛かりが無い論文が488本あり、**それを沈めること自体が効いている**（スコア0 になるだけで
候補列からは消えないので、ランキングB が押し上げれば戻ってくる）。

**表チャンクを evidence 用途で落としてはいけない。** 重みを掛けるのは論文の代表スコア
だけで、チャンクプールは無変更。表チャンクは読解 LLM にそのまま渡り `evidence` にも
出せる（gold の `primary_evidence_type` は table が17件で最多）。実装は
`RetrievalResult.chunk_type == "table"` を見る（`chunk_id` の接頭ではなく）。

**むしろ読解には表を積極的に見せたほうがよい**（代表スコアの話と逆向きに見えるが両立する。
表は「論文が主題でなくても高くスコアされる」ので順位付けには雑音だが、
「その論文が本当に正解なら具体的な数値を含む最良の証拠」になる）:

| 読解 LLM に表チャンクを | 確認 | 見落とし | 確認率 |
|---|---|---|---|
| 見せた | 29 | 12 | **71%** |
| 見せない | 23 | 22 | **51%** |

**プールに表があるのに見せていない gold が33本（うち見落とし18本）。** 1論文あたり
プールには中央15本のチャンクがあるのに `chunks_per_paper: 2` で上位2本しか見せていない。
「表が1本でもあれば2本のうち1本を表にする」規則が候補（因果ではなく相関である点に注意）。

#### 表除外は**単独で入れる**（`max_steps: 2` と抱き合わせない）— フル走行で確定

3構成を同じ search_style で走らせた実測（55件、`--production-input`）:

| | multi cr@20 | multi ecr@20 | ecr@20 | ecr@50 | single cr@1 |
|---|---|---|---|---|---|
| `verdict_anchor` | 0.839 | 0.908 | 0.952 | 0.980 | 0.885 |
| **`notable`（表除外のみ）** | **0.859** | **0.925** | **0.961** | **0.989** | 0.885 |
| `steps2_notable`（表除外 + `max_steps: 2`） | 0.833 | 0.899 | 0.947 | 0.974 | 0.885 |

**表除外は +0.020、`max_steps: 2` はそれを打ち消して -0.026。** 表除外だけが正解で、
`max_steps: 2` は**採用しない**。@10 だけ横ばい（-0.002〜-0.011）で他は全部改善し、
**single は1桁も動かない**（cr@1〜@50 が完全に不変）。

#### ⚠ オフラインの土台は**生成日を必ず確認する**（この誤りの原因）

`runs_fat.jsonl` は **2026-08-03** の走行で、`subquery_count: 4` の導入（08-05）より前。
step1 のサブクエリが305本（平均5.5本/クエリ）あり、現行の4本上限で走る run とは
**別物のランキングA** を作っていた（現行は step1 が133本）。

この土台1本で「`max_steps: 2` ＋ 表除外」を選んでしまった。**4分割検証で過学習は
潰していたのに、土台そのものが現行と別物という一段上の問題を見落とした。**
「step2 を削っても平気」は step0/1 が今の2倍太い土台の上でしか成立しなかった。

**2本目の土台で測り直すと効果量が変わる:**

| 打ち手 | 土台1（08-03） | 土台2（08-08） |
|---|---|---|
| 表を代表から除外 | multi@5 +0.049 / 悪化 0/8 | multi@5 +0.017 / 悪化 0/8 |
| `agg="sum"` | +0.026 / 悪化 0/8 | +0.037 / 悪化 0/8 |
| `neighbors: 50` | 悪化 6/8 | 悪化 2/8 |

**表除外が2本とも正・悪化ゼロだったことがフル走行で報われた**（`agg="sum"` は
土台で優劣が逆転するので未検証のまま）。**今後は土台3本
（`runs_fat` / `runs_steps2_notable` / `runs_notable`）で測る。**

#### `agg="sum"`（単独では改善するが、上の組と役目が重なる）

`runs_fat.jsonl` の全長A・55件での実測（基準は `verdict_anchor.yaml`）:

| | multi@5 | multi@10 | multi@20 | 全体@5 | 全体@10 | 全体@20 |
|---|---|---|---|---|---|---|
| 基準（max_steps 3 / agg=max） | 0.677 | 0.844 | 0.914 | 0.812 | 0.918 | 0.955 |
| **`max_steps: 2`** | **0.735** | **0.861** | **0.931** | **0.860** | **0.927** | **0.964** |
| `to_gold_papers(agg="sum")` | 0.703 | 0.868 | 0.922 | 0.843 | 0.930 | 0.959 |
| 両方 | 0.706 | 0.859 | 0.931 | 0.845 | 0.926 | 0.964 |

**`max_steps: 2` は全指標で改善しつつ検索を327回（全812回の40%）減らす。**
step2 まで走ったのは55件中35件で、そのぶんの LLM 呼び出し2回も消える。
比較は verdict も揃えた（max_steps=2 なら step1 時点の `selected` を使う）。

`agg="sum"` は「何チャンクがヒットしたか」を効かせるもので、`to_gold_papers` は
既に `agg` を受け取るので `_build_prediction()` の呼び出しに1語足すだけ。8指標とも改善する。

**併用は加算的でない**（全体 ecr@5 の差分: sum 単独 +0.032 / max_steps=2 単独 **+0.048** /
両方 +0.033）。**`max_steps: 2` ＋ 表除外の上に `agg="sum"` を足すと4分割で悪化セルが
4つ出る**——表除外が同じ歪み（max 集約）を直しているので役目が重なる。**足すのではなく選ぶ。**

**土台が `runs_fat.jsonl` 1本しか無い**（土台4本はチャンク単位の情報を持たないので
ランキングA の作り方を変える打ち手は土台間で検証できない）。代わりにクエリを4分割した:

| 分割 | `agg=sum` | `max_steps=2` |
|---|---|---|
| 前半 / 後半 | +0.025 / +0.039 | +0.049 / +0.048 |
| 奇数番 / 偶数番 | +0.054 / +0.009 | +0.077 / +0.019 |

@5 は両方とも4分割すべてで正だが、@10 以降は `agg=sum` が2セルで負になる。
**`max_steps: 2` はどの分割・どの k でも負にならない**ので、先に走らせるならこちら。

#### オフライン再生 `scripts/replay_merge.py`（本走行しない）

`scripts/run_search.py --dump-runs runs.jsonl` でサブクエリ単位の検索結果を落とし、
そこから候補列だけを組み直す。土台は `reading_normal/fat.yaml`（`retrieve_top_k: 100`）で
**1回だけ**フル走行すればよく、以後100以下の任意の `retrieve_top_k` と
上表の設定を数十秒で振れる。`replay_expansion.py` と同じく **ReadingAgent 本体の
メソッド（`_merged_results` / `_depth_for`）をそのまま呼ぶ**。

```
uv run python scripts/replay_merge.py --runs runs_fat.jsonl --pred predictions_fat.jsonl \
  --agent configs/agent_style/reading.yaml --set subquery_merge=rrf --ks 5,10,20,50
```

**`grounded_refine` だけは再生できない**（サブクエリ自体が変わるのでフル走行が要る）。
`pool_rescore` も GPU が要るので再生対象外（`pool_prune_to` は振れる）。
他も「サブクエリを固定した条件下」の数字なので**下限として読む**。

### 検索エージェント不要のシステム（最優先。`configs/agent_style/agentless/`）

**LLM を1回も呼ばずに候補論文の順位を出す構成を最優先で開発する。** 仕様と現在地は
`docs/agentless_spec.md`。エントリポイントは `scripts/eval_retrieval.py --agent` で、
これは**候補列の並べ替えだけで完結する部分**（A/B の RRF 統合と
`paper_score_skip_chunk_types`）しか載せないので、分解・反復・読解は走らない。

現在地（55件、`bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100` +
`agentless/agentless.yaml`）と、フル走行との差:

| | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| **エージェント抜き ecr** | **0.637** | 0.775 | 0.889 | 0.958 | 0.979 |
| フル走行 `notable` ecr | 0.605 | **0.842** | 0.915 | 0.961 | 0.989 |

**@20 以降はほぼ互角**（cr@50 は 0.9591 で完全一致）、**@1 はエージェント抜きが上**、
**@5 だけ -0.067**。差の出所は「サブクエリ4本の融合」と「`anchor_from: verdict`」の
2つだけなので、**この2つを LLM 抜きで埋めるのが中心課題**。

- `agentless/agentless.yaml` : 素の版（`notable` から `anchor_from: verdict` を外しただけ）
- `agentless/score_anchor.yaml` : + `anchor_from: score`。**verdict の LLM 不要版**で、
  reranker の yes 確率がしきい値を超えた論文を起点にする。エージェント側の土台では
  verdict の伸びの大半を回収する（@20 0.938 -> 0.956、verdict は 0.961）が、
  **エージェント抜きの土台では起点が平均1.2本にしか増えず効かない**。
  原因はプールの深さと見られる（予備測定の土台は top-20、本番は 200 件要求）。
  **深い土台で測り直すまで採否を決めない。**

**エージェント側で確定したつまみをそのまま持ち込まない。** エージェント抜きの A は
「1本のクエリの reranker 順位」で分布が違う。予備測定では表除外・`combine_rrf_k`・
`related_weight` の3つで最適値が食い違った（詳細は `docs/agentless_spec.md` §4）。

⚠ **エージェント抜きでは `evidence` を出せない**（読解を走らせないので
`evidence_chunk_ids` が無い）。渡せるのは候補列＝論文の順位まで。

### 生質問1位のピン留め（`rawq_pin`、既定オフ）

**元の質問そのものを step0 に5本目のサブクエリとして足し、その検索が返した論文1位を
ランキングA の先頭に固定する。** 実装は `agent/reading.py` の `_rawq_ranking()` /
`_pin_front()`、詳細な実測は `docs/offline_findings_spec.md` §8。

`_decompose()` が作る4本は元の質問の**語の組み合わせ**を保たない（step0 に質問文
そのものが入るのは 0/55 件）。LLM を一度も呼ばない `eval_retrieval.py` が
**@1 でフル走行を上回る**のはそのため（ecr@1 0.637 vs 0.605 / cr@1 0.557 vs 0.529）。
1位の中身を突き合わせると 40件は同じで、違う15件の内訳は
**生質問だけ gold 4 / 土台だけ gold 1 / 両方 6 / どちらも違う 4 = 正味 +3件**。

**プールに足すだけでは1桁も動かない。** `subquery_merge: "max"` は chunk_id ごとに
最高スコアを残すだけなので、生質問が引くチャンクはすでにどれかのサブクエリが
同等以上のスコアで持っている。実測でも**候補列は 34/55 件で並びが変わるのに
cr / ecr は全 k で小数4桁まで不変**だった。**「足す」と「順位を動かす」は別の打ち手。**

**固定先は統合の前（ランキングA の先頭）。** 統合後の候補列に置くだけだと @1 しか
動かないが、A の先頭に置くと `_anchor_papers()` 経由で**ランキングB の起点も変わる**
ぶん @5 以降にも効く（土台 notable の ecr@5: 統合後 0.8465 / A の先頭 **0.8556** /
固定なし 0.8419）。**2本以上は固定しない**（ecr@5: N=1 0.8556 / N=2 0.8247 / N=3 0.7975）。

土台7本での差分（total_ecr。上3本は全長A、下4本は候補列を A にした過小評価版）:

| 土台 | @1 | @5 | @10 | @20 | @50 | 土台自身の ecr@1 |
|---|---|---|---|---|---|---|
| `notable` | +0.032 | **+0.014** | +0.011 | +0.006 | 0.000 | 0.605 |
| `steps2_notable` | +0.036 | **+0.015** | +0.006 | -0.006 | 0.000 | 0.601 |
| `fat`(08-03) | +0.041 | **+0.026** | -0.006 | -0.005 | 0.000 | 0.596 |
| `k100_cand50` | 0.000 | **+0.015** | +0.017 | +0.006 | +0.006 | 0.637 |
| `fat`(候補列) | +0.005 | **+0.023** | 0.000 | +0.005 | -0.005 | 0.632 |
| `chunk_cand50` | -0.032 | **+0.009** | -0.003 | +0.006 | +0.006 | 0.669 |
| `b_merged` | -0.020 | **+0.009** | -0.005 | 0.000 | -0.005 | 0.657 |

**@5 が7土台すべてで改善するのが採用根拠**（+0.009〜+0.026）。@5 は外部チームに
0.145pt 離されている位置。@10 / @20 は符号が割れるのでノイズとして読む。

**⚠ @1 は土台次第で符号が変わる。** 生質問1本の ecr@1 は **0.637 で固定**なので、
**土台自身の @1 がそれを上回っていると負ける**（上表の右端と符号が完全に対応）。
既定構成（`notable`, 0.605）では勝つが、`cand50` 系のように @1 が 0.66 を超える
構成に足してはいけない。「生質問の1位は常に強い」ではなく
**「生質問の1位は 0.637 で頭打ち」**と読む。

コストは step0 の検索が1本増えるだけで LLM 呼び出しは増えない。足した run は普通の
サブクエリなので `chunks` に積まれ、**生質問が引いたチャンクも evidence に出せる**。
ただし**プールが増えるぶん読解 LLM が読む候補は変わる**ので、
**この打ち手はオフライン厳密ではない**（統合層のつまみと違う）。数字は下限として読む。

### サブクエリ生成プロンプト（`agent/reading.py` の `CORPUS_NOTE`）

**「Web検索エンジンではない」と書かないと LLM は Google 検索クエリを書く。**
予測ファイルの trace を集計した実測で、`_refine()`（2周目以降の再分解）が作る
サブクエリの **29〜41%** が `site:arxiv.org` / `filetype:pdf` /
`site:openaccess.thecvf.com/content/CVPR2025/html "avg.Col." "UniAD"` のような
Web検索演算子付きだった。投げ先はローカルの BM25 と faiss なのでこれらは1件も
ヒットせず、`max_steps: 3` の2周目・3周目の検索が丸ごと空振りしていた。

| 構成 | step0 | step1 | step2 | 影響したクエリ |
|---|---|---|---|---|
| `bm25_qwen3_8b_rerank_qwen3_8b_paper_attrfilter` | 0.0% | 35.6% | 40.5% | 37/55 |
| `bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter` | 0.0% | 32.3% | 31.0% | 30/55 |
| `bm25_specter2_body_qwen3` | 0.0% | 29.6% | 32.5% | 39/55 |

step0（`_decompose()`）では0%だったが同じ誤解が起きうるので、`CORPUS_NOTE` は
分解・再分解の両方の先頭に置いてある。

**属性制約が取れたときは、サブクエリの先頭に `[NAACL 2025]` を付けさせる**
（`_constraint_note()`）。絞り込み自体は `attribute_filter` が担当するので、これは
**検索語としての**制約。title_abstract チャンクの本文は実際に
`[ACL 2025] タイトル…` とこの表記で始まる（`preprocess/mineru_chunker.py`）ので、
同じ表記が BM25 の語として効く。`_decompose()` は自発的に会議名を残していたが
`_refine()` では落ちていたため、両方に明示する。

**`reading` の打ち切りは task_family に依存しない。** 反復検索の停止条件は
`_read_and_judge()` が返す LLM の `sufficient` 判定のみ（`src/littraceqa/di_pipeline/agent/reading.py`）。
本番入力に `task_family` が無く、推定しても正解率0.67程度で当てにならないため、
本数決定の経路から task_family を外した。

### 提出論文は選定しない（`submit_from: candidates`、既定）

**どれを提出するかを決めるのは読解チーム側の担当**なので、検索エージェントは
候補列の順位を渡すところで止める。`gold_papers` は `candidate_papers` の順位そのまま
（`max_papers: 10` で頭打ち）で、読解 LLM が返した `paper_ids` は使わない。

**それでも `_read_and_judge()` は呼ぶ。** 1回の LLM 呼び出しが返す3つのうち、
選定（`paper_ids`）以外の2つは別の役割を持っているため:

- `sufficient` … **反復の停止条件そのもの**。これが無いと `max_steps` 固定になる
- `evidence_chunk_ids` … 根拠チャンク（`evidence_f1`）。CLAUDE.md 冒頭のとおり
  evidence の特定は ReadingAgent の担当

`submit_from: llm` にすると従来どおり LLM の選定結果を提出する（選定込みで測りたい
ablation 用）。**指標の読み方が変わる点に注意**——選別しないので `paper_recall` は
上がり `paper_precision` は下がる。`candidate_recall` / `evidence_candidate_recall`
は候補列を見る指標なので**一切変わらない**（打ち手の評価は従来どおりこちらで読む）。

### 回答は生成しない（`_generate_answer` は削除済み）

`Prediction.answer` は常に空（`Answer()`）。freeform / multiple_choice / table を
埋めるのは読解チーム側の担当なので、検索エージェントが渡すのは **candidate_papers と
evidence まで**。

- **LLM 呼び出しが1クエリにつき1回減った**（1周あたり最大2回＝分解1 + 読解1）。
- `--options-file`（multiple_choice の選択肢を結合する oracle 実行）も一緒に削除した。
  選択肢は回答生成にしか使い道が無く、残しても黙って無視されるだけになるため。
- 回答生成の上限を測りたくなったら `scripts/generate_oracle_answers.py`
  （gold paper を渡して回答と evidence を作らせる単体スクリプト）が別にある。

### サブクエリの本数は4本固定（`SUBQUERY_COUNT`）

`_decompose()` は task_family で件数を振り分けない。以前は single「1〜3個」/
multi「3〜6個」と分けていたが、**その分岐のためだけに `TaskFamilyClassifier` が
クエリ1件につき LLM を1回呼んでいた**（本番入力に task_family が無いため）。

買えていたのは実測で平均0.58本（`predictions_8b_chunk_b_merged.jsonl` の trace、
single 26件が平均3.08本・25件が上限の3本、multi 29件が平均3.66本・17件が下限の3本）。
推定精度も **LLM 0.67 / ヒューリスティック 0.673**（55件実測）で差が無い。
両者を挟む4本に固定し、LLM 呼び出しを1回減らした。

これで**1クエリあたりの LLM 呼び出しは分解1 + 読解1（+ 再分解と再読解が最大2周）**
になった。`TaskFamilyClassifier` 自体は `paper_cutoff: task_family` モードのために
残してあるが、現行の yaml は全部 `paper_cutoff: llm` なので呼ばれない。

#### 本数は返り値も切る（`subquery_count`、既定4）

**プロンプトで本数を頼むだけでは守られない。** `_decompose()` は本数を書いていたので
概ね守っていた（実測 平均3.3本・最大6本）が、**`_refine()` は本数を一言も書いて
いなかったため平均8.2〜9.3本・最大20本**まで膨らんでいた（`runs_fat.jsonl`、55件812本。
step0 180本 / step1 305本 / step2 327本）。**サブクエリ1本 = 検索1回 = reranker が
`pool_k` 件を推論する量**なので、これがそのまま走行時間になる（`pool_k: 1000` の構成で
実測 1.73分/本 → 55件で23時間）。

膨らんだぶんは**検索力に一切効いていない**。812本を1本ずつ抜いて候補列を組み直すと、
**抜くと ecr@50 の gold が減るのは5本だけ**（0.6%）:

| | 抜くと gold が減る |
|---|---|
| step0 `_decompose()` | 4/180 (2.2%) |
| step1 `_refine()` | **0/305 (0.0%)** |
| step2 `_refine()` | 1/327 (0.3%) |

各ステップを先頭N本に絞った再生（`replay_merge` と同じ経路、`subquery_merge: rrf`）:

| N | 総本数 | 削減 | cr@20 | ecr@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|---|
| 2 | 253 | -69% | **0.796** | **0.878** | 0.836 | 0.912 |
| 3 | 375 | -54% | 0.770 | 0.847 | 0.841 | 0.918 |
| **4（既定）** | 461 | -43% | 0.774 | 0.852 | 0.836 | 0.914 |
| 6 | 611 | -25% | 0.770 | 0.841 | 0.841 | 0.912 |
| 上限なし | 812 | 0% | 0.779 | 0.847 | 0.841 | 0.914 |

`max` マージでも同じ向き（N=2 で ecr@20 0.874 vs 上限なし 0.850）。**増やすほど
良くなる関係にはなっていない**——サブクエリを足すほど比較可能でないスコアと裾の雑音が
混ざって上位が薄まる（`subquery_merge` の節と同じ理屈）。N=3〜8 の差は55件では
誤差幅なので `_decompose()` と同じ4本に揃えた。**「4より上に価値が無い」ほうが
確かな結論**で、2 が両マージ方式で @20 最良だったのはノイズと区別が付かない。

問題は重複ではなく**総当たり**だった。正規化トークンの Jaccard≥0.6 のペアは
step0 21.3% / step1 10.2% / step2 4.6% で再分解ほど低い。実際に出ていたのは
q_021 step1 の `SimLingo trained on Bench2Drive Base split` /
`SimLingo only uses the Bench2Drive Base dataset` … を手法名 × 言い回しで20本、という形。
重複除去では削れないので**本数の上限**で切る。プロンプトにも
「言い換えを並べるな」と明示してある。

**この変更は既定の挙動を変える**（`subquery_count` を書かなくても4本で切られる）。
2026-08-05 より前の `results/experiments.jsonl` の行は上限なしで走っているので、
本数まわりを比べるときは実行日を見る。

### 3.1 評価の作法

**目標は `candidate_recall` を上げること。提出物側の指標は既定で出さない。**
`evaluate.py` が返すのは `candidate_recall` / `evidence_candidate_recall` の系列だけで、
`paper_precision` / `paper_recall` / `paper_f1` / `evidence_*` / 回答系
（`multiple_choice_accuracy` / `freeform_exact_match` / `table_*`）は
**`--metrics all` を付けたときだけ**足される。提出論文の選定も回答生成も読解チーム側の
担当（`submit_from: candidates`）で、我々が動かせない数字を並べると、その上下を
改善・悪化として読んでしまうため。

`results/experiments.jsonl` と `report/*.md` も自動的に検索側だけになる
（`run_search.py` は `evaluate.py` を既定のまま呼ぶ）。過去の行には提出物側の指標が
残っているので、`experiments_report.py` の詳細展開ではそのまま読める。一覧表と
横棒グラフの主役は cr@20 に差し替えてある。

**実験は必ず tmux セッションの中で回す。** 1構成4〜5時間かかるので、端末やエージェントの
セッションが切れた時点でプロセスごと落ちると数時間が丸ごと消える（実測で、索引読み込み中に
セッション終了で殺されて0件のまま消滅した）。`nohup` / `setsid` でも切り離せるが、
tmux ならあとから `attach` して生の進捗を見られるので tmux に統一する。

```
mkdir -p logs
tmux new-session -d -s littrace-exp \
  "PYTHONUNBUFFERED=1 uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/{検索構成}.yaml \
  --agent configs/agent_style/{エージェント}.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions_{識別子}.jsonl \
  --production-input 2>&1 | tee logs/{識別子}.log"

tmux ls                        # 生きているか
tmux attach -t littrace-exp    # 進捗を直接見る（抜けるのは Ctrl-b d）
tail -f logs/{識別子}.log      # アタッチせずに追う
```

**`PYTHONUNBUFFERED=1` を付ける。** stdout がパイプ（`tee`）に繋がると Python は
ブロックバッファリングになり、`N/55 完了` の進捗が数十分ぶん溜まってから一気に出る。
走っているのか固まっているのか区別できなくなるので必ず付ける。

ログは `logs/`（`.gitignore` 済み）に置く。予測ファイルは全55件が終わってから一括で
書き出される実装なので、**途中経過は `wc -l predictions_*.jsonl` では測れない**——
進捗はログの `N/55 完了` で見る。

**評価は `--production-input` を付けて回す。** `data/validation_inputs.jsonl` は55件
すべてに `task_family` が入っているが、本番入力には無い（`query_id` / `question` /
`answer_types` / `table_schema` の4つだけ）。与えたまま評価すると「正解を教えてもらった
状態」の点数になり本番と乖離する（`reading` は `task_family` を提出本数に使わないため
影響は小さいが、`_decompose()` のサブクエリ分解の文言分岐にはまだ使っている）。

結果は `results/experiments.jsonl` に自動で追記される（config名 + metrics + timestamp）。
加えて実行1回につき、設定と指標とLLMコメントをまとめた Markdown が
`report/{timestamp}_{process名}_{search名}_{agent名}.md` として1枚書き出される
（`scripts/run_search.py` の `write_report()`）。`results/` `report/` はどちらも
各自のローカルな実行記録で `.gitignore` 対象（チーム共有はしない、生成物なので消えても
再実行すれば復元できる）。
LLM は非決定的（Opus 4.8 は temperature を受け付けない）でクエリは55件しかないので、
数ポイントの差はノイズの可能性がある。結論を出す前に複数回まわすこと。

**`report/*.md` にクエリ診断は書かない。** レポートに載せるのは設定・指標・LLMコメント
だけで、1クエリ1行の内訳は `scripts/audit_report.py` が作る単一HTMLに集約した。
実験のたびに55行の表が `report/` に積み上がっても、実験どうしを見比べられないため。

集約先のHTMLには**実験セレクタ**があり、`results/experiments.jsonl` の各行のうち
`output` の予測 JSONL が実在するものが自動で並ぶ。同じクエリを実験間で切り替えて
「どの構成なら gold が何位に来たか」を直接比べられる（`by_exp` / `top_by_exp`）。
**実験を1本回したら、このHTMLを作り直して公開先に反映する**:

```
uv run python scripts/audit_report.py --audit audits/query_audit.jsonl \
  --output report/query_audit.html
```

公開先（Artifact）: https://claude.ai/code/artifact/3065240a-0bd9-4337-b922-186d7902241d
更新は同じURLに再公開する（新しいURLを作らない）。**スクリプトからは自動更新できない**
ので、再生成と公開は手動の1ステップとして残る。

1件ずつ深掘りするときは従来どおり `scripts/inspect_candidate_recall.py` を使う。
`scripts/evaluate.py --per-query` は残してあるが、`run_search.py` は付けずに呼ぶ。

#### 論文→論文展開の調整は `scripts/replay_expansion.py` で（本走行しない）

展開・統合は `candidate_papers`（論文IDの列）を入れ替えるだけの処理で、LLM も
検索索引も要らない。既存の予測ファイルを土台にすれば**15秒**で回るので、
`related_offset` / `neighbors` / ソース構成の比較は本走行（4〜5時間）ではなくここで詰める。

```
uv run python scripts/replay_expansion.py \
  --paths configs/paths/default.yaml --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter.yaml \
  --agent configs/agent_style/reading_expand_rrf/rrf.yaml \
  --pred predictions_8b_chunk_cand50.jsonl \
  --output predictions_rrf_offline.jsonl \
  --set related_offset=25          # agent yaml の expansion を1つだけ上書き（振るとき用）
```

**ReadingAgent 本体のメソッド（`_combine_rrf` / `_expand_candidates`）をそのまま呼ぶ。**
ロジックを書き写すとオフラインの結論が本走行に効かなくなる。

**限界**: 土台の予測に残っているのは候補上位50本なので、ランキングA は50位までしか
復元できない。本走行は打ち切り前の全長を使うので、ここの数字は**下限**として読む。

**展開まわりは実行のたびにブレていた。** `bib_coupling` の近傍スコアは set を走査して
作るため、同点の並びが文字列ハッシュの乱択でプロセスごとに変わっていた（実測で
55件中23件の候補列が入れ替わり、cr@20 が 0.4pt 動いた）。同点を paper_id で決めるよう
直してあるので、いまは同じ入力なら常に同じ候補列が出る。

#### `evidence_candidate_recall`（ecr@k）— 取りに行ける gold だけの検索力

**multi_paper の gold には、検索では原理的に取れない論文が混ざっている。**
`gold_papers` に名前はあるのに `evidence` が1件も紐づいていない論文が
**gold 120本中29本（24%）**あり、中身は「質問文が名指ししていない同トピックの
ピア論文」だった。q_036「TCM の batch size は？」の gold に IMM / sCT /
Consistency Models Made Easy が並び、q_039「IMM の kernel function は？」の gold が
**まったく同じ4本**、という作りになっている。質問文が求めているのは
「TCM の batch size」なので、埋め込みを大きくしても reranker を強くしても
そのクエリベクトルの近傍にピア論文は来ない。

実測でこの2群は当たり方がはっきり違う（`predictions_8b_chunk_b_merged.jsonl`, micro）:

| 分母 | @10 | @20 | @50 |
|---|---|---|---|
| 根拠付き 91本 | 0.615 | 0.736 | **0.813** |
| 根拠なし 29本 | 0.103 | 0.207 | **0.345** |

そこで **`candidate_recall` の分母を根拠付き gold に絞った `evidence_candidate_recall`**
を並べて出す（`evaluate.py` の `evidence_backed_paper_ids()`。`eval_retrieval.py` にも
同じ定義で `evidence_recall@k` 列がある）。マクロ平均は:

| | cr@10 | ecr@10 | cr@50 | ecr@50 |
|---|---|---|---|---|
| multi (29件) | 0.475 | **0.650** | 0.690 | **0.825** |
| total (55件) | 0.723 | **0.816** | 0.836 | **0.908** |

**使い分け:** 索引・fuser・reranker を変えた効果を読むときは ecr を見る。
gold 全体で測ると取れない29本が常に混ざって天井が張り付き、改善が薄まって見える。

**paper_recall / paper_f1 の分母は従来どおり gold 全件**（採点仕様なので変えない。
ただし `--metrics all` を付けたときだけ出る）。つまり根拠付きを完璧に拾っても
**multi の paper_recall は 0.836、全体で 0.914 が上限**で、1.0 には構造上届かない。
ecr は打ち手を選ぶための診断指標であって提出スコアの予測値ではない。

根拠付き gold が1本も無いクエリは分母が空になるので集計から除外する
（`recall_at_k()` は gold が空だと 1.0 を返す仕様なので、入れると満点が水増しされる）。
除外件数は `details.evidence_candidate_recall_counts` と
`candidate_recall_counts` の差で分かる。

**外部チームと single/multi を突き合わせるときは `..._by_backed_...` を見る。**
同じ ecr を、**single/multi の振り分けだけ**「根拠付き gold の本数」でやり直した
系列を並べて出している（`evidence_candidate_recall_by_backed_at{k}_{scenario}_macro`）。

外部チームの評価は「回答の根拠になる論文だけ残した gold」で single/multi を数え直して
いるため、同じ55問でも **single 43 / multi 12**（我々は `task_family` 基準で 26 / 29）と
分類が逆転している。`task_family` 基準の数字をそのまま並べると single 比率の違いだけで
差が出る。振り分けを揃えると **single 32 / multi 23** になり、比較できる形になる。
`total` は既存系列と完全に一致する（振り分け先が変わるだけで分母も分子も同じ）。

#### クエリ品質監査（`scripts/audit_queries.py` / `scripts/audit_report.py`）

仕様は `docs/query_audit_spec.md`。gold の各論文を `(query_id, paper_id)` 単位で
LLM に判定させ（relevance / noise_type 等）、クエリラベル（良問/やや良問/悪問）と
補正指標 `paper_recall_macro_clean`（分母を supporting/partial に絞った recall）を出す。

```
uv run python scripts/audit_queries.py --pred predictions_xxx.jsonl \
  --output audits/query_audit.jsonl            # LLM 判定（--resume で追記再開、--llm fake でドライラン）
uv run python scripts/audit_report.py --audit audits/query_audit.jsonl \
  --pred predictions_xxx.jsonl --output report/query_audit.html   # 集計 + 単一HTMLビューア
```

守るべき設計: **判定（audit_queries）と描画・集計（audit_report）を分離する**
（HTML は JSONL のビューアで判定ロジックを持たない）。**`relevance = no_evidence` は
「その論文由来の evidence_id が無い」というデータセットの事実なので LLM に判定させず
機械的に確定する**（アノテ漏れの疑いは別フィールド `body_supports_answer` に残す）。
判定プロンプトには論文全文を入れず abstract + evidence 周辺抜粋に絞る。

**分割実行（val_a / val_b）は必ず `--merge-with` で結合してから採点する。**
1構成4〜5時間かかるので `data/split/val_a.jsonl`(28件) と `val_b.jsonl`(27件) に
分けて回すことがあるが、`evaluate.py` は常に55件の gold と突き合わせるので、
片側だけを採点すると**全 macro 指標が網羅率のぶん薄まる**（val_a 単独なら約51%。
実測で paper_f1 0.636 が 0.367、evidence_f1 0.206 が 0.103 に見えた）。
2本目の実行に1本目の予測を渡せば、結合ファイルを作って55件で採点した行が
`results/` と `report/` に残る:

```
uv run python scripts/run_search.py ... \
  --queries data/split/val_b.jsonl --output predictions_xxx_b.jsonl \
  --merge-with predictions_xxx_a.jsonl
```

網羅率が足りない実行は stderr に警告が出て、`results/experiments.jsonl` に
`coverage: {covered, gold_total}`、レポート冒頭に「比較してはいけない」旨が入る。
`generate_comment()` に渡す過去記録も `n_queries` が一致する行だけに絞ってある
（絞らないと分割実行の片割れを「前回」として渡してしまい、LLM が
「28件の薄まった値 -> 55件の値」を改善として書く）。

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
