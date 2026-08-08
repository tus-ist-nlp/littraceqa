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
- git commit と git push は実行せず、変更を未コミットで残す

## 検索手法を追加するときのルール

新しい Indexer / Preprocessor / Agent を実装したときは、
必ず以下も合わせて作成・更新すること。

### 1. configs/ は5フォルダに分離されている
前処理・検索手法・エージェント・提出方法・共有パスはそれぞれ独立したyamlファイルで、
実行時に1ファイルずつ選んで組み合わせる（`src/littraceqa/di_pipeline/config.py` の
`compose_config()` が合成する）。1ファイルに全部詰め込まない。

- `configs/paths/{名前}.yaml`: 実行環境ごとの共有パス（pdf_dir, index_dirのルート等）
- `configs/process_style/{preprocessor名}.yaml`: 前処理（`{name, params}`）
- `configs/search_style/{組み合わせ名}.yaml`: 検索手法（indexer群 + fuser + reranker）
- `configs/agent_style/{agent名}.yaml`: エージェント（`{name, llm?, params}`）
- `configs/select_style/{構成名}.yaml`: 提出する論文集合の決め方（`{name, params}`、省略可）

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
  --search configs/search_style/seed_expansion_structured_filter.yaml \
  --agent configs/agent_style/reading.yaml \
  --select configs/select_style/f1_balanced.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --build
```

### 2. 推奨デフォルトの組み合わせ
新しい手法をデフォルト（推奨組み合わせ）にする場合は、この節の記載を更新する。
ablation 用なら触らない。

現在のデフォルト: `process_style/mineru.yaml` + `search_style/seed_expansion_structured_filter.yaml`
+ `agent_style/reading.yaml`。27,487件の索引（`bm25s` / `paper_bm25` /
`specter2_paper_embeddings`）が構築済みで、`--build` なしですぐ検索できる。

構成は「chunk-level BM25 + paper-level BM25 を PaperRank RRF で論文単位に統合 →
seed expansion → 候補補充4レーン → Qwen3-Reranker-4B で50件を採点して Top 50」。
候補補充は次の4つで、いずれも上限付きで既存順位を壊さない。

| レーン | 発火条件 | 効果 |
|---|---|---|
| method dense tail | 手法名ヒントあり | SPECTER2 近傍から最大3本 |
| open-set slot | 列挙型質問 | rank 20 に1枠 |
| structured filter | 列挙型 かつ 会場・年・モダリティを明示 | 会場×モダリティで絞った候補を6位以降へ昇格 |
| exact method search | 質問中の固有名が1論文を一意に指す | その論文を候補へ追加（昇格はしない） |

validation 55問（answer-bearing gold 87本）での実測は R@1 0.7808 / R@10 0.9848 /
R@20 1.0000 / All-Gold@20 1.0000。

**主要な設定値はいずれも掃引で選んでいる。変更するなら測り直すこと。**

- `reranker.model: Qwen3-Reranker-4B` — 0.6B / 4B / 8B の比較で 4B が最良。
  8B は 4B より遅く R@1 も低い（0.7581 対 0.7808）。大きいほど良いわけではない。
- `base_rank_weight: 0.52` — 0.30〜0.80 の5点掃引で最良。0.30 は R@10 だけ僅かに
  上回るので、後段が上位10本しか使わない設計に変えるなら再検討する。
- `final_rerank_protected_top_k: 20` — 0 / 5 / 10 / 20 の掃引で最良。この保護は
  reranker が上位と判定した論文を21位以降へ落とすと同時に、reranker が下位と
  判定した元上位を救っており、validation では後者の利得が上回る。
- `structured_filter` は**列挙型質問であることを必須**にしている。会場・年・
  モダリティの3条件だけで発火させると「For the two ICCV 2025 papers ...」のような
  比較質問にも作動し、質問が名指しした論文を3位から21位へ落とした。

**無効だったレーンは削除済み。** two-lane rerank / paper neighborhood /
method relation / method bridge / dense reciprocal / dense consensus /
paper dense tail / local expansion は重みゼロで素通りしていたので、
コンストラクタ引数ごと消した（60引数 → 31引数）。実装は git 履歴に残り、
結果と考察は `docs/retrieval_negative_results.md` にまとめてある。
再導入するなら「validation の改善」だけでなく「held-out での発火内容が
想定どおりか」まで確認すること。

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
│   ├── abstract_specter2_body_qwen3.yaml : BM25 + SPECTER2(title_abstractのみ) + Qwen3-Embedding-0.6B(本文のみ)
│   ├── bm25_paper_rank_seed_expansion_qwen3_reranker.yaml : 上の構成から候補補充を外した比較基準
│   └── seed_expansion_structured_filter.yaml : dual BM25 + seed expansion + 候補補充4レーン
│         + Qwen3-Reranker-4B（デフォルト、構築済み）
├── agent_style/
│   └── reading.yaml      : 分解→読解→不足分の再検索を繰り返す唯一の本命。evidence も埋める（デフォルト）
└── select_style/
    ├── high_precision.yaml : 質問が明示した本数だけ出す。evidence 必須（Run B）
    ├── f1_balanced.yaml    : 上に open-set列挙3本を足したもの（Run A、推奨）
    └── high_recall.yaml    : 複数論文問題で多めに出す（Run C）
```

`iterative.yaml` / `reading_llmcount.yaml` / `simple.yaml` / `verifying.yaml` は削除済み
（`iterative` は停止条件が「見つかった論文の本数」で top_k=20 の時点で初回から満たされ、
反復ループが事実上空回りしていた）。以後 agent_style は `reading` 一本で運用する。

**`reading` の打ち切りは task_family に依存しない。** 反復検索の停止条件は
`_read_and_judge()` が返す LLM の `sufficient` 判定のみ（`src/littraceqa/di_pipeline/agent/reading.py`）。
提出本数も `paper_cutoff: llm` にしてあるので、LLM が「これで十分」と判断した時点の
選定をそのまま出す（`max_papers: 10` で頭打ち）。本番入力に `task_family` が無く、
推定しても正解率0.67程度で当てにならないため、本数決定の経路から task_family を外した。

### 2.1 提出論文の絞り込み（Paper Selector）

**Recall@k は公式の採点指標ではない。** 公式は提出した paper_id の**集合**と gold
集合を問題ごとに突き合わせ、precision / recall / F1 を出してマクロ平均する
（`scripts/evaluate.py` の `paper_f1_macro`）。順位も「20位以内か」も見ない。
gold 1本の問題に20本出すと F1 は 0.095 で頭打ちになる。

そのため候補生成と提出は目的が違う。

| 段階 | 目的 | 指標 |
|---|---|---|
| 候補生成 Top50 | gold を漏らさない | Recall@50 / All-Gold@50 |
| 読解 Top20 | 読解予算 | Recall@20 |
| **提出集合** | **gold 集合と一致させる** | **paper_f1_macro** |

`src/littraceqa/di_pipeline/select/` が提出集合を決める。
`CardinalityPaperSelector` は**質問文が明示している本数**で順位を切る
（"For the two ICCV 2025 papers" → 2本、"the X paper and the Y paper" → 2本、
明示が無ければ1本）。`--select configs/select_style/{構成}.yaml` で選ぶ。

**閾値は validation で当てていない。3構成でトレードオフを挟み、本番で決める。**
validation の multi 29問の gold はクラスタ注釈なので、そこに合わせて閾値を
最適化すると「質問が言及していない論文も出す」設定が選ばれてしまう。

| 構成 | 明示なし | 明示あり | open-set列挙 | evidence必須 |
|---|---|---|---|---|
| `high_precision` (Run B) | 1本 | そのまま | 1本 | あり |
| `f1_balanced` (Run A、既定) | 1本 | そのまま | 3本 | なし |
| `high_recall` (Run C) | 2本 | +1本 | 5本 | なし |

`require_evidence` は reading agent が根拠を取れた論文だけに絞る。根拠が1本も
取れなかったときは検索順位のまま出す（空提出は precision も recall も0になり、
必ず損をするため）。

**`answer_types` で本数を判定してはいけない。** validation 55問では
`freeform` を含む＝single、`multiple_choice` 単独＝multi がほぼ完全に成立するが、
**実際の test / test_extra には `freeform` が1問も無い**。validation に当てた規則は
転移しない。質問文の明示表現だけを使うのはこのため。

validation 55問（公式 gold 146本）での実測:

| 提出方法 | P | R | **F1** |
|---|---|---|---|
| 固定 top20 | 0.0991 | 0.8318 | 0.1686 |
| 固定 top10 | 0.1836 | 0.7939 | 0.2750 |
| 固定 top1 | 0.9273 | 0.5444 | 0.6200 |
| `high_recall` | 0.5188 | 0.6732 | 0.5302 |
| `f1_balanced` | 0.8455 | 0.6101 | 0.6410 |
| **`high_precision`** | 0.8758 | 0.6081 | **0.6437** |
| 上限（top50から完璧に選ぶ） | 1.0000 | 0.9091 | 0.9406 |

**この順位を鵜呑みにしない。** validation がクラスタ注釈である以上、precision 寄りが
有利に出るのは当然で、test で同じとは限らない。`eval_paper_selection.py` は
evidence 判定を動かせない（reading agent が要る）ので、`require_evidence` 付きの
構成については recall の上限・precision の下限を出しているだけである。

`uv run python scripts/eval_paper_selection.py --retrieval {検索出力}.json
--gold data/validation.jsonl` で GPU も LLM も使わず即座に測り直せる。

**残りの差 0.64 → 0.94 は本数ではなく「どの論文か」で、読解が要る。** single(26問)は
top1 で F1 0.8846 とほぼ上限だが、multi(29問)は順位ベースだと 0.42〜0.45 で頭打ち
（precision と recall がちょうど相殺する）。method alias graph はクラスタを繋いで
おらず、SPECTER2 近傍でも +0.03 しか出ない。

### 3.1 評価の作法

**gold paper の評価は公式の `data/validation.jsonl`（146本）を使う。**
`validation_answer_bearing_gold_draft.jsonl`（87本）は回答に必要な論文だけを
残した監査結果で、公式採点の対象ではない。公式 gold は「4本の関連論文クラスタを
選び、それについて複数問を作る」方式で付いており、質問が TCM 1本しか名指しして
いなくても gold はクラスタ4本全部になる（q_031〜q_042 の12問が同一の4本を共有）。
answer-bearing gold で R@50 = 1.0000 でも、公式 gold では 0.9091 しかない。

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
`MinerUChunker` が自動導出する）。27,487件で 4GPU 約25時間。変換済みの論文は
飛ばすので、中断しても同じコマンドで再開できる。

### 5. registry への登録確認
@register("indexer", "xxx") のデコレータが付いているか確認する。
付いていないと config から呼び出せない。
