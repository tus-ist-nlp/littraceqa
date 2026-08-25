# 現行ベストの検索フロー（全体）

2026-08-06 時点で最も良い実測を出している構成を、上から下まで1本に書き下したもの。
`docs/search_agent2_spec.md`（新方式の設計）の比較対象となる土台。
**ここに書いてあるのは実際に走って測った値だけ**で、未実測のものは末尾の5節に隔離してある。

## 0. 何がベストか

| | |
|---|---|
| 実行 | `2026-08-03T14:37:41`（`results/experiments.jsonl`） |
| paths | `configs/paths/default.yaml` |
| process | `configs/process_style/mineru.yaml` |
| search | `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml` |
| agent | `configs/agent_style/reading_expand_rrf/rrf.yaml` |
| 予測 | `predictions_8b_chunk_k100_rrf.jsonl`（55件フル、`--production-input`） |
| git | `a0206042354c` |

実測（macro）:

| 指標 | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| cr single | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cr multi | 0.205 | 0.505 | 0.647 | 0.775 | 0.871 |
| **cr total** | 0.581 | 0.739 | 0.814 | **0.881** | 0.932 |
| ecr single | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ecr multi | 0.383 | 0.640 | 0.750 | 0.855 | 0.937 |
| **ecr total** | 0.675 | 0.810 | 0.868 | **0.924** | **0.967** |

**single は @1 から飽和している。** 動かせるのは multi の29件だけ。

### 再現コマンド

```bash
mkdir -p logs
tmux new-session -d -s littrace-exp \
  "PYTHONUNBUFFERED=1 uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
  --agent configs/agent_style/reading_expand_rrf/rrf.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions_k100_rrf_repro.jsonl \
  --production-input 2>&1 | tee logs/k100_rrf_repro.log"
```

**このコマンドで 08-03 の数字は完全には再現しない。** 当時は `SUBQUERY_COUNT`
（本数の機械的な切り詰め）が入る前で、step0 が平均3.3本・`_refine()` が平均8.2〜9.3本・
最大20本まで膨らんだまま走っていた。現在のコードは全ステップ4本で切る。
再生実測ではこの切り詰めはほぼ無損失（3.2節）なので**悪化はしないはず**だが、同一条件ではない。

---

## 1. フロー全体

```
Query { query_id, question, answer_types, table_schema }     ← 本番入力はこの4つだけ
  │
  │ ── 属性制約の抽出（1回だけ） ──────────────────────────────
  ├─ _extract_attribute_filter(question)
  │     正規表現で会議名・年を取る。**一意に取れたときだけ**発火し、
  │     以降すべてのステップ・すべてのサブクエリで同じ制約を使い回す。
  │     （サブクエリから抽出すると `_decompose()` が会議名を落として発火しない）
  │
  │ ── 分解（LLM 1回目） ─────────────────────────────────
  ├─ _decompose(question, filter)
  │     入力 = 質問文のみ
  │       + CORPUS_NOTE（「Web検索エンジンではない。site:/filetype: は何にも当たらない」）
  │       + _constraint_note（「全サブクエリを "[ICML 2025]" で始めろ」）
  │     出力 = サブクエリ 4本（プロンプトで指定し、返り値も [:4] で切る）
  │
  └─ for step in 0, 1, 2                                     ← max_steps: 3
       │
       │ ── 検索（サブクエリ1本ごとに1回） ───────────────────
       ├─ _retrieve(subquery) → HybridRetriever.retrieve()
       │     ├ bm25s        → per_index_k = 100 チャンク
       │     ├ faiss_qwen3  → per_index_k = 100 チャンク
       │     │    ※ attribute_filter が効くときは多めに取ってから落とす
       │     │      （取得件数 = per_index_k / 選択率 × safety、上限 max_fetch_k = 3000）
       │     ├ RRFFuser(k=60, weights すべて1.0) で融合 → pool_k = 200 チャンク
       │     └ Qwen3-Reranker-8B が 200 件を推論し、**順位を完全に置き換える**
       │     受け取り = retrieve_top_k = 20 チャンク
       │
       │     → runs に SubqueryRun(step, subquery, results) を積む（順位を保存）
       │     → chunks[chunk_id] に貯める。**同じチャンクはスコアが高いほう勝ち**
       │
       ├─ _merged_results(runs, chunks) … subquery_merge = "max"（既定）
       │     chunks.values() をそのまま返す（= スコア最大マージ）
       │
       ├─ _candidate_papers(merged)
       │     論文単位に畳み、上位 max_candidates = 20 本
       │     各論文 chunks_per_paper = 2 チャンク、1チャンク snippet_chars = 1800 字
       │
       │ ── 読解（LLM、ステップごとに1回） ──────────────────
       ├─ _read_and_judge(query, candidates, chunks)
       │     20本 × 2チャンクを読ませて JSON:
       │       { papers[].evidence_chunk_ids, sufficient, missing }
       │     候補に無い paper_id / chunk_id は捨てる（捏造チェック）
       │
       ├─ sufficient == true なら break     ← **停止条件はこれだけ。本数では止めない**
       │
       │ ── 再分解（LLM、最大2回） ────────────────────────
       └─ _refine(question, missing, tried, filter, runs, merged)
             入力 = 質問 + missing + 投げ済みサブクエリ一覧（「似たものを繰り返すな」）
             **候補論文の本文は入らない**（grounded_refine を有効にして初めてタイトルだけ）
             出力 = 新サブクエリ 4本。空リストなら break
```

ループ後:

```
_build_prediction()
  merged
    │
    ├─ to_gold_papers(merged)            ← **50本で切る前の全長**をランキングA にする
    │                                       （51位の論文をB が押し上げられるように）
    │
    ├─ _combine_rrf(A, B)                ← expansion.combine: rrf
    │     A = 質問→論文（上の検索。BM25 + 埋め込み → RRF → reranker）
    │     B = 論文→論文
    │         anchor = 候補1位の論文（anchors: 1）
    │         3ソースの近傍を各 neighbors=50 本取り、RRFFuser(k=60) で融合
    │           · specter2      … faiss_specter2_abstract を再利用（追加構築ゼロ）
    │           · bib_coupling  … 参考文献 arXiv ID の Jaccard、min_shared 2
    │           · bm25_mlt      … anchor の title+abstract で bm25s_paper を引く
    │         **anchor 自身を B の先頭に置く**（置かないと single の cr@20 が 1.000→0.923）
    │         **B は reranker に通さない**（名指しされないピア gold を必ず下げるため）
    │
    │     score(p) = 1.0/(60 + rank_A) + 1.0/(60 + rank_B)
    │
    ├─ [:50]  = candidate_papers          ← **実験の評価対象はここ**
    └─ [:10]  = gold_papers（提出）        ← max_papers。paper_cutoff は単純な切り詰め

  evidence = verdict の evidence_chunk_ids のうち、提出10本に残った論文のぶんだけ
  answer   = 空（回答生成は読解チーム側の担当）
```

**1クエリあたりの LLM 呼び出しは最大6回**: 分解1 + 読解3 + 再分解2。
`TaskFamilyClassifier` は呼ばない（`paper_cutoff: llm` は実体としては `[:max_papers]`）。

**reranker の推論量 = サブクエリ本数 × pool_k**。ここが走行時間のほぼ全部で、
4本 × 3ステップ × 200件が上限。

---

## 2. 設定値の一覧

### search_style（`bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml`）

| | 値 |
|---|---|
| per_index_k | 100 |
| pool_k | 200（= 索引2本 × 100。融合プールを切り捨てずに全部 reranker に見せる） |
| indexer 1 | `bm25s`（params なし） |
| indexer 2 | `faiss_qwen3` / index_name `faiss_qwen3_8b` / `Qwen/Qwen3-Embedding-8B` / devices `cuda:0` / fp16 / max_tokens 8192 / doc_prefix `"passage: "` / query_prefix `"query: "` / batch_size 8 |
| fuser | `rrf` k=60、weights すべて 1.0 |
| reranker | `qwen3` / `Qwen/Qwen3-Reranker-8B` / devices `cuda:1,cuda:2` / fp16 / max_batch_tokens 2048 / batch_size 4 / max_tokens 2048 |
| attribute_filter | enabled / safety 1.5 / max_fetch_k 3000 / min_results 10 |

**検索時の埋め込みは `devices: cuda:0` の1枚だけにする。** `_embed_query` は
devices[0] しか使わないので、残りを reranker に空ける。

**`max_fetch_k` を上げてはいけない。** per_index_k に合わせて 40000 にしたところ
faiss search が 1.5秒 → 91.1秒（61倍）に膨らんだ。件数が足りなければ `min_results` の
fail-open で「絞り込みなし」に戻るだけなので、小さく抑えるのが正しい。

### agent_style（`reading_expand_rrf/rrf.yaml`）

| | 値 |
|---|---|
| llm | `azure_openai`、reasoning_effort `medium` |
| max_steps | 3 |
| retrieve_top_k | 20 |
| max_candidates | 20（LLM が読む論文数） |
| chunks_per_paper | 2 |
| snippet_chars | 1800 |
| paper_cutoff / max_papers | `llm` / 10 |
| subquery_count | 4（yaml 未記載、`SUBQUERY_COUNT` の既定） |
| subquery_merge | `max`（yaml 未記載の既定） |
| expansion.sources | specter2(`faiss_specter2_abstract`) / bib_coupling(min_shared 2) / bm25_mlt(query_chars 1200) |
| expansion.neighbors / anchors | 50 / 1 |
| expansion.combine / rrf_k / combine_rrf_k | `rrf` / 60 / 60 |
| expansion.related_weight / related_offset | 1.0 / 0 |

既定オフのまま（yaml に書いていない）: `grounded_refine` / `adaptive_depth` /
`title_protect` / `pool_rescore` / `rerank_blend` / `consensus`。

---

## 3. なぜこの形なのか（実測の根拠）

### 3.1 展開は「位置挿入」ではなく「順位融合」

| | cr@20 | ecr@20 | multi@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|
| 展開なし | 0.789 | 0.868 | 0.601 | 0.832 | 0.889 |
| specter2 のみ | 0.816 | 0.870 | 0.650 | 0.908 | 0.945 |
| 書誌結合 のみ | 0.829 | 0.880 | 0.675 | 0.876 | 0.921 |
| 全文MLT のみ | 0.857 | 0.910 | 0.728 | 0.898 | 0.950 |
| **3ソース（現行）** | **0.879** | **0.926** | **0.770** | **0.917** | **0.956** |

3ソース併用の根拠は「違う gold を拾う」こと。候補圏外 gold 37本の回収は
specter2 15 / 書誌結合 11 / 全文MLT 16 で、MLT だけが拾えたのが2本、
既存2つだけが拾えたのが6本、重複14本。

**スコアで混ぜてはいけない**（cr@20 0.822 → 0.773 に悪化）。RRF は順位しか見ないので
この問題が起きない。`related_weight` 1.0 / `related_offset` 0 が最良
（w=0.5 で 0.817、w=2.0 で 0.830、offset=15 で 0.839）。

### 3.2 サブクエリの本数は step0 の幅がすべて

`runs_fat.jsonl`（55件812本）から step0 の本数だけ削って候補列を組み直した実測
（`subquery_merge: rrf` で再生）。

**step0 だけ残し、再分解を落とした場合**:

| step0 の本数 | ecr@5 | ecr@10 | ecr@20 | ecr@50 |
|---|---|---|---|---|
| 1本（分解しない） | 0.748 | 0.782 | 0.796 | 0.796 |
| 2本 | 0.742 | 0.793 | 0.830 | 0.840 |
| 3本 | 0.746 | 0.810 | 0.847 | 0.858 |
| 4本 | 0.742 | 0.797 | 0.852 | 0.858 |
| 上限なし | 0.737 | 0.797 | 0.852 | 0.858 |

**全ステップ残し、step0 の幅だけ変えた場合**:

| step0 の本数 | 総本数 | ecr@10 | ecr@20 | ecr@50 |
|---|---|---|---|---|
| 1本 | 687 | 0.777 | 0.818 | 0.858 |
| 2本 | 741 | 0.792 | 0.847 | 0.868 |
| 4本 | 805 | 0.815 | 0.852 | 0.873 |
| 上限なし | 812 | 0.815 | 0.852 | 0.873 |

読み取れること:

- **@5 は分解しても伸びない**（むしろ1本が最良）。主役の論文は1本目で取れている。
- 効くのは **@20 以降**（1本→4本で ecr@20 +5.6pt / ecr@50 +6.2pt）で、
  **3本でほぼ飽和**する。4本目が買うのは @20 で +0.5pt、@50 では 0。
- **現行の再分解では埋め合わせられない。** step0 を1本にすると、あとに630本以上の
  `_refine()` が続いても ecr@20 が 3.4pt 戻らない。
- leave-one-out では **step1 の305本は1本抜いても ecr@50 の gold が減らない（0/305）**。
  step0 は 4/180（2.2%）。

つまり**サブクエリの価値はほぼ全部 step0 の幅に乗っている**。

注意（この表の限界）:

- 土台の走行は `subquery_count` 導入前なので step0 は最大6本しかなく、
  **「4本より上」は測れていない**（表の4本と上限なしが 173/180本でほぼ同じ）。
- 再生では step1/step2 のサブクエリを**固定**している。実際に step0 を削れば
  `_refine()` の入力も変わるので、「再分解が埋め合わせられない」は厳しめの見積もり。
  **とくに、チャンクを見て書き換える再分解（現行には無い）についてはこの表は何も言っていない。**

### 3.3 プロンプトに `CORPUS_NOTE` が要る

これが無いと LLM は Google 検索クエリを書く。実測で `_refine()` のサブクエリの
**29〜41%** が `site:arxiv.org` / `filetype:pdf` 付きで、投げ先はローカルの BM25 と
faiss なので1件もヒットせず、2周目・3周目の検索が丸ごと空振りしていた。

| 構成 | step0 | step1 | step2 |
|---|---|---|---|
| paper_attrfilter | 0.0% | 35.6% | 40.5% |
| chunk_attrfilter | 0.0% | 32.3% | 31.0% |
| bm25_specter2_body_qwen3 | 0.0% | 29.6% | 32.5% |

### 3.4 属性フィルタは「会議名が一意に取れたとき」だけ発火

年しか書かれていない / `all venues` を含む / 会議名が2種類以上、のときは抽出せず
従来と完全に同一のコードパスを通る。検証55件で発火5件、**gold が制約を満たす率 18/18 = 100%**。

年で絞る意味が薄いのは、コーパスで (venue, year) の組が9通りしかなく
**ECCV だけが 2024 で残り8会議が全部 2025** だから。

---

## 4. 分かっている弱点

### 4.1 multi の伸びしろは「質問が名指ししていない兄弟 gold」91本にある

08-03 の候補列で、multi 29件・gold 120本を「各クエリで最も上位の gold」と
「残り」に分けた分布:

| | 1位 | top5 | top20 | 圏外 | 中央値 |
|---|---|---|---|---|---|
| 各クエリで最上位の gold（29本） | 83% | 97% | 100% | 0本 | 1位 |
| **残りの gold（91本）** | 0% | **36%** | 71% | **15本** | **6位** |

29件すべてで1本は必ず top20 に入り、8割強はぴったり1位。**質問は gold 1本の語彙で
書かれていて、残りは性質でしか繋がっていない。**

（「最上位の gold」は事後に最小順位を選んでいるので小さく出るバイアスがある。
ただし *ぴったり1位が83%* と *圏外0本* は選び方だけでは説明が付かない。）

**上位5本しか読まない設計を考えるときは、この行の `top5 = 36%` を見ること。**
兄弟 gold の6割強は5位より下にいる。

### 4.2 gold どうしで語彙が共有されない

q_022「ICML 2025 で reference-free な preference optimization を提案した論文を列挙せよ」。
gold の本文を数えると:

| 論文 | chunk数 | `reference-free` を含む chunk | SimPO 言及 |
|---|---|---|---|
| LOGO（1位で取れる） | 101 | 5 | 8 |
| AlphaPO（28位） | 123 | **1**（参考文献リスト内） | **33** |

AlphaPO は自分を "reference-free" と一度も呼ばない（自称は
`Direct Alignment Algorithm` / `reward shape` / `likelihood displacement`）。
BM25 単体で書き方を変えて投げた実測:

| クエリ | AlphaPO | LOGO |
|---|---|---|
| 実際のサブクエリ（reference-free 系） | 圏外 | 12 |
| 「Direct Alignment Algorithm、reward の shape を変える」 | **1** | 圏外 |
| 「eliminates the reliance on a reference model …」（本文の言い回し） | **2** | 圏外 |

**どの1本も3本まとめては取れない。** そして現行のサブクエリ生成の入力は
「質問文」と「読解 LLM の `missing`」だけで、**候補論文の本文は一度も入らない**。
q_022 で23本すべてが同じ語彙ファミリーの言い換えになったのはこの構造の帰結。

### 4.3 展開で拾った論文は evidence を持てない

`evidence` は読解 LLM が返した `evidence_chunk_ids` からしか作られず、
`_combine_rrf()` は**読解が終わったあと**に `candidate_papers`（論文IDの列）だけを
入れ替える。つまり展開で入った論文はチャンクを1本も持たず、読まれることもない。

multi の gold で候補 top20 に居る94本のうち:

| | 本数 |
|---|---|
| 検索でも取れていた（チャンクあり → evidence 可） | 76 |
| **展開でしか入っていない（evidence 不可）** | **18** |

### 4.4 展開の anchor が質問の軸とズレると雑音しか入らない

同じ q_022 で anchor は候補1位の LOGO（正式名 "**Long cOntext aliGnment** via
efficient preference Optimization"）。長文脈の論文が preference optimization を
道具に使っているだけなので、3ソースとも長文脈クラスタを返した。

| | AlphaPO | 上位20の長文脈系 |
|---|---|---|
| 展開なし | 14位 | 1本 |
| 展開あり（現行） | **28位** | **10本** |

展開ありの top20 のうち9本が、展開なしの候補列50本に一度も現れない長文脈論文だった。
q_022 の cr@20 は 2/3 → 1/3 に落ちている。

**ただし multi 全体では展開が大きくプラス**（cr@50 0.664 → 0.871）なので、
この1件のために切るのは損。

---

## 5. 未実測

以下は測っていないか、オフライン再生でしか測っていない。**上の表と混ぜて読まないこと。**

- **`anchors: 1` → `3`**。`replay_expansion.py` の再生（土台
  `predictions_8b_chunk_k100_cand50.jsonl`）で cr@20 0.861→0.874 /
  ecr@20 0.904→0.915 / ecr@50 0.950→0.967 / multi ecr@20 0.852→0.873。
  `anchors: 5` は悪化（ecr@20 0.892）。q_022 の長文脈汚染も8本→4本に減るが、
  AlphaPO は28位のまま。**フル走行はまだ無い。**
- **`grounded_refine: true`**。オフライン再生できない（サブクエリ自体が変わる）。
  `k100_rrf_grounded` として走らせたが途中で落ちた。
- **`rerank_blend`**。`chunk_attrfilter_blend` + `stacked` の1本にしか含まれておらず、
  他の5変更と同時なので切り分けできていない。その実行は ecr@20 0.905 / ecr@50 0.939 と
  現行ベストより低い。
- **チャンク接地のクエリ書き換え**（`docs/search_agent2_spec.md` の新方式）。
  3.2 の表は「質問と missing だけを見る現行の再分解」についての数字なので、
  **チャンクを見て書き換える方式については何も言っていない。** 未実装・未実測。
