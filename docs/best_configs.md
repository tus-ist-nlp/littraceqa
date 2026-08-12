# `candidate_recall`（multi_paper）が最良の構成 — エージェントあり / なし

**指標**: `candidate_recall_at20_multi_macro`（`evaluate.py` / `eval_retrieval.py`）。
multi_paper の29件だけを分母にした、候補列 top20 の gold 論文 recall。
**打ち手の効きは multi にしか乗らない**（single は cr@20 が 1.000 で飽和済み）。

検証データは 55 件（single 26 / multi 29、`data/validation_inputs.jsonl` を
`--production-input` で投入）。前処理はどちらも MinerU（27,489 論文）。

---

## 1. 結論

| | 構成 | **multi cr@20** | multi ecr@20 | 測り方 |
|---|---|---|---|---|
| **エージェントあり** | `chunk_attrfilter_k100` + `reading_expand_rrf/notable` | **0.8592** | 0.9253 | フル走行の実測 |
| **エージェントなし** | `chunk_attrfilter_k100` + `agentless/agentless` | **0.8247** | 0.8994 | オフライン再生（**過小評価**） |

どちらも **ランキングA（質問→論文）とランキングB（論文→論文）を RRF 統合する**型。
**B を使わない「A のみ」の構成は §6** に分けて書いた（本番でピア論文を取る必要が
薄い場合の検討用）。結論だけ先に言うと、**B はピア論文専用の仕掛けではない**——
B が top20 に足す gold の 56% は根拠付き gold で、ピアを完全に無視して測っても
multi ecr@20 が +0.190 ある。

⚠ **この2つは測り方が違う。** あり側は 55 件のフル走行（`results/experiments.jsonl`
の 2026-08-09T02:31:18）。なし側は `runs_rawq.jsonl`（`--top-k 20` で採った土台）からの
再生で、**ランキングA が20チャンク分しかない**。フル深度（200件要求）で測った総合値は
`ecr@20 0.9576` あり、そちらはフル走行の 0.9606 とほぼ互角なので、
**なし側の multi 0.8247 は下振れした値**と読むべき。フル深度の multi 内訳は
深いダンプ（`runs_rawq_deep.jsonl`、GPU 待ち）を採ってから確定する。

---

## 2. エージェントあり — `reading_expand_rrf/notable`

```
uv run python scripts/run_search.py \
  --paths   configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search  configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
  --agent   configs/agent_style/reading_expand_rrf/notable.yaml \
  --queries data/validation_inputs.jsonl \
  --output  predictions_k100_notable.jsonl --production-input
```

### 指標（55件、フル走行）

| k | multi cr | multi ecr | single cr | total cr | total ecr |
|---|---|---|---|---|---|
| 1 | 0.2107 | 0.3544 | 0.8846 | 0.5293 | 0.6051 |
| 5 | 0.5910 | 0.7002 | 1.0000 | 0.7843 | 0.8419 |
| 10 | 0.7529 | 0.8391 | 1.0000 | 0.8697 | 0.9152 |
| **20** | **0.8592** | **0.9253** | 1.0000 | 0.9258 | 0.9606 |
| 50 | 0.9224 | 0.9799 | 1.0000 | 0.9591 | 0.9894 |

**multi cr@20 は全フル走行中の最高値**（次点 `verdict_anchor` 0.8391、
`steps2_notable` 0.8333、`rewrite` 0.8161）。@1〜@50 のすべての k でも最高。

### 構成図

```
                        質問（本番入力は query_id / question / answer_types / table_schema の4つ）
                          |
              +-----------+------------------------------+
              |                                          |
     [属性フィルタの抽出]                         [LLM: サブクエリ分解]
     会議名・年を正規表現で1回だけ                  Azure OpenAI（4本固定）
     取る（取れた時だけ発火）                        ※ここで LLM 1回
              |                                          |
              +-----------+------------------------------+
                          v
        +-----------------------------------------------------------+
        |  反復ループ（最大3周。停止条件は読解 LLM の sufficient）      |
        |                                                             |
        |   サブクエリ4本 --> それぞれ検索 -----------------------+    |
        |                                                        |    |
        |     [BM25 chunk]      [faiss Qwen3-Embedding-8B]        |    |
        |      bm25s             27,489論文のチャンク索引          |    |
        |          \                    /                         |    |
        |           +--> RRF (k=60, 等重み) --> 属性フィルタで絞込 |    |
        |                        |                                |    |
        |                        v                                |    |
        |            [Qwen3-Reranker-8B]  pool_k=200               |    |
        |             cuda:1,2 / fp16 / yes確率で並べ替え           |    |
        |                        |                                |    |
        |                   上位20チャンク --> チャンクプールに蓄積 <+    |
        |                                        |                     |
        |                                        v                     |
        |                          [LLM: 読解・根拠判定]                |
        |                          上位20論文×2チャンクを読む            |
        |                          -> sufficient / evidence / paper_ids |
        |                             ※ここで LLM 1回                   |
        |                                        |                     |
        |                        足りなければ [LLM: 再分解] して次の周へ  |
        +-----------------------------------------------------------+
                          |
                          v
        チャンクプール --> to_gold_papers(agg=max, **表チャンクは代表スコアに使わない**)
                          |
                          v  ランキングA（質問→論文。50件で切る前の全長）
        +-----------------------------------------------------------+
        |  A/B の RRF 統合   score(p) = 1/(10+rank_A) + 1/(10+rank_B) |
        |                                                             |
        |  ランキングB（論文→論文。**reranker に通さない**）            |
        |    起点 = 候補1位 ∪ 読解LLMが根拠を確認した論文               |
        |    +-- SPECTER2(title+abstract) 近傍100                     |
        |    +-- 書誌結合（参考文献の arXiv ID の Jaccard）近傍100      |
        |    +-- 全文MLT（bm25s_paper への more-like-this）近傍100      |
        |         -> 3ソースを RRF(k=60) で融合                        |
        +-----------------------------------------------------------+
                          |
                          v
             candidate_papers（上位50）  +  evidence（読解が指名したチャンク）
                          |
                          v
                    読解チーム側へ受け渡し（提出論文の選定と回答生成は担当外）
```

**LLM 呼び出しは1クエリあたり最大6回**（分解1 + 読解1 が1周、最大3周）。
実行時間は 55 件で 4〜5 時間。

---

## 3. エージェントなし — `agentless/agentless`

```
uv run python scripts/eval_retrieval.py \
  --paths   configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search  configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
  --agent   configs/agent_style/agentless/agentless.yaml \
  --ks 1,5,10,20,50
```

`--agent` を渡しても、`eval_retrieval.py` は**候補列の並べ替えだけで完結する部分**
（A/B の RRF 統合と表チャンク除外）しか載せない。分解・反復・読解は走らず、
**LLM を1回も呼ばない**。

### 指標

総合はフル深度の実測（2026-08-10）、multi/single はオフライン再生（**過小評価**、§1 参照）:

| k | multi cr | multi ecr | single cr | total cr | total ecr |
|---|---|---|---|---|---|
| 1 | 0.2280 | 0.3803 | 0.9231 | 0.5566 | 0.6369 |
| 5 | 0.5182 | 0.6561 | 0.9231 | 0.7005 | 0.7747 |
| 10 | 0.7079 | 0.8285 | 0.9615 | 0.8298 | 0.8889 |
| **20** | **0.8247** | **0.8994** | 0.9615 | 0.9182 | 0.9576 |
| 50 | 0.9224 | 0.9713 | 1.0000 | 0.9591 | 0.9788 |

次点は `agentless/score_anchor`（`anchor_from: score` を足した版）で、
multi cr@20 は同値 0.8247、multi cr@5 だけ 0.5182 -> 0.5268 と僅かに上。
**採否は深いダンプで測り直してから決める。**

### 構成図

```
                        質問（そのまま1本のクエリとして使う）
                          |
                          v
              [属性フィルタの抽出]  会議名・年を正規表現で取る（LLM 不使用）
                          |
        +-----------------+-----------------+
        |                                   |
  [BM25 chunk]                 [faiss Qwen3-Embedding-8B]
   bm25s                        27,489論文のチャンク索引
        \                              /
         +--> RRF (k=60, 等重み) --> 属性フィルタで絞込
                          |
                          v
              [Qwen3-Reranker-8B]  pool_k=200
               cuda:1,2 / fp16 / yes確率で並べ替え
                          |
                          v  （検索1回だけ。反復も分解もしない）
        to_gold_papers(agg=max, **表チャンクは代表スコアに使わない**)
                          |
                          v  ランキングA（切る前の全長）
        +-----------------------------------------------------------+
        |  A/B の RRF 統合   score(p) = 1/(10+rank_A) + 1/(10+rank_B) |
        |                                                             |
        |  ランキングB（論文→論文）                                    |
        |    起点 = **候補1位のみ**（verdict が無いのでここが減る）      |
        |    +-- SPECTER2(title+abstract) 近傍100                     |
        |    +-- 書誌結合 近傍100                                      |
        |    +-- 全文MLT 近傍100                                       |
        |         -> 3ソースを RRF(k=60) で融合                        |
        +-----------------------------------------------------------+
                          |
                          v
             candidate_papers（上位50）      ※ evidence は出せない
```

**LLM 呼び出しゼロ。** 実行時間は 55 件で十数分（索引ロードを除く）。

---

## 4. 2つの差はどこから来るか

あり／なしの差は **@5 に集中していて、@20 以降はほぼ無い**（total で見ると
cr@50 は 0.9591 で小数4桁まで完全一致、@1 は**なし側が上**）。

| | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| total ecr（あり） | 0.605 | **0.842** | **0.915** | 0.961 | 0.989 |
| total ecr（なし） | **0.637** | 0.775 | 0.889 | 0.958 | 0.979 |

差の出所は**2つしかない**:

1. **サブクエリ4本の融合** — 1本のクエリより上位が厚くなる。
   逆に @1 でなし側が勝つのは、分解した4本が元の質問の語の組み合わせを保たないため
   （step0 に質問文そのものが入るのは 0/55 件）。
2. **ランキングB の起点**（`anchor_from: verdict`） — あり側は「読解 LLM が根拠を
   確認した論文」を起点に加えるので、multi のトピッククラスタを複数展開できる。
   なし側は候補1位のみ。

**この2つを LLM 抜きで埋めるのが、エージェント不要システムの中心課題**（現在地と
打ち手は `docs/agentless_spec.md`）。2 については `anchor_from: score`
（reranker の yes 確率で起点を選ぶ）を実装済みで、**エージェント側の土台では
verdict の伸びの大半を回収する**が、エージェント抜きの土台では起点が
平均1.2本にしか増えず効いていない。原因はプールの深さと見ており、
深いダンプで測り直す。

---

## 5. 共通部分（どちらの構成でも同じ）

| 段 | 中身 |
|---|---|
| 前処理 | MinerU（`process_style/mineru.yaml`）。27,489論文をチャンク化 |
| 索引1 | `bm25s`（chunk 単位、5.1GB） |
| 索引2 | `faiss_qwen3_8b`（Qwen3-Embedding-8B、fp16、max_tokens 8192） |
| 融合 | RRF（k=60、全索引 weight 1.0）、`per_index_k: 100` |
| 絞込 | 属性フィルタ（会議名・年。正規表現のみ、`max_fetch_k: 3000`、fail-open） |
| 並べ替え | Qwen3-Reranker-8B（fp16、`max_batch_tokens: 2048`、`pool_k: 200`） |
| 論文化 | `to_gold_papers(agg="max")` + **表チャンクを代表スコアに使わない** |
| 展開 | SPECTER2 / 書誌結合 / 全文MLT を RRF(k=60) 融合、`neighbors: 100` |
| 統合 | A/B を RRF(`combine_rrf_k: 10`)、`related_weight: 1.0`、`related_offset: 0` |

**GPU は 3枚**（埋め込み cuda:0 / reranker cuda:1,2）。索引はすべて構築済みで、
`--build` は不要。

---

## 6. ランキングB を使わない構成（A のみ）

**動機**: 本番データではピア論文（gold に名前はあるが evidence が1件も紐づかない
同トピック論文）をあまり取らなくてよい可能性がある。ランキングB（論文→論文展開）は
もともとピア gold を拾うために入れたので、それなら B ごと外せるのではないか、という問い。

### 6.1 先に結論 — **B はピア論文専用の仕掛けではない**

同じ土台（`notable` のフル走行）で **`_combine_rrf` を通すか通さないかだけ**を
切り替えた純粋な対照（他は完全に同一）:

| multi_paper (29件) | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| A のみ  cr | 0.2107 | 0.4253 | 0.4866 | 0.5862 | 0.6466 |
| **A+B**  cr | 0.2107 | **0.5910** | **0.7529** | **0.8592** | **0.9224** |
| A のみ  **ecr** | 0.3544 | 0.5977 | 0.6590 | 0.7356 | 0.7759 |
| **A+B**  **ecr** | 0.3544 | **0.7002** | **0.8391** | **0.9253** | **0.9799** |

**`ecr` はピア gold を分母から完全に除いた指標**（根拠付き gold だけが分母）。
それでも **multi ecr@20 で +0.190**、@10 で +0.180 ある。
2本目の土台（`steps2_notable`）でも同じ向きで、multi ecr@20 は 0.6944 -> 0.8994（+0.205）。

top20 に入った gold を実数で数えると（multi 29件、micro）:

| | A のみ | A+B | 差 | 母数 |
|---|---|---|---|---|
| **根拠付き gold** | 66 | **84** | **+18** | 91 |
| ピア gold | 6 | 20 | +14 | 29 |

**B が足す32本のうち18本（56%）は根拠付き gold。** ピアが要らないとしても、
B を外すと根拠付き gold を 84 -> 66 本に落とすことになる。

なぜそうなるかは B の作りから説明できる。B は「候補1位（と読解 LLM の確認済み論文）に
近い論文」を推すランキングで、**近さの判定に質問文を使わない**。multi の gold は
「1本目は1位で取れているが2本目以降が沈む」形をしており（クエリ内の順位で見ると
2本目が中央4位・3本目が中央8位・4本目が中央14位）、**沈んでいるのはピアではなく
根拠付き gold のほう**。B はそこを引き上げている。

**@1 は A のみと A+B で完全に同値**（0.2107 / 総合 0.5293）。統合では anchor を
B の先頭に固定するので、候補1位は必ず1位のまま残る設計になっている。

### 6.2 エージェントあり × A のみの最良 — `reading_normal/cand50`

```
uv run python scripts/run_search.py \
  --paths   configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search  configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml \
  --agent   configs/agent_style/reading_normal/cand50.yaml \
  --queries data/validation_inputs.jsonl \
  --output  predictions_8b_chunk_k100_cand50.jsonl --production-input
```

`reading.yaml` の `retrieve_top_k` と `max_candidates` を両方 50 に広げた版
（**LLM が読む本数そのもの**が増える）。expansion ブロックを持たないので
ランキングB は一切走らない。

| k | multi cr | multi ecr | total cr | total ecr |
|---|---|---|---|---|
| 1 | 0.2050 | 0.3803 | 0.5292 | 0.6369 |
| 5 | 0.4358 | 0.6169 | 0.6980 | 0.7616 |
| 10 | 0.5038 | 0.6705 | 0.7495 | 0.8081 |
| **20** | **0.6264** | **0.7644** | 0.8232 | 0.8758 |
| 50 | 0.6954 | 0.8247 | 0.8636 | 0.9076 |

A のみのフル走行では **multi cr@20 が最高**（次点 `reading_normal/cand50` ×
`chunk_attrfilter`(k1000) 0.6006、`reading_normal/topk50` 0.5891、
`reading_normal/fat` 0.5661、`reading.yaml` × `paper_attrfilter` 0.5010）。

**§6.1 の対照（0.5862）より高いのは幅の違い**——こちらは `retrieve_top_k` /
`max_candidates` が 50 で、`notable`（20）より広い候補を LLM に読ませている。
B を捨てるなら幅で少しだけ取り返せる、という関係になっている。

#### 構成図

```
                        質問
                          |
              +-----------+------------------+
              |                              |
     [属性フィルタの抽出]              [LLM: サブクエリ分解]（4本固定）
              |                              |
              +-----------+------------------+
                          v
        +-----------------------------------------------------------+
        |  反復ループ（最大3周。停止条件は読解 LLM の sufficient）      |
        |                                                             |
        |   サブクエリ4本 --> それぞれ検索                              |
        |                                                             |
        |     [BM25 chunk]      [faiss Qwen3-Embedding-8B]            |
        |          \                    /                             |
        |           +--> RRF (k=60) --> 属性フィルタで絞込              |
        |                        |                                     |
        |            [Qwen3-Reranker-8B]  pool_k=200                   |
        |                        |                                     |
        |            上位**50**チャンク --> チャンクプールに蓄積          |
        |                        |                                     |
        |            [LLM: 読解・根拠判定] 上位**50**論文を読む           |
        |            -> sufficient / evidence / paper_ids              |
        +-----------------------------------------------------------+
                          |
                          v
        チャンクプール --> to_gold_papers(agg=max)
                          |
                          v
             candidate_papers（上位50）  +  evidence
                          |
                          x  ←★ ランキングB（SPECTER2 / 書誌結合 / 全文MLT）は無し
                          v
                    読解チーム側へ受け渡し
```

A+B 型（§2）との違いは**最後の1段だけ**。検索・reranker・反復・読解はすべて同じで、
`expansion` ブロックの有無しか変わらない。

### 6.3 エージェントなし × A のみ — **まだ測れていない**

**現在の土台では測定不能。** `runs_rawq.jsonl` は `--top-k 20` で採ったもので、
**A に載る論文が中央7本（最小1・最大18）しかない**。@10 以降は「A の全部」を
数えているだけになり、@20 と @50 が同値（ecr 0.8126）になってしまう。

比較のため、エージェント側の土台では A の長さは中央39本（最大168）。
**A のみの実力を測るには A が長い土台が要る。**

GPU が空き次第 `runs_rawq_deep.jsonl`（`--top-k 300`、`eval_retrieval.py` が
要求する件数と同じ）を採るジョブが tmux `littrace-agentless` で待機中で、
その土台が採れれば A のみ（`--agent` を渡さない `eval_retrieval.py` と同じ経路）が
そのまま測れる。掃引は tmux `littrace-sweep` が自動で走る。

いま言えるのは **@1 と @5 だけ**（A が中央7本なのでここは切られていない）:

| | @1 | @5 |
|---|---|---|
| A のみ（total ecr） | 0.6369 | 0.7646 |
| A+B（total ecr） | 0.6369 | 0.7823 |

@1 が同値なのは §6.1 と同じ理由（anchor は必ず1位に残る）。

参考として、純 BM25 の ablation（LLM も埋め込みも reranker も無し、A のみ、
`eval_retrieval.py` の実測、total）は次のとおり:

| 構成 | recall@5 | ecr@5 | ecr@20 | ecr@50 |
|---|---|---|---|---|
| `bm25.yaml`（chunk） | 0.606 | 0.697 | 0.797 | 0.847 |
| `bm25_paper.yaml`（論文単位） | 0.634 | 0.723 | **0.825** | **0.865** |
| `bm25_dual.yaml`（両方 RRF） | **0.663** | **0.748** | 0.815 | 0.843 |

### 6.4 判断

**ピア論文が要らないとしても、B を外す根拠にはならない**（§6.1）。外すなら
根拠付き gold を multi top20 で 18本失う覚悟が要る。

一方で **B は本番でコストがほぼゼロ**でもある——SPECTER2 索引は構築済みで、
書誌結合と全文MLT はキャッシュ後 CPU のみ、LLM 呼び出しは1回も増えない。
外して得られるのは実行時間の数秒だけなので、**取捨は精度だけで決めてよい。**

ピアを取らないことが本当に有利になるのは `paper_precision` を見るときだが、
**提出論文の選定は読解チーム側の担当**（`submit_from: candidates`）なので、
検索側の看板は引き続き `candidate_recall` 系で読む。
