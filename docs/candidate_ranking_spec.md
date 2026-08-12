# 候補順位付けの拡張（関係信号の導入）仕様

作成日: 2026-08-03 / 状態: **実装済み・測定済み**（3.1〜3.7 すべて。既定オフなので
既存構成の挙動は不変）

## 結論（先に）

| 節 | 打ち手 | 判定 | 根拠 |
|---|---|---|---|
| 3.1 | Paper BM25 併用 | **採用候補** | ecr@5 0.697 -> **0.748**（BM25 単体 ablation） |
| 3.2 | reranker の RRF 融合 | 未測定 | GPU が要る。実装とテストは済み |
| 3.3 | Consensus | **不採用** | multi ecr@20 0.852 -> 0.784 |
| 3.4 | 名指し保護 | **不採用** | 名前特定は 28/28 精度100%だが、28本とも既に候補1〜4位 |
| 3.5 | タイトル言及グラフ | **不採用** | 独自に拾う gold **0本**（既存3ソースが全部拾える） |
| 3.6 | 手法共言及グラフ | **不採用** | 同上、独自 **0本** |
| 3.7 | 評価の追補 | **採用** | single 32 / multi 23 に再分類。外部と比較可能に |

**当たったのは検索の土台（3.1）だけで、関係信号（3.4〜3.6）は3つとも空振りした。**
理由は一貫している——**このコーパスでは「質問が名指しした論文を検索が取り逃す」
ことも「明示的な言及でしか届かない gold」も、ほとんど存在しない**。
外部チームがそこに仕組みを入れているのは向こうの @1 が 0.767 だからで、
我々の single@1 は 1.000。**同じ打ち手でも、弱点が違えば効かない。**

実装で判明した重要な事実を2つ、先に書いておく:

- **本文中の名前照合は大文字小文字を捨てると壊れる。** 小文字化して照合した最初の
  実装では、`MoRE` / `MoST` / `DeFine` / `DIFFER` / `RANGE` / `CLEAR` といった手法名が
  本文の普通の英単語 more / most / define … に当たり、コーパスの21%（5,738論文）で
  **96,662本の偽の辺**が張られた（`HTML` は123本から「名指し」されていた）。
  大文字小文字を含めた一致とハブ名の除去を入れて 987本まで落ちた。詳細は
  `retrieve/paper_titles.py` のモジュール docstring。
- **手法名の正規表現抽出は MinerU の出力では成立しない**（3.6 の設計変更を参照）。

外部チームの検索システム（BM25 主体 + 関係グラフ + 順位保護、ReadingAgent 不使用）と
我々の現行系を突き合わせて出てきた移植案6件 + 評価の追補1件。
実装するときはこのファイルを読めば着手できるように書いてある。

---

## 1. なぜやるか

### 1.1 比較の前提（数字をそのまま並べてはいけない）

外部系の gold は「answer-bearing gold 87本 / single 43問 / multi 12問」。
我々は gold 120本（うち evidence 付き 91本）で **single 26問 / multi 29問**。
同じ55問なのに分類がほぼ逆転しているのは、外部系が「回答の根拠になる論文だけ残す」
段階で multi の多くが single に落ちたため。

つまり**外部系の指標は我々の `ecr` よりさらに甘い分母**（single 比率 78% vs 我々 47%）。
並べるなら `cr` ではなく `ecr` を使い、**見た目の差より実差は小さい**前提で読む。
分類を揃える手当ては 3.7 に書く。

我々の現状ベスト（`bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100`
+ `reading_expand_rrf/rrf`, 2026-08-03T14:37, 55件フル）と並べる:

| K | 我々 ecr@K (total) | 外部系 Recall@K |
|---|---|---|
| 1 | **0.675** | 0.638 |
| 5 | 0.810 | **0.955** |
| 10 | 0.868 | **0.970** |
| 20 | 0.924 | **0.989** |
| 50 | 0.967 | **1.000** |

**@1 は我々が勝ち、@5 で 0.145pt 突き放される。** 分母の甘さを差し引いてもこの形は
一貫している。single@1 が 1.000（外部系 0.767）なのがその裏返しで、
**我々は1本目を当てる力は足りていて、2〜5本目を積み上げる力が無い**。

### 1.2 確認済みの構造的欠陥（コード根拠）

- **reranker が RRF 順位を完全に捨てている**（`retrieve/hybrid.py:74`
  `return self.reranker.rerank(query, fused, top_k)`）。融合ではなく置換。
  CLAUDE.md には既に「8B は ecr が上がり cr が下がる」「reranker は
  『質問に答えるか』で判定するのでピア gold を必ず下げる」と記録があり、
  だから `reading_expand_rrf/rrf` ではランキングB を reranker に通さない設計にした。
  **同じ危険がランキングA の内部では無防備なまま残っている。**
  外部系はここを 0.59:0.41 の RRF + top20 集合保護で解いている。
- **ランキングB のソースが全部トピック類似性**。`specter2` / `bib_coupling` /
  `bm25_mlt` はどれも「内容が近い」。外部系は**明示的な関係**
  （タイトル言及・手法所有・手法利用）を持っていて、我々にはゼロ。
- **複数 anchor の合意が捨てられている**。`paper_expander.py:102 _interleave()` は
  各 anchor の近傍を交互配置して1本にするので、重複した論文は1回しか置かれない。
  「2本の anchor が揃って推した」という情報が消える（外部系の Consensus / Reciprocal
  が見ているのはまさにそこ）。
- **`bm25s_paper` 索引（2.9G, 構築済み）が検索本線で使われていない**。
  `bm25_paper.yaml` の単体 ablation にしか登場せず、chunk BM25 との併用構成が無い。

### 1.3 伸びしろの所在

現状ベストの multi は ecr@5 0.640 / @10 0.750 / @20 0.855。single は @1 で既に 1.000。
**打ち手は multi の @5〜@20 を動かせるかだけで評価する。**

ピア gold（`gold_papers` に名前はあるが evidence が1件も無い論文、120本中29本 = 24%）は
CLAUDE.md の記録どおり「埋め込みを大きくしても reranker を強くしても取れない」。
3.6 の手法グラフはここに届く唯一の新経路として設計する。

---

## 2. 方針

**すべて既定オフの任意キーにし、キーが無ければ現行と完全に同一の経路を通す。**
`Retriever` / `Indexer` / `Expander` の Protocol は変えない。

新規の関係信号は `registry.build("expander", ...)` に載せる。既存の
`reading_expand_rrf/rrf.yaml` の `expansion.sources` に1行足すだけで使える形にし、
`replay_expansion.py`（15秒）で本走行なしに順位が付くようにする。

検証は**本走行（4〜5時間）を使い切る前にオフラインで潰す**:

| 検証手段 | 対象 | 所要 |
|---|---|---|
| `scripts/replay_expansion.py` | 3.3 / 3.5 / 3.6（候補列の組み替えのみ） | 15秒 |
| `scripts/eval_retrieval.py` | 3.1 / 3.2（LLM 不要の純検索） | 数分 |
| オフライン再スコア（新規、3.4） | 3.4（候補列の並べ替えのみ） | 数十秒 |
| `scripts/run_search.py`（tmux） | 勝ち筋のみ最終確認 | 4〜5時間 |

---

## 3. 打ち手

### 3.1 Chunk BM25 + Paper BM25 の併用（S-1）

**追加構築ゼロ。yaml 1枚。**

`bm25s`（chunk 単位）と `bm25s_paper`（論文単位）は両方構築済み
（`/data2/iseakira/pdfs/index/mineru/` に 5.1G / 2.9G）。外部系の土台はこの2本立て
RRF で、「両方から支持された論文が上がる」効果を狙っている。我々は片方しか
検索本線に載せていない。

```yaml
# configs/search_style/bm25_dual_qwen3_8b_rerank_qwen3_8b.yaml
indexers:
  - name: bm25s
    params: {}
  - name: bm25s_paper          # 索引は構築済み。index_name 不要
    params: {}
  - name: faiss_qwen3
    index_name: faiss_qwen3_8b
    params: { ... 既存構成のまま ... }
per_index_k: 100
fuser: { name: rrf, params: { k: 60 } }
```

**注意**: `bm25s_paper` の `chunk_id` は `"{paper_id}#paper"` の擬似 ID で
evidence 用途に使えない（CLAUDE.md 冒頭の方針どおり）。ReadingAgent は
chunk_id から evidence を引くので、**この索引由来のチャンクが LLM 可視域に
入ったときの挙動を確認する**こと（`candidate_recall` は論文単位なので指標は動くが、
`evidence_f1` が下がらないかを見る）。下がるなら `pool_k` までは載せて
LLM に渡す前に落とす、という切り分けをする。

**判定**: `eval_retrieval.py` で `recall@k` / `evidence_recall@k` が現行構成を
上回るか。上回らなければ捨てる（索引が既にあるので損失は yaml 1枚）。

---

### 3.2 reranker を「置換」から「RRF 融合 + 上位集合保護」へ（S-2）

**1.2 の第1項に直接効く。@5 の差を最も説明できる打ち手。**

`HybridRetriever` に既定オフの params を足す。**書かなければ現行と1ビットも
変わらない**（既定は純置換のまま）。

```yaml
# search_style のトップレベル
pool_k: 1000
rerank_blend:
  original_weight: 0.6   # 融合前（RRF 直後）の順位
  rerank_weight: 0.4     # reranker の順位
  rrf_k: 60
  protect_top: 20        # 元 top20 の「集合」は reranker に壊させない
```

実装は `retrieve/hybrid.py:67-75` を差し替える:

```python
fused = self.fuser.fuse(runs, top_k=fuse_k)
if self.reranker is None:
    return fused[:top_k]

reranked = self.reranker.rerank(query, fused, len(fused))
if self.rerank_blend is None:          # 既定。現行と同一
    return reranked[:top_k]

# 順位融合。スコアは見ない（reranker の yes 確率と RRF スコアは
# 比較可能でない——同じ理由で _combine_rrf も順位しか使っていない）
k = self.rerank_blend["rrf_k"]
wa = self.rerank_blend["original_weight"]
wb = self.rerank_blend["rerank_weight"]
scores = {}
for rank, r in enumerate(fused):
    scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + wa / (k + rank + 1)
for rank, r in enumerate(reranked):
    scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + wb / (k + rank + 1)
ordered = sorted(by_id.values(), key=lambda r: -scores[r.chunk_id])

# 上位集合保護: 融合前 top-N の「集合」を先頭に残す（並び順は融合結果に従う）
if protect_top:
    protected = {r.chunk_id for r in fused[:protect_top]}
    ordered = ([r for r in ordered if r.chunk_id in protected]
               + [r for r in ordered if r.chunk_id not in protected])
return ordered[:top_k]
```

**重みは外部系の 0.59 をそのまま借りない。** 外部系自身が「同じ validation で
選んだので過学習の可能性」と明記している。`original_weight` を
0.0（= 現行の純置換）/ 0.4 / 0.6 / 1.0（= reranker 無視）で振り、
我々の55件で選び直す。`protect_top` は 0（無効）/ 10 / 20 を振る。

**判定**: `eval_retrieval.py` の `recall@5` / `evidence_recall@5`。
**cr と ecr を必ず並べて読む**——reranker を弱めると
「質問に答える論文」の選別が緩むので、ecr が下がって cr が上がる向きに
出る可能性がある（CLAUDE.md の `pool_rescore` と逆向きの現象）。
その形になったら**打ち手の評価は ecr で見る**という既定方針に従って捨てる。

---

### 3.3 展開を anchor ごとの独立ランキングにする（Consensus）（S-3）

**最安（replay 15秒）。まずこれを測る。**

現状 `Specter2PaperExpander.rank()` 等は `_interleave(self._pools(...), neighbors)` を
返すので、複数 anchor が同じ論文を推しても**候補列には1回しか現れない**。
`_combine_rrf` から見ると「2本の anchor が揃って推した論文」と
「1本だけが推した論文」が区別できない。

RRF は同じ論文が複数のランキングに出れば自然に加点するので、
**anchor ごとに別ランキングとして渡せば Consensus がタダで手に入る**。

expander に `rank_pools()` を足す（`rank()` は残す。既存経路は不変）:

```python
def rank_pools(self, ranked_paper_ids: list[str]) -> list[list[str]]:
    """anchor ごとの近傍ランキングを潰さずに返す。"""
    return self._pools(ranked_paper_ids)      # 既に各クラスにある
```

`FusedPaperExpander.rank_pools()` は各ソースの pools を**連結**して返す
（ソース×anchor の直積ぶんのランキングになる。RRF は本数に比例して
スコアが増えるので、`related_weight` で正規化する余地は残す）。

`reading.py:_combine_rrf` 側:

```python
if getattr(expander, "consensus", False) and hasattr(expander, "rank_pools"):
    pools = expander.rank_pools(candidate_papers)
else:
    pools = [expander.rank(candidate_papers)]
anchors = candidate_papers[: getattr(expander, "anchors", 1)]
pools = [anchors + [p for p in pool if p not in anchors] for pool in pools]
for pool in pools:
    for rank, paper_id in enumerate(pool):
        scores[paper_id] = scores.get(paper_id, 0.0) + weight / (k + offset + rank + 1)
```

anchor 自身を各 pool の先頭に置く扱いは現行と同じ理由で維持する
（置かないと single_paper の候補1位が top20 から落ちる。`_combine_rrf` の
コメント参照）。

yaml は `expansion` ブロックに2キー:

```yaml
expansion:
  combine: rrf
  consensus: true      # 新規。既定 false = 現行と同一
  anchors: 3           # 既定 1。consensus 時のみ意味を持つ
  neighbors: 20
```

`consensus` は `_COMBINE_DEFAULTS`（`paper_expander.py:83`）に足して
`_set_combine()` 経由で全 expander に配る。

**CLAUDE.md の「anchor を3本に増やしても ecr は動かず候補だけ増えた」は
*位置挿入方式*での実測**であって、RRF 統合方式では結論が変わりうる。
位置挿入では合意情報を使う場所が無かったが、RRF には有る。

**判定**: `replay_expansion.py --set consensus=true --set anchors=3`。
`anchors` を 1/2/3/5 で振る。multi の ecr@5〜@20 が動くか。

---

### 3.4 質問が名指しした論文の保護（S-4）

質問文に `D-FINE` のような論文識別子があり、候補内にその論文が居るのに
低順位なら上位へ引き上げる。single@1 は既に 1.000 なので**効くとしたら
multi の2本目以降**（「FooNet と BarTune を比較せよ」型）。

論文識別子の作り方（外部系の設計をそのまま借りる）:

- コーパス各論文の `title_abstract` チャンクからタイトルを取り、
  **コロン前の部分**を識別子候補にする（`D-FINE: Redefine Regression...` → `D-FINE`）
- 正式タイトル全文も識別子にする
- **汎用語を除外**: `BERT` / `RAG` / `LLM` / `GPT` / `CLIP` など。
  除外リストは固定リストではなく**コーパス内で複数論文が同じ識別子を持つなら捨てる**
  （一意性で機械的に決める。固定リストは保守が要るので補助に留める）
- 大文字・数字・ハイフンを含む2〜20文字程度に制限し、英単語だけの語は捨てる

実装は `retrieve/` に新モジュール（`title_protect.py`）。
**`attribute_filter` と同じ位置づけ**——索引の改修も再構築も不要で、
検索結果の後処理として順位を触るだけ。

```yaml
# agent_style の任意キー（元の質問から1回だけ抽出するので retriever ではなく agent 側）
title_protect:
  enabled: true
  max_papers: 4        # 保護する上限
  promote_to: 10       # 引き上げ先の順位
```

**元の質問から1回だけ抽出してステップ全体で使い回す**（`attribute_filter` と同じ理由。
サブクエリは識別子を落とすことがある）。

**判定**: 候補列の並べ替えだけなので、`replay_expansion.py` と同型の
オフライン再スコアで測れる（土台の予測ファイルの `candidate_papers` を組み替える）。
発火件数と、発火したクエリの ecr@5 / @20 を見る。
**発火0件なら即捨てる**——実装の複雑さに見合わない。

---

### 3.5 タイトル言及グラフ（A-1）

**新規 expander。既存結論を上書きする可能性がある本命。**

コーパス全論文のタイトル（と 3.4 の識別子）を辞書化し、
**各論文の本文にどの論文のタイトルが出現するか**を1走査で集める。
A の本文に B のタイトルがあれば A→B のリンク。

**CLAUDE.md の「引用グラフはほぼ張れない（実測でコーパス内引用1本）」は
arXiv ID 解決ベースの結論**。参考文献の arXiv ID が取れなくても、
本文に正式タイトルが書かれていれば繋がる。**この結論は上書きされうるので検証する。**

実装は `BibCouplingExpander`（`paper_expander.py:247`）をテンプレにする
（同じ「chunks.jsonl 1走査 → pickle キャッシュ」構造、GPU 不要）:

```python
@register("expander", "title_mention")
class TitleMentionExpander:
    def __init__(self, chunks, cache_path, neighbors=20, anchors=1,
                 two_hop_weight=0.05, max_hub_degree=4, **combine_kwargs): ...
```

索引構築（1走査）:

1. 全論文の `title_abstract` からタイトルと識別子を集める（3.4 と共通化する）
2. **Aho-Corasick か、識別子長でバケットした正規表現**で本文を走査する
   （27,487 × 27,487 の総当たりは不可）。`pyahocorasick` を足すか、
   まず素朴に「識別子を小文字化して set 化 → 本文をトークン化して照合」で
   十分速いか測る
3. `mentions: dict[paper_id, set[paper_id]]` と転置 `mentioned_by` を pickle 保存

近傍の返し方:

- **直接リンク**（A↔B、方向は問わない）: 重み `1.0`
- **2ホップ**（A→C→B）: 重み `two_hop_weight`（既定 0.05）。
  ただし**中継論文 C の次数が `max_hub_degree`(4) 以下のときだけ**
  （survey を経由して無関係な論文が大量に上がるのを防ぐ）
- **References より前の本文だけ**を対象にする（参考文献リストに名前が
  あるだけの論文で繋がるのを防ぐ）。chunk の `chunk_type` / セクション情報で切る

**References を含めるか否かは実測で決める。** 外部系は Paper BM25 で
`exclude_references: false` にしていて「引用関係を聞く問題に有利だが
参考文献に名前があるだけの論文も拾う」と認めている。
**含む版・含まない版の両方を replay で測る。**

```yaml
expansion:
  combine: rrf
  sources:
    - { name: specter2, params: { neighbors: 20 } }
    - { name: bib_coupling, params: { neighbors: 20 } }
    - { name: bm25_mlt, params: { neighbors: 20 } }
    - { name: title_mention, params: { neighbors: 20 } }   # 追加はこの1行
```

**判定**: `replay_expansion.py`。既存3ソースに足して multi の ecr@20 が動くか。
併せて**「title_mention だけが拾えた gold は何本か」を数える**
（CLAUDE.md が3ソースについて出している「SPECTER2 15本 / 書誌結合 11本 /
全文MLT 16本、MLT だけが2本」と同じ集計）。**独自に拾う gold が0本なら
既存3ソースと冗長なので捨てる。**

---

### 3.6 手法グラフ（A-2）

**ピア gold 29本（gold の 24%）に届く唯一の新経路。**

> **実装時の設計変更（2026-08-03）**: 正規表現による手法名抽出をやめ、
> **タイトルのコロン前の見出し**を手法名として使い、**共言及（co-mention）**の
> グラフにした。registry 名は `method_comention`。理由は2つ:
>
> 1. `We propose X` の X をトークン列から復元する方式は **MinerU の出力では
>    成立しない**。タイトルの実データが `M o RE :` / `T oken S hapley:` /
>    `D e F ine:` のように大文字の前で分かち書きが壊れており、本文側の名前も
>    同じ崩れ方をする。コロン前の見出しなら `paper_titles` の正規化
>    （英数字以外を落として連結）でそのまま吸収できる。65%のタイトルが
>    コロンを持ち、その前が手法名になっている。「提案元がコーパス内で一意」
>    という要件は、正規表現ではなく `TitleIndex` の一意性フィルタが満たす。
> 2. **owner への辺は `title_mention` の辺と同じものになる**（「手法 X を挙げる
>    論文」→「X の提案元」）。足しても情報が増えない。ピア gold に届くのは
>    「**A と B が同じ手法群を挙げている**」ほうで、これは title_mention では
>    作れない辺。Jaccard で測る（`bib_coupling` と同じ形、`min_shared: 2`）。

（当初案）`We propose X` / `We introduce X` / `we call our method X` /
`Our proposed framework, X,` を正規表現で抽出し、手法名 → 提案元論文（owner）の
対応を作る。他論文の本文に X が出れば relation。

我々の最大の未解決課題は「q_036 TCM の batch size を聞く質問の gold に
IMM / sCT / Consistency Models Made Easy が並ぶ」型のピア gold で、
CLAUDE.md は「そのクエリベクトルの近傍にピア論文は来ない」と結論している。
手法グラフは**「TCM 論文が比較対象として IMM を明示している」という別経路**で
ここに繋がる。

```python
@register("expander", "method_graph")
class MethodGraphExpander:
    def __init__(self, chunks, cache_path, neighbors=20, anchors=1,
                 max_owners_per_method=1, max_papers_per_method=10,
                 **combine_kwargs): ...
```

**誤検出の防御（外部系の設計をそのまま借りる）**:

- `LLM` / `RAG` / `BERT` のような汎用語を除外（3.4 と同じ一意性ルール）
- **手法の提案元がコーパス内で一意な場合だけ採用**（`max_owners_per_method: 1`）
- **References より前の本文**でだけ手法言及を探す
- 短い略称（3文字以下等）は周辺に手法固有の文脈語が必要
- **同じ手法名に繋がる論文が `max_papers_per_method`(10) を超えたら
  曖昧な名前としてグラフから外す**

近傍の返し方: anchor の論文が提案・言及している手法を集め、
その手法の owner と、その手法を言及している他論文を近い順に返す。

**判定**: `replay_expansion.py`。**ピア gold（evidence を持たない gold 29本）を
何本拾えたかを個別に数える**——ここが設計の目的なので、
全体の ecr が動かなくてもピア gold の回収が増えるなら意味がある。
ただし CLAUDE.md の方針どおり、**ピア gold は `evidence_f1` に寄与しない**ので
`paper_precision` を下げないかも併せて見る。

---

### 3.7 評価の追補: evidence-backed で single/multi を再分類

`scripts/evaluate.py:461` は single/multi を `gold.task_family` で振り分けている。
外部系は「answer-bearing gold に絞った後の本数」で分類しているので、
**現状のままでは single/multi 別の数字を突き合わせられない**（1.1）。

`ecr` のシナリオ振り分けだけを「`evidence_backed_paper_ids(gold)` の本数が
1本なら single、2本以上なら multi」に変えた系列を**並べて出す**
（既存の `evidence_candidate_recall_at{k}_{single,multi}_macro` は
`task_family` ベースのまま残す。指標の意味を変えない）。

```
evidence_candidate_recall_at{k}_{single,multi}_macro          # 現行: task_family 基準
evidence_candidate_recall_by_backed_at{k}_{single,multi}_macro # 新規: evidence 本数基準
```

**`paper_recall` / `paper_f1` は採点仕様なので触らない。**

---

## 4. 見送り

- **属性フィルタのソフト化**（外部系は ±1 × 0.75 の加点、我々はハードフィルタ）。
  我々は既に `min_results` の fail-open があり、実測で発火5件・gold 満足率 18/18 = 100%。
  **壊れていないものは触らない。**
- **SPECTER2 の `adhoc_query` による質問→論文の直接 dense 検索**。
  外部系自身が未検証かつ「候補 Recall が上がる一方で不要論文も増える」と
  リスクを認めている。我々は既に `faiss_specter2_abstract` を持っているので
  やるなら安いが、優先度は上記6件の後。
- **図表を画像のまま扱う経路の強化**。外部系も「Qwen3 Reranker は画像を見ていない」
  と認めていて、両者共通の未解決領域。`bm25_qwen3_vl_8b_rerank_qwen3vl_8b.yaml` が
  既にあるのでこの spec の範囲外とする。

---

## 5. 実行順

| 順 | 節 | 打ち手 | 検証 | 期待 |
|---|---|---|---|---|
| 1 | 3.3 | anchor 別 RRF（Consensus） | replay 15秒 | multi@5〜20 |
| 2 | 3.5 | タイトル言及グラフ | 1走査 + replay | multi、既存結論の上書き |
| 3 | 3.6 | 手法名グラフ | 1走査 + replay | ピア gold 29本 |
| 4 | 3.1 | Paper BM25 併用 | eval_retrieval 数分 | 土台の底上げ |
| 5 | 3.2 | reranker 融合 + 集合保護 | eval_retrieval 数分 | **@5 の本丸** |
| 6 | 3.4 | 名指し保護 | オフライン再スコア | multi の2本目 |
| — | 3.7 | 評価の追補 | 即時 | 外部系と同じ土俵 |

1〜3 と 6 は `replay_expansion.py` 系の土台に載るので**本走行を1回も使わずに
順位が付く**。4〜5 は検索側なので `eval_retrieval.py`。
全部済んでから勝ち筋だけ tmux で本走行する（CLAUDE.md 3.1 の作法に従う）。

**LLM は非決定的でクエリは55件しかないので、数ポイントの差はノイズ。**
結論を出す前に複数回まわす。
