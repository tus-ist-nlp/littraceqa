# anchor 相対のピア判定器（`peer_rerank`）— 仕様

**未実装。** 走行中はコードを触らない方針のため仕様だけ先に置く。
根拠となる実測は `docs/offline_findings_spec.md` の §3.9。

## 何を解くのか

multi の `candidate_recall@20` を @50 に近づけたい。@21-50 に居る gold は
土台1で9本 / 土台2で7本しかなく、**取り切れれば @20 が @50 に並ぶ**。

だが**枠はゼロサム**で、11-20位に押し込むには同じ数だけ既存を押し出す。
そこの gold 密度＝**損益分岐は4%**。ランキングB の既存信号は最良でも
「2ソースに居る」の **3%** で、分岐を割っている（`docs/offline_findings_spec.md` §3.9）。
だから重み付け（凍結+裾の再融合）も枠の予約も**12通り試して全滅**した。

**必要なのは重み付けではなく B の解像度。** 4% を超える判定器を1つ足せば勝てる。

## なぜ既存の reranker ではダメで、これならいけるのか

CLAUDE.md の「**ランキングB は reranker に通さない**」は**質問相対**の話。
reranker の instruction は

    Given a scientific question, retrieve passages from research papers
    that help identify or support the answer

で、「質問に答えるか」を測る。ピア gold は**定義上その質問に答えない**
（`evidence` が1件も紐づかない同トピック論文）ので、必ず下がる。
実測でも `reading_expand_insert/rerank.yaml`（質問相対で展開を rerank）は
cr@20 0.813 止まりだった。

**anchor 相対なら測る量が違う**:

    この論文は〈確認済みの正解論文〉と同じ問題設定・同じ手法族を扱っているか

質問文に現れない語彙（自称のカテゴリ名）でも判定できる。
q_036「TCM の batch size は？」の gold に IMM / sCT / Consistency Models Made Easy が
並ぶ構造は、**anchor（TCM）との関係**でしか説明できない。

## ⚠ 先に上限を知っておく — **効果は最大でも gold 5〜6本**

判定器がどれだけ賢くても、**ランキングB に入っていない gold は拾えない**。
融合 top20 に無い multi gold がどこに居るかを数えると:

| | 土台1（fat） | 土台2（steps2） |
|---|---|---|
| 融合 top20 に無い multi gold | 19本 | 20本 |
| **B にまったく居ない（原理的に不可能）** | **8本** | **10本** |
| B top10 に居る | **0本** | **0本** |
| B top20 | 3本 | 2本 |
| **B top30（`top_k: 30` の上限）** | **6本** | **5本** |
| B top50 | 8本 | 8本 |

**B の top10 には1本も居ない。** 取りたい gold は B の 15〜82位に散っている
（実例: q_051 15位 / q_036 19位 / q_048 25位 / q_038 34位・82位 / q_044 58位）。

したがって **multi cr@20 の上限は +0.05 程度**（29クエリで gold 5〜6本）。
押し出される既存 gold を差し引いた正味はさらに小さい。
**実行間ばらつきが multi@20 で ±0.01〜0.02 観測されている**ので、
判定器が完璧でも正味の改善はノイズの2〜3倍にとどまる。

`top_k: 50` に広げると拾える上限は8本に増えるが、**同じ `keep` で50本から選ぶ**ことに
なるので精度要求が上がる。単純に得とは限らない。

### 優先度についての判断

**読解側（オラクル上限 multi@5 +0.150）のほうが桁が違う**ので、そちらを先にやるべき。
ただし `peer_rerank` は**判定スコアを1度キャッシュすれば `keep` / `insert_at` /
`top_k` を数秒で振れる**ので実装コスト自体は小さい。

**進め方**: まず判定スコアだけ作って**当たり率が4%を超えるか**を見る。
超えなければそこで打ち切る（並べ替えの実験まで進めない）。

## 設計

### 入力

* **anchor**: `_anchor_papers()` が返す集合（候補1位 ∪ 読解 LLM の確認済み論文）。
  `anchor_from: verdict` で既に実装済み。**確認済みが混ざるのが要点**——
  「正解と分かっている論文の仲間」を探すことになる。
* **候補**: ランキングB の上位 `peer_rerank.top_k`（既定30）本。
  1,700本全部は要らない——gold は B の上位30本の中では上の方に居る
  （@21-50 のピア gold は B 中央15〜19位）。

### 判定

`Qwen3Reranker` をそのまま使い、`instruction` と「クエリ」を差し替える:

```yaml
peer_rerank:
  enabled: true
  top_k: 30            # B の上位何本を判定にかけるか
  keep: 5              # 判定を通した上位何本を昇格させるか
  insert_at: 10        # 昇格先の位置（融合後の候補列）
  model: Qwen/Qwen3-Reranker-0.6B   # 8B は不要（下記）
  instruction: >
    Given a reference paper, judge whether the candidate paper studies the same
    problem setting or belongs to the same family of methods, so that a survey of
    this topic would list them together. Ignore whether it answers any question.
```

`rerank(query=anchor の title+abstract, docs=B候補の title+abstract)` の形で呼ぶ。
**anchor が複数あるときは anchor ごとに判定して max を取る**（どれか1本の仲間なら良い）。

### 昇格の仕方

**スコアで既存候補と混ぜてはいけない。** CLAUDE.md に実測がある——展開を
スコアで混ぜると cr@20 が 0.822 -> 0.773 に壊れた。reranker の絶対スコアは
既存候補と比較可能でない。**本数で絞ってから位置挿入する。**

    candidate_papers[:insert_at] + 昇格 keep 本 + 残り

`insert_at: 10` は「A の強い論文（1-10位）を邪魔しない」という制約から。
§3.9 の案X・案Yで single cr@1/@5 が全条件で不変だったので、この位置なら安全。

### コスト

1クエリにつき **reranker 推論30件 × anchor 数**。anchor は中央1本・平均1.2本なので
実質30〜40件で、本番の検索1本（`pool_k: 200`）より軽い。**0.6B で十分**——
判定するのは「同じ手法族か」で、8B の細かい弁別は要らない。実測でも展開の rerank は
0.6B で代用できている（23秒 vs 8B の147秒）。

## 採否の判定基準

**4% を超えるかどうか**。昇格した `keep` 本のうち gold の割合を数える:

* `keep: 5` × 29クエリ = 145本を昇格 → **gold が6本以上なら分岐超え**
* 同時に **multi cr@20 と single cr@1 / @5 を必ず並べて読む**
  （§3.9 の全案は single を守れていたが multi が上がらなかった）

**土台3本すべてで悪化しないこと**を条件にする（`runs_fat` / `runs_steps2_notable` /
`runs_notable`）。土台1本での結論が転んだ前例があるため（§冒頭の訂正）。

## オフラインで測れる範囲

昇格の判定は GPU が要るので `replay_expansion.py` のようには回せないが、
**判定結果さえ1度作ればあとは候補列の並べ替え**なので、
`(query_id, paper_id) -> score` を1度だけ計算してキャッシュすれば
`keep` / `insert_at` / `top_k` は数秒で振れる。**キャッシュを先に作る。**

## やらないこと

* **B 全体（1,700本）を判定にかけない。** gold は B の上位に居るので上位30本で足りる。
* **質問相対の reranker と混ぜない。** 測る量が違うので、片方の順位にもう片方の
  スコアを足すと両方壊れる。
* **`keep` を増やして当てにいかない。** 損益分岐は本数に依存しないので、
  精度が4%を割る本数まで増やした時点で純損になる。
