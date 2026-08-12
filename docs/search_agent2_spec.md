# 検索エージェント v2 アーキテクチャ

**サブクエリを、質問文からではなく「コーパスが返した本文」から作る。**

材料が無いのは step0 だけ。それ以降のクエリは全部、検索で当たった論文と
論文→論文展開で拾った論文の本文から書き換えて作る。

現行フローと実測は `docs/search_flow_current.md`。この文書は差分だけを書く。

---

## 1. 全体のループ

```
Query { query_id, question, answer_types, table_schema }
  │
  ├─ _extract_attribute_filter(question)        （現行のまま。会議名・年を1回だけ取る）
  │
  ├─ step0: 質問文からサブクエリ                 （唯一の非接地ステップ）
  │     → 重複除去（2.6）
  │
  └─ for step in 0 .. max_steps-1:
        │
        ├─ 各サブクエリを検索                     （現行のまま）
        │     BM25 + 埋め込み → RRF → reranker → chunks に貯める
        │
        ├─ 読解 _read_and_judge()                （現行のまま、max_candidates 本）
        │     → evidence_chunk_ids / sufficient / missing
        │
        ├─ sufficient なら終了
        │
        └─ ★ 次のサブクエリ = クエリ書き換え（_refine() を置き換える）
              材料A ← 検索の上位論文
              材料B ← 論文→論文展開の上位論文
              → LLM 1回 → 書き換えクエリ
              → 重複除去（2.6）→ 残ったものだけ検索する
```

ループ後:

```
_combine_rrf(ランキングA, ランキングB)            （現行のまま）
  → candidate_papers[:50] → gold_papers[:10]
  → evidence（読解が選んだ chunk_id から）
```

---

## 2. クエリ書き換え

### 2.1 材料A — 検索の上位論文（2段）

チャンクを**論文単位に畳む**（`_candidate_papers()` と同じ処理：`paper_id` でまとめ、
各論文の最高スコアで並べる）。そのうえで**深さの違う2つのビューを渡す**。

**詳細ビュー（上位 `from_a.detail_papers` 本）** — 「いま何を引き当てているか」

| 材料 | 出所 |
|---|---|
| タイトル・会議・年 | `RetrievalResult.metadata` |
| abstract | `ChunkStore.load_paper()` の `title_abstract` チャンク |
| **ヒットしたチャンク** | その論文で実際に検索に当たったチャンク上位 `hit_chunks_per_paper` 本 |

**俯瞰ビュー（上位 `from_a.listing_papers` 本、タイトルのみ）** — 「軸がズレていないか」

```
1. [ICML 2025] LOGO --- Long cOntext aliGnment via efficient preference Optimization
2. [ICML 2025] Larger or Smaller Reward Margins to Select Preferences
...
20. [ACL 2025] How to Train Long-Context Language Models (Effectively)
```

**2段にする理由。** 詳細ビューだけだと分布の異常が分布に見えない。
q_022 の候補列では上位5本に長文脈論文が1本しか無く（ノイズに見える）、
**20本まで広げて初めて10本が長文脈だと分かる**——つまり展開の軸がズレている、
という事実が5本の窓には映らない。俯瞰ビューはタイトルだけなので追加コストはほぼゼロ。

実装は `_grounding_note()` がすでに同じ整形をしている（上位N本を
`[venue year] title` で並べ、**上位に1本も残せなかったサブクエリも名指しする**）。
本数を `listing_papers` にして流用する。

### 2.2 材料B — 論文→論文展開の上位論文

```
展開（specter2 + bib_coupling + bm25_mlt を RRF 融合、anchors: 3）
  → 関連論文の順位列
  → 上位 from_b.top_papers 本
  → ChunkStore.load_paper(paper_id)                     （0.7ms/本）
  → title_abstract チャンクだけを取る
```

**B は本文チャンクを見せず、本数を稼ぐ。** 欲しいのは「この論文は自分を何と呼ぶか」
という語彙だけで、そのために本文は要らない。abstract だけなら20本でも
5本×本文2チャンクとほぼ同じ分量に収まる。

**`anchors: 3` が前提。** anchors=1 だと B の上位に取りたい論文がほとんど入らない
（実測: ランキングA の top20 に入っていない multi の gold 26本のうち、
B の上位5本に入るのが anchors=1 で0本、anchors=3 で5本）。

**展開はループの中に移す**（現行は `_build_prediction()` の中で最後に1回だけ）。

### 2.3 A と B の非対称性

| | A（検索） | B（展開） |
|---|---|---|
| チャンクが手元にあるか | **ある**（ヒットしたもの） | **無い**（paper_id だけ） |
| 何本見せるか | 詳細5本 ＋ タイトルのみ20本 | **20本** |
| 各論文に見せるもの | title + abstract + ヒットしたチャンク | **title + abstract のみ** |
| 材料が答える問い | 「いま何を引き当てているか」「軸はズレていないか」 | 「この論文は自分を何と呼ぶか」 |

### 2.4 書き換え（LLM 1回）

```
入力:
  元の質問
  材料A 詳細（5本：タイトル / abstract / ヒットしたチャンク）
  材料A 俯瞰（20本：[venue year] タイトルのみ）
  材料B（20本：タイトル / abstract）
  すでに投げたサブクエリ一覧
  そのうち上位に1本も残さなかったもの（＝効かなかったクエリ）
  CORPUS_NOTE（Web検索エンジンではない）
  _constraint_note（"[ICML 2025]" を先頭に付けろ）

指示:
  これらの論文が**自分自身を説明するときに使っている語**で、
  元の質問と同じ性質を満たす論文を探すクエリを R 本書け。
  質問文の言い換えを書くな。性質は変えるな、語彙だけ変えろ。

出力:
  {"subqueries": ["...", "..."]}      ← このあと 2.6 の重複除去にかける
```

### 2.5 書き換えクエリの行き先

**通常のサブクエリとして扱う。** 検索 → chunks に積む → 読解の対象になる。
これで展開由来の論文もチャンクを持ち、`evidence` に出られる。

### 2.6 重複除去 — 本数は固定せず「中身が重ならない数」にする

**本数を先に決めない。** LLM に N 本と指定しても、返ってくるのは
言い回しを変えただけの同じクエリになりがちで、そのぶん検索と reranker が
空回りする。**何本作らせるかではなく、何本残すかを中身で決める。**

判定は**引いてくる論文が重なるか**で行う。文字列の重複では捕まえられない
（`reference-free …` と `Direct Alignment Algorithm …` は文字列としては全く別だが、
逆に言い回し違いで同じ論文しか引かないクエリは文字列上は別物に見える）。

```
候補クエリを順に処理:
  1. **BM25 だけ**で引く（埋め込みも reranker も使わない。CPU、1本あたり数十ms）
  2. 上位 probe_k 本の paper_id 集合を取る
  3. すでに採用したクエリ・すでに投げたクエリの集合と Jaccard を比べる
  4. max_overlap を超えたら捨てる
  5. 残ったものだけを本番の検索（埋め込み + reranker）に回す
```

**BM25 で先に篩う理由**は、コストが桁で違うから。本番の検索1本は
reranker が `pool_k` 件を推論するが、BM25 の引き当ては索引を1回叩くだけ。
**重複したクエリに reranker を1回も走らせない。**

安全弁として `max_queries` で上限も切る（LLM が大量に返したとき用）。
step0 の非接地クエリにも同じ除去をかける。

---

## 3. やらないこと

- **ランキングB は reranker に通さない。** `_combine_rrf()` に入る B の順位は現行のまま。
  reranker がかかるのは書き換えクエリで回した検索の結果だけ。
- **`_combine_rrf()` は置き換えない。** 書き換えで拾えなかった関連論文は、
  これまでどおり順位融合だけで押し上げる。
- **提出論文の選定と回答生成はしない**（現行のまま）。

---

## 4. パラメータ（agent yaml）

```yaml
expansion:                  # 現行のまま。anchors だけ変える
  sources:
    - { name: specter2, index_name: faiss_specter2_abstract }
    - { name: bib_coupling, min_shared: 2 }
    - { name: bm25_mlt, query_chars: 1200 }
  neighbors: 50
  anchors: 3
  combine: rrf
  combine_rrf_k: 60
  related_weight: 1.0
  related_offset: 0

rewrite:                    # ★新規。書かなければ現行と同一のコードパス
  enabled: true
  at_step: 1                # この step 以降、_refine() を置き換える
  chunk_store: /data2/iseakira/pdfs/chunks/mineru_chunks.jsonl

  from_a:
    detail_papers: 5        # title + abstract + ヒットしたチャンク
    hit_chunks_per_paper: 2
    include_abstract: true
    listing_papers: 20      # タイトルのみの俯瞰ビュー（軸ズレ検出用）

  from_b:
    top_papers: 20          # 語彙が欲しいだけなので本数を稼ぐ
    include_abstract: true
    body_chunks_per_paper: 0  # 本文チャンクは見せない

subquery_dedup:             # ★新規。step0 と書き換えの両方にかける
  method: bm25_overlap      # bm25_overlap | none
  probe_k: 20               # BM25 上位N論文の集合で比べる
  max_overlap: 0.7          # Jaccard がこれを超えたら捨てる
  max_queries: 4            # 安全弁（除去後の上限）

params:
  max_steps: 3
  retrieve_top_k: 20
  max_candidates: 20        # 読解が読む本数。書き換え材料の5本とは別物
  chunks_per_paper: 2
  snippet_chars: 1800
  paper_cutoff: llm
  max_papers: 10
```

---

## 5. 実装の差分

| 箇所 | 変更 |
|---|---|
| `agent/reading.py` `_refine()` | `rewrite.enabled` なら書き換え経路へ分岐 |
| `agent/reading.py` `run()` | 展開をループ内で呼べるようにする（現行は `_build_prediction()` のみ） |
| `agent/reading.py` 新規 | 材料A / 材料B の組み立て、書き換えプロンプト |
| `chunk_store.py` | 変更なし（`load_paper()` をそのまま使う） |
| `retrieve/paper_expander.py` | 変更なし |
| `retrieve/hybrid.py` | 変更なし |
| `configs/agent_style/` | `rewrite` ブロックを持つ yaml を1本追加 |
