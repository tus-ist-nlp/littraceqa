# src/littraceqa/di_pipeline/agent/

`Query` を受け取り `Prediction` を返す検索エージェント層。`retriever`（と必要なら
`llm`）を DI で注入して使う。retriever が「どの論文か」までを担い、agent が
「その中のどこが根拠か（evidence）」を担う。**回答（freeform / multiple_choice /
table）は生成しない**——回答生成も提出論文の選定も読解チーム側の担当で、
我々が渡すのは候補列と evidence まで。

## ファイル

- `base.py` — `SearchAgent` Protocol（`run(query) -> Prediction`）
  検索そのものの素の実力を測る基準線として `scripts/eval_retrieval.py` が使う
- `reading.py` — `ReadingAgent`: 検索→読解→不足分の再検索を反復する本命（デフォルト）
- `task_family.py` — `TaskFamilyClassifier`（single/multi 推定）と `apply_paper_cutoff`
- `evidence.py` — `RetrievalResult` から提出用 `Evidence`（locator 付き）を組み立てる
- `json_utils.py` — LLM 応答からの JSON 抽出

---

# ReadingAgent の設計

## 一言で

**「検索 → LLM に候補を読ませて根拠を確定 → 足りなければ何が欠けているか聞いて
再検索」を繰り返す、反復型の検索エージェント。**

## 全体フロー（`run()`）

```
run(query):
  0. _extract_attribute_filter()  会議名・年の制約を「元の質問から1回だけ」抽出
  1. _decompose()                 質問を検索用サブクエリに分解
  ┌─ for step in range(max_steps=3):              ← 反復ループ
  │  2. retriever.retrieve(subquery, retrieve_top_k) を各サブクエリで実行
  │     → 結果を chunks: dict[chunk_id -> RetrievalResult] に蓄積
  │  3. _candidate_papers()        チャンクを論文単位に集約、上位を候補に
  │  4. _read_and_judge()          LLM に候補を読ませて判定
  │     → {paper_ids, evidence_chunk_ids, sufficient, missing}
  │  5. verdict["sufficient"] が true なら break
  │  6. _refine(missing)           欠けているものを聞いて新サブクエリ → 2 へ
  └─
  7. _build_prediction()          提出物（Prediction）を組み立て
                                  （answer は空。回答生成は読解チーム側の担当）
```

LLM 呼び出しは **1周につき最大2回**（`_decompose` / `_refine` が1回、`_read_and_judge`
が1回）。回答生成を外したので、以前あった「クエリ1件につき+1回」は無い。

## 各ステップの設計判断

### 0. 属性フィルタの抽出（`_extract_attribute_filter`）

質問が「Which NAACL 2025 papers ...」のように検索範囲を会議名で明示している場合、
その制約を**元の `query.question` から1回だけ**取り出し、反復全体で使い回す。

- **サブクエリからは取らない。** `_decompose()` は「NAACL 2025」のような制約語を
  落とすことがあるため、サブクエリから抽出すると発火しない。
- 制約が取れたときだけ `retrieve(..., attribute_filter=...)` に渡す。取れなければ
  引数自体を渡さず、**制約が無い質問の挙動は従来と完全に同一**に保つ。
- 抽出器は retriever が持つ（search_style の `attribute_filter` 設定で構築される）。
  無効な構成では None を返す。詳細は `retrieve/attribute_filter.py`。
- 抽出器が `extract_with_llm()` を持つ構成（`attribute_filter.llm_extract: true`）では
  そちらを使う。正規表現で取れなかった質問だけ LLM に判定させる経路で、**元の質問に
  対して1回だけ**呼ぶ（サブクエリごとには呼ばない）。

### 1. 分解（`_decompose`）

single/multi の**両方で常に分解する**。single でも「どの論文か」と「その中の
どの表か」は別々の検索語になりうるため。

**件数は `SUBQUERY_COUNT`（4本）に固定で、task_family では分岐しない。** 以前は
single「1〜3個」/ multi「3〜6個」と分けていたが、そのためだけに
`TaskFamilyClassifier.infer()` が**クエリ1件につき LLM を1回**呼んでいた
（本番入力に task_family が無いため）。実測で買えていたのは平均0.58本:

| gold | step0 のサブクエリ数 | 分布 |
|---|---|---|
| single 26件 | 平均 3.08 | 26件中25件が上限の3本 |
| multi 29件 | 平均 3.66 | 29件中17件が下限の3本 |

推定精度も LLM 0.67 / ヒューリスティック 0.673（55件実測）で差が無く、分岐する
根拠が無かった。両者を挟む4本に固定して LLM 呼び出しを1回減らした。

指示文も「1本の論文で足りることも複数にまたがることもある。主役の論文を引き当てる
言い換えと、必要な事実ごとの分解の両方を書け」という1文に統一してある
（片方に決め打ちすると、外したときに他方が丸ごと欠けるため）。

LLM が空を返したら元の質問1本にフォールバックする。

**プロンプトの先頭に `CORPUS_NOTE`（「投げ先はローカル索引であって Web検索エンジン
ではない」）を必ず置く。** これが無いと LLM は `site:arxiv.org` / `filetype:pdf` の
ような Google 検索クエリを書く。実測では `_refine()` の出力の 29〜41% がそれで、
ローカルの BM25/faiss には1件もヒットせず 2〜3周目の検索が空振りしていた
（構成別の内訳は CLAUDE.md）。`_decompose` / `_refine` の両方に置いてある。

**属性制約が取れているときは `_constraint_note()` でサブクエリの先頭に
`[NAACL 2025]` を付けさせる。** 絞り込み自体は attribute_filter が担当するので、
これは検索語としての制約。title_abstract チャンクの本文が実際に
`[ACL 2025] タイトル…` で始まるため BM25 の語として効く。

### 2. 検索と蓄積

各サブクエリの結果を `chunks` dict に貯める。同じ chunk_id が複数のサブクエリで
当たったら**スコアの高い方を残す**（後勝ちにすると、サブクエリ1で最上位だった
チャンクがサブクエリ3の低スコアで上書きされ、論文順位が「最後に投げた
サブクエリ」に引きずられるため）。

### 3. 候補集約（`_candidate_papers`）

チャンクを `paper_id` でグループ化し、論文ごとに最高スコアで並べる。上位
`max_candidates`（既定20）論文まで、各論文 `chunks_per_paper`（既定2）チャンクを
候補にする。**この件数が「LLM が読める上限」になる**点に注意（ステップ2で拾って
いないチャンクは候補に入らない）。

### 4. 読解と判定（`_read_and_judge`）— 心臓部

候補チャンクを**全文**（`snippet_chars`、既定1800文字）LLM に見せ、1回の呼び出しで
4つを同時に返させる。

```json
{"papers": [{"paper_id": "...", "evidence_chunk_ids": ["..."]}],
 "sufficient": true, "missing": ""}
```

| フィールド | 用途 |
|---|---|
| `paper_ids` | 提出する論文。**既定では使わない**（`submit_from: candidates`。選定は読解チーム側の担当） |
| `evidence_chunk_ids` | 提出する evidence の元チャンク |
| `sufficient` | **反復の停止条件**。ここが true で break |
| `missing` | 次に何を検索するか（`_refine` に渡す） |

**捏造ガード**: 候補一覧に無い paper_id、実在しない chunk_id、他論文の chunk_id は
すべて弾く。`_format_paper()` が各候補を「`[paper_id] title (venue year)` + 各
チャンクの chunk_id・type・page/table_id/figure_id/equation_id・本文」の形で
提示するので、LLM は locator まで見て根拠を選べる。

**本数の打ち切りはここではやらない。** `paper_cutoff` で一括して決める（比較実験で
本数の決め方を揃えられるようにするため）。

### 5. 停止判定（`run()` 内）

```python
if verdict is not None and verdict["sufficient"]:   # LLMが「足りた」
    break
if step == self.max_steps - 1:                       # 反復上限
    break
subqueries = self._refine(query, missing, tried)
if not subqueries:                                   # LLMが「これ以上探しても無い」
    break
```

**検索が何本返したかではなく、LLM が根拠として確認できた論文で足りているかで
打ち切る。** これが「本数で止める反復」との決定的な違い。

### 6. 再分解（`_refine`）

`missing`（何が欠けているか）と `tried`（既に投げたサブクエリ）を渡し、
**重複しない**新しいサブクエリを作らせる。これ以上探しても見つからないと LLM が
判断すれば空リストを返し、ループを抜ける。

### 7. 提出物の組み立て（`_build_prediction`）

Prediction に**2つの論文リスト**を乗せる。ここが評価と直結する。

```python
gold_papers       = candidate_papers の順位 + apply_paper_cutoff  # 提出セット
candidate_papers  = to_gold_papers(全チャンク, max=50)            # 検索が拾えた候補
```

- `candidate_papers` は**打ち切り前**の「検索が拾えた候補」。`candidate_recall@k` は
  これで測る。**検索力（拾えたか）と選定力（LLM がどれだけ絞れたか）を分離**する
  ため。indexer / reranker / 属性フィルタの改善はこの指標に出る。
- `gold_papers` は cutoff の後。実提出の paper P/R/F1 はこちら。**既定では選定しない**
  （`submit_from: candidates`）——どれを提出するかは読解チーム側の担当なので、
  候補列の順位をそのまま渡す。verdict が一度も返らなかった場合も同じ経路。
- 各ステップの `trace`（subqueries / attribute_filter / n_chunks / selected /
  sufficient / missing）も残す。後から挙動を追える。

evidence は `evidence_from_result()` で locator 付き `Evidence` に変換する。ただし
cutoff で落ちた論文の evidence は出さない。

## 提出論文の決め方（`submit_from`）

- `candidates`（**既定**）: 候補列の順位そのまま。**選定は読解チーム側の担当**なので
  検索エージェントは順位を渡すところで止める。
- `llm`: 読解 LLM が選んだ `paper_ids` を使う（選定込みで測りたい ablation 用）。

どちらでも `_read_and_judge()` は呼ぶ。1回の LLM 呼び出しが返す3つのうち、選定以外の
2つ（`sufficient` = 反復の停止条件、`evidence_chunk_ids` = 根拠チャンク）は選定とは
別の役割を持っているため。選定を外すと `paper_recall` は上がり `paper_precision` は
下がるが、`candidate_recall` / `evidence_candidate_recall` は候補列を見る指標なので
変わらない。

## 提出本数の決め方（`paper_cutoff`）

`apply_paper_cutoff`（`task_family.py`）が担当。モードは2つ。

- `llm`（**現在の運用**）: LLM が `sufficient` と判断した時点の選定をそのまま出す。
  `max_papers`（既定10）で頭打ち。**task_family を使わない。**
- `task_family`: single なら1本、multi なら複数、と task_family で本数を決める。

**現在 `llm` にしている理由**（CLAUDE.md にも記載）: 本番入力に task_family が無く、
質問から推定しても正解率0.67程度で当てにならない。本数決定の経路から
task_family を外し、LLM の `sufficient` 判定だけに寄せた。

## 主要パラメータ（`agent_style/reading.yaml`）

| param | 既定 | 意味 |
|---|---|---|
| `max_steps` | 3 | 反復の上限 |
| `retrieve_top_k` | 20 | 1サブクエリで受け取る**チャンク**数（論文数ではない） |
| `max_candidates` | 20 | LLM に見せる**論文**数。`candidate_recall@20` と揃える |
| `chunks_per_paper` | 2 | 1論文あたり LLM に見せるチャンク数 |
| `snippet_chars` | 1800 | 1チャンクを何文字まで見せるか |
| `paper_cutoff` | `llm` | 提出本数の決め方 |
| `max_papers` | 10 | 提出本数の上限 |
| `submit_from` | `candidates` | 提出論文をどのランキングから作るか（選定は読解チーム側） |

`max_candidates: 20` は評価指標 `candidate_recall@20` と揃えてある。ここが15だと
16〜20位の論文は検索で拾えても LLM が見られず提出候補に入らないため、指標上の
改善が実スコアに乗らないズレが生じる。

## 検索側との関係

agent は retriever の出力（chunk 単位の `RetrievalResult` 列）を受け取るだけで、
検索側の強化はそのまま `candidate_papers`（＝`candidate_recall`）に効く。
retriever の中で何が起きているかは後半の
**「# 検索パイプライン（`HybridRetriever`）」** に分けて書く。agent 側のコード
（`_decompose` / `_read_and_judge` 等）は無変更のまま、indexer / fuser / reranker /
属性フィルタの差分がすべて retriever の中で完結する。

## 評価時の注意（CLAUDE.md 3.1 も参照）

- **見るのは `candidate_recall` / `evidence_candidate_recall` だけ。** `evaluate.py`
  は提出物側の指標（paper_* / evidence_* / 回答系）を既定で出さない
  （`--metrics all` で足せる）。
- `scripts/run_search.py` は `--production-input` を付けて回す。手元の
  `validation_inputs.jsonl` には task_family が入っているが本番には無い。付けたまま
  評価すると `_decompose()` の分岐が「正解を教えてもらった」状態になる。
- LLM は非決定的（Opus 4.8 は temperature を受け付けない）でクエリは55件しか
  ないので、数ポイントの差はノイズの可能性がある。結論の前に複数回まわす。

---

# 検索パイプライン（`HybridRetriever`）

ReadingAgent の**ステップ2**（`self.retriever.retrieve(subquery, retrieve_top_k, ...)`）が
呼ぶ検索の実体。実装は agent 層ではなく `src/littraceqa/di_pipeline/retrieve/` に
あるが、agent の反復ループの一部としてここに併記する。入力は**サブクエリ1本**、
出力は**chunk 単位の `RetrievalResult` 列**（論文単位に畳むのは agent 側の
`_candidate_papers` / `to_gold_papers` の仕事）。

## agent → retriever → agent の全体像

```
ReadingAgent.run(query)
  step0 _extract_attribute_filter(query.question)  → attribute_filter | None
  step1 _decompose(query)                          → [subquery, ...]
  ┌ for step in range(max_steps):
  │   for subquery in subqueries:
  │     results = HybridRetriever.retrieve(subquery, top_k, attribute_filter)  ← ★ここ
  │       │
  │       │  === retrieve() の中身（retrieve/hybrid.py）===
  │       │  a. attribute_filter が None かつ extractor があれば subquery から抽出
  │       │     （eval_retrieval.py 用のフォールバック。agent は step0 の値を渡す）
  │       │  b. _run_indexers(): 各 indexer を引く（下記）      → runs: list[list]
  │       │  c. fuser.fuse(runs, top_k=fuse_k):  RRF で1本のランキングに融合
  │       │       fuse_k = reranker あり ? (pool_k or top_k*3) : top_k
  │       │  d. reranker あり: reranker.rerank(query, fused, top_k) で並べ替え
  │       │     reranker なし: fused[:top_k]
  │       ▼
  │     results（chunk 単位）を chunks dict に蓄積（同じ chunk_id はスコア高い方を残す）
  │   _candidate_papers() → _read_and_judge() → sufficient 判定 → _refine()
  └
  _build_prediction()
```

retriever が返すのは常に **chunk 単位**。「どの論文か」に畳むのは agent
（`candidate_papers` は `to_gold_papers`、提出は `_read_and_judge` の LLM 選定）。
この分担が「retriever は gold paper の特定、evidence は agent」という
プロジェクト方針（CLAUDE.md）の実装上の現れ。

## b. 各 indexer の実行（`_run_indexers`）

indexer 群は `search_style` の yaml で決まり（例: `bm25s` + `faiss_specter2` +
`faiss_qwen3`）、それぞれ `search(query, k) -> list[RetrievalResult]` を持つ。
**制約の有無で2つのコードパスに分かれる**。

**制約なし**（本番の質問の大多数）:
```
runs = [indexer.search(query, per_index_k) for indexer in indexers]   # 各索引 per_index_k=100 件
```
従来と完全に同一。属性フィルタ機能を足しても、制約が取れない質問は損失ゼロ。

**制約あり**（「Which NAACL 2025 papers ...」など会議名が一意に取れたとき）:
```
fetch_k = _fetch_k(filter)                 # 絞った後に per_index_k 残るよう選択率から逆算
for indexer in indexers:
    raw  = indexer.search(query, fetch_k)  # 多めに取る
    kept = filter_results(raw, filter)     # metadata の venue/year で落とす
    if len(kept) < min_filtered_results:   # 枯れたら fail-open（そのランだけ制約なしに戻す）
        kept = raw
    runs.append(kept[:per_index_k])
```
- **索引は無改修。** `RetrievalResult.metadata` に venue/year が既に載っているので、
  取得後に落とすだけ。どの indexer にも同じように効く。
- `fetch_k = per_index_k / selectivity * fetch_safety`、上限 `max_fetch_k`
  （`_fetch_k`）。コーパスは 2025 が 91.3% なので年だけでは絞らない設計。
- 発火条件は「会議名が一意に取れたとき」だけ。年のみ / `all venues` / 会議名が
  2種類以上 のときは抽出せず制約なしパスを通る（`attribute_filter.py`）。

## c. 融合（`RRFFuser`, `retrieve/rrf.py`）

複数索引のランキングを Reciprocal Rank Fusion で1本に統合する。`rrf(k=60)`、
全索引 weight 1.0（`search_style` 共通）。スコアの絶対値ではなく**順位**で混ぜるので、
BM25（語彙）と埋め込み（意味）のようにスコールが違う索引を素直に合成できる。
`fuse_k`（融合後に残す件数）は reranker の有無で変わる（上の図 c）。

## d. reranker（任意, `retrieve/reranker.py`）

`search_style` に reranker を書いた構成だけ発火。RRF 後の候補を `pool_k` 件プールし、
クエリ×チャンクを cross-encoder 系モデルで採点し直して `top_k` に絞る。
`pool_k` を書かないと候補が増えず reranker の効果が出ない（CLAUDE.md 参照）。
`none` の構成では融合結果の上位 `top_k` をそのまま返す。

## この層の主要パラメータ（`HybridRetriever.__init__`）

| param | 既定 | 意味 |
|---|---|---|
| `per_index_k` | 100 | 1索引が返す chunk 数（融合前） |
| `pool_k` | None | reranker に渡す候補数。None なら `top_k*3` |
| `fetch_safety` | 1.5 | 制約あり時の取得件数の逆算係数 |
| `max_fetch_k` | 5000 | 制約あり時の1索引あたり取得上限 |
| `min_filtered_results` | 10 | 絞り込み後これ未満なら fail-open |
| `rerank_blend` | None | 融合前の順位と reranker の順位を RRF で混ぜる設定。None なら reranker が順位を置き換える（従来） |

`per_index_k` / `pool_k` / 属性フィルタ関連はすべて `search_style` の yaml から
`compose_config()` → `build_pipeline()` 経由で注入される。agent 側の `top_k`
（1サブクエリで受け取る chunk 数）とは別物なので混同しないこと。
