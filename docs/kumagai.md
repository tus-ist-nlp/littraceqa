## 1. 現在の検索アルゴリズム

```
質問
  ↓
会議名・年・手法名を抽出
  ↓
Chunk BM25 ─────┐
                 ├─ 論文単位RRF
Paper BM25 ─────┘
  ↓
1位論文を使ったSeed Expansion
  ↓
初回検索と拡張検索を論文単位RRF
  ↓
限定的な候補補充
  ↓
50論文をQwen3-Reranker-4Bで再順位付け
  ↓
Top50
  ↓
F1向けPaper Selector
  ↓
任意: MinerU Evidence Coverage
  ↓
提出する少数のpaper_id
```

設定の正本は seed_expansion_structured_filter.yaml です。

### 1.1 Chunk BM25

MinerUから作った最大約2,000文字のChunkを検索します。Chunkには次のprefixが付いています。

```
[venue year] paper title
chunk body
```

本文、表、図キャプション、数式などを個別に検索でき、最終的な代表Evidenceを残しやすい検索です。

設定は次のとおりです。

```
method       = lucene
idf_method   = lucene
k1           = 1.5
b            = 0.75
delta        = 0.5
top_k        = 100
```

実装は bm25_index.py (line 76) です。

### 1.2 Paper BM25

同じ論文の全Chunkを1つの文書にまとめて検索します。

Chunk BM25が「質問に一致する箇所」を探すのに対し、Paper BM25は「論文全体として質問に近いか」を調べます。

現在は参考文献も含めます。

```
exclude_references = false
top_k              = 100
```

さらに、本文から「この論文自身が提案している手法名」を抽出し、`method_alias_graph.json`へ保存します。

実装は以下です。

- paper_bm25.py (line 154)
- method_matching.py
- method_sidecar.py

### 1.3 論文単位RRF

Chunk BM25とPaper BM25の結果は、scoreの値を直接混ぜません。両者のscore尺度が異なるため、順位のみを使います。

論文 \(p\) の融合scoreは次です。

```
score(p) =
    1.0 / (60 + chunk_bm25_paper_rank(p))
  + 1.0 / (60 + paper_bm25_rank(p))
```

同じ論文にChunk hitが何件あっても、1つの検索run内では1票です。このため、Chunk数が多い長い論文だけが有利になるのを防げます。

代表結果にはChunk BM25のEvidence候補を使い、Paper BM25の本文はSeed Expansion用metadataに保持します。

実装は paper_rank_rrf.py (line 9) です。

### 1.4 属性のsoft boost

質問に会議名・年が明示されている場合、BM25後にsoft boostします。

```
明確な一致   = +1
明確な不一致 = -1
metadata欠損 =  0
```

元順位を0〜1へ正規化し、次のscoreで再整列します。

```
attribute_score =
    normalized_base_rank
  + 0.75 * attribute_signal
```

欠損論文は除外しません。これはRecallを守るためです。

実装は attributes.py (line 350) です。

### 1.5 Seed Expansion

初回検索の1位論文から、次を取得します。

```
論文タイトル
Paper BM25文書の先頭512文字
```

これを元質問へ追加して再検索します。

```
expanded_query =
    original_question
  + top1_paper_title
  + top1_paper_head_512_chars
```

初回順位と拡張順位を再びRRFします。

```
score(p) =
    1 / (60 + initial_rank(p))
  + 1 / (60 + expanded_rank(p))
```

これはLLMによるクエリ分解ではなく、上位論文からコーパス内の語彙を借りるpseudo relevance feedbackです。

中心実装は retriever.py (line 100) と query.py です。

### 1.6 候補補充

候補補充は強くしすぎず、発火条件と追加件数を限定しています。

#### 一意な手法名・タイトル

質問が `AD-GS` のような識別的な手法名を直接書いていて、それがコーパス内で1論文だけを指す場合、その論文が検索候補から漏れていれば追加します。

```
最大追加 = 5本
追加位置 = rerank poolの後方
上位への直接昇格はしない
```

exact_match.py

#### 会議名・年・modalityによる構造化検索

次のようなopen-set質問だけで動きます。

```
Which CVPR 2025 papers ...
in their main comparison tables?
```

会議名、年、表・図・数式の種類がすべて明示された場合、Chunk BM25を最大5,000件まで調べ、条件に一致する論文を最大20件抽出します。

元の上位5本は保護し、6位以降だけを組み直します。

structured_filter.py

#### SPECTER2近傍

これは通常の「質問Embedding→論文検索」ではありません。

```
質問中の手法名
  ↓
その手法を所有する論文を特定
  ↓
その論文のSPECTER2近傍を取得
  ↓
関連論文候補としてtailへ補充
```

設定は次です。

```
owner seed       = 上位1本
SPECTER2近傍     = 7本
新規候補追加上限 = 3本
```

検索時にはEmbeddingモデルを動かさず、構築済みの27,487×768ベクトルをmmapで読みます。

- paper_embedding.py
- dense_tail.py

#### open-set consensus

open-set質問だけ、初回上位5論文をSeedとして追加検索します。

候補論文が次を満たした場合だけ採用します。

```
2つ以上のSeed検索で出現
かつ
どれかの検索で2位以内
```

該当する1本だけを最終20位へ入れます。1〜19位は維持されます。

### 1.7 Qwen3 Reranker

確定した50論文について、Paper BM25文書の先頭2,000文字をQwenへ渡します。

```
model       = Qwen/Qwen3-Reranker-4B
dtype       = bfloat16
batch       = 4
max_tokens  = 1024
```

Qwen単独順位ではなく、元順位と融合します。

```
score(p) =
    0.52 / (60 + pre_qwen_rank(p))
  + 0.48 / (60 + qwen_rank(p))
```

元のTop20の「集合」は保護します。QwenはTop20内部を並べ替えられますが、21位以下をTop20へ入れられません。候補集合自体も増減しません。

- qwen3_reranker.py (line 280)
- final_rerank.py

## 2. F1向け最終論文選択

検索Top50と、実際に提出するpaper_id集合は分けています。

```
Top50検索
  ↓
質問文から必要本数を推定
  ↓
手法ownerで限定的に順位調整
  ↓
必要本数だけ選択
  ↓
任意: MinerU Evidence Coverage
```

### Paper Selector

設定は f1_method_owner.yaml です。

```
通常質問       = 1本
open-set       = 1本
明示された本数 = その本数
最大           = 10本
```

質問中の手法ownerがTop50内にあれば優先しますが、候補外論文は追加しません。

- selector.py
- cardinality.py
- owner_aware.py (line 131)

### Evidence Coverage

MinerUのTop20論文だけを遅延読込し、非常に強い条件が成立した場合に限って選択集合を補正します。

現在の4ルールは次です。

1. 明示された論文・Table番号・複数式が同一表にある
2. 質問の全dataset行と値列を1つの表が被覆する
3. 2つの回答slotを、Evidenceを持つ異なる2論文が被覆する
4. 引用と主比較表の両方を満たすopen-set論文を列挙する

上から順に試し、最初に選択を変更した1ルールだけを採用します。曖昧・欠損・複数解なら元の選択を維持します。

- evidence_coverage.py (line 41)
- paper_tables.py
- `select/table_rules.py`
- `select/two_slot_question.py`
- `select/multi_paper_coverage.py`
- `select/citation_table_coverage.py`

これは検索コアではなく、最終選択のopt-in補正です。

## 3. クエリ分解エージェントとの推奨構成

一番重要なのは、クエリ分解を現在の検索器の内側へ混ぜ込まないことです。

推奨構成は次です。

```
Original Query
  ↓
Query Planner
  ├─ anchor: 元質問
  ├─ slot-1 subquery
  ├─ slot-2 subquery
  └─ slot-N subquery
  ↓
各laneでCandidate Retrieval（Qwenなし）
  ↓
paper_id単位のMulti-query RRF
  ↓
候補補充
  ↓
元質問でQwenを1回だけ実行
  ↓
Top50
  ↓
Evidence Reader / 再検索判定
  ↓
F1 Paper Selector
  ↓
Evidence Coverage
```

### 推奨する新しいデータ型

```
@dataclass(frozen=True)
class SubquerySpec:
    id: str
    text: str
    facet: str
    hints: SearchHints
    required: bool = True

@dataclass(frozen=True)
class SearchPlan:
    original_question: str
    subqueries: tuple[SubquerySpec, ...]

@dataclass(frozen=True)
class PaperRun:
    lane_id: str
    query: str
    weight: float
    results: tuple[RetrievalResult, ...]
```

agentはIndexerやQwenを直接呼ばず、`SearchPlan`を作ることだけを担当させます。

### Multi-query融合

異なるサブクエリのscoreは直接比較できません。必ず順位で融合します。

初期値としては、元質問全体に重み1.0、サブクエリ群全体に重み1.0が安全です。

```
N個のsubqueryがある場合

score(p) =
    1.0 / (60 + anchor_rank(p))
  + Σ [ (1/N) / (60 + subquery_rank_i(p)) ]
```

これなら、サブクエリを多く生成した質問だけ総scoreが大きくなるのを防げます。

metadataには次を残してください。

```
{
  "query_hits": [
    {
      "lane_id": "anchor",
      "rank": 3,
      "chunk_id": "...",
      "source": "paper_rank_rrf"
    },
    {
      "lane_id": "slot-2",
      "rank": 1,
      "chunk_id": "...",
      "source": "paper_rank_rrf"
    }
  ],
  "support_count": 2,
  "best_rank": 1,
  "anchor_rank": 3
}
```

### Seed Expansionとの関係

Seed Expansionとquery decompositionは別物なので併用できます。

ただし、すべてのサブクエリへ現在のfull retrieverを適用すると、各サブクエリでSeed ExpansionとQwenが動きます。

例えば4サブクエリなら、最大200ペアをQwenで評価します。

推奨は次です。

```
anchor lane:
    Seed Expansionあり

subquery lanes:
    Chunk BM25 + Paper BM25のみ
    Seed Expansionなし

全lane融合後:
    Qwenを元質問で1回だけ
```

### 追加する薄い実装

移植先では以下の2ファイル程度に抑えるのがよいです。

```
src/littraceqa/di_pipeline/agent/search_plan.py
src/littraceqa/di_pipeline/retrieve/multi_query.py
```

`search_plan.py`:

- LLMの英語プロンプト
- 原質問を独立したEvidence slotへ分解
- venue/year/method hintsを構造化
- task_family、primary_evidence_typeは使わない
- 最大3〜4サブクエリ
- 重複クエリ除去

`multi_query.py`:

- anchorとsubqueryの実行
- paper_id単位RRF
- provenance保持
- 候補重複除去
- Qwenの1回実行
- 検索budget管理

### 既存コードに必要な小変更

現在の `SeedExpansionRetriever` は候補生成と最終Qwenが1つの `retrieve()`にまとまっています。

後方互換を維持しながら、次を公開すると接続しやすくなります。

```
retrieve_candidates(
    query,
    top_k,
    *,
    hints=None,
    seed_expand=True,
) -> list[RetrievalResult]

rerank_candidates(
    original_query,
    candidates,
    top_k,
) -> list[RetrievalResult]
```

既存の `retrieve()`は内部でこの2つを順に呼ぶfacadeとして残します。

```
def retrieve(...):
    candidates = self.retrieve_candidates(...)
    return self.rerank_candidates(query, candidates, top_k)
```

これにより、単一クエリの既存挙動を維持したまま、multi-query時だけQwenを1回にできます。

## 4. 現在のReadingAgentをそのまま使わない理由

reading.py (line 44) はすでにクエリ分解を持ちますが、移植先では次を変更すべきです。

- サブクエリ間でraw scoreの最大値を比較している
- full retrieverをサブクエリごとに呼ぶ
- 元質問のanchor検索が必ず残る保証がない
- lane別provenanceが失われる
- LLM verdict後のpaper_idだけをSelectorへ渡す
- Evidence Coverageが接続されていない
- prompt/commentが日本語

したがって、検索Planner、Multi-query Retriever、Evidence Reader、Stop Policyを別クラスに分ける方が分かりやすくなります。

### 停止条件

最大2〜3roundとし、次のいずれかで終了します。

- 全必須facetにEvidenceがある
- Top20 paper_id集合が前roundと同一
- 新しいpaper_idが0本
- 新しいsubqueryが生成されない
- 最大roundまたは検索budgetへ到達

open-setでは必要本数を知らないので、「N本見つけた」だけでは終了しません。

## 5. 移植するソースコード

### 候補Top50検索の必須runtime

```
src/littraceqa/di_pipeline/
  contracts.py
  config.py
  registry.py

  index/
    bm25_index.py
    chunk_store.py
    resumable_bm25.py
    bm25_checkpoint_format.py
    bm25_checkpoint_layout.py
    bm25_completion_check.py
    paper_bm25.py
    method_matching.py
    method_sidecar.py
    paper_embedding.py

  retrieve/
    base.py
    hybrid.py
    attributes.py
    method_aliases.py
    paper_rank_rrf.py
    qwen3_reranker.py
    seed_expansion/   # 現在存在するファイル一式
```

構築済みBM25は`CURRENT.json`とoffset mmapを使うため、`origin/main`の旧BM25ローダーでは読めません。checkpoint関連も一緒に移してください。

対象branch側に同等のRegistryやConfigがあるなら、`config.py`を丸ごと置き換えず、次の登録だけ追加します。

```
indexer: paper_bm25
fuser: paper_rank_rrf
reranker: qwen3
retriever_wrapper: seed_expansion
```

### F1選択

```
configs/select_style/f1_method_owner.yaml

src/littraceqa/di_pipeline/select/
  __init__.py
  selector.py
  cardinality.py
  owner_aware.py
```

追加依存:

```
retrieve/seed_expansion/question_entities.py
retrieve/method_aliases.py
index/method_sidecar.py
method_alias_graph.json
```

### Evidence Coverageを使う場合

```
retrieve/paper_tables.py

select/
  evidence_coverage.py
  table_rules.py
  two_slot_question.py
  multi_paper_coverage.py
  citation_table_coverage.py

evaluation/
  evidence_coverage_input.py
  selection_input.py
```

提出JSONLまで作るなら build_submission.py を使うか、同じ呼び出しを対象branchのwriterへ組み込みます。

### 評価専用

```
scripts/eval_retrieval.py
scripts/eval_paper_selection.py
scripts/build_candidate_lists.py
scripts/report_paper_selection_errors.py
```

これらは本番推論の必須runtimeではありません。

## 6. 統合の推奨順序

1. 対象branchへ単一クエリRetrieverだけ移植
2. 同じ構築済み索引を読み、現在のTop50と完全一致させる
3. `retrieve_candidates`と`rerank_candidates`へ後方互換分割
4. `SearchPlan`と`MultiQueryCoordinator`を追加
5. original-only planで従来Top50と完全一致することを確認
6. クエリ分解ON/OFFを比較
7. F1 Selectorを元質問＋Top50へ接続
8. Evidence Coverageは最後にopt-inで接続
9. 読解エージェントと停止条件を追加

最初から検索、Agent、Evidence Coverageを同時に統合すると、精度変化の原因が分からなくなるため、段階ごとの評価が重要です。

## 7. 回帰試験

最低限、次を固定してください。

- original-only planのTop50が現在と完全一致
- 同じpaperが3サブクエリに出ても最終1件
- 3run分のprovenanceが残る
- サブクエリ順や重複で結果が変わらない
- Qwen呼び出しが1質問1roundにつき1回
- 候補集合50本をQwenが増減させない
- `task_family`と`primary_evidence_type`を削除しても同一
- Selectorには元質問とTop50を渡す
- Evidence CoverageはTop20外を追加しない

現在のAnswer-bearing validationの回帰基準は次です。

```
候補検索:
Recall@10 = 0.9848
Recall@20 = 1.0000
Recall@50 = 1.0000
All-Gold@20 = 1.0000

F1選択:
Precision = 0.9909
Recall    = 0.9515
F1        = 0.9600
```

クエリ分解ON/OFFは、single/multi別に次を比較してください。

```
Recall@5/10/20/50
All-Gold@20/50
Paper Precision/Recall/F1
全facetを回収した質問数
検索回数
Qwen推論時間
最大GPUメモリ
```

## 8. 移植前に判断・修正が必要な点

- 4B configにはモデルrevisionがありません。現在のcacheは`22e683669bc0f0bd69640a1354a6d0aebcfeede5` なので、移植時にpinするのが安全です。
- `cuda:0`は設定へ固定せず、環境変数やCLIで選べる方が安全です。
- `SearchHints`を1項目でも渡すと質問文からの自動抽出がすべて止まります。Planner hintとliteral hintを明示mergeしてください。
- structured filterは文字列から`Which papers`、会議、年、表・図を読むため、サブクエリ生成時にこれらを消さないでください。
- 同一サーバーでもSPECTER2ディレクトリは現在熊谷さん以外のOSユーザーから読めません。同じユーザーの別branchなら利用できます。
- Evidence Coverageはvalidationで効果がありますが、test/test_extraでは発火0件でした。検索コアとは分け、experimentalなopt-inとして扱うのが安全です。
- retrieve/README.md は削除済みの旧モジュール名を一部含むため、移植ファイル一覧には上記の実ファイルを使用してください。

この分解なら、検索エージェントは「何を検索するか」に集中し、現在の検索器は「候補を高Recallで集めて順位付けする」、Selectorは「最終的に何本提出するか」を担当できます。

### 今の評価精度の寄与度合い

はい。保存済みのアブレーション結果を同じ条件で再集計すると、主要部分の寄与はかなり切り分けられます。

主評価は、MinerU全27,487論文・validation 55問・Answer-bearing gold 87本です。以下の差は相対%ではなく「ポイント差」です。

### 候補検索の段階別結果

| 構成 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | All-Gold@20 |
| --- | --- | --- | --- | --- | --- |
| Chunk BM25のみ | 0.6566 | 0.8379 | 0.9242 | 0.9394 | 0.8727 |
| + Paper BM25・Paper RRF | 0.7500 | 0.9222 | 0.9394 | 0.9505 | 0.8909 |
| + Seed Expansion | 0.7520 | 0.8934 | 0.9586 | 0.9798 | 0.9273 |
| 現行候補・Qwen前 | 0.7520 | 0.9116 | 0.9682 | 0.9955 | 0.9818 |
| 現行4B・全処理後 | **0.7808** | **0.9323** | **0.9848** | **1.0000** | **1.0000** |

BM25のみから現行構成までで、Recall@10は`+6.06`ポイント、Recall@20も`+6.06`ポイント、All-Gold@20は`+12.73`ポイント改善しています。

### 各部分の貢献

#### 1. Paper BM25＋Paper単位RRF

最も大きな基礎改善です。

- Recall@1: `+9.34`ポイント
- Recall@5: `+8.43`
- Recall@10: `+1.52`
- Recall@20: `+1.11`

Chunk検索だけだと同じ論文の複数Chunkが候補を占有します。論文単位BM25と論文単位RRFによって、上位の論文識別が大きく安定しています。

#### 2. Seed Expansion

上位論文のタイトル・本文を使って再検索する部分です。

- Recall@5: `−2.88`
- Recall@10: `+1.92`
- Recall@20: `+2.93`
- All-Gold@20: `+3.64`

浅い順位は少し不安定になりますが、複数goldを20位までに集める用途には有効です。特にmulti問題のRecall@20は約`+13.4`ポイント改善しています。

#### 3. Qwen3-4B Reranker

現行候補集合に対するQwen前後の比較です。

- Recall@1: `+2.88`
- Recall@5: `+2.07`
- Recall@10: `+1.67`

multi問題だけでは、

- Recall@1: `+13.19`
- Recall@5: `+9.49`
- Recall@10: `+7.64`

と大きく効いています。

ただし、Qwen順位だけを使うとRecall@1は`0.7045`まで低下します。現在の「BM25系の元順位52%＋Qwen順位48%」の融合が重要です。

モデル比較では4Bが最良でした。

| モデル | Recall@1 | Recall@10 | 秒/問 |
| --- | --- | --- | --- |
| 0.6B | 0.7202 | 0.9808 | 2.24 |
| 4B | **0.7808** | **0.9848** | 7.15 |
| 8B | 0.7581 | 0.9763 | 9.22 |

8Bへ増やしても改善せず、現在は4Bが妥当です。

#### 4. Structured Filter

会議・年・表／図などが明示されたopen-set質問に限定した補充です。

- Recall@10: `+1.11`
- Recall@20: `+0.66`
- All-Gold@20: `+3.64`
- Recall@5: `−0.20`

主にq020・q023の複数論文回収へ効きました。無条件で使うとノイズになるため、現在の限定的な発火条件は残した方がよいです。

#### 5. Open-set slot

複数seedから支持された候補を20位へ1本だけ追加します。

- Recall`@20/50`: `+0.45`
- All-Gold`@20/50`: `+1.82`
- 悪化した質問: 0

実際にはq025のScaleKVを候補外から20位へ入れた効果です。

#### 6. Exact Method Search

validationの集計値は全cutoffで差がありません。q030のgoldを拾っていますが、Method Dense Tailも同じ論文を既に回収していたためです。

一方、予想testではAD-GSを候補外から4位へ救った例があります。validationでは冗長ですが、低コストな保険として働いています。

#### 7. Method Dense Tail（SPECTER2）

現行出力では2,750候補中115件を生成していますが、最終的にこのsourceを持つAnswer-bearing goldは0件です。以前はq030を候補外から16位へ救いましたが、現在はExact Method Searchと重複しています。

ここは「Exactを有効にしたままDense TailだけOFF」にする比較がまだなく、削除候補ではあるものの、現時点では寄与を数値で断定できません。

### F1向け最終選択の寄与

同じTop50から提出論文を選ぶ処理も、きれいに分離できます。

| 選択処理 | Precision | Recall | F1 | 前段からのF1差 |
| --- | --- | --- | --- | --- |
| 常に検索1位を提出 | 0.9273 | 0.7808 | 0.8139 | — |
| 質問から論文本数を推定 | 0.8970 | 0.8490 | 0.8421 | +0.0282 |
| + Method Owner | 0.9591 | 0.8899 | 0.8982 | +0.0561 |
| + Evidence Coverage | **0.9909** | **0.9515** | **0.9600** | +0.0618 |

固定Top1から最終構成までで、F1は`+14.61`ポイントです。

Evidence Coverageの内訳は次のとおりです。

| ルール | 対象 | F1差 |
| --- | --- | --- |
| Table番号・式の一致 | q033 | +0.0182 |
| 単一表による全項目被覆 | q052 | +0.0109 |
| 2論文による回答slot被覆 | q042・q043・q045 | +0.0182 |
| 引用＋主比較表による列挙 | q023 | +0.0145 |

ただしEvidence Coverageはtest/test_extraでは発火0件です。validationでは6問すべて改善していますが、汎化性能が確認できたとはまだ言えません。

### 現時点の判断

精度への貢献が明確なのは次です。

1. Paper BM25＋Paper RRF
2. Seed Expansionによる深いRecall改善
3. Qwen3-4Bと元順位の融合
4. Structured Filterによるopen-set補充
5. Method OwnerとEvidence CoverageによるF1改善

寄与がまだ不明確なのは次です。

- Method Dense Tail
- 属性boost単体
- タイトル保護単体
- Exact Method Searchのvalidation上の純増分

次に実行すべき最小アブレーションは、現行4B設定を固定して、`Method Dense Tail`、`Exact Method Search`、`Structured Filter`、`Open-set slot`を1つずつOFFにする比較です。特にDense Tailは、現在も必要か判断するための優先度が高いです。

なお、以上はAnswer-bearing goldで最適化した結果です。同じ現行順位でもRecall@50はEvidence-backedで`0.9545`、公式goldで`0.9091`になるため、報告時にはgold定義を必ず併記する必要があります。

13:27

一回それでやってみようか

13:39

48m 49s間作業しました

実行しました。結論として、Method Dense Tailは外す方向がかなり有力です。Exact・Structured・Open-setは残す価値があります。

評価条件は現在のHEAD `d014eab`、全27,487論文、Answer-bearing validation 55問です。GPUが全4枚とも他ユーザーのVLLM処理で100%使用中だったため、まずQwen前の候補生成を厳密に比較しました。

### validationのleave-one-out

| 構成 | Recall@10 | Recall@20 | Recall@50 | All-Gold@20 |
| --- | --- | --- | --- | --- |
| 全機能ON | 0.9682 | **1.0000** | **1.0000** | **1.0000** |
| Dense Tail OFF | 0.9682 | **1.0000** | **1.0000** | **1.0000** |
| Exact Method OFF | 0.9636 | 1.0000 | 1.0000 | 1.0000 |
| Structured Filter OFF | 0.9631 | 0.9934 | 1.0000 | 0.9636 |
| Open-set slot OFF | 0.9682 | 0.9955 | 0.9955 | 0.9818 |

Recall`@20/50`は、QwenのTop20集合保護により最終4Bでも変わらない確定値です。Recall@10以下はQwen前の値です。

各機能の実際の効果は次のとおりです。

- Method Dense Tail
    - 全cutoffで改善0
    - 23問の候補集合を変えるが、Answer-bearing goldの回収には貢献なし
    - q029のgoldは、Denseを外した方が17位→14位へ上昇
- Exact Method Search
    - q030のgoldを16位→10位
    - Recall@10 `+0.45`ポイント
    - All-Gold@10 `+1.82`ポイント
- Structured Filter
    - q020・q023を改善
    - Recall@20 `+0.66`ポイント
    - All-Gold@20 `+3.64`ポイント
- Open-set slot
    - q025のScaleKVを候補外→20位へ追加
    - Recall`@20/50` `+0.45`ポイント
    - All-Gold`@20/50` `+1.82`ポイント

### 予想test 71問

非公式pseudo-goldでの候補生成結果です。

| 構成 | Recall@10 | Recall@20 | Recall@50 | All-Gold@20 |
| --- | --- | --- | --- | --- |
| 全機能ON | 0.8991 | 0.9460 | 0.9953 | 0.8873 |
| Dense Tail OFF | 0.8991 | **0.9531** | 0.9953 | **0.9014** |
| Exact Method OFF | 0.8920 | 0.9390 | 0.9883 | 0.8732 |
| Structured Filter OFF | 0.8991 | 0.9460 | 0.9953 | 0.8873 |
| Open-set slot OFF | 0.8991 | 0.9460 | 0.9953 | 0.8873 |

Dense Tailを外すと、予想testではむしろ、

- Recall@20: `+0.70`ポイント
- All-Gold@20: `+1.41`ポイント

となりました。1問でgoldが24位→20位へ上がっています。

Exactを外すと、AD-GSが10位からTop50圏外へ消えました。そのためExactは維持すべきです。

StructuredとOpen-setは厳格な発火条件によりtestでは0件発火し、出力を壊していません。

### 処理時間

Qwenを除いた候補生成のwarm-cache比較です。

| 対象 | 全機能ON | Dense OFF | 短縮 |
| --- | --- | --- | --- |
| validation 55問 | 87.5秒 | 30.9秒 | 約65% |
| test 71問 | 61.4秒 | 33.3秒 | 約46% |

Dense Tailを外すとSPECTER2近傍探索も不要になり、別ブランチへの移植も簡単になります。

### Qwen3-4Bについて

既存GPUスコアを使った部分再生では、Dense OFFの感度値は次でした。

- Recall@1: `0.7808 → 0.7808`
- Recall@5: `0.9323 → 0.9505`
- Recall@10: `0.9848 → 0.9848`

ただし未採点のtail候補が74件あるため、これは厳密なGPUアブレーションではありません。正式に確認するにはGPUが空いた時に「基準」と「Dense OFF」の2条件だけを再実行すれば十分です。

### 判断

現時点の推奨は次です。

- Method Dense Tail: 削除候補
- Exact Method Search: 維持
- Structured Filter: 維持
- Open-set slot: 維持

canonical設定はまだ変更していません。最終F1 `0.9600`への回帰がないことをGPUで確認してからDenseを無効化するのが安全です。

全結果はablations_20260811に保存しています。全10条件で失敗0件、リポジトリの追跡対象ファイルは変更していません。