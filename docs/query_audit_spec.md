# クエリ品質監査システム 仕様書

## 1. 背景

現行の検索システムを改善するにあたり、評価データセット側に以下の問題が確認されている。

**1. gold_paper に回答と無関係な論文が含まれる**`task_family = multi_paper` の問題で顕著。問題文のテーマは共有しているが、回答の根拠にはならない論文が gold として登録されている。むしろ誤答の根拠として機能しているケースがある。

**2. gold_paper 数と evidence 数が対応していない**
gold_paper が複数あるにもかかわらず、そのうち 1 つの論文由来の evidence_id しか付与されていない問題が存在する。残りの gold_paper は、回答生成に寄与しているかを検証する手段がない。

**3. 検索失敗とデータセットのノイズが切り分けられない**
上記の結果、`paper_recall` が低いとき、それが retriever の性能不足なのか、そもそも取る必要のない論文を分母に含めているせいなのかを判別できない。改善の方向を決められない状態にある。

## 2. 目的

クエリ単位・gold_paper 単位でデータセットの品質を判定し、以下の 2 つを可能にする。

- **評価指標の補正** — 回答に不要な gold_paper を分母から除いた recall を算出し、retriever の実性能を測る
- **失敗分析** — 検索できなかった gold_paper について、取るべきだったのか否かを即座に判断する

## 3. 判定モデル

### 3.1 判定の最小単位

判定は `(query_id, paper_id)` のペア単位で行う。クエリ単位のラベルはここから導出する。

理由は 2 つ。第一に、「悪問」の中でもノイズが 1 件だけの問題と 3 件混ざっている問題は改善上の意味が異なり、クエリラベルだけでは区別できない。第二に、recall の補正には論文単位の可否判定が必要であり、クエリラベルからは復元できない。

### 3.2 gold_paper 単位の判定項目

各 gold_paper に対し、入力（question / answer / gold_paper 本文 / 当該論文由来の evidence 全文）から次を判定する。

| 項目 | 型 | 内容 |
| --- | --- | --- |
| `relevance` | enum | evidence が回答の根拠としてどう機能するか（下表） |
| `evidence_role` | text | 回答のどの主張をどう支えるか。supporting 以外なら「どうずれているか」 |
| `noise_type` | enum | null | 回答に不要な場合、なぜ混入したかの推定（下表）。supporting / partial では null |
| `relation_to_gold` | text | supporting な gold_paper との具体的な関係 |
| `confidence` | enum | high / medium / low |

**`relevance` の値**

| 値 | 定義 |
| --- | --- |
| `supporting` | 回答の記述を直接支持する。この evidence を除くと回答が成立しない |
| `partial` | 文脈や前提を与えるが、回答の主張そのものは含まない |
| `irrelevant` | 回答と論理的なつながりがない。テーマが同じなだけ |
| `contradicting` | 回答と矛盾する。誤答の根拠になりうる |
| `no_evidence` | この gold_paper 由来の evidence_id が存在しない |

**`noise_type` の値**

| 値 | 定義 |
| --- | --- |
| `same_topic_different_finding` | 研究テーマは同じだが、問われている知見を含まない |
| `same_method_different_task` | 手法・モデルが共通。タスクや対象データが異なる |
| `citation_neighbor` | 正解論文の引用元 / 引用先。文献グラフ上の近傍 |
| `shared_author_or_venue` | 著者・会議が共通。内容的な必然性は薄い |
| `distractor_by_design` | 誤答を誘発する設計に見える。表層的に答えに近い記述を含む |
| `annotation_error` | 関連が説明できない。アノテーションミスと考えられる |

自由記述は `evidence_role` と `relation_to_gold` に限定し、集計対象の軸は必ず enum で持つ。自由記述だけでは分布が出せず、「このデータセットのノイズは主に X 型」という主張ができない。

### 3.3 クエリラベルの導出規則

gold_paper 単位の判定から、機械的に決定する。

| ラベル | 条件 |
| --- | --- |
| **良問** (`good`) | 全 gold_paper の `relevance` が `supporting` |
| **やや良問** (`fair`) | 全 gold_paper が evidence を持つが、`supporting` 以外を含む |
| **悪問** (`noisy`) | `relevance = no_evidence` の gold_paper を 1 件以上含む |

`contradicting` を含むクエリは、ラベルとは独立にフラグを立てて抽出できるようにする。誤答の根拠が gold に混入している事例は、それ自体が報告価値を持つため。

## 4. 出力仕様

### 4.1 一次成果物

`(query_id, paper_id)` 単位の JSONL。判定結果・データセットの事実・自システムの検索結果を join した 1 レコードとする。

```json
{
  "query_id": "...",
  "paper_id": "...",
  "task_family": "multi_paper",
  "relevance": "irrelevant",
  "evidence_role": "...",
  "noise_type": "same_topic_different_finding",
  "relation_to_gold": "...",
  "confidence": "high",
  "evidence_ids": [],
  "retrieval": { "rank": null, "score": null },
  "judge_model": "...",
  "judged_at": "..."
}
```

クエリラベル・集計値・レポートはすべてこの JSONL から再生成できる派生物として扱う。

### 4.2 補正指標

以下を並記する。両者の差そのものが、データセットのノイズ量を示す指標となる。

- `paper_recall_macro` — 従来定義。全 gold_paper を分母とする
- `paper_recall_macro_clean` — `relevance ∈ {supporting, partial}` の gold_paper のみを分母とする

分母が 0 になるクエリはマクロ平均から除外する。

### 4.3 レポート

ラベルごとに以下を記述させる。いずれも 3.2 の判定結果を根拠として参照し、判定と矛盾しないこと。

**良問**
どの evidence が回答のどの主張を支えているかを、evidence → 主張の対応として明示する。ここで対応が書けないクエリは良問判定が誤っている可能性が高く、再判定の対象とする。

**やや良問**`supporting` でない evidence について、回答の根拠としてどの程度ずれているかを記述する。「回答の前提にあたる背景情報」「同じ手法を別条件で適用した結果」など、ずれの方向を具体化する。

**悪問**`no_evidence` の gold_paper について次を記述する。

- その論文に何が書かれているか
- 回答に必要な gold_paper との関係（`noise_type` の判定根拠を、本文の記述に即して説明する）
- 誤答の根拠になりうるか。なりうる場合、どの記述がどう誤読を誘うか

### 4.4 閲覧形式

単一 HTML ファイル（外部依存なし）として生成する。1 クエリ 1 行を基本とし、展開すると evidence 本文・gold_paper 該当箇所・レポートを並べて表示する。

備えるフィルタ:

- クエリラベル
- `task_family`
- `noise_type`
- 検索結果の有無（未検出の gold_paper を持つクエリのみ表示）

HTML は JSONL のビューアであり、判定ロジックを含まない。判定の誤りと描画の誤りを切り分け可能にするため、生成処理は判定処理から分離する。

## 5. 想定される利用フロー

1. 全クエリを判定し、JSONL を生成する
2. `paper_recall_macro` と `paper_recall_macro_clean` を比較し、現行スコアの何割がデータセット由来の損失かを確定する
3. HTML レポートで「未検出かつ `supporting`」の gold_paper を絞り込み、retriever の実際の失敗事例のみを分析する
4. `noise_type` の分布を集計し、データセットの性質として報告する

## 6. 検討事項

- **判定の信頼性検証** — 一定数のクエリに人手ラベルを付け、LLM 判定との一致率を測る。特に `partial` / `irrelevant` の境界は揺れやすい
- **判定コスト** — gold_paper 本文の全文投入は現実的でない。evidence 周辺と abstract に絞る方針を要検討
- **`no_evidence` の扱い** — 単に evidence アノテーションが漏れているだけの可能性がある。本文を確認し、回答を支持する記述が存在するかまで判定するか、evidence の有無のみで切るかを決める必要がある