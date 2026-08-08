# 検索手法の否定的結果メモ

採用に至らなかった手法と、その判断根拠の記録。論文の考察・ablation を書くとき、
および「これは試したのか」と問われたときの一次資料として残す。

対象は LitTraceQA の gold paper 検索タスク。評価は validation 55問、gold は
`data/validation_answer_bearing_gold_draft.jsonl`（answer-bearing 87本）。
コーパスは 27,487論文、前処理は MinerU。

最終構成は `configs/search_style/seed_expansion_structured_filter.yaml`
（dual BM25 → PaperRank RRF → seed expansion → 候補補充4レーン →
Qwen3-Reranker-4B で50件採点 → Top 50）。validation で R@1 0.7808 /
R@10 0.9848 / R@20 1.0000 / All-Gold@20 1.0000。

---

## 1. 実装済みだが最終構成で無効だったレーン

いずれも重みゼロ・seed数ゼロで素通りする状態だった。採用構成では一切実行され
ないため、提出コードから削除した（実装は git 履歴に残る）。削除の内訳は
モジュール7本 1,701行 + テスト6本 1,857行、コンストラクタ引数は 60 → 31。
`SeedExpansionRetriever` から `PaperTwoLaneReranker` / `PaperNeighborhoodExpansion` /
`MethodRelationExpansion` / `MethodBridgeExploration` / `DenseReciprocalExploration` /
`DenseConsensusExploration` の配線と、`DenseTailFusion` の paper レーン、
`CandidateGeneration` の local expansion レーン、`protection.restore_method_protected_candidates`、
`evaluation/diagnostics.py` の死んだ provenance キー51個が消えた。

| レーン | 設計意図 | 無効化された経緯 |
|---|---|---|
| paper neighborhood | 取得済み論文間の引用リンクで候補を再スコア | 重み 0.2 で使われていた時期があるが、候補補充を method dense tail と open-set に絞る過程で 0 に |
| method owner / relation / topic | 手法の所有関係と手法間エッジを辿って候補を補充 | 同上。weight 0.35 から 0 へ |
| method bridge | 共有手法で繋がる論文に最終1枠を使う | 同上 |
| paper dense tail | 上位論文の SPECTER2 近傍で末尾を再構築 | method dense tail のみ残す判断で 0 に |
| dense reciprocal | 相互参照が閾値以上の論文に最終1枠を使う | seed_k 0 |
| dense consensus | 複数 seed が合意した論文に最終1枠を使う | seed_k 0 |
| two-lane rerank | 語彙レーンと拡張レーンを論文単位で融合 | `two_lane_rerank: false` |
| local expansion | 旧方式の seed クエリで3本目の検索 | weight 0 |

**注意**: これらは「効果がないと測定された」ものと「構成を絞る過程で外された」
ものが混在している。個別の寄与を測り直した記録は残っていないため、論文で
「効果がなかった」と書くなら再測定が必要。

---

## 2. 実装して測定し、採用しなかった手法

### 2.1 Query Decomposition（LLM による質問分解）

**手法**: 質問を最大4つの SubQuery へ分解し、各々で BM25 検索して結果を統合。
分解には Qwen3-0.6B をローカルで使用（`llm/local_hf.py`）。

**結果**: 3種の gold 定義すべてで **改善ゼロ**。

| gold | baseline R@10 | + SubQuery R@10 |
|---|---|---|
| answer-bearing 87本 | 0.9697 | 0.9697 |
| evidence 117本 | 0.8424 | 0.8424 |
| 原本 146本 | 0.7606 | 0.7606 |

**分かったこと**

1. **列挙型質問には原理的に効かない。** 「Which CVPR 2025 papers cite UniAD ...」
   のような質問は「述語 P を満たす論文を列挙せよ」という形で、分割すべき対象が
   質問文に存在しない。実際に LLM は質問を1行に言い換えるだけだった。
   分解が成立するのは「TCM, sCT, ECM-XL, IMM の FID を比べよ」のように
   **質問が対象を列挙している**場合に限られる（55問中9問）。

2. **小さいモデルは few-shot をプロンプト本文に置くと解答を丸写しする。**
   0.6B では55問中ほぼ全部が偽の分解になった。few-shot を対話ロール
   （user/assistant の完了済みやり取り）として分離すると停止した。

3. **RRF による全面再構成は有害。** 4本の SubQuery 結果を主ランキングと RRF で
   融合したところ、55問で221本が入れ替わり、rank 21 と 12 にいた gold 2本が
   top50 から押し出された。上限付きの追加（新規3本まで）に変えると悪化は
   消えたが、改善も出なかった。

**構造的な対策**: SubQuery の内容語がすべて元の質問に含まれることを必須にすると、
丸写しと幻覚が同時に排除でき、固有名の保持も担保できる。分解とは質問の語彙で
言い換えることなので、この制約は本質的。

### 2.2 Citation lane（引用関係による証拠索引）

**手法案**: 「UniAD を引用している」という条件を citation_context チャンクの
検索で判定する。

**結果**: **実装不可**。コーパスに `citation_context` チャンクが **0件**。
MinerU の現行設定では生成されていない。実装するには 27,487論文の再前処理が必要。

なお q_023（UniAD を引用し比較表で baseline に使う CVPR 2025 論文の列挙）は、
table チャンクの検索だけで gold 9本すべてを取得できたため、citation lane が
無くても解けている。

### 2.3 Selective Figure VLM（図画像の VLM 処理）

**手法案**: 図中に描かれた文字（MinerU が抽出しない）を VLM で読む。対象は
q_020 の `naacl2025_00895`（証拠が `region: "Figure"` で値が `MCTS Procedure`）。

**結果**: **不要だった**。図中にしか無いと判断していたが、**同じ論文の本文にも
MCTS が出ていた**。モダリティ検索の後に text_span 検索を連結するだけで rank 11
で取得でき、VLM は要らなかった。

**教訓**: 「この証拠はこのモダリティにしか無い」という判断は、他モダリティを
実際に検索して確かめるまで信用しない。

### 2.4 Entity Family Expansion（派生表記の展開）

**手法案**: `UniAD` → `UniAD-Base` / `UniAD-B` のような派生表記を辞書展開する。

**結果**: **不要**。bm25s のトークナイザは `UniAD-Base` を `["uniad","base"]` に
分割するため、`UniAD` での検索が既に派生表記に一致している。実際
`cvpr2025_02317`（UniAD-Base 表記）は展開なしで取得できた。

### 2.5 リランカーの大型化（8B）

**結果**: **8B は 4B に劣る**。

| reranker | R@1 | R@3 | R@10 | 秒/問 |
|---|---|---|---|---|
| 0.6B | 0.7202 | 0.9010 | 0.9808 | 2.24 |
| **4B** | **0.7808** | **0.9056** | **0.9848** | 7.15 |
| 8B | 0.7581 | 0.9010 | 0.9763 | 9.22 |

8B は遅いうえに R@1 も低く、0.6B からの gold 順位変化も 改善18 / 悪化17 と
ほぼ相殺だった。`base_rank_weight: 0.52` が 0.6B 向けに調整された値である点は
交絡要因になりうるが、4B では同じ 0.52 が最適だったので、設定の不整合だけでは
説明しきれない。

### 2.6 リランク保護の解除（`final_rerank_protected_top_k: 0`）

**手法案**: `_protect_prefix` が元の上位20を強制的に前へ固定するため、rank 21
以降の論文はリランカーがどれだけ高く評価しても20位内に入れない。この保護を
外して50件全体の順位をリランカーに委ねる。

**結果**: **悪化**。

| 保護 | R@5 | R@8 | R@10 | 現行比 改善/悪化 |
|---|---|---|---|---|
| 0 | 0.9551 | 0.9652 | 0.9697 | 0 / 8 |
| 5 | 0.9116 | 0.9652 | 0.9697 | 10 / 15 |
| 10 | 0.9505 | 0.9545 | 0.9636 | 3 / 8 |
| **20（採用）** | **0.9551** | **0.9672** | **0.9808** | — |

**理由**: 保護は対称的に働く。test 71問で計測すると、保護は「リランカーが top20
と判定した論文を324本落とす」と同時に「リランカーが21位以降と判定した元 top20 を
322本救って」いた。ほぼ同数で、validation では後者の利得が上回る。

Qwen3-Reranker のような比較的小さいモデルの判断を全面採用するより、語彙検索の
候補順位を信じるほうが良い、という結果。

---

## 3. 採用したが、当初の判断を修正した手法

### 3.1 Structured Filter の発火条件

**当初**: 会場・年・モダリティの3条件で発火。validation の2問（q_020, q_023）で
R@20 0.9934 → 1.0000、All-Gold@20 0.9636 → 1.0000 と改善したため採用した。

**held-out で判明した問題**: 発火条件が広すぎた。

| データ | 発火 | うち誤発火（非列挙型） |
|---|---|---|
| validation | 2問 | 0問 |
| test | 3問 | **3問（全部）** |
| test_extra | 186問 | **185問（99.5%）** |

「For the two ICCV 2025 papers, compare ...」のように**答えとなる論文が既に
2本に限定されている比較質問**にも作動し、会場全体の候補を前方へ昇格させて
元の良い順位を壊していた（質問が名指しした論文が3位→21位、2位→22位）。

**修正**: 既存の `is_open_set_enumeration()` を発火条件に追加。発火は
validation 2問 / test 0問 / test_extra 1問になり、validation の改善は維持された。

**教訓**: validation で偶然2問とも列挙型だったため問題が表面化しなかった。
新しいレーンは「validation での改善」だけでなく「**held-out での発火内容が
想定どおりか**」を確認する必要がある。

### 3.2 Exact Method Search の一度目の削除

**経緯**: 質問中の固有名から論文を一意に引く索引を実装。validation では
適合率 1.000（21本ヒット・誤ヒット0）だったが **Recall は3種の gold すべてで
改善0**だったため、一度削除した。

**復活の根拠**: held-out で実際の取りこぼしを救えることが判明。test の
`ltqa_a2c8b9763a7ce26e` では、質問が `AD-GS` と明記しているのに
`iccv2025_00058` が候補50件に入っていなかった。Exact Search を戻すと
**候補外 → 4位**。

**なぜ validation で測れなかったか**: baseline が既に R@50 = 1.0000 に達しており、
候補補充で入れられる gold が残っていなかった。**「効果がない」と「効果を測れ
ない」は別物**だった。

**設計上の要点**: alias は「1本の論文を一意に指す」場合のみ採用する。2本まで
許すと `RAG` や指標の `mAP` が混入し、適合率が 1.000 → 0.913 に落ちた。

---

## 4. 評価指標についての注意

validation の answer-bearing gold では **R@50 = 1.0000、R@20 = 1.0000、
All-Gold@20 = 1.0000** に達しており、**この gold ではこれ以上の改善を検出できない**。
候補補充系の手法を評価するときは、指標が飽和していないか先に確認すること。

原本 gold（146本）では R@50 = 0.9091 と余地があるが、top50 で取りこぼす20本は
**すべて answer-bearing ではない**（監査で supportive / irrelevant と分類した
対照用の論文、および evidence が無い論文）。回答に必要な論文の取りこぼしは 0本
なので、この差を埋める意味は薄い。

held-out（test / test_extra）には公式 gold が無いため、Recall は測れない。
代わりに「質問が名指しした一意論文が候補に入っているか」「質問が明示した
会場・年に候補が合致しているか」で検証した。
