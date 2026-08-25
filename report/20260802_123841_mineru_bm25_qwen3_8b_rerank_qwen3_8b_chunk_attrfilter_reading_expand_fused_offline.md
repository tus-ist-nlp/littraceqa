# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_expand_fused（オフライン適用）

- 実行日時: 2026-08-02T12:38:41
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_expand_fused.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_expand_fused_offline.jsonl`
- git: `a0206042354c`
- **オフライン適用**: 土台 `predictions_8b_chunk_b_merged.jsonl`。`compose_config()` で
  組んだ expander に `reading.py._expand_candidates()` と同じ手順を適用。
- **注意: reranker は 0.6B で代用**（本番構成は 8B。GPU 空き待ち）。

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `1000` |
| pool_k | `1000` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| reranker_model_本番 | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_model_本実験 | `"Qwen/Qwen3-Reranker-0.6B"` |
| attribute_filter_enabled | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |
| expansion_sources | `["specter2(faiss_specter2_abstract)", "bib_coupling(min_shared=2)"]` |
| expansion_neighbors | `20` |
| expansion_anchors | `1` |
| expansion_rrf_k | `60` |
| expansion_rerank | `true` |
| expansion_rerank_top_k | `5` |
| expansion_insert_at | `15` |
| bib_coupling_索引 | `"25,012論文 / 68,418 arXiv ID、構築29〜47秒（以後キャッシュ）"` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7824 |
| paper_recall_macro | 0.6475 |
| paper_f1_macro | 0.5841 |
| candidate_recall_at1_single_macro | 0.9615 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at100_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.4071 |
| candidate_recall_at10_multi_macro | 0.4751 |
| candidate_recall_at20_multi_macro | 0.6628 |
| candidate_recall_at50_multi_macro | 0.8190 |
| candidate_recall_at100_multi_macro | 0.8534 |
| candidate_recall_at1_total_macro | 0.5611 |
| candidate_recall_at5_total_macro | 0.6874 |
| candidate_recall_at10_total_macro | 0.7232 |
| candidate_recall_at20_total_macro | 0.8222 |
| candidate_recall_at50_total_macro | 0.9045 |
| candidate_recall_at100_total_macro | 0.9227 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at100_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3831 |
| evidence_candidate_recall_at5_multi_macro | 0.5910 |
| evidence_candidate_recall_at10_multi_macro | 0.6504 |
| evidence_candidate_recall_at20_multi_macro | 0.7749 |
| evidence_candidate_recall_at50_multi_macro | 0.8937 |
| evidence_candidate_recall_at100_multi_macro | 0.9282 |
| evidence_candidate_recall_at1_total_macro | 0.6566 |
| evidence_candidate_recall_at5_total_macro | 0.7843 |
| evidence_candidate_recall_at10_total_macro | 0.8157 |
| evidence_candidate_recall_at20_total_macro | 0.8813 |
| evidence_candidate_recall_at50_total_macro | 0.9439 |
| evidence_candidate_recall_at100_total_macro | 0.9621 |
| evidence_precision_macro | 0.2404 |
| evidence_recall_macro | 0.2328 |
| evidence_f1_macro | 0.2205 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.4945 |
| table_cell_accuracy_macro | 0.2900 |
| table_cell_accuracy_micro | 0.2963 |

## 比較

| 指標 | 展開なし | SPECTER2のみ+rerank | **融合+rerank（本実行）** |
|---|---|---|---|
| paper_f1_macro | 0.5841 | 0.5841 | **0.5841** |
| candidate_recall_at20_total_macro | 0.7833 | 0.8131 | **0.8222** |
| candidate_recall_at20_multi_macro | 0.5891 | 0.6456 | **0.6628** |
| candidate_recall_at50_total_macro | 0.8364 | 0.8955 | **0.9045** |
| candidate_recall_at50_multi_macro | 0.6897 | 0.8017 | **0.8190** |
| evidence_candidate_recall_at20_total_macro | 0.8697 | 0.8753 | **0.8813** |
| evidence_candidate_recall_at50_total_macro | 0.9076 | 0.9379 | **0.9439** |

## 所見

**全指標で SPECTER2 単独を上回った。** cr@20 0.813 -> 0.822、multi@20 0.646 -> 0.663、
cr@50 0.895 -> 0.905、multi@50 0.802 -> 0.819。展開なしからは cr@20 +3.9pt / multi@50 +12.9pt。

**併用の根拠は「違う gold を拾う」こと。** 候補圏外 gold の回収を数えると
両方7本 / 書誌結合のみ5本 / SPECTER2のみ10本 で、重複は3割程度しかない。
意味的な近さ（SPECTER2）と引用文献の共有（書誌結合）は独立した信号として働く。

**書誌結合は引用グラフではない。** A が B を引く関係はこのコーパスでは張れない
（2024〜2025年のみで同時期の論文は互いに引用できない。anchor から解決できた
コーパス内引用は実測1本）。共有している**古い文献**で繋ぐ。TCM とピア3本の
Jaccard は 0.19〜0.24、無作為30本は中央値 0.000・最大 0.054 と明確に分離する。

**コストはほぼゼロ。** 書誌結合索引はコーパス1走査（29〜47秒）で作ってキャッシュし、
クエリ時はメモリ上の集合演算のみ。GPU も追加モデルも要らない。

## 次に検証すべきこと

- **8B reranker での再検証**（GPU 空き待ち）。本実験は 0.6B 代用。
- min_shared / rrf_k の掃引。現状は min_shared=2（1本だけの共有は Adam 等の
  汎用引用で繋がるため切る）、rrf_k=60（検索側の fuser と同値）を根拠なく踏襲している。
- 提出に載せるにはループ内へ組み込む改修が必要（cr は上がったが paper_f1 は不変）。

## 訂正（2026-08-02 追記）: 本レポートの @100 の値は比較に使えない

`candidate_recall@100` / `evidence_candidate_recall@100` として載せた値は、
**実際には「@min(100, 候補列の長さ)」**である。予測に残す候補は `reading.py` の
`CANDIDATE_PAPERS_LIMIT`（既定50）で切られるため:

- 展開なしの実験は候補列が **50本** -> その「@100」は @50 と同値
- 展開ありの実験は候補列が **最大70本** -> その「@100」は実質 @70

つまり「@100 が 0.836 -> 0.914 に伸びた」という記述は、**@50 と @70 を比べたもの**で
あり、展開なし側は構造的に51位以降が空欄だった。候補列に gold が増えたこと自体は
事実だが、`@100` という指標名で比較したのは誤り。

**公平に比較できるのは cr@50 / ecr@50 まで**（どちらの実験も50本以上ある）。
本レポート内の @50 以下の数値と結論は影響を受けない。

## クエリ品質ラベル別の candidate_recall（2026-08-03 追記）

`audits/query_audit.jsonl`（146行 = gold 論文1本1行）のラベルで層別した。判定は再実装せず
既存のものを import している——`scripts/audit_report.py` の `query_label()`
（spec 3.3。`no_evidence` が1本でも混ざれば `noisy`、全部 `supporting` なら `good`、
それ以外は `fair`）と、`scripts/evaluate.py` の `recall_at_k()` /
`candidate_paper_ids()` / `evidence_backed_paper_ids()`。k=5 は
`CANDIDATE_RECALL_KS` に無いのでこの集計でだけ足している。

本レポートの構成（`predictions_8b_chunk_expand_fused_offline.jsonl`）:

| 集合 | 件数 (single/multi) | cr@5 | ecr@5 | cr@10 | ecr@10 | cr@20 | ecr@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|---|---|---|---|
| 全体 | 55 (26/29) | 0.687 | 0.784 | 0.723 | 0.816 | 0.822 | 0.881 | 0.905 | 0.944 |
| **良問+やや良問** | **39 (26/13)** | 0.841 | 0.841 | 0.885 | 0.885 | 0.935 | 0.935 | **0.981** | 0.981 |
| 良問のみ | 32 (25/7) | 0.924 | 0.924 | 0.977 | 0.977 | 0.989 | 0.989 | 0.992 | 0.992 |
| やや良問のみ | 7 (1/6) | 0.464 | 0.464 | 0.464 | 0.464 | 0.690 | 0.690 | 0.929 | 0.929 |
| あほ問 | 16 (0/16) | 0.312 | 0.646 | 0.328 | 0.646 | 0.547 | 0.750 | 0.719 | 0.854 |

土台（`predictions_8b_chunk_b_merged.jsonl`）:

| 集合 | 件数 (single/multi) | cr@5 | ecr@5 | cr@10 | ecr@10 | cr@20 | ecr@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|---|---|---|---|
| 全体 | 55 (26/29) | 0.687 | 0.784 | 0.723 | 0.816 | 0.783 | 0.870 | 0.836 | 0.908 |
| 良問+やや良問 | 39 (26/13) | 0.841 | 0.841 | 0.885 | 0.885 | 0.919 | 0.919 | 0.955 | 0.955 |
| 良問のみ | 32 (25/7) | 0.924 | 0.924 | 0.977 | 0.977 | 0.992 | 0.992 | 0.992 | 0.992 |
| やや良問のみ | 7 (1/6) | 0.464 | 0.464 | 0.464 | 0.464 | 0.583 | 0.583 | 0.786 | 0.786 |
| あほ問 | 16 (0/16) | 0.312 | 0.646 | 0.328 | 0.646 | 0.453 | 0.750 | 0.547 | 0.792 |

### 読み取れること

1. **あほ問を除くと cr@50 は 0.981 でほぼ天井。** 39件中、@50 に gold が入らなかったのは
   実質1件ぶん。全体値 0.905 は取れないものを分母に入れているせいで11ポイント低く見えている。
2. **あほ問16件は全部 multi_paper**（single/multi = 0/16）。`noisy` の条件が
   「evidence の無い gold が1本でも混ざる」なので、質問文が名指ししないピア gold を含む
   multi がそのまま落ちる。single 26件は良問25 + やや良問1 であほ問ゼロ。
3. **良問+やや良問では cr と ecr が完全に一致する**（この39件に根拠なし gold が無いため）。
   つまり **ecr を見る意味があるのはあほ問16件だけ**で、そこでは cr 0.312 に対し
   ecr 0.646 と倍以上開く（@5）。
4. **やや良問は @5 と @10 が同値（0.464）。** 6位〜10位の枠に gold が1本も増えておらず、
   拾えているものは5位以内、拾えていないものは11位以降にある。中間が空なので、
   top_k を10前後で刻む調整には意味が無い。
5. **`fused` の改善は @20 以降にしか出ない。** @5 / @10 は土台と完全に同値
   （0.841 / 0.885）で、伸びているのは @20 0.919->0.935 と @50 0.955->0.981。
   展開した論文の挿入位置が 15位（`insert_at`）なので当然だが、
   **@10 を看板にすると `fused` の効果は1ポイントも見えない**。

### task_family 別（良問+やや良問に絞った内訳）

本レポートの構成:

| 集合 | 件数 | cr@5 | ecr@5 | cr@10 | ecr@10 | cr@20 | ecr@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|---|---|---|---|
| **良問+やや良問 / single** | **26** | **1.000** | 1.000 | **1.000** | 1.000 | **1.000** | 1.000 | **1.000** | 1.000 |
| **良問+やや良問 / multi** | **13** | 0.524 | 0.524 | 0.656 | 0.656 | 0.806 | 0.806 | 0.942 | 0.942 |
| 良問+やや良問 / 全体 | 39 | 0.841 | 0.841 | 0.885 | 0.885 | 0.935 | 0.935 | 0.981 | 0.981 |
| （参考）全体 / single | 26 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| （参考）全体 / multi | 29 | 0.407 | 0.591 | 0.475 | 0.650 | 0.663 | 0.775 | 0.819 | 0.894 |
| （参考）あほ問 / multi | 16 | 0.312 | 0.646 | 0.328 | 0.646 | 0.547 | 0.750 | 0.719 | 0.854 |

土台:

| 集合 | 件数 | cr@5 | ecr@5 | cr@10 | ecr@10 | cr@20 | ecr@20 | cr@50 | ecr@50 |
|---|---|---|---|---|---|---|---|---|---|
| 良問+やや良問 / single | 26 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 良問+やや良問 / multi | 13 | 0.524 | 0.524 | 0.656 | 0.656 | 0.756 | 0.756 | 0.865 | 0.865 |
| 良問+やや良問 / 全体 | 39 | 0.841 | 0.841 | 0.885 | 0.885 | 0.919 | 0.919 | 0.955 | 0.955 |
| （参考）全体 / multi | 29 | 0.407 | 0.591 | 0.475 | 0.650 | 0.589 | 0.753 | 0.690 | 0.825 |
| （参考）あほ問 / multi | 16 | 0.312 | 0.646 | 0.328 | 0.646 | 0.453 | 0.750 | 0.547 | 0.792 |

**single_paper は @5 で既に 1.000。26件すべてで gold が候補上位5本に入っている。**
しかもこれは良問+やや良問に絞る前から 1.000（single にあほ問が1件も無いため、26件が
そのまま全体と一致する）。**single_paper の検索側にはもう伸びしろが無い**ので、
索引・fuser・reranker をいじる価値はここには残っていない。改善余地は評価の下流
（提出本数の決定、evidence 抽出）にしかない。**pool_k や top_k を削って高速化する
余地がある**とも言える——single に関しては上位5本で足りている。

**残る課題は multi_paper 13件に完全に集約される。** 良問+やや良問の multi は
@5 0.524 -> @50 0.942 で、**50位以内には居るのに上位に上げられていない**という形。
@5 と @50 の差 0.418 が丸ごと順位付けの取りしろで、索引を替える話ではない。

`fused` の効き方も multi にしか出ていない（single は 1.000 で据え置き、
良問+やや良問の multi が @50 0.865 -> 0.942、あほ問が 0.547 -> 0.719）。
論文→論文展開は multi_paper 専用の打ち手として機能している。

## candidate_recall（後から追記）

予測ファイル `predictions_8b_chunk_expand_fused_offline.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 70 本なので、それを超える k は @70 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.8534 |
| candidate_recall_at70_total_macro | 0.9227 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.9282 |
| evidence_candidate_recall_at70_total_macro | 0.9621 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_expand_fused_offline.jsonl, max_candidates=70 -->
