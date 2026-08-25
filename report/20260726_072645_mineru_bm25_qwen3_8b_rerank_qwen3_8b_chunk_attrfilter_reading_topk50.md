# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_topk50

- 実行日時: 2026-07-26T07:26:45
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_topk50.yaml`
- queries: `data/split/val_b.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_b.jsonl`
- git: `f53e1da3a4b5`
- 分割実行を結合して採点: `predictions_8b_chunk_a.jsonl`

> **このレポートが同一構成の 2026-07-25T20:14:46 の実行（`20260725_201446_..._topk50.md`）を置き換える。**
> 旧版は val_a の28件しか予測が無い状態で55件の gold に採点していたため、全 macro 指標が
> 網羅率(約51%)のぶん薄まっていた（paper_f1 0.3297 -> 実際 0.5841、evidence_f1 0.1272 -> 0.2205）。
> 旧版のコメントもその薄まった数字に基づいていたので破棄した。旧ファイルは削除済み。

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `1000` |
| pool_k | `1000` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| fuser_k | `60` |
| fuser_weights | `{"bm25s": 1.0, "faiss_qwen3": 1.0}` |
| reranker | `"qwen3"` |
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_devices | `"cuda:1,cuda:2,cuda:3"` |
| reranker_fp16 | `true` |
| reranker_max_batch_tokens | `2048` |
| reranker_batch_size | `4` |
| reranker_max_tokens | `2048` |
| reranker_device | `"cuda"` |
| reranker_instruction | `"Given a scientific question, retrieve passages from research papers that help identify or support the answer"` |
| reranker_compile | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_top_k | `50` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

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
| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.4071 |
| candidate_recall_at10_multi_macro | 0.4751 |
| candidate_recall_at20_multi_macro | 0.5891 |
| candidate_recall_at50_multi_macro | 0.6897 |
| candidate_recall_at1_total_macro | 0.5611 |
| candidate_recall_at5_total_macro | 0.6874 |
| candidate_recall_at10_total_macro | 0.7232 |
| candidate_recall_at20_total_macro | 0.7833 |
| candidate_recall_at50_total_macro | 0.8364 |
| evidence_precision_macro | 0.2404 |
| evidence_recall_macro | 0.2328 |
| evidence_f1_macro | 0.2205 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.4945 |
| table_cell_accuracy_macro | 0.2900 |
| table_cell_accuracy_micro | 0.2963 |

## クエリ診断

- 候補上位10本に gold が1本も入らなかった: **1件** (q_022)
- 一部しか入らなかった(multi の取りこぼし): **24件** (q_023, q_025, q_029, q_031, q_032, q_033, q_034, q_035, q_036, q_037, q_038, q_039, q_040, q_041, q_042, q_043, q_044, q_045, q_046, q_047, q_048, q_049, q_050, q_051)

`gold順位` は候補列(`candidate_papers`)で gold 論文が何位だったか。`-` は候補に入っていない。`cr@k` はその上位k本での recall。

| query_id | family | gold | 提出 | 候補 | gold順位 | cr@10 | cr@50 | paper_f1 | ev_f1 | 回答 | attrfilter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q_001 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_002 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | ff:× |  |
| q_003 | single | 1 | 3 | 3 | 1 | 1.00 | 1.00 | 0.50 | 0.00 | ff:○ |  |
| q_004 | single | 1 | 10 | 32 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_005 | single | 1 | 10 | 20 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_006 | single | 1 | 1 | 24 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_007 | single | 1 | 10 | 41 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_008 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_009 | single | 1 | 1 | 21 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_010 | single | 1 | 1 | 13 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_011 | single | 1 | 1 | 30 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_012 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:○ |  |
| q_013 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_014 | single | 1 | 1 | 39 | 1 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_015 | single | 1 | 1 | 11 | 1 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_017 | single | 1 | 1 | 45 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_018 | single | 1 | 10 | 30 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_019 | single | 1 | 10 | 35 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_020 | multi | 4 | 3 | 50 | 3, 1, 8, 5 | 1.00 | 1.00 | 0.29 | 0.29 | ff:× / table row_f1:0.29 cell:0.29 | NAACL 2025 |
| q_021 | single | 1 | 3 | 50 | 2 | 1.00 | 1.00 | 0.50 | 0.25 | mc:× / ff:× |  |
| q_022 | multi | 3 | 3 | 50 | 28, 44, 15 | 0.00 | 1.00 | 0.33 | 0.00 | table row_f1:0.00 cell:0.00 | ICML 2025 |
| q_023 | multi | 9 | 7 | 49 | 1, 4, 11, 3, 5, 8, 10, 16, 2 | 0.78 | 1.00 | 0.88 | 0.56 | table row_f1:0.75 cell:0.75 | CVPR 2025 |
| q_024 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× | NeurIPS |
| q_025 | multi | 4 | 9 | 50 | 5, 8, -, 2 | 0.75 | 0.75 | 0.46 | 0.00 | ff:× / table row_f1:0.15 cell:0.15 |  |
| q_026 | single | 1 | 1 | 18 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_027 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | ff:× / table row_f1:0.00 cell:0.00 | CVPR 2025 |
| q_028 | multi | 4 | 3 | 50 | 1, 5, 3, 9 | 1.00 | 1.00 | 0.57 | 0.29 | table row_f1:0.75 cell:0.75 |  |
| q_029 | multi | 4 | 2 | 33 | 2, 8, 7, 15 | 0.75 | 1.00 | 0.33 | 0.29 | table row_f1:0.75 cell:0.75 |  |
| q_030 | multi | 4 | 3 | 50 | 4, 1, 2, 3 | 1.00 | 1.00 | 0.86 | 0.00 | table row_f1:1.00 cell:0.50 |  |
| q_031 | multi | 4 | 1 | 37 | 17, 19, 1, - | 0.25 | 0.75 | 0.40 | 0.00 | mc:× |  |
| q_032 | multi | 4 | 1 | 50 | 44, -, 1, 22 | 0.25 | 0.75 | 0.40 | 0.00 | mc:× |  |
| q_033 | multi | 4 | 1 | 38 | 1, 5, 6, 14 | 0.75 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_034 | multi | 4 | 1 | 40 | 13, 29, -, 1 | 0.25 | 0.75 | 0.40 | 0.00 | mc:× |  |
| q_035 | multi | 4 | 1 | 50 | 20, 1, 42, - | 0.25 | 0.75 | 0.40 | 0.00 | mc:× |  |
| q_036 | multi | 4 | 1 | 50 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.40 | mc:× |  |
| q_037 | multi | 4 | 1 | 41 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_038 | multi | 4 | 1 | 26 | 1, -, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_039 | multi | 4 | 1 | 8 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.40 | mc:× |  |
| q_040 | multi | 4 | 1 | 50 | 32, -, -, 1 | 0.25 | 0.50 | 0.40 | 0.00 | mc:× |  |
| q_041 | multi | 4 | 1 | 50 | 3, 1, 4, 47 | 0.75 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_042 | multi | 4 | 2 | 50 | -, 2, -, 3 | 0.50 | 0.50 | 0.67 | 0.00 | mc:× |  |
| q_043 | multi | 4 | 2 | 50 | 1, 4, 5, - | 0.75 | 0.75 | 0.67 | 0.33 | mc:× |  |
| q_044 | multi | 4 | 2 | 50 | 4, 2, 1, - | 0.75 | 0.75 | 0.33 | 0.00 | mc:× |  |
| q_045 | multi | 4 | 2 | 50 | 3, 38, 37, 12 | 0.25 | 1.00 | 0.33 | 0.00 | mc:× |  |
| q_046 | multi | 4 | 1 | 50 | 1, 17, 15, 19 | 0.25 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_047 | multi | 4 | 1 | 50 | -, 1, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_048 | multi | 4 | 1 | 50 | 1, -, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_049 | multi | 4 | 2 | 50 | 43, -, -, 1 | 0.25 | 0.50 | 0.33 | 0.00 | mc:× |  |
| q_050 | multi | 4 | 1 | 50 | 13, -, 1, - | 0.25 | 0.50 | 0.40 | 0.33 | mc:× |  |
| q_051 | multi | 4 | 1 | 50 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_052 | single | 1 | 1 | 44 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:1.00 cell:0.00 |  |
| q_053 | single | 1 | 1 | 24 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_054 | single | 1 | 1 | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:0.00 cell:0.00 |  |
| q_055 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_056 | multi | 4 | 1 | 50 | 10, 3, 1, 2 | 1.00 | 1.00 | 0.40 | 0.00 | table row_f1:0.75 cell:0.00 |  |

## コメント

候補生成はかなり強く、single では Recall@1=0.962、@5 以降は 1.0 なので、少なくとも正解文献を候補に入れる力は十分あります。一方で最終的な paper F1=0.584 は precision 0.782 に対して recall 0.647 がやや低く、特に multi 条件の candidate recall@50=0.690 が低めで、多文献系クエリの取りこぼしが主なボトルネックに見えます。evidence F1=0.221、freeform exact match=0.192、multiple choice accuracy=0.0 から、候補以降の根拠抽出・読解・解答生成がかなり弱く、table cell accuracy も約0.30で表構造の扱いも改善余地が大きいです。次は multi クエリ向けに候補数や多様化を強めること、rerank/reader のプロンプトや根拠抽出条件を見直すこと、表を含む文書の chunking・属性フィルタ設定を緩めて再評価するのがよさそうです。

## candidate_recall（後から追記）

予測ファイル `predictions_8b_chunk_b.jsonl + predictions_8b_chunk_a.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 50 本なので、それを超える k は @50 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.6897 |
| candidate_recall_at70_total_macro | 0.8364 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3831 |
| evidence_candidate_recall_at5_multi_macro | 0.5910 |
| evidence_candidate_recall_at10_multi_macro | 0.6504 |
| evidence_candidate_recall_at20_multi_macro | 0.7529 |
| evidence_candidate_recall_at50_multi_macro | 0.8247 |
| evidence_candidate_recall_at70_multi_macro | 0.8247 |
| evidence_candidate_recall_at1_total_macro | 0.6566 |
| evidence_candidate_recall_at5_total_macro | 0.7843 |
| evidence_candidate_recall_at10_total_macro | 0.8157 |
| evidence_candidate_recall_at20_total_macro | 0.8697 |
| evidence_candidate_recall_at50_total_macro | 0.9076 |
| evidence_candidate_recall_at70_total_macro | 0.9076 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_b.jsonl + predictions_8b_chunk_a.jsonl, max_candidates=50 -->
