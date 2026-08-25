# mineru + bm25_qwen3_8b_rerank_qwen3_8b_paper_attrfilter + reading

- 実行日時: 2026-07-25T12:21:48 (val_a) / 2026-07-25T12:32 (val_b)、55件で採点し直したのが 2026-07-25T16:41:21
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_paper_attrfilter.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/split/val_a.jsonl` + `data/split/val_b.jsonl` (55件, production_input=True)
- output: `predictions_8b_paper_attrfilter.jsonl`（val_a 28件 + val_b 27件を結合）
- git: `f53e1da3a4b5`

> **このレポートは55件フルで採点し直したもの。** 生成時(12:21)の版は val_a の28件しか予測が無い状態で55件の gold に採点していたため、全 macro 指標が約半分に薄まっていた（例: paper_f1 0.3669 -> 実際 0.6360、evidence_f1 0.1025 -> 0.2062）。旧版のコメントもその薄まった数字に基づいていたので破棄した。

## 設定（この実行時の実際の値）
##　論文単位のrerank

| パラメータ | 値 |
|---|---|
| per_index_k | `1000` |
| pool_k | `1000` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| fuser_k | `60` |
| fuser_weights | `{"bm25s": 1.0, "faiss_qwen3": 1.0}` |
| reranker | `"qwen3_paper"` |
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_devices | `"cuda:1,cuda:2,cuda:3"` |
| reranker_fp16 | `true` |
| reranker_max_batch_tokens | `2048` |
| reranker_batch_size | `4` |
| reranker_max_tokens | `2048` |
| reranker_chunks_per_paper | `3` |
| reranker_device | `"cuda"` |
| reranker_instruction | `"Given a scientific question, retrieve passages from research papers that help identify or support the answer"` |
| reranker_compile | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.8728 |
| paper_recall_macro | 0.6394 |
| paper_f1_macro | 0.6360 |
| candidate_recall_at1_single_macro | 0.9615 |
| candidate_recall_at5_single_macro | 0.9615 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2193 |
| candidate_recall_at5_multi_macro | 0.4033 |
| candidate_recall_at10_multi_macro | 0.4377 |
| candidate_recall_at20_multi_macro | 0.5010 |
| candidate_recall_at50_multi_macro | 0.5182 |
| candidate_recall_at1_total_macro | 0.5702 |
| candidate_recall_at5_total_macro | 0.6672 |
| candidate_recall_at10_total_macro | 0.7035 |
| candidate_recall_at20_total_macro | 0.7369 |
| candidate_recall_at50_total_macro | 0.7460 |
| evidence_precision_macro | 0.2446 |
| evidence_recall_macro | 0.2010 |
| evidence_f1_macro | 0.2062 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.5303 |
| table_cell_accuracy_macro | 0.2955 |
| table_cell_accuracy_micro | 0.3704 |

## 注記

- `multiple_choice_accuracy` が 0.0 なのは不具合ではない。`f53e1da` で multiple_choice の options 結合を oracle 実験専用にしたため、`--production-input` の実行では選択肢が与えられない。0.6B ベースライン(2026-07-21, sha `291ca06`) の 0.5854 は options を結合した実行の値なので、この指標は両者で比較できない。

## クエリ診断

- 候補上位10本に gold が1本も入らなかった: **1件** (q_022)
- 一部しか入らなかった(multi の取りこぼし): **25件** (q_023, q_025, q_028, q_029, q_031, q_032, q_033, q_034, q_035, q_036, q_037, q_038, q_039, q_040, q_041, q_042, q_043, q_044, q_045, q_046, q_047, q_048, q_049, q_050, q_051)

`gold順位` は候補列(`candidate_papers`)で gold 論文が何位だったか。`-` は候補に入っていない。`cr@k` はその上位k本での recall。

| query_id | family | gold | 提出 | 候補 | gold順位 | cr@10 | cr@50 | paper_f1 | ev_f1 | 回答 | attrfilter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q_001 | single | 1 | 1 | 27 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_002 | single | 1 | 10 | 35 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_003 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | ff:× |  |
| q_004 | single | 1 | 7 | 7 | 1 | 1.00 | 1.00 | 0.25 | 0.00 | mc:× / ff:× |  |
| q_005 | single | 1 | 3 | 3 | 1 | 1.00 | 1.00 | 0.50 | 0.00 | ff:× |  |
| q_006 | single | 1 | 1 | 22 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_007 | single | 1 | 3 | 3 | 1 | 1.00 | 1.00 | 0.50 | 0.00 | mc:× / ff:× |  |
| q_008 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_009 | single | 1 | 1 | 2 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_010 | single | 1 | 10 | 16 | 1 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_011 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_012 | single | 1 | 1 | 19 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:○ |  |
| q_013 | single | 1 | 1 | 4 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_014 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_015 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_017 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_018 | single | 1 | 7 | 7 | 1 | 1.00 | 1.00 | 0.25 | 0.00 | mc:× / ff:× |  |
| q_019 | single | 1 | 1 | 9 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:○ |  |
| q_020 | multi | 4 | 5 | 24 | 1, 5, 3, 8 | 1.00 | 1.00 | 0.67 | 0.36 | ff:× / table row_f1:0.75 cell:0.75 | NAACL 2025 |
| q_021 | single | 1 | 1 | 9 | 7 | 1.00 | 1.00 | 1.00 | 0.50 | mc:× / ff:× |  |
| q_022 | multi | 3 | 1 | 17 | -, -, 12 | 0.00 | 0.33 | 0.50 | 0.40 | table row_f1:0.50 cell:0.00 | ICML 2025 |
| q_023 | multi | 9 | 3 | 10 | 1, 3, 4, 5, -, -, -, -, - | 0.44 | 0.44 | 0.50 | 0.31 | table row_f1:0.33 cell:0.33 | CVPR 2025 |
| q_024 | single | 1 | 1 | 8 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× | NeurIPS |
| q_025 | multi | 4 | 4 | 22 | -, 4, -, 5 | 0.50 | 0.50 | 0.25 | 0.00 | ff:× / table row_f1:0.00 cell:0.00 |  |
| q_026 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_027 | single | 1 | 1 | 8 | 1 | 1.00 | 1.00 | 1.00 | 0.00 | ff:× / table row_f1:0.00 cell:0.00 | CVPR 2025 |
| q_028 | multi | 4 | 2 | 12 | 1, 11, 2, 4 | 0.75 | 1.00 | 0.00 | 0.00 | table row_f1:0.75 cell:0.75 |  |
| q_029 | multi | 4 | 1 | 11 | 2, 5, 8, 11 | 0.75 | 1.00 | 0.40 | 0.40 | table row_f1:0.75 cell:0.75 |  |
| q_030 | multi | 4 | 1 | 11 | 2, 3, 4, 1 | 1.00 | 1.00 | 0.40 | 0.00 | table row_f1:1.00 cell:0.00 |  |
| q_031 | multi | 4 | 1 | 8 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_032 | multi | 4 | 1 | 17 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_033 | multi | 4 | 1 | 4 | 1, 4, -, - | 0.50 | 0.50 | 0.40 | 0.00 | mc:× |  |
| q_034 | multi | 4 | 1 | 1 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_035 | multi | 4 | 1 | 20 | -, 1, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_036 | multi | 4 | 1 | 9 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.40 | mc:× |  |
| q_037 | multi | 4 | 1 | 14 | -, 3, 1, - | 0.50 | 0.50 | 0.40 | 0.00 | mc:× |  |
| q_038 | multi | 4 | 1 | 1 | 1, -, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_039 | multi | 4 | 1 | 32 | 17, -, -, 1 | 0.25 | 0.50 | 0.40 | 0.00 | mc:× |  |
| q_040 | multi | 4 | 1 | 36 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_041 | multi | 4 | 2 | 37 | 6, 1, 12, - | 0.50 | 0.75 | 0.67 | 0.00 | mc:× |  |
| q_042 | multi | 4 | 2 | 11 | -, 1, -, 3 | 0.50 | 0.50 | 0.67 | 0.00 | mc:× |  |
| q_043 | multi | 4 | 2 | 3 | 1, 2, 3, - | 0.75 | 0.75 | 0.67 | 0.33 | mc:× |  |
| q_044 | multi | 4 | 2 | 33 | 11, 2, 1, - | 0.50 | 0.75 | 0.67 | 0.29 | mc:× |  |
| q_045 | multi | 4 | 2 | 39 | 18, -, 1, 24 | 0.25 | 0.75 | 0.67 | 0.00 | mc:× |  |
| q_046 | multi | 4 | 1 | 10 | 1, -, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_047 | multi | 4 | 1 | 45 | 27, 1, 4, - | 0.50 | 0.75 | 0.40 | 0.00 | mc:× |  |
| q_048 | multi | 4 | 1 | 10 | 1, -, -, - | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_049 | multi | 4 | 1 | 4 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_050 | multi | 4 | 1 | 2 | -, -, 1, - | 0.25 | 0.25 | 0.40 | 0.40 | mc:× |  |
| q_051 | multi | 4 | 1 | 41 | -, -, -, 1 | 0.25 | 0.25 | 0.40 | 0.00 | mc:× |  |
| q_052 | single | 1 | 1 | 34 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:1.00 cell:0.67 |  |
| q_053 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_054 | single | 1 | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:0.00 cell:0.00 |  |
| q_055 | single | 1 | 1 | 24 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_056 | multi | 4 | 2 | 23 | 7, 3, 4, 1 | 1.00 | 1.00 | 0.67 | 0.29 | table row_f1:0.75 cell:0.00 |  |

## コメント

ベースライン比では paper_precision が 0.678→0.873、paper_f1 も 0.513→0.636 と大きく改善しており、paper_recall はほぼ横ばいなので、全体としてかなり良い結果です。candidate recall は single では @1 と @10 が改善し、multi も概ね同程度ですが、multi/total の @50 がやや悪化しているため、広めに拾う場面での取りこぼしが少し気になります。evidence 系は precision/recall/f1 が大きく伸び、table 指標も改善していますが、絶対値としては evidence_f1=0.206 とまだ低く、根拠抽出がボトルネックです。次は attrfilter の厳しさや候補数上限を見直して multi-hop の候補網羅性を改善しつつ、reranker 後の根拠文抽出・読解プロンプトを重点的に詰めるのがよさそうです。

## candidate_recall（後から追記）

予測ファイル `predictions_8b_paper_attrfilter.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 45 本なので、それを超える k は @45 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.5182 |
| candidate_recall_at70_total_macro | 0.7460 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 0.9615 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.4090 |
| evidence_candidate_recall_at5_multi_macro | 0.5872 |
| evidence_candidate_recall_at10_multi_macro | 0.6130 |
| evidence_candidate_recall_at20_multi_macro | 0.6705 |
| evidence_candidate_recall_at50_multi_macro | 0.6906 |
| evidence_candidate_recall_at70_multi_macro | 0.6906 |
| evidence_candidate_recall_at1_total_macro | 0.6702 |
| evidence_candidate_recall_at5_total_macro | 0.7641 |
| evidence_candidate_recall_at10_total_macro | 0.7960 |
| evidence_candidate_recall_at20_total_macro | 0.8263 |
| evidence_candidate_recall_at50_total_macro | 0.8369 |
| evidence_candidate_recall_at70_total_macro | 0.8369 |

<!-- candidate_recall backfill: pred=predictions_8b_paper_attrfilter.jsonl, max_candidates=45 -->
