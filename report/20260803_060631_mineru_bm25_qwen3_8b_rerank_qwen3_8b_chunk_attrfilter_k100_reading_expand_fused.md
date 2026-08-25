# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_fused

- 実行日時: 2026-08-03T06:06:31
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_fused.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_k100_expand_fused.jsonl`
- git: `a0206042354c`

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `100` |
| pool_k | `200` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| fuser_k | `60` |
| fuser_weights | `{"bm25s": 1.0, "faiss_qwen3": 1.0}` |
| reranker | `"qwen3"` |
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_devices | `"cuda:1,cuda:2"` |
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
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7745 |
| paper_recall_macro | 0.6495 |
| paper_f1_macro | 0.5899 |
| candidate_recall_at1_single_macro | 0.8462 |
| candidate_recall_at5_single_macro | 0.9231 |
| candidate_recall_at10_single_macro | 0.9615 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1964 |
| candidate_recall_at5_multi_macro | 0.3927 |
| candidate_recall_at10_multi_macro | 0.4866 |
| candidate_recall_at20_multi_macro | 0.6667 |
| candidate_recall_at50_multi_macro | 0.7672 |
| candidate_recall_at70_multi_macro | 0.8017 |
| candidate_recall_at1_total_macro | 0.5035 |
| candidate_recall_at5_total_macro | 0.6434 |
| candidate_recall_at10_total_macro | 0.7111 |
| candidate_recall_at20_total_macro | 0.8242 |
| candidate_recall_at50_total_macro | 0.8773 |
| candidate_recall_at70_total_macro | 0.8955 |
| evidence_candidate_recall_at1_single_macro | 0.8462 |
| evidence_candidate_recall_at5_single_macro | 0.9231 |
| evidence_candidate_recall_at10_single_macro | 0.9615 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3716 |
| evidence_candidate_recall_at5_multi_macro | 0.5766 |
| evidence_candidate_recall_at10_multi_macro | 0.6561 |
| evidence_candidate_recall_at20_multi_macro | 0.7672 |
| evidence_candidate_recall_at50_multi_macro | 0.8592 |
| evidence_candidate_recall_at70_multi_macro | 0.8707 |
| evidence_candidate_recall_at1_total_macro | 0.5960 |
| evidence_candidate_recall_at5_total_macro | 0.7404 |
| evidence_candidate_recall_at10_total_macro | 0.8005 |
| evidence_candidate_recall_at20_total_macro | 0.8773 |
| evidence_candidate_recall_at50_total_macro | 0.9258 |
| evidence_candidate_recall_at70_total_macro | 0.9318 |
| evidence_precision_macro | 0.2094 |
| evidence_recall_macro | 0.2101 |
| evidence_f1_macro | 0.2005 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.5279 |
| table_cell_accuracy_macro | 0.3249 |
| table_cell_accuracy_micro | 0.4074 |

## クエリ診断

- 候補上位10本に gold が1本も入らなかった: **1件** (q_021)
- 一部しか入らなかった(multi の取りこぼし): **26件** (q_022, q_023, q_025, q_028, q_029, q_031, q_032, q_033, q_034, q_035, q_036, q_037, q_038, q_039, q_040, q_041, q_042, q_043, q_044, q_045, q_046, q_047, q_048, q_049, q_050, q_051)

`gold順位` は候補列(`candidate_papers`)で gold 論文が何位だったか。`-` は候補に入っていない。`cr@k` はその上位k本での recall。`gold` の括弧内は evidence が紐づいている本数で、`ecr@k` はそこだけを分母にした recall（取りに行ける gold だけの検索力）。

| query_id | family | gold | 提出 | 候補 | gold順位 | cr@10 | ecr@10 | cr@50 | ecr@50 | paper_f1 | ev_f1 | 回答 | attrfilter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_001 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_002 | single | 1 | 10 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:× |  |
| q_003 | single | 1 | 10 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_004 | single | 1 | 10 | 68 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_005 | single | 1 | 10 | 57 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_006 | single | 1 | 1 | 26 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_007 | single | 1 | 10 | 70 | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_008 | single | 1 | 1 | 49 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_009 | single | 1 | 1 | 32 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_010 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_011 | single | 1 | 1 | 26 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_012 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:○ |  |
| q_013 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_014 | single | 1 | 1 | 38 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_015 | single | 1 | 1 | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_017 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_018 | single | 1 | 1 | 67 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_019 | single | 1 | 10 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_020 | multi | 4 | 5 | 46 | 7, 2, 4, 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.22 | 0.18 | ff:× / table row_f1:0.22 cell:0.22 | NAACL 2025 |
| q_021 | single | 1 | 2 | 53 | 13 | 0.00 | 0.00 | 1.00 | 1.00 | 0.67 | 0.33 | mc:× / ff:× |  |
| q_022 | multi | 3 | 1 | 70 | 22, 40, 1 | 0.33 | 0.33 | 1.00 | 1.00 | 0.50 | 0.00 | table row_f1:0.40 cell:0.00 | ICML 2025 |
| q_023 | multi | 9 | 8 | 38 | 3, 4, 12, 2, 1, 7, 11, 10, 5 | 0.78 | 0.78 | 1.00 | 1.00 | 0.94 | 0.59 | table row_f1:0.82 cell:0.82 | CVPR 2025 |
| q_024 | single | 1 | 1 | 70 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× | NeurIPS |
| q_025 | multi | 4 | 9 | 70 | 2, 4, -, 12 | 0.50 | 0.50 | 0.75 | 0.75 | 0.46 | 0.00 | ff:× / table row_f1:0.11 cell:0.11 |  |
| q_026 | single | 1 | 1 | 38 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_027 | single | 1 | 1 | 48 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | ff:× / table row_f1:0.00 cell:0.00 | CVPR 2025 |
| q_028 | multi | 4 | 3 | 70 | 1, 11, 5, 4 | 0.75 | 0.75 | 1.00 | 1.00 | 0.57 | 0.29 | table row_f1:0.75 cell:0.75 |  |
| q_029 | multi | 4 | 2 | 70 | 6, 7, 3, 30 | 0.75 | 0.75 | 1.00 | 1.00 | 0.67 | 0.57 | table row_f1:0.75 cell:0.75 |  |
| q_030 | multi | 4 | 3 | 61 | 3, 4, 2, 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.00 | table row_f1:1.00 cell:0.25 |  |
| q_031 | multi | 4 (3) | 1 | 23 | -, 4, 1, 14 | 0.50 | 0.67 | 0.75 | 0.67 | 0.40 | 0.00 | mc:× |  |
| q_032 | multi | 4 (1) | 1 | 48 | -, 18, 1, 19 | 0.25 | 1.00 | 0.75 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_033 | multi | 4 (1) | 1 | 41 | 1, 3, 7, 31 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_034 | multi | 4 (1) | 1 | 36 | 10, 16, 19, 1 | 0.50 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_035 | multi | 4 | 1 | 54 | 30, 1, 22, 17 | 0.25 | 0.25 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_036 | multi | 4 | 1 | 48 | -, 16, 1, 19 | 0.25 | 0.25 | 0.75 | 0.75 | 0.40 | 0.40 | mc:× |  |
| q_037 | multi | 4 (1) | 10 | 70 | 58, 62, 1, 67 | 0.25 | 1.00 | 0.25 | 1.00 | 0.14 | 0.00 | mc:× |  |
| q_038 | multi | 4 (1) | 1 | 24 | 1, -, -, - | 0.25 | 1.00 | 0.25 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_039 | multi | 4 (3) | 1 | 70 | -, -, -, 4 | 0.25 | 0.33 | 0.25 | 0.33 | 0.40 | 0.00 | mc:× |  |
| q_040 | multi | 4 (1) | 1 | 70 | 35, 16, 17, 1 | 0.25 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_041 | multi | 4 (3) | 1 | 43 | 4, 2, 8, 24 | 0.75 | 0.67 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_042 | multi | 4 (2) | 2 | 60 | -, 2, -, 3 | 0.50 | 1.00 | 0.50 | 1.00 | 0.67 | 0.00 | mc:× |  |
| q_043 | multi | 4 | 2 | 46 | 1, 3, 4, 20 | 0.75 | 0.75 | 1.00 | 1.00 | 0.67 | 0.33 | mc:× |  |
| q_044 | multi | 4 | 2 | 70 | 6, 2, 1, - | 0.75 | 0.75 | 0.75 | 0.75 | 0.67 | 0.00 | mc:× |  |
| q_045 | multi | 4 | 1 | 70 | 8, 13, 11, 25 | 0.25 | 0.25 | 1.00 | 1.00 | 0.00 | 0.00 | mc:× |  |
| q_046 | multi | 4 (3) | 1 | 68 | 1, 19, 43, 17 | 0.25 | 0.33 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_047 | multi | 4 (3) | 1 | 28 | -, 1, 15, - | 0.25 | 0.33 | 0.50 | 0.67 | 0.40 | 0.00 | mc:× |  |
| q_048 | multi | 4 (3) | 1 | 53 | 1, -, -, 42 | 0.25 | 0.33 | 0.50 | 0.67 | 0.40 | 0.00 | mc:× |  |
| q_049 | multi | 4 (3) | 1 | 70 | -, -, -, 1 | 0.25 | 0.33 | 0.25 | 0.33 | 0.00 | 0.00 | mc:× |  |
| q_050 | multi | 4 (3) | 10 | 70 | 58, 17, 1, - | 0.25 | 0.33 | 0.50 | 0.67 | 0.14 | 0.00 | mc:× |  |
| q_051 | multi | 4 (3) | 1 | 70 | -, 20, -, 1 | 0.25 | 0.33 | 0.50 | 0.33 | 0.40 | 0.00 | mc:× |  |
| q_052 | single | 1 | 1 | 49 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:1.00 cell:0.67 |  |
| q_053 | single | 1 | 1 | 32 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_054 | single | 1 | 1 | 30 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:0.00 cell:0.00 |  |
| q_055 | single | 1 | 10 | 70 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_056 | multi | 4 | 1 | 63 | 7, 6, 1, 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | table row_f1:0.75 cell:0.00 |  |

## コメント

候補段階は比較的良好で、single では candidate/evidence recall@20 が 1.0、total でも evidence recall@10 が 0.80 と、関連文献を拾う力はかなりあります。一方で最終的な paper F1=0.590 は precision 0.775 に対して recall 0.649 がやや弱く、evidence F1=0.200・multiple_choice_accuracy=0.0 から、候補以降の絞り込みや根拠抽出、回答生成で大きく落としていそうです。特に multi 系は candidate recall@10=0.487、evidence recall@10=0.656 と single よりかなり低く、複数文献・複数根拠が必要な問題への弱さが目立ちます。次は attrfilter の閾値緩和や top-k 拡大、multi-hop/複数根拠向けの rerank・fuse 改善、加えて evidence 抽出プロンプトや chunk 粒度の見直しを試すのがよさそうです。
