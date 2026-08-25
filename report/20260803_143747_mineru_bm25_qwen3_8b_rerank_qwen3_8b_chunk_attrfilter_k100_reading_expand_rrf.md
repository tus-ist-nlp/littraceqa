# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_rrf

- 実行日時: 2026-08-03T14:37:47
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_rrf.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_k100_rrf.jsonl`
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
| paper_precision_macro | 0.7958 |
| paper_recall_macro | 0.6566 |
| paper_f1_macro | 0.5975 |
| candidate_recall_at1_single_macro | 1.0000 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2050 |
| candidate_recall_at5_multi_macro | 0.5048 |
| candidate_recall_at10_multi_macro | 0.6466 |
| candidate_recall_at20_multi_macro | 0.7749 |
| candidate_recall_at50_multi_macro | 0.8707 |
| candidate_recall_at70_multi_macro | 0.8707 |
| candidate_recall_at1_total_macro | 0.5808 |
| candidate_recall_at5_total_macro | 0.7389 |
| candidate_recall_at10_total_macro | 0.8136 |
| candidate_recall_at20_total_macro | 0.8813 |
| candidate_recall_at50_total_macro | 0.9318 |
| candidate_recall_at70_total_macro | 0.9318 |
| evidence_candidate_recall_at1_single_macro | 1.0000 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3831 |
| evidence_candidate_recall_at5_multi_macro | 0.6398 |
| evidence_candidate_recall_at10_multi_macro | 0.7500 |
| evidence_candidate_recall_at20_multi_macro | 0.8554 |
| evidence_candidate_recall_at50_multi_macro | 0.9368 |
| evidence_candidate_recall_at70_multi_macro | 0.9368 |
| evidence_candidate_recall_at1_total_macro | 0.6747 |
| evidence_candidate_recall_at5_total_macro | 0.8101 |
| evidence_candidate_recall_at10_total_macro | 0.8682 |
| evidence_candidate_recall_at20_total_macro | 0.9237 |
| evidence_candidate_recall_at50_total_macro | 0.9667 |
| evidence_candidate_recall_at70_total_macro | 0.9667 |
| evidence_precision_macro | 0.2144 |
| evidence_recall_macro | 0.2035 |
| evidence_f1_macro | 0.1929 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.2692 |
| table_row_f1_macro | 0.5049 |
| table_cell_accuracy_macro | 0.3003 |
| table_cell_accuracy_micro | 0.2963 |

## コメント

候補取得はかなり強く、single では paper/evidence ともに recall@1〜70 が 1.0、total でも recall@20 が高いため、検索段階での取りこぼしは比較的小さいです。一方で最終的な paper_f1_macro=0.598、evidence_f1_macro=0.193 と特に根拠抽出の精度・再現率が低く、取得した候補を回答・根拠に変換する段階が主なボトルネックに見えます。multiple_choice_accuracy=0.0 と freeform_exact_match=0.269 も気になり、reader/aggregation の不安定さや出力形式の不一致を疑いたいです。次は multi 系で recall の伸びがまだ大きいので rerank 後の上位件数や RRF の重み、根拠抽出プロンプト／chunk 粒度を見直しつつ、MC 問題のフォーマット遵守エラーを優先的に誤り分析するとよさそうです。
