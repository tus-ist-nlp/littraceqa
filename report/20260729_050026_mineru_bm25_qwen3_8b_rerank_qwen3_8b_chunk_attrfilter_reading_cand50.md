# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_cand50

- 実行日時: 2026-07-29T05:00:26
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_cand50.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_cand50.jsonl`
- git: `f53e1da3a4b5`

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
| agent_max_candidates | `50` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7555 |
| paper_recall_macro | 0.6505 |
| paper_f1_macro | 0.5522 |
| candidate_recall_at1_single_macro | 1.0000 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1935 |
| candidate_recall_at5_multi_macro | 0.4013 |
| candidate_recall_at10_multi_macro | 0.5345 |
| candidate_recall_at20_multi_macro | 0.6006 |
| candidate_recall_at50_multi_macro | 0.6810 |
| candidate_recall_at1_total_macro | 0.5747 |
| candidate_recall_at5_total_macro | 0.6843 |
| candidate_recall_at10_total_macro | 0.7545 |
| candidate_recall_at20_total_macro | 0.7894 |
| candidate_recall_at50_total_macro | 0.8318 |
| evidence_candidate_recall_at1_single_macro | 1.0000 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3716 |
| evidence_candidate_recall_at5_multi_macro | 0.5680 |
| evidence_candidate_recall_at10_multi_macro | 0.6897 |
| evidence_candidate_recall_at20_multi_macro | 0.7500 |
| evidence_candidate_recall_at50_multi_macro | 0.7902 |
| evidence_candidate_recall_at1_total_macro | 0.6687 |
| evidence_candidate_recall_at5_total_macro | 0.7722 |
| evidence_candidate_recall_at10_total_macro | 0.8364 |
| evidence_candidate_recall_at20_total_macro | 0.8682 |
| evidence_candidate_recall_at50_total_macro | 0.8894 |
| evidence_precision_macro | 0.1840 |
| evidence_recall_macro | 0.1556 |
| evidence_f1_macro | 0.1583 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1538 |
| table_row_f1_macro | 0.4766 |
| table_cell_accuracy_macro | 0.3457 |
| table_cell_accuracy_micro | 0.4444 |

## candidate_recall（後から追記）

予測ファイル `predictions_8b_chunk_cand50.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 50 本なので、それを超える k は @50 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.6810 |
| candidate_recall_at70_total_macro | 0.8318 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.7902 |
| evidence_candidate_recall_at70_total_macro | 0.8894 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_cand50.jsonl, max_candidates=50 -->
