# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_rrf_rewrite

- 実行日時: 2026-08-06T15:39:48
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_rrf/rewrite.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_k100_rewrite.jsonl`
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
| candidate_recall_at1_single_macro | 0.8077 |
| candidate_recall_at5_single_macro | 0.9615 |
| candidate_recall_at10_single_macro | 0.9615 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1877 |
| candidate_recall_at5_multi_macro | 0.4904 |
| candidate_recall_at10_multi_macro | 0.6753 |
| candidate_recall_at20_multi_macro | 0.8161 |
| candidate_recall_at50_multi_macro | 0.8793 |
| candidate_recall_at70_multi_macro | 0.8793 |
| candidate_recall_at1_total_macro | 0.4808 |
| candidate_recall_at5_total_macro | 0.7131 |
| candidate_recall_at10_total_macro | 0.8106 |
| candidate_recall_at20_total_macro | 0.9030 |
| candidate_recall_at50_total_macro | 0.9364 |
| candidate_recall_at70_total_macro | 0.9364 |
| evidence_candidate_recall_at1_single_macro | 0.8077 |
| evidence_candidate_recall_at5_single_macro | 0.9615 |
| evidence_candidate_recall_at10_single_macro | 0.9615 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3630 |
| evidence_candidate_recall_at5_multi_macro | 0.6025 |
| evidence_candidate_recall_at10_multi_macro | 0.7586 |
| evidence_candidate_recall_at20_multi_macro | 0.8879 |
| evidence_candidate_recall_at50_multi_macro | 0.9511 |
| evidence_candidate_recall_at70_multi_macro | 0.9511 |
| evidence_candidate_recall_at1_total_macro | 0.5732 |
| evidence_candidate_recall_at5_total_macro | 0.7722 |
| evidence_candidate_recall_at10_total_macro | 0.8545 |
| evidence_candidate_recall_at20_total_macro | 0.9409 |
| evidence_candidate_recall_at50_total_macro | 0.9742 |
| evidence_candidate_recall_at70_total_macro | 0.9742 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.8438 |
| evidence_candidate_recall_by_backed_at5_single_macro | 0.9688 |
| evidence_candidate_recall_by_backed_at10_single_macro | 0.9688 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.1969 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.4988 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.6957 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.8587 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.9384 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.9384 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.5732 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.7722 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.8545 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9409 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9742 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9742 |

## コメント

全体として single 系は非常に良好で、candidate/evidence  専用のクエリ分解・補助検索を試すとよさそうです。ともに@20でほぼ頭打ち、single では @20 で 1.0 に到達しており上位想起は強いです。一方で multi 系は弱く、candidate_recall_at1_multi_macro=0.188、@10でも0.675にとどまっていて、複数根拠が必要な質問で取りこぼしが目立ちます。evidence 系が candidate 系をやや上回っているので候補の質自体は悪くないですが、上位順位への寄せ方や multi-hop 的な網羅性に改善余地があります。次は rewrite の多様化強化、chunk 粒度や attrfilter の緩和、rerank 前の取得件数拡大や multi
