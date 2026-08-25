# mineru + bm25_qwen3_8b_rerank_qwen3_8b_k100_external_all + reading_expand_rrf_notable

- 実行日時: 2026-08-15T11:21:39
- paths: `configs/paths/nlp02.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_external_all.yaml`
- agent: `configs/agent_style/reading_expand_rrf/notable.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_val_external_all.jsonl`
- git: `07d286054731`

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `100` |
| pool_k | `200` |
| indexers | `["bm25s", "bm25s_paper", "faiss_qwen3"]` |
| fuser | `"paper_rrf"` |
| fuser_k | `60` |
| fuser_chunks_per_paper | `3` |
| fuser_weights | `{"bm25s": 1.0, "bm25s_paper": 1.0, "faiss_qwen3": 1.0}` |
| reranker | `"qwen3"` |
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_devices | `"cuda:3"` |
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
| agent_paper_score_skip_chunk_types | `["table"]` |
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| candidate_recall_at1_single_macro | 0.8462 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2222 |
| candidate_recall_at5_multi_macro | 0.5852 |
| candidate_recall_at10_multi_macro | 0.7462 |
| candidate_recall_at20_multi_macro | 0.8362 |
| candidate_recall_at50_multi_macro | 0.9397 |
| candidate_recall_at70_multi_macro | 0.9397 |
| candidate_recall_at1_total_macro | 0.5172 |
| candidate_recall_at5_total_macro | 0.7813 |
| candidate_recall_at10_total_macro | 0.8662 |
| candidate_recall_at20_total_macro | 0.9136 |
| candidate_recall_at50_total_macro | 0.9682 |
| candidate_recall_at70_total_macro | 0.9682 |
| evidence_candidate_recall_at1_single_macro | 0.8462 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.4090 |
| evidence_candidate_recall_at5_multi_macro | 0.7117 |
| evidence_candidate_recall_at10_multi_macro | 0.8467 |
| evidence_candidate_recall_at20_multi_macro | 0.9080 |
| evidence_candidate_recall_at50_multi_macro | 0.9799 |
| evidence_candidate_recall_at70_multi_macro | 0.9799 |
| evidence_candidate_recall_at1_total_macro | 0.6157 |
| evidence_candidate_recall_at5_total_macro | 0.8480 |
| evidence_candidate_recall_at10_total_macro | 0.9192 |
| evidence_candidate_recall_at20_total_macro | 0.9515 |
| evidence_candidate_recall_at50_total_macro | 0.9894 |
| evidence_candidate_recall_at70_total_macro | 0.9894 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.8750 |
| evidence_candidate_recall_by_backed_at5_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.2548 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.6365 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.8068 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.8841 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.9746 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.9746 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6157 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.8480 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.9192 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9515 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9894 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9894 |

## コメント

全体としてはかなり良く、single は candidate/evidence ともに@5で 1.0 に到達しており、単一正解クエリには非常に強いです。 一方で multi は @1 が低く、candidate_recall_at1_multi_macro=0.222、evidence_candidate_recall_at1_multi_macro=0.409 なので、複数文書・複数根拠が必要なケースで上位順位づけにまだ課題があります。 @20 以降では multi も 0.84〜0.91 程度まで伸び、@50 では 0.94〜0.98 と十分高いため、取りこぼしというより rerank や RRF 後の並び順最適化が主な改善ポイントに見えます。 次は multi クエリだけを切り出して失敗例を確認し、query expansion の増やし方、multi-hop 向けの reranker 強化、あるいは top-k を 10〜20 程度まで使う設定での downstream 影響を比較するとよさそうです。
