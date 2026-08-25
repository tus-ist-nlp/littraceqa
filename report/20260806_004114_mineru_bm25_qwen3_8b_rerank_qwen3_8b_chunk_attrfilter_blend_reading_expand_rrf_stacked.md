# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_blend + reading_expand_rrf_stacked

- 実行日時: 2026-08-06T00:41:14
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_blend.yaml`
- agent: `configs/agent_style/reading_expand_rrf/stacked.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_stacked_blend1000.jsonl`
- git: `a0206042354c`

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
| agent_retrieve_top_k | `100` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |
| agent_subquery_merge | `"rrf"` |
| agent_subquery_rrf_k | `60` |
| agent_grounded_refine | `true` |
| agent_grounded_refine_top_n | `10` |
| agent_adaptive_depth | `{"enabled": true, "probe_rank": 4, "gap_threshold": 0.15, "shallow_k": 10, "deep_k": 100}` |
| agent_title_protect | `{"enabled": true, "max_papers": 4, "promote_to": 10, "chunks": "/data2/iseakira/pdfs/chunks/mineru_chunks.jsonl", "cache_path": "/data2/iseakira/pdfs/index/mineru/relations/mentions.pkl"}` |
| agent_pool_rescore | `false` |

## 指標

| 指標 | 値 |
|---|---|
| candidate_recall_at1_single_macro | 0.9615 |
| candidate_recall_at5_single_macro | 0.9615 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1762 |
| candidate_recall_at5_multi_macro | 0.4962 |
| candidate_recall_at10_multi_macro | 0.6705 |
| candidate_recall_at20_multi_macro | 0.7644 |
| candidate_recall_at50_multi_macro | 0.8534 |
| candidate_recall_at70_multi_macro | 0.8534 |
| candidate_recall_at1_total_macro | 0.5475 |
| candidate_recall_at5_total_macro | 0.7162 |
| candidate_recall_at10_total_macro | 0.8263 |
| candidate_recall_at20_total_macro | 0.8758 |
| candidate_recall_at50_total_macro | 0.9227 |
| candidate_recall_at70_total_macro | 0.9227 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 0.9615 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3602 |
| evidence_candidate_recall_at5_multi_macro | 0.5996 |
| evidence_candidate_recall_at10_multi_macro | 0.7567 |
| evidence_candidate_recall_at20_multi_macro | 0.8190 |
| evidence_candidate_recall_at50_multi_macro | 0.8851 |
| evidence_candidate_recall_at70_multi_macro | 0.8851 |
| evidence_candidate_recall_at1_total_macro | 0.6444 |
| evidence_candidate_recall_at5_total_macro | 0.7707 |
| evidence_candidate_recall_at10_total_macro | 0.8717 |
| evidence_candidate_recall_at20_total_macro | 0.9045 |
| evidence_candidate_recall_at50_total_macro | 0.9394 |
| evidence_candidate_recall_at70_total_macro | 0.9394 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.9688 |
| evidence_candidate_recall_by_backed_at5_single_macro | 0.9688 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.1932 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.4952 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.6932 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.7717 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.8551 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.8551 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6444 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.7707 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.8717 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9045 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9394 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9394 |

## コメント

全体として single 系は非常に良く、candidate/evidence ともに@10でほぼ 1.0 に到達しているため、単一正解の取りこぼしはかなり少ないです。 一方で multi 系は弱く、candidate_recall_at1_multi=0.176、@5でも0.496に留まっており、複数根拠を含む質問で上位順位に必要文書を揃え切れていません。 evidence 系は candidate 系より multi でやや良いので、文書自体は拾えているが順位付けや多様性確保に課題がある可能性があります。 次は multi クエリ向けに取得件数拡大後の多様化（MMR/RRF 重み調整、属性フィルタの緩和、chunk 粒度見直し）や、reranker を single 最適化から複数根拠カバレッジ重視に寄せる設定を試すとよさそうです。
