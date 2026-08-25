# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_rrf_verdict_anchor

- 実行日時: 2026-08-08T20:16:06
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_rrf/verdict_anchor.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_k100_verdict_anchor.jsonl`
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
| candidate_recall_at1_single_macro | 0.8846 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |

| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.5536 |
| candidate_recall_at10_multi_macro | 0.7548 |
| candidate_recall_at20_multi_macro | 0.8391 |
| candidate_recall_at50_multi_macro | 0.9224 |

| candidate_recall_at1_total_macro | 0.5247 |
| candidate_recall_at5_total_macro | 0.7646 |
| candidate_recall_at10_total_macro | 0.8707 |
| candidate_recall_at20_total_macro | 0.9152 |
| candidate_recall_at50_total_macro | 0.9591 |

| evidence_candidate_recall_at1_single_macro | 0.8846 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |

| evidence_candidate_recall_at1_multi_macro | 0.3803 |
| evidence_candidate_recall_at5_multi_macro | 0.6715 |
| evidence_candidate_recall_at10_multi_macro | 0.8496 |
| evidence_candidate_recall_at20_multi_macro | 0.9080 |
| evidence_candidate_recall_at50_multi_macro | 0.9626 |
| evidence_candidate_recall_at70_multi_macro | 0.9626 |
| evidence_candidate_recall_at1_total_macro | 0.6187 |
| evidence_candidate_recall_at5_total_macro | 0.8268 |
| evidence_candidate_recall_at10_total_macro | 0.9207 |
| evidence_candidate_recall_at20_total_macro | 0.9515 |
| evidence_candidate_recall_at50_total_macro | 0.9803 |
| evidence_candidate_recall_at70_total_macro | 0.9803 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.9062 |
| evidence_candidate_recall_by_backed_at5_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.2186 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.5857 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.8104 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.8841 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.9529 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.9529 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6187 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.8268 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.9207 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9515 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9803 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9803 |

## コメント

全体として単一正解系は非常に強く、candidate/evidence ともに @5 で 1.0 に到達しており、上位少数件でほぼ取り切れています。 一方で multi 系は弱く、candidate_recall@1=0.20、@5=0.55 と初段の取りこぼしが大きく、複数根拠を要する問い合わせで順位付けや拡張の不足が見えます。 evidence 側は multi でも @10=0.85、@50=0.96 まで伸びるので、候補生成自体はある程度できており、最終的には上位への押し上げが課題です。 次は multi クエリ向けにクエリ展開の強化、chunk 粒度や attrfilter の見直し、rerank の多様性確保（重複抑制や coverage 重視）を試すとよさそうです。
