# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_expand（max_candidates=50, オフライン適用）

- 実行日時: 2026-08-02T12:03:35
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_expand.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_cand50_expand_offline.jsonl`
- git: `a0206042354c`
- **オフライン適用**: 土台 `predictions_8b_chunk_cand50.jsonl`（7/29 実行、reading_cand50.yaml =
  retrieve_top_k/max_candidates ともに本 yaml と同値）の candidate_papers に、
  `Specter2PaperExpander(neighbors=20, anchors=1)` を `reading.py` と同じ規則
  （挿入位置 = `min(max_candidates, 候補数)`）で適用。検索と LLM は再実行していない。

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
| reranker_max_tokens | `2048` |
| attribute_filter_enabled | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_retrieve_top_k | `50` |
| agent_max_candidates | `50` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |
| expansion_index | `"faiss_specter2_abstract"` |
| expansion_neighbors | `20` |
| expansion_anchors | `1` |
| expansion_inserted_at | `50` |
| expansion_added_per_query_avg | `15.5` |

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
| candidate_recall_at100_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1935 |
| candidate_recall_at5_multi_macro | 0.4013 |
| candidate_recall_at10_multi_macro | 0.5345 |
| candidate_recall_at20_multi_macro | 0.6006 |
| candidate_recall_at50_multi_macro | 0.6897 |
| candidate_recall_at100_multi_macro | 0.8103 |
| candidate_recall_at1_total_macro | 0.5747 |
| candidate_recall_at5_total_macro | 0.6843 |
| candidate_recall_at10_total_macro | 0.7545 |
| candidate_recall_at20_total_macro | 0.7894 |
| candidate_recall_at50_total_macro | 0.8364 |
| candidate_recall_at100_total_macro | 0.9000 |
| evidence_candidate_recall_at1_single_macro | 1.0000 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at100_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3716 |
| evidence_candidate_recall_at5_multi_macro | 0.5680 |
| evidence_candidate_recall_at10_multi_macro | 0.6897 |
| evidence_candidate_recall_at20_multi_macro | 0.7500 |
| evidence_candidate_recall_at50_multi_macro | 0.7902 |
| evidence_candidate_recall_at100_multi_macro | 0.8966 |
| evidence_candidate_recall_at1_total_macro | 0.6687 |
| evidence_candidate_recall_at5_total_macro | 0.7722 |
| evidence_candidate_recall_at10_total_macro | 0.8364 |
| evidence_candidate_recall_at20_total_macro | 0.8682 |
| evidence_candidate_recall_at50_total_macro | 0.8894 |
| evidence_candidate_recall_at100_total_macro | 0.9455 |
| evidence_precision_macro | 0.1840 |
| evidence_recall_macro | 0.1556 |
| evidence_f1_macro | 0.1583 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1538 |
| table_row_f1_macro | 0.4766 |
| table_cell_accuracy_macro | 0.3457 |
| table_cell_accuracy_micro | 0.4444 |

## 比較

| 指標 | max_cand=20 | max_cand=20+展開 | max_cand=50 | **max_cand=50+展開（本実行）** |
|---|---|---|---|---|
| paper_precision_macro | 0.7824 | 0.7824 | 0.7555 | **0.7555** |
| paper_recall_macro | 0.6475 | 0.6475 | 0.6505 | **0.6505** |
| paper_f1_macro | 0.5841 | 0.5841 | 0.5522 | **0.5522** |
| candidate_recall_at20_total_macro | 0.7833 | 0.7879 | 0.7894 | **0.7894** |
| candidate_recall_at50_total_macro | 0.8364 | 0.8803 | 0.8318 | **0.8364** |
| candidate_recall_at100_total_macro | 0.8364 | 0.9136 | 0.8318 | **0.9000** |
| candidate_recall_at20_multi_macro | 0.5891 | 0.5977 | 0.6006 | **0.6006** |
| candidate_recall_at50_multi_macro | 0.6897 | 0.7730 | 0.6810 | **0.6897** |
| evidence_candidate_recall_at50_total_macro | 0.9076 | 0.9242 | 0.8894 | **0.8894** |
| evidence_candidate_recall_at100_total_macro | 0.9076 | 0.9561 | 0.8894 | **0.9455** |
| evidence_f1_macro | 0.2205 | 0.2205 | 0.1583 | **0.1583** |

## 所見（この構成は狙いを達成していない）

**1. max_candidates を上げても LLM は展開論文を読まない。**
展開は `_build_prediction()`（反復ループが終わった後）で candidate_papers に適用されるため、
`max_candidates` の値に関わらず LLM の読解対象には最初から入らない。「LLM が読む範囲を広げれば
展開分が可視域に入る」という想定は、現在の実装では成り立たない。

**2. max_candidates=50 にすると挿入位置が50位へ後退し、展開の効果が cr@50 から cr@100 に逃げる。**
挿入規則が `min(max_candidates, 候補数)` のため。実測で cr@50 は 0.832 -> 0.836 とほぼ動かず、
効果は cr@100 0.832 -> 0.900 に現れた。max_candidates=20 の構成なら同じ展開で
cr@50 が 0.836 -> 0.880 まで伸びていた（そちらが優る）。

**3. 展開と無関係に、max_candidates=50 自体が paper_f1 を下げている。**
paper_f1 0.5841(20本) -> 0.5522(50本)。recall は +0.0030 だが
precision が 0.0270 落ちる。読ませる本数を増やすと選定が緩くなる。

**結論: 提出指標の観点では max_candidates=20 + 展開（cr@50 0.880）を採用すべきで、本構成は不採用。**

## 次に検証すべきこと

- 展開論文を LLM に読ませるには、展開をループ内（`_read_and_judge` の前）へ移し、
  paper_id からチャンク本文を取得して候補に混ぜる実装が要る（config では届かない）。
- ただし展開候補の gold 率は 2.0%（860本中17本）で既存 top20 の 9.4%（1053枠中99本）より低く、
  素のまま LLM の枠に混ぜると選定が荒れて f1 を下げる公算が大きい（本実行の 3. がその予兆）。
- 先に **展開論文を Qwen3-Reranker に通して絞る**（GPU 必要）を試し、精度を既存枠と同等まで
  上げてから統合するのが筋。プールを完璧に並べ替えた場合の cr@20 天井は 0.914（ecr 0.956）で、
  現状 0.789 との差13ポイントは候補生成ではなく並べ替えの問題として残っている。

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

## candidate_recall（後から追記）

予測ファイル `predictions_8b_chunk_cand50_expand_offline.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 70 本なので、それを超える k は @70 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.8103 |
| candidate_recall_at70_total_macro | 0.9000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.8966 |
| evidence_candidate_recall_at70_total_macro | 0.9455 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_cand50_expand_offline.jsonl, max_candidates=70 -->
