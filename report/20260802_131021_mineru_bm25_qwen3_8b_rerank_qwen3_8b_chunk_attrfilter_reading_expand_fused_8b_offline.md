# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_expand_fused（8B reranker・オフライン適用）

- 実行日時: 2026-08-02T13:10:21
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_expand_fused.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_expand_fused_8b_offline.jsonl`
- git: `a0206042354c`
- **オフライン適用**: 土台 `predictions_8b_chunk_b_merged.jsonl`。展開論文の rerank だけを
  **Qwen3-Reranker-8B（cuda:0, fp16, max_batch_tokens=2048）**で回した。所要 147秒/55件。
  前回の 0.6B 代用（23秒）に対する検証。

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_device | `"cuda:0"` |
| reranker_fp16 | `true` |
| reranker_max_batch_tokens | `2048` |
| reranker_batch_size | `4` |
| reranker_max_tokens | `1024` |
| expansion_sources | `["specter2(faiss_specter2_abstract)", "bib_coupling(min_shared=2)"]` |
| expansion_neighbors | `20` |
| expansion_anchors | `1` |
| expansion_rrf_k | `60` |
| expansion_rerank | `true` |
| expansion_rerank_top_k | `5` |
| expansion_insert_at | `15` |
| agent_max_steps | `3` |
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |
| rerank所要 | `"147秒/55件（0.6Bは23秒）"` |

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
| candidate_recall_at100_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.4071 |
| candidate_recall_at10_multi_macro | 0.4751 |
| candidate_recall_at20_multi_macro | 0.6542 |
| candidate_recall_at50_multi_macro | 0.8103 |
| candidate_recall_at100_multi_macro | 0.8534 |
| candidate_recall_at1_total_macro | 0.5611 |
| candidate_recall_at5_total_macro | 0.6874 |
| candidate_recall_at10_total_macro | 0.7232 |
| candidate_recall_at20_total_macro | 0.8177 |
| candidate_recall_at50_total_macro | 0.9000 |
| candidate_recall_at100_total_macro | 0.9227 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at100_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3831 |
| evidence_candidate_recall_at5_multi_macro | 0.5910 |
| evidence_candidate_recall_at10_multi_macro | 0.6504 |
| evidence_candidate_recall_at20_multi_macro | 0.7864 |
| evidence_candidate_recall_at50_multi_macro | 0.8937 |
| evidence_candidate_recall_at100_multi_macro | 0.9282 |
| evidence_candidate_recall_at1_total_macro | 0.6566 |
| evidence_candidate_recall_at5_total_macro | 0.7843 |
| evidence_candidate_recall_at10_total_macro | 0.8157 |
| evidence_candidate_recall_at20_total_macro | 0.8874 |
| evidence_candidate_recall_at50_total_macro | 0.9439 |
| evidence_candidate_recall_at100_total_macro | 0.9621 |
| evidence_precision_macro | 0.2404 |
| evidence_recall_macro | 0.2328 |
| evidence_f1_macro | 0.2205 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.4945 |
| table_cell_accuracy_macro | 0.2900 |
| table_cell_accuracy_micro | 0.2963 |

## 0.6B との比較

| 指標 | 展開なし | SPECTER2+0.6B | SPECTER2+8B | 融合+0.6B | **融合+8B** |
|---|---|---|---|---|---|
| cr@20 | 0.7833 | 0.8131 | 0.8131 | 0.8222 | **0.8177** |
| cr@20 multi | 0.5891 | 0.6456 | 0.6456 | 0.6628 | **0.6542** |
| cr@50 | 0.8364 | 0.8955 | 0.8955 | 0.9045 | **0.9000** |
| cr@50 multi | 0.6897 | 0.8017 | 0.8017 | 0.8190 | **0.8103** |
| ecr@20 | 0.8697 | 0.8753 | 0.8874 | 0.8813 | **0.8874** |
| ecr@50 | 0.9076 | 0.9379 | 0.9439 | 0.9439 | **0.9439** |

## 所見

**8B は ecr で改善、cr で微減。**（ecr@20 0.881 -> 0.887、ecr@50 0.944 で同値／
cr@20 0.822 -> 0.818、cr@50 0.905 -> 0.900）。SPECTER2 単独でも同じ傾向で、
ecr@20 が 0.875 -> 0.887 と改善する一方 cr は 0.813 で同値だった。

**この差は「8B の方が質問に答える論文を選べている」ことの表れ。** top20 に押し上げた
論文300本の内訳を数えると:

| reranker | 非gold | gold(evidence持ち) | gold(no_evidence) |
|---|---|---|---|
| 0.6B | 286 | 7 | **7** |
| 8B | 287 | **8** | 5 |

8B は evidence 持ち gold を1本多く拾い、no_evidence gold（質問に答えない同トピックの
ピア論文）を2本減らしている。cr は no_evidence gold も分母に数えるので、**賢くなるほど
cr が下がる**という逆説が起きる。**打ち手の評価には ecr を見るべき**という従来の方針
（CLAUDE.md の使い分け）がここでも当てはまる。

**差はいずれも1〜2本ぶんで小さい。** 実運用では 0.6B（23秒）と 8B（147秒）のコスト差に
見合うかは微妙だが、**展開の rerank は 0.6B で代用しても結論は変わらない**ことが確認できた
（前回レポートの「8Bなら上振れの余地」という留保は解消。上振れは ecr 側にわずかにあった）。

## 次に検証すべきこと

- 本構成でのライブ実行（experiments.jsonl への正式登録、GPU 4〜5時間）。
- 拾った論文を提出に載せる改修（展開をループ内に移す）。cr/ecr は上がったが paper_f1 は不変。

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

予測ファイル `predictions_8b_chunk_expand_fused_8b_offline.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 70 本なので、それを超える k は @70 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.8534 |
| candidate_recall_at70_total_macro | 0.9227 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.9282 |
| evidence_candidate_recall_at70_total_macro | 0.9621 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_expand_fused_8b_offline.jsonl, max_candidates=70 -->
