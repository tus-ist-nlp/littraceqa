# mineru + bm25_paper + reading

- 実行日時: 2026-07-18T10:36:29
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_paper.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_paper.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.2848 |
| paper_recall_macro | 0.5727 |
| paper_f1_macro | 0.2671 |
| evidence_precision_macro | 0.0000 |
| evidence_recall_macro | 0.0000 |
| evidence_f1_macro | 0.0000 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文レベルでは再現率 0.573 に対して適合率 0.285、F1 0.267 なので、関連論文をある程度拾えている一方でノイズがかなり多い結果です。特に evidence 系、multiple choice、freeform、table 系がすべて 0.0 で、論文は取れても根拠抽出や回答生成までまったくつながっていない点が強い懸念です。まずは検索結果から evidence へのマッピング処理、本文抽出・チャンク化、リーダーの参照単位が崩れていないかを優先的に確認するとよさそうです。次の試行としては、上位件数やチャンクサイズの調整、evidence 抽出を意識した search/agent 設定への変更、失敗事例の少数目視分析を行うのが有効だと思います。


考えられる要因:
  - 候補プールが1論文=1候補になり、per_index_k=100件を素朴に取っても多様性が下がる:
  chunk単位なら同じ論文の別々の観点（abstract・手法名の言及箇所・表など）が別々にヒットして複数の弱いシグナ
  ルが積み上がるが、論文単位だと1論文=1スコアなので、部分一致による「掘り出し」効果が働きにくい。
  - 長文連結によるBM25スコアの希薄化: 論文全体（数十チャンク分）を1ドキュメントにすると、クエリ語に対する相
  対的なtf/文書長比が下がり、b（長さ正規化）の効きで短い論文が不当に有利/不利になっている可能性。
  - ReadingAgentは候補チャンクのtextを見て評価するため、論文単位だと1候補のtextが非常に長くなり、LLMへの情報
  提示のされ方も変わっている（今回はper_index_k=100のままなので、候補が論文100件分の全文になっている点も影響
  大きそう）。

  CLAUDE.mdに書いた通りLLMは非決定的でクエリ55件なのでノイズの余地はありますが、chunk版との差（0.5前後 vs 0.
  27）はノイズだけでは説明しにくい規模です。1回だけの結果なので断定はできませんが、現状では論文単位化はgold
  paper特定においてもむしろ不利という結果になりました。もう一度回すか、この結果を受けてCLAUDE.mdやconfig構成
  をどう扱うか、方針を確認したいです。