# LitTraceQA RAG読解系：方法論、実装記録、再現可能な評価手順

## 0. 文書の目的と記述上の区分

本書は、LitTraceQAに対して構築した科学論文RAG読解系を、論文のMethod節へ転用できる粒度で記録する。対象は、検索結果または外部で選定された論文集合を受け取り、論文・根拠位置・最終回答を結び付ける読解部分である。検索器そのものの新規性と、検索後の読解・検証・直列化の新規性を混同しない。

本書では、機能の状態を次の3種類に区別する。

- **実装済み**：現在のリポジトリの読解ランタイムまたはvalidatorに存在する。
- **運用済み**：今回の実験・提出作成では行ったが、再利用可能なランタイム機能としては統合されていない。
- **将来実装**：今回の観察から導いた設計案であり、現時点の公式結果を生んだ実装とは主張しない。

記録時点は2026-08-14である。公式入力は `LitTraceQA/LitTraceQA` の revision `bd35dc14cf0483e0ffa51fa2a54d2689c13f9845` に固定されている。作業ブランチは `OzakiHisanori/selected-paper-reader`、記録時のHEADは `28d35ba5f2b102f01e8b39ca4812e62394425d1c` である。ただし、記録時のworktreeには未コミット変更が含まれるため、最終論文ではcommitだけでなく、実行manifest、prompt version、入力・出力SHA-256も併記する必要がある。

## 1. 論文向け要約

本システムは、科学論文QAを、(i) 関連論文集合の決定、(ii) 各論文内の回答根拠の抽出、(iii) 根拠に拘束された回答生成、の3部分へ分解する。読解器は二段階構成であり、第1段階では質問と各候補論文を独立に読み、回答関連性と利用可能な根拠を分離して判定するか、外部で固定された論文集合の場合には論文ごとの原子事実と正確なchunk IDを抽出する。第2段階では、検証済みchunkだけを統合し、原子事実、決定論的演算、最終回答へのbindingを同時に生成する。Python側は、論文間のchunk混同、存在しない根拠ID、未添付画像からの視覚的主張、算術・比較・カウントの不整合、選択肢labelと本文の不一致、表schema・型・row keyの破損を拒否する。

few-shot例はすべて合成例とし、公開入力の質問文、回答形式、選択肢、表schemaだけから質問タイプを決定する。図・引用・計算・その他の4分類により、Stage 1とStage 2へ必要な読解規則のみを追加する。視覚問題ではMinerU由来の画像を信頼境界内で検証し、実際に添付された画像に基づく事実だけを `visual` として認める。最終的な根拠は、論文ID、source type、およびsource typeごとに必要なpage/section・object IDを持つcoarse locatorへ直列化する。

公開testのaggregate feedbackは、意味的な回答内容だけでなく、表のrow keyの表面形と根拠locatorの直列化が主要な誤差源であることを示した。この観察は、意味推論と出力直列化を分離する設計、および表問題だけを複数回生成して原文で裁定する将来設計の動機として用いた。最終的な根拠locatorはsource typeごとの公式eligibilityに従い、table/figureではlocationとobject IDを、equation/citationではlocationまたは対応object IDを用いる。hidden testの正解、query ID別の正解辞書、または評価器を迂回する出力は、共有promptにもランタイムにも埋め込まない。

## 2. タスク定式化

質問を \(q\)、検索対象となる論文集合を \(\mathcal{C}\)、予測する関連論文集合を \(\hat{P}\)、根拠集合を \(\hat{E}\)、回答を \(\hat{Y}\) とする。システムは次の写像を実現する。

\[
\hat{P}=R(q,\mathcal{C}),\qquad
\hat{E}=G(q,\hat{P}),\qquad
\hat{Y}=A(q,\hat{P},\hat{E}).
\]

ここで、`R` は検索・論文選定、`G` は論文内grounding、`A` は回答構成である。今回のselected-paper readerでは `R` の出力を外部入力として固定し、読解器は `G` と `A` を担当する。pairwise候補モードでは、Stage 1が `R` の最終選別にも関与する。

重要なのは、最終回答だけでなく、次の追跡可能な鎖を保持することである。

```text
question
  -> paper_id
  -> source chunk / attached image
  -> atomic fact
  -> deterministic operation (必要な場合のみ)
  -> answer fragment
  -> submission evidence locator
```

公式composite scoreは、paper F1、evidence F1、利用可能な回答指標の平均の平均である。

\[
S=\frac{F_{paper}+F_{evidence}+S_{answer}}{3}.
\]

公開testはmultiple-choice 50問とtable 21問のみであるため、freeformが存在しない今回の設定では、概念上、

\[
S_{answer}=\frac{Acc_{MC}+F_{row}+Acc_{cell,macro}}{3}
\]

となる。`table_cell_accuracy_micro` は重要な診断指標だが、compositeの直接項ではない。この構造から、検索、grounding、回答を同じ「正解率」としてまとめず、各チャネルを独立に改善・ablationする必要がある。

## 3. 設計原則

### 3.1 検索と読解の責任分離

外部選定モードでは、渡されたpaper IDを読解モデルが追加・削除・並べ替えしない。提出schema上のフィールド名は `gold_papers` だが、実際にはシステムが予測した論文集合である。固定モードでは、この集合をそのまま出力し、paper F1を上流検索器の結果、evidence・answerを読解器の結果として解釈できる。

候補選別モードでは、次の3集合を区別する。

1. Stage 1で回答関連と判定した論文集合
2. 利用可能な根拠chunkがありStage 2へ渡す論文集合
3. 最終回答を直接支える最小evidence集合

「論文は関連するが、現在の抽出contextに答えがない」という状態を表現できるように、論文関連性と根拠利用可能性を同じBooleanへ潰さない。

### 3.2 LLMと決定論的処理の責任分離

LLMに任せるのは、科学的文章・表・図の意味読解と、原文中の答えに必要な事実の同定である。次はPython側で検証または再計算する。

- JSON構造とschema
- paper/chunkの所有関係
- chunk IDの存在性
- 画像が実際に添付されたか
- 数値演算、割合変化、平均、count、argmax/argmin、比較
- multiple-choiceのlabelとoption textの対応
- tableの列名、型、row keyの重複
- final answerの各断片とsource factまたはoperationのbinding
- submission用coarse evidence locatorの有効性

この分離により、「根拠は正しいが計算が違う」「選択肢本文は正しいがlabelが違う」「表の値は正しいが別の行に入った」といった静かな矛盾を、回答採用前に検出する。

### 3.3 fail-closed

存在しないchunk ID、別論文のchunk、読み取れない必須画像、原文中に見つからない引用excerpt、算術不一致、無効locatorは、無根拠の推測へ変換しない。修復promptで直せない場合は、その処理を失敗として記録する。

## 4. 公開入力・gold-free契約

### 4.1 入力射影

読解器が受け取る公式queryは、次の公開フィールドに限定する。

- 共通：`query_id`, `benchmark`, `question`, `answer_types`
- multiple-choice：`multiple_choice_options`
- table：`table_schema`

`task_family`、`primary_evidence_type`、正解論文、正解根拠、正解回答は、本番promptへ入れない。`candidate_handoff.py` は入力をこの公開形へ射影し、development-only値が残っていればfailする。

### 4.2 候補論文sidecar

検索器から読解器へのhandoffは、query本文と分離したsidecarとする。

```json
{"query_id":"query_example","candidate_papers":[{"paper_id":"paper_example_001","rank":1,"title":"...","venue":"ExampleVenue","year":2025}]}
```

loaderは次を拒否する。

- `_gold`, `gold_papers`, `evidence`, `answer`
- development-only hint
- sidecarへコピーされた選択肢
- 未知または重複query ID
- 重複paper ID
- 1から連続しないrank
- 公式queryの欠落または余分なquery

この境界により、validationの豊富なレコードを誤ってそのままpromptへ渡すことを防ぐ。

### 4.3 prompt内の信頼境界

質問、選択肢、メタデータ、論文本文、画像mappingは、delimiterで囲まれたuntrusted dataとしてsystem promptへ渡す。論文本文や質問文に命令形の文字列が含まれても、agentの役割、出力schema、根拠境界を変更できないものとして扱う。

## 5. 二段階読解器

### 5.1 Stage 1：論文単位の読解

Stage 1は、1つのqueryと1つのpaperを組にして処理する。長い論文を意味的に複数requestへ分割せず、質問に関係するchunkと最大10枚の画像を決定論的に選び、文字数上限内の1つのcontextへ圧縮する。JSON repairやprovider retryは失敗時だけに発生し、通常の論文partitionではない。

#### pairwise候補モード（実装済み）

モデルは次を返す。

- `is_relevant_to_answer`
- `has_usable_answer_evidence`
- `evidence_chunk_ids`

Pythonはchunk IDを候補論文内の実在IDに限定し、`A=true, B=true, valid IDs` の論文だけをStage 2へ渡す。明示的に名指しされたFigure/Table/Equation/Referenceの所有論文が候補集合内で一意に解決できる場合、明らかな別所有者はLLM callなしで除外できる。ただし、曖昧なtitle、単なる引用先のtitle、複合名の一部分にはこの破壊的gateを適用しない。

#### fixed-selectedモード（実装済み、今回の主読解経路）

外部paper集合はauthoritativeなので、Stage 1はrelevanceを再判定しない。各paperから、次の4フィールドを持つ `evidence_facts` を抽出する。

- `chunk_id`
- `purpose`: `answer_value`, `comparison_operand`, `eligibility_condition`, `table_row`, `visual_fact`, `citation_fact`
- `fact`
- `source_excerpt`

`source_excerpt` は引用chunk内に可視な形で存在しなければ採用しない。画像だけから得たfactは、画像が実際にrequestへ添付された場合に限り、空excerptを許可する。抽出が空でもpaper集合は変更せず、Stage 2だけに同一paper内のquery-ranked fallback chunkを最大3件追加する。このfallbackは「そのchunkが正解根拠である」と主張するものではなく、Stage 2に再読解機会を与えるためのcontextである。

### 5.2 Stage 2：query単位の統合回答

Stage 2は、Stage 1で検証されたchunkと、明示的に区別して記録されたPython supplemental contextをquery単位で統合する。モデルは次を含む構造化応答を返す。

- answerを直接支えるpaperとその役割
- 原子事実
- 必要な場合の演算
- final answerへのbinding
- answer object
- support mapping
- completeness情報

回答は最大5回の構造修復を許す。修復ごとに、失敗理由、raw response、provider call metadataを残す。最終的にvalidatorを通らなければ採用しない。

#### 設定スナップショット

今回のfixed-selected設定は、概ね次の上限を持つ。

| 項目 | 値 |
|---|---:|
| paper context | 220,000 characters |
| Stage 1 prompt guard | 240,000 characters |
| Stage 1 completion | 4,096 tokens |
| Stage 2 completion | 12,000 tokens |
| images per request | 10 |
| Stage 2 papers | 50 |
| submission evidence | 32 |
| evidence per paper | 4 |

値は再現用manifestに記録し、将来変更した場合は同じrunの結果として混ぜない。

## 6. test-aware質問taxonomy

### 6.1 分類方法

質問分類は公開queryの文字列とanswer schemaだけで行い、gold labelや開発専用task familyを使わない。主分類は優先順位付きの4種類である。

1. `visual`
2. `citation`
3. `calculation`
4. `other`

さらに、複合選択肢、表回答、明示的row inventory、symbolic exact、percent change、mean、ordinal reference、bibliography title、axis extentなどの補助tagを付ける。

### 6.2 公開testでの構成

固定した公式releaseでは、公開test 71問のanswer typeと主分類は次の通りである。

| 区分 | 問題数 |
|---|---:|
| multiple-choiceのみ | 50 |
| tableのみ | 21 |
| visual | 10 |
| citation | 8 |
| calculation | 1 |
| other | 52 |

43件のmultiple-choiceは、複数の値・条件・論文を1つのoptionへ照合する複合型として検出される。21件すべてのtable問題には、質問が要求するunitを列挙してschema cellへ対応付ける合成例を選択する。

### 6.3 few-shot設計

few-shotは、実在するvalidation/testの答えを含まない合成例である。Stage 1は通常、共通negative 1件と質問タイプ別のusable/not-usable例を用いる。Stage 2はtagが一致する例だけを最大12件選ぶ。表問題にMCのlabel例を混ぜず、MC問題にtable専用例を混ぜない。

記録時の主要versionは次である。

- pairwise judgment: `pairwise-paper-judge-v30-validation-name-free-examples`
- selected evidence: `fixed-selected-evidence-v4-candidate-local-visual`
- generic answer: `accepted-evidence-answer-v46-gold-free-table-contract`
- fixed-selected answer: `fixed-selected-answer-v26-gold-free-table-contract`
- taxonomy: `question-only-four-way-v2-test-wording`
- derivation: `answer-derivation-v8-number-word-and-contrast-bindings`

## 7. 視覚情報とPDF処理

### 7.1 MinerU画像経路（実装済み）

本文・table・figureはMinerU chunkとして読み、table/figure chunkは `image_path` を持ち得る。画像は、明示的な `image_root` の下に `paper_id/auto/images/filename` として再配置する。元データ内の絶対pathをそのまま信頼しない。

preflightは、次を検査する。

- candidate paperがcorpusに存在するか
- submission可能なpage/section/object IDを持つchunkがあるか
- 明示的visual queryに読めるfigure/chart画像があるか
- path traversalまたはroot外symlinkがないか
- 画像が存在し、完全decodeできるか
- 画像が20 MiB以下、40 million pixels以下か
- 対応MIMEが許可されているか

明示的visual queryに画像がなければ、通常のanswer実行は停止する。診断用overrideは `judge` stageだけに限定し、提出回答の生成には使わない。

### 7.2 実際に添付された画像の証明（実装済み）

モデルが `visual_fact` を返すには、そのfactのpaper-local chunkに対応する画像が実際のprovider requestへ添付されていなければならない。単にcorpusにpathがある、caption textがある、または別論文の画像を添付した、という状態ではvisual evidenceと認めない。

### 7.3 PDFからの画像回復（運用済み、コア未統合）

今回のtest監査では、MinerU cropが欠落していた一部visual ownerについて、公式paper URLからPDFを取得し、PDFのpageをrenderし、MinerUのbboxに基づいて必要なobjectだけをcropする補助スクリプトを用いた。取得PDFのhash、page数、render元、crop先をmanifestへ残した。

これは `work/test_submission_v44/recover_visual_images.py` にある対象限定の運用コードであり、一般のpairwise readerが任意PDFから自動復旧する機能ではない。論文では「対象限定のsource recoveryを行った」と明記し、自動システムの標準能力として主張しない。一般化されたPDF fallback、page-to-chunk alignment、再現可能なcrop generationは将来実装である。

## 8. 原子事実と決定論的derivation

### 8.1 fact表現

Stage 2のfactは、少なくとも次を持つ。

```json
{
  "id": "f1",
  "name": "descriptive source fact",
  "value": 14.7,
  "value_kind": "reported",
  "paper_id": "paper_example_001",
  "chunk_ids": ["paper_example_001#tab0002"]
}
```

`value_kind` は `reported`, `visual`, `text` に限定する。計算結果をsource factとして偽装できないよう、`computed` factは許可しない。各factは一意のID・説明名・paper ID・1件以上の同一paper chunkを持つ。

### 8.2 operation表現

sourceから直接読んだ値と導出値を分離する。現在のvalidatorは、次を含む操作を再計算する。

- `add`, `subtract`, `multiply`, `divide`
- `mean`
- `percent_change`
- `count`
- `argmax`, `argmin`
- `compare`
- 複数labelの条件一致を選ぶ `select_where`

演算のoperandはfact valueと一致しなければならない。countは明示的なitem inventoryを要求し、argmax/argminは全候補のlabel/valueがfactと一致することを要求する。平均やpercentage changeは、質問がその演算を要求している場合だけ許す。

### 8.3 answer binding

各final answer fragmentは、factまたはoperationへbindingする。たとえば複合MCで2つの数値を含むoptionを選ぶ場合、片方だけをgroundしてoption全体を採用できない。tableでは各rowまたは各leaf cellを、対応するpaper/chunkへ結び付ける。

この構造により、同じsource factを複数の回答形式に再利用しつつ、freeform、MC、table間の値の不一致を防ぐ。

## 9. 回答形式別の戦略

### 9.1 multiple-choice

読解モデルはsemantic answerを先に決め、`label` と `selected_option_text` の両方を返す。Pythonは、labelが公開option集合に存在し、そのlabelの実テキストと `selected_option_text` が一致することを検証し、提出JSONLにはlabelを保存する。option内の値や論文名は、source evidenceとして扱わない。

今回の公開testではmultiple-choice accuracyが1.0000に到達した。その後の表面形実験では、既存MCレコードをbyte-level diffで不変に保つ「MC freeze」を運用した。reader CLIに独立した `--freeze-mc` optionはないが、`--answer-type table` で実行対象をtable queryへ限定し、後段のreview/compose gateでMCを含む全非table answerをbaseと同一に保つ実装へ一般化した。

### 9.2 table：実装済みの制約

表回答は、公開 `table_schema` の列名をそのまま使い、余分な列を許さず、`string`, `number`, `boolean` の型を検査する。明示row keyは空にできず、同じkeyの重複行を拒否する。明示的に列挙されたrowがある場合は、質問だけからinventoryを作り、未ground rowを黙って落とさない。open-ended multi-paper tableでは、固定selected paperごとのgrounded row coverageを確認する。

さらに、現在のprompt rendererは、質問文と `table_schema` だけからgold-freeな列別output contractを構成する。`Paper Title` のrow keyは入力metadata titleの完全一致、その他のrow keyは質問中の最短の明示label、非keyのstring cellはsource上の表面形、number/booleanは対応するnative JSON型を既定方針とする。このcontractはquery ID、candidate、予測answerを参照しない。これは生成時の表面形を統制する実装であり、生成後にsourceと照合して最適文字列を自動選択する裁定器とは区別する。

### 9.3 row identityとcell valueの分離

公式table metricのcell分母は、すべてのgold rowと非key列の直積である。予測row keyが一致した場合だけ対応cellの値を比較し、一致する予測rowがないgold rowでは、そのrowの全非key cellを不正解として数える。したがって、意味的に同じrowであってもrow keyの表面形が異なると、そのrowの非key cellはすべて0点になる。一方、row keyを直しても対応cellの文字列・型・値が一致しなければcell scoreは上がらない。

内部表現では、次を分離すべきである。

```text
semantic_row_identity
source_row_label
question_surface_candidates
serialized_row_key
typed_non_key_cells
supporting_fact_ids
```

安全なserializerは、row identityを質問中の明示名、source tableのrow label、canonical paper/method metadataから決める。略称化、引用番号の削除、ハイフンの変更、単位の追加・削除は、一般規則として盲目的に行わない。number列はnative numberへ変換できるが、string列はsourceまたは質問に支持されたlexical formを保持する。

### 9.4 table-only multi-sample adjudication（裁定・合成は実装済み、候補生成は未統合）

reader本体は1つのbase Stage 2 answer callと、validator失敗時のrepairを行う。`scripts/run_aoai_pairwise_reader.py --stage answer --answer-type table` は公開answer typeだけからtable queryを選び、既存Stage 1 checkpointを使ったanswer-only再実行を可能にする。独立した \(N\) 個のrun directoryを一括生成するmulti-sample orchestratorと、自動的にsource supportを採点するmodelは未実装である。一方、手動または外部で生成した複数のfullまたはtable-only JSONLを比較し、表回答だけを安全に昇格させるfail-closedなreview/compose機能は `src/littraceqa/table_adjudication.py` と `scripts/adjudicate_table_answers.py` に実装した。

review段階では、公式入力、base、各candidate、query順、table schemaをSHA-256で封印し、21件のtable queryごとにrow count、row key、table hash、完全なtable objectを並べる。compose段階では各queryについて、選択candidate、`source_checked=true`、理由、baseのfrozen evidence集合に実在する公式形locatorを要求する。採用時に置換できるのは `answer.table` だけであり、paper、evidence、MC/freeform、非table query、query順をfreezeする。同一baseを全件選んだ場合は、出力をbaseとbyte-for-byteで一致させる。この実装はsource reviewの記録と差分隔離を保証するが、`source_checked` の真偽や科学的なcell内容を自動検証するものではない。

生成から裁定までを含む完全な将来プロトコルは次である。

1. MC、paper集合、既存evidenceを固定し、table queryだけを対象にする。
2. queryごとに同一公開入力・同一source contextから \(N\) 個の独立候補を生成する。
3. 各候補をschema、型、row-key重複、fact bindingで先にfilterする。
4. row identityを意味的にclusterするが、row-key文字列はまだ多数決しない。
5. 各row/cellをsource excerptまたは添付画像へ戻し、支持される候補だけを残す。
6. row key候補は「質問での名称」「source row label」「canonical metadata」の優先規則で裁定する。
7. non-key cellはrow keyとは別に裁定し、数値、単位、引用番号、hyphenを独立に検査する。
8. 完全なgrounding chainを持つ表だけを、実装済みcompose gateで提出候補にする。

単純なstring多数決は、複数sampleが同じOCR誤りを繰り返す場合に失敗する。裁定基準は生成頻度ではなく、source supportと公開schemaへの適合でなければならない。

## 10. coarse evidenceの生成

### 10.1 locator形

submission evidenceは、chunk IDそのものではなく、公式metricが比較するcoarse locatorへ変換する。基本形は次である。

| source type | location | object field |
|---|---|---|
| text span | page、なければsection | なし |
| table | page、なければsection | `table_id` |
| figure | page、なければsection | `figure_id` |
| equation/algorithm | pageまたはsection。対応object IDがあればlocation単独は不要 | `equation_id` または `algorithm_id` |
| citation context | pageまたはsection。`citation_id` があればlocation単独は不要 | `citation_id`（存在時） |

`title_abstract` はsubmission時にtext spanへ写像する。References/Bibliography sectionまたはcitation IDを持つtext chunkはcitation contextへ写像する。

### 10.2 object locator recovery

MinerUが隣接したtable/figure captionを1 chunkへ結合する場合がある。実装は、複数captionが可視で、metadata IDもその集合に含まれ、質問がそのうち1つを明示するかquery tokenで一意に選べる場合に限り、最終出力object IDを保守的に補正する。曖昧な場合は元metadataを保持する。

隣接table chunk間でcaptionとbodyがずれた狭いpatternについても、同一paper・同一page・複数caption・Markdown table bodyという観測可能な条件を満たす場合だけtable IDを回復する。

### 10.3 minimality

evidence precisionとrecallを同時に維持するため、読んだchunkをすべて提出evidenceへ入れない。final answerのfactまたはoperationを直接支える最小chunkを選ぶ。Stage 1 handoff、Python supplemental context、最終evidenceをtrace上で区別する。

## 11. validation/test mismatchの定量記録

### 11.1 answer type分布

公開validation 55問とtest 71問は、answer typeの構成が大きく異なる。

| split | freeformを含む | MCを含む | tableを含む | 純MC | 純table |
|---|---:|---:|---:|---:|---:|
| validation | 26 | 41 | 11 | 21 | 8 |
| test | 0 | 50 | 21 | 50 | 21 |

validationには `freeform+MC` 20問、`freeform+table` 3問があるが、testは全問が単一answer typeである。このため、validation全体で最良のpromptを、そのままtestに最良とみなすことはできない。特に、combined-answer用のfew-shotやbinding要件はtestでは不要であり、表問題の比率はtestで高い。

### 11.2 validation結果の読み方

同一日に保存されたvalidation auditのv25 runでは、paper F1 0.7307、evidence F1 0.5454、MC 0.9756、table row F1 0.6255、cell macro 0.4697であった。test-aware prompt系のv35 runではvalidation上の複数指標が低下した。これはtest上の優劣を意味しない。入力分布、候補paper集合、prompt版、画像availabilityが異なるため、validation scoreをleaderboard test scoreへ直接外挿しない。

この比較の出典は `runs/audit_v25_vs_v30_v35_validation_tuned_20260812/official_metrics_v25.json`（SHA-256 `d5d490357f16f1153ca1c4666b2b69cac7e5991b9176b8f263a3287ea391db59`）と、同directoryの `official_metrics_v35.json`（SHA-256 `c99bd25f100a99ed5939fa6465094de6f1f289d93ab90d5a5bfd96ed2aa27ff9`）である。

今後は、validation内でもanswer typeと主質問タイプで層別し、全体平均だけでなく、MC、table、visual、citation、calculationごとのerrorを報告する。

## 12. aggregate leaderboard feedbackから得た設計上の知見

### 12.1 観測結果

online evaluatorが表示したaggregate metricのみを記録した。hidden goldレコードやquery別正誤は取得していない。

| 段階 | Composite | Paper F1 | Evidence F1 | MC | Table row F1 | Cell macro | Cell micro |
|---|---:|---:|---:|---:|---:|---:|---:|
| source-audited初期提出 | 0.743246 | 0.9817 | 0.7070 | 1.0000 | 0.3849 | 0.2381 | 0.2414 |
| table修正済みcheckpoint | 0.747301 | 0.9845 | 0.7070 | 1.0000 | 0.3955 | 0.2556 | 0.2644 |
| row-key単独検証後 | 0.749946 | 0.9845 | 0.7070 | 1.0000 | 0.4193 | 0.2556 | 0.2644 |
| row/value統制bundle後 | 0.753701 | 0.9845 | 0.7070 | 1.0000 | 0.4412 | 0.2675 | 0.2759 |

`cell micro=0.2759` は、表示丸め前のmetric構造から87 comparable cells中24 cellsの一致に対応する。row-keyだけを変えた提出でrow F1が上昇し、cell macro/microが変わらなかったことは、新たに一致したrow keyに対応するnon-key cellがすべて不正解だった場合と整合する。この1回のprobeだけでrow identityとcell valueの誤差が統計的に独立とは結論できないが、両者を別々に監査する設計動機にはなる。

### 12.2 設計への反映

このaggregate feedbackは次の動機として使う。

- semantic reasoningとsubmission serializationを別moduleにする。
- row keyの採択とnon-key valueの採択を別々に監査する。
- MCが満点の間は、表・evidence実験からMCを隔離する。
- compositeだけでなく、paper/evidence/row/cellの全vectorで仮説を判定する。
- 1つの提出variantには、可能な限り1つの機構だけを入れる。

このfeedbackをquery ID別の答え辞書へ変換しない。共有promptには、特定test問題の正解語、数値、paper ID、row keyをfew-shotとして入れない。個別提出のsource auditを行った場合は、automatic readerの結果と分けて記録する。

### 12.3 score headroom

0.753701時点の表示値から、他チャネルを固定して各部分だけを完全化した場合のcomposite改善余地は概ね次である。

- paper：+0.0052
- evidence：+0.0977
- table row/cell：+0.1435

したがって、今後の主対象はretrieval prompt全体ではなく、coarse evidence alignmentとtable serializationである。ただし、このheadroomは改善可能性を示すだけで、hidden goldを復元できることを意味しない。

## 13. 再現可能な実験プロトコル

### 13.1 inference前

1. 公式release revisionと各ファイルhashを固定する。
2. query inputとcandidate sidecarを別ファイルにする。
3. gold-free loaderとcorpus/image preflightを実行する。
4. prompt version、few-shot ID、config、model/deployment、token上限、temperature、worker数をmanifestへ記録する。
5. 変更する機構と、変化を期待するmetric channelを事前に書く。
6. validation gold由来の固有名、数値、方向、question-specific構造がfew-shotへ混入していないか監査する。

### 13.2 inference

1. query-paperのStage 1結果を個別checkpointする。
2. Stage 2 answerとrepair attemptをcheckpointする。
3. provider invocationごとにPREPARE/FINALIZE ledgerを残す。
4. crash後は `--resume` で成功済みcallを再利用する。
5. image attachment、chunk ID、fact、operation、bindingをtraceへ保存する。

### 13.3 submission gate

1. 公式query 71件と完全一致することを確認する。
2. query IDの順序・一意性・欠落・余分を検査する。
3. paper、evidence、answerのshapeを検査する。
4. table column、cell type、row key非空・重複なしを検査する。複数候補を使う場合はsealed reviewを生成し、各table decisionへsource check、理由、locatorを記録する。
5. evidence source typeとlocator形を検査する。
6. official validatorとrepository strict validatorを通す。
7. baselineとの差分query、変更field、出力SHA-256、builder SHA-256をmanifestへ残す。
8. MC freezeを用いる実験では、全MC answerがbaselineと一致することを機械的に証明する。

### 13.4 online評価後

1. compositeだけでなく全metric vectorを保存する。
2. artifact filename、hash、submission ID、時刻、残quotaを対応付ける。
3. 期待したmetric signatureと一致しない場合は、原因を決め打ちせず停止する。
4. 次variantは、固定baselineと確認済みpositive componentから再構築する。
5. negativeまたは不明なcomponentを、惰性で次の提出へcarryしない。
6. leaderboardがbest scoreを保持しても、bestを生んだ正確なJSONLを別途保存する。

## 14. ablation計画

以下は、実装済み機構の寄与と将来機構の寄与を分けて測るための推奨ablationである。

| ID | 条件 | 状態 | 主な観測指標 |
|---|---|---|---|
| A0 | full two-stage reader | 実装済み | 全指標、cost、repair率 |
| A1 | question taxonomyなし、共通few-shotのみ | 実装可能、未集計 | type別paper/evidence/answer |
| A2 | text-only、visual imageを渡さない | 診断のみ | visual subset accuracy、fail率 |
| A3 | deterministic derivation validationなし | 実験用branchが必要 | 算術・MC整合性、invalid率 |
| A4 | query-aware object locator recoveryなし | 実装可能、未集計 | evidence P/R/F |
| A5 | fixed-selected fallbackなし | 実装可能、未集計 | answer completion、evidence recall |
| A6 | model raw table surface vs row/value serializer | serializerは将来実装 | row F1、cell macro/micro |
| A7 | single table sample vs source-adjudicated multi-sample | review/composeは実装済み、独立生成と自動source scorerは将来実装 | table metrics、cost |
| A8 | MC freezeなし vs freezeありの表実験 | review/compose gateで実装済み | MC保全、差分局所性 |
| A9 | automatic output vs human source-audited post-processing | 運用比較 | 全指標、human time |

validation ablationでは、prompt選択に使うfoldと最終評価foldを分ける。少数のvisual/citation/table問題しかないため、単純random splitだけでなくtype-stratified評価とleave-one-query-outの両方を報告する。testのaggregate scoreは最終確認として扱い、同じtest feedbackへ反復適合した回数と変更内容を開示する。

## 15. 制約と失敗モード

### 15.1 lexical metricへの依存

表のrow keyとstring cellは、意味的同値ではなく表面一致に強く依存する。citation number、hyphen、略称、単位、OCR空白の差で、source-groundedな回答でもscoreが0になり得る。公開schemaだけではhidden annotationの全表面形を一意に決められない。

### 15.2 evidence annotationとの差

source factが正しくても、page/section/object IDの付け方がgold annotationと異なればcoarse evidenceは一致しない。複数locatorを無差別に追加するとrecallは上がってもprecisionが下がる。根拠はtargeted replacementで直す必要がある。

### 15.3 OCRとobject segmentation

MinerUはtable header、数式、hyphen、連続captionを誤ることがある。query-aware recoveryは曖昧な場合に補正しないため、安全だがrecallを失う。PDF fallbackは一部運用したものの、一般化・自動化されていない。

### 15.4 fixed-selected仮定

fixed-selected readerは、上流のpaper集合を正しいものとして読む。paper F1の残差は、読解promptだけでは修正できない。end-to-endシステムとして報告する場合は、paper sidecarを作ったretrieval componentを別途説明・評価する必要がある。

### 15.5 computeとAPI依存

query-paper単位の読解は候補数に比例してcall数が増える。multi-sample adjudicationは表問題だけに限定しても追加costを要する。provider failure、429、image policy rejectionを通常の意味判定と区別しなければならない。

### 15.6 leaderboard overfitting

aggregate feedbackを繰り返し使うと、公開testに対する適応が起こる。特に、query ID別例外や1つのtest表面形だけを条件分岐へ追加すると、一般的なRAG能力ではなくtest artifactを測ることになる。今回の方法論では、feedbackを「row/value/evidenceを分離する」という一般設計へ還元し、特定test answerをpromptへ埋め込まない。

## 16. 倫理・開示・再現性

最終システム論文では、少なくとも次を開示する。

- 使用したofficial release revision
- candidate paper集合を作成した検索器または外部手順
- 使用したLLM、API、prompt version、few-shotが合成であること
- MinerU、PDF取得元、画像回復・crop手順
- 人手によるsource auditとpost-processingの有無、対象範囲、作業時間
- leaderboard feedbackを見た回数と、それに基づく設計変更
- test-specific routingを行ったが、hidden goldとtest answer hardcodeを使っていないこと
- 提出artifactとcode/hashの対応
- 外部PDFの著作権・利用条件に従い、PDFそのものを再配布しないこと

今回のonline提出には、自動reader出力を基にしたsource auditと、公開論文に照らした局所的な表面形修正を含むartifactがある。これを報告する場合、完全自動system scoreと同一視しない。自動run、source-audited run、aggregate feedback後のsubmission variantを別行で報告するのが望ましい。

次は行わない。

- hidden goldへのアクセス
- 複数accountによるquota回避
- malformed JSON、重複ID、NaN、Unicode collision等の評価器exploit
- test query IDから正解を返す辞書
- negative結果や手動修正の非開示

## 17. 実装対応表

| 機能 | 主な実装・記録 | 状態 |
|---|---|---|
| gold-free query/candidate loader | `src/littraceqa/candidate_handoff.py` | 実装済み |
| pairwise/fixed-selected two-stage reader | `src/littraceqa/aoai_pairwise_reader.py` | 実装済み |
| prompt taxonomy・synthetic few-shot | `src/littraceqa/pairwise_prompts.py` | 実装済み |
| atomic derivation validation | `src/littraceqa/answer_derivation.py` | 実装済み |
| visual corpus preflight | `src/littraceqa/corpus_preflight.py` | 実装済み |
| coarse/query-aware locator | `src/littraceqa/mineru_record.py`, `src/littraceqa/citation_locator.py` | 実装済み |
| resumable orchestration・attempt ledger | `scripts/run_aoai_pairwise_reader.py`, `src/littraceqa/pairwise_run_store.py` | 実装済み |
| selected-paper config | `configs/agent_style/aoai_selected_paper_reader.yaml` | 実装済み |
| test taxonomy回帰検査 | `tests/test_official_test_prompt_taxonomy.py` | 実装済み |
| table-only candidate review・safe compose | `src/littraceqa/table_adjudication.py`, `scripts/adjudicate_table_answers.py` | 実装済み（source判断は人手） |
| targeted PDF image recovery | `work/test_submission_v44/recover_visual_images.py` | 運用済み、一般化未実装 |
| answer-type実行filter | `scripts/run_aoai_pairwise_reader.py --stage answer --answer-type table` | 実装済み |
| MC・paper・evidence freeze | `src/littraceqa/table_adjudication.py` | compose gateで実装済み |
| table-only multi-sample adjudication | `src/littraceqa/table_adjudication.py`、本書 Section 9.4 | review/compose実装済み、候補生成は未統合 |
| reusable row/value serializer | `src/littraceqa/query_requirements.py`、本書 Section 9.3 | gold-free prompt contract実装済み、生成後の自動裁定は将来実装 |

### 17.1 検証スナップショット

2026-08-14 15:14 JST時点で `uv run pytest -q` を実行し、838 testsが成功した。これには、公開test taxonomy、gold-free handoff、二段階reader、atomic derivation、visual/corpus preflight、coarse locator、answer-type filter、およびtable adjudicationの回帰検査を含む。worktreeが未コミットであるため、このtest数はcommitの恒久的属性ではなく、Section 0のHEAD、prompt version、artifact hashと組にしたスナップショットとして扱う。

## 18. 論文Method節へ転用する場合の短縮版

以下は、本書の内容を短いMethod節へ圧縮した日本語草案である。

> 我々は、LitTraceQAを論文選定、根拠抽出、回答構成の3段階に分解し、検索後の読解に二段階readerを用いた。第1段階では、各query-paper pairを独立に処理し、候補選別モードでは回答関連性と利用可能な根拠を分離して判定し、外部選定モードでは各selected paperから回答に必要な原子事実と正確なchunk IDを抽出した。第2段階では、検証済みchunkのみを統合し、source fact、決定論的演算、final answer fragmentへのbindingを含む構造化回答を生成した。Python validatorは、paper/chunk所有関係、未添付画像からの視覚的主張、算術・count・比較、選択肢label/text、table schema・型・row key、およびsubmission evidence locatorを検査した。
>
> Prompt selectionにはgold labelを用いず、公開質問文とanswer schemaだけからvisual、citation、calculation、otherの4分類と補助tagを決定した。few-shotはすべて合成し、validation/testの具体的な答えを含めなかった。視覚問題ではMinerU画像を明示的なtrust root下で検証し、実際にproviderへ添付された画像だけをvisual evidenceとして認めた。根拠はpaper ID、source type、pageまたはsection、object IDからなる公式coarse locatorへ変換した。
>
> 公開testのaggregate feedbackは、table row keyの表面形とnon-key cellの表面形が独立した誤差源であることを示した。この結果を受け、multiple-choice出力を固定したまま、tableのsemantic row identityとsubmission serializationを分離して分析した。複数のtable候補を比較する場合は、入力と候補をhashで封印し、source確認済みの `answer.table` だけを置換するfail-closedなcompose gateを用いた。feedbackは一般的なserializer設計の動機としてのみ使用し、hidden test answerやquery ID別の正解をpromptへ埋め込まなかった。自動出力、source-audited post-processing、feedback後のsubmission variantは別々に記録した。

## 19. 次の実装優先順位

1. table-only multi-sample generatorと自動source-support scorerを、既存review/compose moduleへ統合する。
2. semantic row identity、row-key surface、typed cell valueを分離した中間表現を導入する。
3. public validationだけでserializer規則を学習・固定し、test固有語を規則へ入れない。
4. evidence locatorのpage/section/object候補をsource metadataとPDFから1対1で監査する。
5. targeted PDF recoveryを、hash・page alignment・crop manifest付きの一般fallbackへする。
6. 複数候補生成を1 commandで再現し、既存answer-type filterとfreeze/diff gateへ接続する。
7. automatic、multi-sample、human-auditedの3条件を同一validation protocolでablationする。

本書は、現在の自動readerが既に実装する範囲と、leaderboard診断から導いた次段階の設計を意図的に分けている。将来機構を実装した後は、対応表、version、ablation結果を更新してから論文へ転用する。
