# **2. Method**

## **2.1 Overview**

```
                              Question
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      [LLM] query decomposition        attribute constraint
           (4 subqueries)                  (venue / year)
                 │
 ════════════════╪═══ iterative loop (up to 3 rounds) ═══════════════
                 ▼
     chunk BM25      paper BM25      dense (Qwen3-Embedding-8B)
          │               │               │
          └──── attribute filter (per index) ────┘
                          │
                paper-level RRF (k = 60)
                          │
                 Seed Expansion  (re-retrieve → filter → fuse)
                          │
            Qwen3-Reranker-8B   (rank fusion, not replacement)
                          │
                     chunk pool ──────► [LLM] Evidence Verifier
                          │                   ├ sufficient → stop
                          │                   ├ missing    → refine
                          │                   └ paper_ids  → anchors
 ═════════════════════════╪══════════════════════════╪══════════════
                          ▼                          ▼
              Ranking A (question → paper)   anchors = top-1 candidate
              chunk pool → paper scores               ∪ verifier-supported
              (table chunks excluded)                 │
                          │             ┌─────────────┼─────────────┐
                          │             ▼             ▼             ▼
                          │        SPECTER2   biblio coupling  full-text MLT
                          │             └─────────────┼─────────────┘
                          │                   RRF (k = 60)
                          │                           │
                          │              Ranking B (paper → paper)
                          │              *not passed to the reranker*
                          │                           │
                          └──────── rank fusion ──────┘
                                    (k = 10)
                                        │
                                        ▼
                          Top-50 candidate papers
                                        │
                                        ▼
                       Reader:  paper IDs / evidence / answers
```

**Figure 1: System architecture.** Verifier の `missing` が次周のサブクエリ生成を駆動する（§2.2.3）。 Retriever は質問起点のランキング A（§2.4.1）と 論文間展開のランキング B（§2.4.2）を別々に構成し、§2.4.3 で順位融合する。**ランキング B は Reranker に通さない**（§2.4.2）。Retriever と Reader は候補論文50本の順位列だけで接続し、 Reader はこの50本を順位で閾値化せず候補集合として利用する。

---

## **2.2 Retriever**

Retriever は**2本のランキングを別々に構成し、最後に順位融合する**。

| ランキング | 起点 | 構成 | 節 |
| --- | --- | --- | --- |
| **A** (question → paper) | 質問 | 3索引の融合 → 反復検索 → チャンクプールを論文へ集約 | §2.2.1–§2.3、§2.4.1 |
| **B** (paper → paper) | 初期候補と verifier が支持した論文 | 3種類の近傍を RRF で融合 | §2.4.2 |

2本を分ける理由は §2.4.2 に述べる。**ランキング B は Reranker に通さない。**

### **2.2.1 Corpus and Indexes**

コーパスは2024〜2025年の主要会議27,487本である。各論文を最大2,000文字のチャンクに分割し、 全体で2,564,545チャンクを索引する。索引は3本を用いる。

| Index | Unit | Implementation |
| --- | --- | --- |
| chunk BM25 | Chunk | `bm25s` (Lucene-style scoring, k1=1.5, b=0.75) |
| paper BM25 | Paper (all chunks concatenated) | Same as above |
| dense | Chunk | Qwen3-Embedding-8B (4096-dim) + FAISS |

語彙一致を2つの粒度で持つのは、質問語が論文内の離れた箇所に分散する場合にチャンク側の スコアが上がりにくく、逆に論文単位では表の1セルのような局所的な根拠が埋もれるためである。 2つは補い合う。

### **2.2.2 Paper-Level Fusion**

評価は論文単位であるのに索引はチャンク単位であるため、**融合の段で単位を論文へ揃える**。 各索引の結果を論文単位のランキングに畳んでから RRF で融合する。

各索引の結果で**論文が最初に現れた順位だけ**を用い、**等重みの RRF（k = 60）**で統合する。1つの索引の中では、同じ論文に何チャンク当たっても1票である。チャンク単位で融合すると、チャンク数の多い長い論文が、質問との関連とは別の理由で上位に来やすい。

### **2.2.3 Iterative Search**

質問を LLM で4本のサブクエリに分解し、**最大3周の反復検索**を行う。各周では、その周の
サブクエリで検索した結果を**チャンクプールに追加**し、Evidence Verifier（§2.3.2）の出力に
従って停止するか、不足している情報を手がかりに**次周のサブクエリを新たに生成**して次へ進む。
停止を判断するのは reranker ではなく verifier である。

初回の分解では、質問文と（属性制約が取れていれば）その制約だけを入力とし、**根拠が1本の
論文にある場合と複数論文にまたがる場合の両方を覆わせる**。すなわち、**主役の論文を確実に
引き当てる言い換え**と、**回答に必要な事実ごとに分けたサブクエリ**の双方を含ませる。本数は
問題タイプで振り分けず4本に固定し、返り値も4本に切る。あわせて、投げ先がローカルの索引で
あることを明示する（明示しないと、`site:` などの Web 検索演算子を含むサブクエリが生成され、
1件も当たらない）。

**停止しない周では、元の質問・前周までに投げたサブクエリ・Verifier の `missing` を入力として、次周用の新しい4本のサブクエリを生成する。** すなわち次周は初周の4本の言い換えではなく、**不足している情報に絞った別の4本**である。

**サブクエリは周ごとに置き換わるが、検索結果は捨てない。** 再分解は「前周のサブクエリを
作り直す」のではなく、**前周で得られなかった情報を埋めるための新しいサブクエリを作る**
処理である。チャンクプールは全周にわたって蓄積されるため、**候補ランキングは全周の検索結果
の和集合から構成される**。したがって初周の4本が無駄になることはない。前周までに投げた
サブクエリは再分解のプロンプトに渡し、同じ問い合わせを繰り返さないようにする。
反復は最大3周なので、サブクエリは最大12本になる。

あわせて2つの機構を用いる。**適用順は、属性フィルタが先で Seed Expansion が後である。**

第一に、質問が会議名を明示する場合の**属性フィルタ**である。検索結果のメタデータに venue と year が含まれるため、索引を改修せずに絞り込める。**各索引の結果に対して融合の前に適用する。**

第二に、**Seed Expansion** である。融合後の1位論文の title+abstract の先頭512文字を質問に 連結して索引を引き直し（このときも同じ属性フィルタを適用する）、初回の順位と1:1で融合する。 LLM を用いない疑似適合フィードバックであり、質問文に現れない語彙を候補側から補う。

## **2.3 Chunk Reranking and LLM Verification**

検索後の処理は、目的の異なる2つのモジュールからなる。**チャンク単位の cross-encoder reranker** は根拠チャンクを並べ替える。**論文単位の LLM verifier** は反復を制御し、 論文間展開の起点を供給する。**verifier は順位を置き換えず、最終回答も生成しない。**

### **2.3.1 Qwen3-Reranker-8B (Chunk-Level)**

融合後の上位200チャンクを Qwen3-Reranker-8B（fp16, max_tokens 2048）に渡す。 Qwen3-Reranker は yes/no 判定型であり、"yes" トークンの確率をスコアとする。

**Rank fusion instead of replacement.** Reranker に順位を全面的に置き換えさせない。**融合前の順位と reranker の順位を、重み 0.6 対 0.4、k = 60 の RRF で統合する。** あわせて、融合前の上位20件の集合を先頭に残す。

理由は §2.4.2 と同じである。質問との適合のみを最適化する reranker は、質問文が言及しない 論文を下げるリスクがある。融合前の順位には BM25・論文単位 BM25・密ベクトルの合意が 反映されているため、これを捨てずに残す。

### **2.3.2 GPT-5.4 Verifier (Paper-Level)**

上位20論文について、論文あたり最大2チャンク（各1,800文字）を GPT-5.4 に読ませ、 **候補が質問を支持しうるか**を判定させる。返り値は次の3つの検索制御に用いる。

| Return value | Role |
| --- | --- |
| `sufficient` | Stops the retrieval loop |
| `missing` | Guides the query refinement in the next round |
| `paper_ids` | Seeds the paper-to-paper expansion (§2.4.2) |

**No answer generation.** この LLM は最終回答を生成しない。回答と根拠箇所の生成は Reader（§2.5）が担う。 LLM 呼び出しは1クエリあたり最大6回（分解1 + 判定3 + 再分解2）である。

---

## **2.4 Candidate Ranking**

反復が終わった時点で、**2本のランキングを構成して順位融合する**。ランキング A は
チャンクプールを論文へ畳んだもの、ランキング B は論文間展開によるものである。

| ランキング | 起点 | 節 |
| --- | --- | --- |
| **A** (question → paper) | 質問（§2.2 と §2.3 の出力） | §2.4.1 |
| **B** (paper → paper) | 初期候補と Verifier が支持した論文 | §2.4.2 |

### **2.4.1 Ranking A: Chunk-to-Paper Aggregation**

反復が終わると、チャンクプールを論文のランキングへ畳み込む。**論文の代表スコアは、その論文が持つチャンクのうち `table` を除いたものの最大スコア**とする。`table` チャンクしか残っていない論文の代表スコアは0とする。
table チャンクは数値と短いラベルが密集しているため、語の重なりだけで高いスコアが付き、 質問の主題ではない論文の代表スコアを押し上げやすい。ただしチャンクプール自体は変更せず、 table チャンクは Verifier（§2.3.2）にはそのまま渡す。これが**ランキング A** である。

### **2.4.2 Ranking B: Paper-to-Paper Expansion**

複数論文を要する質問には、**質問文との語彙的な一致が弱い論文**が含まれる。この層は質問を 起点とする検索では回収しにくい。そこで**質問を起点としない第2のランキング**を構成する。

**起点（anchor）** は、ランキング A の1位と、Verifier（§2.3.2）が質問を支持しうると判定した 論文の和集合である。1本の起点では1つのトピッククラスタしか展開できないため、Verifier の 判定を起点の拡張に用いる。

各起点について、3種類の近さで近傍論文を取得する。

| Source | Definition of proximity |
| --- | --- |
| SPECTER2 | Semantic similarity from proximity embeddings (title+abstract) |
| bibliographic coupling | Jaccard over the arXiv IDs of the reference lists |
| full-text MLT | More-like-this against the paper-level BM25 index |

同一ソース内では起点ごとの近傍リストを交互配置して1本にまとめ、**ソース間は RRF (k = 60) で融合する**。すなわち、複数の起点に支持されたことは加点せず、**複数のソースに現れたこと だけを加点する**。前者は同じ質問に対する複数の入口であり合意を数える意義が小さいが、 後者は異なる種類の近さであるためである。これが**ランキング B** である。

**ランキング B は Reranker に通さない。** Reranker は質問との適合で判定するため、 質問文が言及しない論文を下げてしまう。ランキング B が対象とするのはまさにその層である。

### **2.4.3 A/B Rank Fusion**

2本のランキングを順位融合して最終的な候補列とする。

**ランキング A と B を等重み、k = 10 の RRF で統合する。** 各融合の式は Supplementary Material に示す。

上位50本を候補列として Reader へ渡す。**スコアではなく順位のみを用いる**——Reranker の yes 確率と近傍の類似度はスケールが異なり、直接加算すると一方が支配するためである。

融合定数を索引融合（§2.2.2、k = 60）より小さくするのは、**両方のランキングに現れることを 過大評価しないため**である。k が大きいと、A と B の双方で下位に現れる論文が、A の1位の 論文を上回りうる。k = 10 では、両方の上位に位置する論文だけが上回る。

---

---

## **2.5 Reader**

Reader は、Retriever が渡した候補論文50本と、MinerU で構造化した論文本文・表・図、および
質問と回答 schema を入力として、**提出論文・根拠箇所・回答**を生成する。以降、
**trusted images** とは、MinerU が論文から抽出し、preflight（§Appendix A.4）を通過した
画像を指す。処理は
**論文単位の読解（Stage 1）と質問単位の統合（Stage 2）**の二段からなり、LLM が論文の意味
読解を担い、**決定論的な検証は Python 側で行う**。

```
  Question + answer schema        Top-50 candidate papers        MinerU chunks
  (question / answer_types /       (from the retriever)          + trusted images
   options / table schema)                 │                            │
            │                              │                            │
            └──────────────┬───────────────┴────────────────────────────┘
                           ▼
 ┌──────────── Stage 1: paper-local reading  (per query–paper pair) ────────────┐
 │  [LLM]  relevance / usable evidence / evidence chunk IDs                      │
 │           │                                                                   │
 │           ▼                                                                   │
 │  [Python] ID and ownership validation                                         │
 │           chunk が対象論文に実在するか、未登録 ID でないか                       │
 │           → 3条件（関連性・利用可能性・有効な chunk）を満たす論文だけ通す         │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
 ┌──────────── Stage 2: cross-paper synthesis  (per query) ─────────────────────┐
 │  [LLM]  原子事実 / 決定論的演算 / 回答断片への binding / 根拠の対応づけ           │
 │           │                                                                   │
 │           ▼                                                                   │
 │  [Python] deterministic validation                                            │
 │           JSON schema・所有関係・画像添付・演算の再計算・                         │
 │           選択肢の label と本文・table の列と型と row key・locator 形式           │
 │           │                          │                                        │
 │           │ pass                     │ fail → 失敗理由つきの修復 prompt         │
 │           │                          └────────► 再検証（不整合なら不採用）      │
 └───────────┼───────────────────────────────────────────────────────────────────┘
             ▼
   Submitted paper IDs  /  evidence locators (page・section・object ID)  /  answer
```

**Figure 2: Reader architecture.** LLM は論文の意味読解と回答構成を担い、**ID の所有関係・演算・出力契約の検査はすべて Python 側で行う**。検査に通らない出力は修復 prompt を経て再検証し、それでも整合しなければ採用しない（fail-closed、§2.5.4）。

### **2.5.1 Inputs and Trust Boundary**

Reader が用いるのは、質問応答時に公開されるフィールドに限る。すなわち `query_id`、
`benchmark`、`question`、`answer_types`、multiple choice の選択肢、table の列 schema である。
**正解論文・正解根拠・正解回答・開発用ラベルは prompt へ渡さない。** few-shot の選択と
質問の routing も、質問文と回答 schema だけから決める。

論文本文とメタデータは **untrusted data** として delimiter 内に置き、本文中の命令文を
system instruction として解釈しない。chunk ID・paper ID・画像パス・page・object ID は
Python 側で照合し、**モデルが生成した未登録の ID は採用しない**。

候補論文は質問本文と分離した sidecar で受け取る。loader は重複 paper、欠落・余分な query、
不連続な rank、および正解・開発専用フィールドの混入を拒否する。

### **2.5.2 Stage 1: Paper-Local Reading**

Stage 1 は各 query–paper ペアを独立に処理する。動作は2つのモードに分かれ、
**提出システムは候補選別モードを用いる。**

| Mode | 動作 | 提出システムでの使用 |
| --- | --- | --- |
| **候補選別モード** | 論文の関連性、利用可能な根拠の有無、根拠 chunk ID を判定する | **使用する** |
| 外部選定モード | 論文集合を**変更せず**、各論文から原子事実と source chunk を抽出する | 分析用 |

**提出論文は Reader が決定する。** Retriever は候補50本を順位付きで渡すが、そのうち
どれを提出するかは Stage 1 の判定に基づく。したがって Retriever の責務は
**50本の候補集合の被覆**であり、順位そのものではない（§3.1、§3.5）。

候補選別モードでは、**論文の関連性と「現在の context で回答できるか」を別々に判定する**。
これにより、質問に関係するが抽出 context だけでは回答できない論文を区別できる。Python は
根拠 chunk が対象論文内に実在することを確認し、3条件（関連性・利用可能性・有効な chunk）を
満たす論文だけを Stage 2 へ渡す。

外部選定モードは分析用である。**渡された論文集合を authoritative な入力として扱い、
追加・削除・再順位付けを行わない**ため、Retriever と Reader の性能を独立に分析できる。

### **2.5.3 Stage 2: Cross-Paper Synthesis**

Stage 2 は、検証済みの論文 context を質問単位で統合し、**原子事実、必要な決定論的演算、
最終回答への binding、根拠の対応づけ**を同時に構成する。原子事実は値・単位・出所の
paper ID と chunk ID を持つ。計算を要する質問では、演算の operand を原子事実に拘束する。
単純な lookup で答えられる質問に演算を追加しない。

対応する演算は、四則演算、平均、割合変化、count、argmax/argmin、比較、条件に合う選択肢の
同定である。**Python が演算結果を再計算し**、丸め、label と値の対応、入力事実との一致を
確認する。

回答形式は質問型に従う。multiple choice では、モデルが選択肢 label と選択肢本文の双方を
返し、Python が公開選択肢との一致を検査する。**選択肢の本文は根拠として扱わず**、選択に
必要な各条件を論文側の事実に結びつける。table では公開 schema の列名と型に従い、各行が
全列を持ち row key が一意であることを検査する。**原文に存在しない行や値を、表を埋める
目的で補わない。**

### **2.5.4 Deterministic Validation and Output**

LLM の出力は、採用前に Python が検査する。検査項目は、JSON schema と必須フィールド、
paper と chunk の所有関係、chunk ID の存在、必須画像の添付実績、事実値と演算 operand の
一致、演算結果と丸め、multiple choice の label と選択肢本文、table の列・型・row key の重複、
回答断片と事実・演算の binding、および根拠 locator の形式である。

検査に失敗した場合は、構造化した失敗理由を添えて修復 prompt を実行する。修復後も整合
しない出力は採用しない。**存在しない根拠を推測で補うよりも、失敗を明示する fail-closed
方針を採る。**

採用しなかった場合の提出物の扱いは §2.5.5 に定める。

視覚情報については、**質問に対応する図表画像が実際に provider へ添付された場合にのみ**
画像由来の事実を採用する。画像が無い状態で、caption や周辺本文だけから画素・軸・色・配置を
断定しない。

出力する根拠は、内部の chunk ID を **coarse locator**（page または section、および table_id /
figure_id / equation_id / citation_id）へ変換したものである。**最終回答を直接支える最小の
locator 集合だけ**を出力し、Stage 1 で読んだ chunk や Stage 2 で追加した補助 context とは
区別する。

質問型の routing は、質問文と回答 schema だけから visual / citation / calculation / other の
主分類を決める。few-shot は**合成例のみ**を用い、実際の評価質問や正解を含めない。

LLM 呼び出しは、Stage 1 が「質問ごとの候補論文数の総和」、Stage 2 が「質問数」である。

### **2.5.5 Submission Contract**

fail-closed（§2.5.4）は幻覚を避けるための方針だが、提出は全 query について有効な
paper IDs・evidence locators・answer を要求する。**採用に至らなかった出力についても、
提出物としては schema を満たすレコードを返す。**

---

# **3. Experimental Results**

## **3.1 Setting**

**Task families.** 本稿ではタスクの公式ラベルをそのまま用いる。**`hidden_source_single_paper`**（26問）は gold 論文が1本の問題であり、**`multi_paper`**（29問）は複数の gold 論文をすべて回収する必要がある問題である（gold は3〜9本、中央値4本）。後者は候補被覆の難しさが大きく異なるため、以降の表では両者を分けて報告する。

**Roles of the two datasets.** validation（55問）は構成選択に、test（71問）は最終評価に用いる。validation には gold が付与されているため、索引構成・融合単位・展開設定はすべてこのデータ上で選んだ。 test には gold がないため、最終的な性能は公式リーダーボードのスコアで判断する。

Retriever の評価には `candidate_recall@k`（cr@k）を用いる。**クエリごとに、そのクエリの gold 論文のうち何本が候補列の上位 k 本に含まれたかの割合を求め、全クエリで平均する**（macro 平均）。gold を1本でも含めれば1とする定義ではない。たとえば gold が4本のクエリで2本しか候補に入らなければ、そのクエリの値は 0.5 である。`hidden_source_single_paper` は gold が1本なので、値は0か1のいずれかになる。

Reader を含む最終提出の評価には、公式オンライン評価器が返す指標をそのまま用いる（§3.6）。

**`candidate_recall@k` は Reader へ渡す候補集合の上流被覆を測る代理指標である。** 最終的な論文選択・根拠位置・回答の性能は、Reader を含む公式の end-to-end 指標（Table 5）でのみ評価する。両者は測っている対象が異なるため、直接比較しない。

再現に必要な設定を次にまとめる。会議の完全な一覧、収集日、欠損 PDF・欠損参考文献の扱い、 prompt と decoding 設定の全文は Supplementary Material に置く。

| Item | Setting |
| --- | --- |
| Retrieval corpus | 27,487 papers / 2,564,545 chunks（会議一覧は Supplementary Material） |
| Chunking | 最大2,000文字、5種類（`text_span` / `table` / `figure` / `equation_algorithm` / `title_abstract`） |
| Lexical retrieval | `bm25s` ライブラリ（**Lucene 互換のスコアリング方式**、k1=1.5, b=0.75）。Lucene 自体はバックエンドとして用いない。チャンク単位と論文単位の2索引 |
| Dense retrieval | Qwen3-Embedding-8B（4,096次元, fp16）、FAISS `IndexFlatIP` |
| Reranker | Qwen3-Reranker-8B（fp16, max input 2,048 tokens）、`pool_k = 200` |
| LLM (decomposition and verifier) | GPT-5.4（Azure OpenAI）、両者とも同一のデプロイ。`reasoning_effort: medium`、`max_completion_tokens: 16000` |
| LLM budget | 1クエリあたり最大6回（分解1 + 判定3 + 再分解2）、反復は最大3周 |
| Candidate budget | Reader へ渡す候補論文50本 |
| Compute | NVIDIA RTX 3090 上で実行。LLM は API 経由 |

**モデルのバージョンと prompt.** 埋め込み・reranker・SPECTER2 の checkpoint revision、 GPT-5.4 のモデル snapshot、API version、prompt version、および `table` チャンクの判定規則は Supplementary Material に記載する。

**decoding の非決定性.** 使用したデプロイは温度指定を受け付けず、seed も固定できない。 したがって**同一入力でも実行間で出力が変わりうる**。本稿の LLM を含む比較は各構成1回の 実行であり、§3.5 の (a) と (c) の対照だけがこの影響を受けない。

## **3.2 Validation Set**

Retriever は **candidate_recall@50 で 0.9682**、@20 で 0.9136 に達する（Table 1）。 問題型別に見ると、`hidden_source_single_paper` は @5 以降 1.0000 に飽和する一方、`multi_paper` は @20 で 0.8362、@50 で 0.9397 であり、**難しさは複数論文問題に集中している**。

論文間展開（§2.4.2）の寄与は、同一の検索結果から候補列のみを再構成する対照で確認できる。 展開を外すと `multi_paper` の cr@20 は 0.8592 から 0.5862 へ低下する。この対照は LLM の 非決定性を含まないため、差は展開の有無のみに由来する。

**Selection criterion.** 構成選択の基準は @50 の被覆とした。Reader が候補列を順位で閾値化せず集合として 利用するため、Retriever が最適化すべきは上位順位の精度ではなく50本の被覆である。

## **3.3 Test Set**

test 71問には gold がないため、我々の手元で recall を測ることはできない。最終的な性能は 公式リーダーボードのスコア（Table 5）で判断する。validation は test と問題型の分布が 異なり（test には freeform が存在しない）、規模も55問と小さいため、validation 上の数値を そのまま test の性能として解釈することはできない。

我々の手元で確認できるのは、提出物として満たすべき最低条件だけである。すなわち、すべての クエリで候補が50本埋まること、および候補の `paper_id` をコーパスのメタデータから解決 できることの2点である。

| Item | Value |
| --- | --- |
| Queries | 71 / 71 |
| Candidate papers per query | 50（全クエリで最小・最大とも50） |
| Candidates with unresolved metadata | 0 |

**Table 4: Output sanity checks on the test inputs (71 queries).** Retriever の出力による測定である。これらは提出物の健全性を示すのみであり、性能を示すものではない。

## **3.4 Validation Results**

本節は**最終構成（§2 で述べたシステム、Table 2 の (d)）の内訳**である。同一の走行を、 Table 2 では構成間の比較のために total のみで示すのに対し、ここでは問題型別に分解して示す。 LitTraceQA の validation は `hidden_source_single_paper` 26問と `multi_paper` 29問からなり、 **両者で難しさが大きく異なる**ためである。

| k | `hidden_source_single_paper` (26) | `multi_paper` (29) | total (55) |
| --- | --- | --- | --- |
| 1 | 0.8462 | 0.2222 | 0.5172 |
| 5 | 1.0000 | 0.5852 | 0.7813 |
| 10 | 1.0000 | 0.7462 | 0.8662 |
| 20 | 1.0000 | 0.8362 | 0.9136 |
| 50 | 1.0000 | 0.9397 | 0.9682 |

**Table 1: `candidate_recall@k` of the final system on the validation set (55 queries, macro).** total の行は Table 2 の (d) と同一の値である。

**`hidden_source_single_paper` は @5 以降で 1.0000 に飽和する。** すなわち、質問が1本の論文を指す場合、 その論文はほぼ確実に候補上位5本に入る。したがって構成間の差はすべて `multi_paper` に現れ、 以降の比較（§3.5）も `multi_paper` を中心に読む必要がある。

## **3.5 System Variants**

**同一の検索結果から候補列のみを再構成した対照では、ランキング A にランキング B を加える ことで、`multi_paper` 問題の `candidate_recall@20` は 0.5862 から 0.8592 へ上昇した。** これは、質問起点のランキング A を論文間展開によるランキング B で補うことが、複数論文問題の 候補被覆に有効であることを示す。

本節では2つの主張を述べるが、**根拠となる比較が異なる**ため区別して扱う。

| 主張 | 根拠となる比較 | 性質 |
| --- | --- | --- |
| (1) ランキング B を加えると `multi_paper` の候補被覆が改善する | (a) と (c) | **単一要素の対照**。差はランキング B の有無のみ |
| (2) 最終構成は @50 で最良である | (d) の値 | **構成全体の到達点**。複数要素を含むため要素別の帰属はしない |

以下の表は構成ごとの `candidate_recall@k`（validation 55問、macro、total）である。 **(a) と (c) の対照を除き、行間の差を単一の要素に帰属させることはできない。**

| Configuration | @1 | @5 | @10 | @20 | @50 |
| --- | --- | --- | --- | --- | --- |
| (a) Ranking A only | 0.5293 | 0.6970 | 0.7293 | 0.7818 | 0.8136 |
| (b) No search agent | 0.5566 | 0.7005 | 0.8298 | 0.9182 | 0.9591 |
| (c) Base A+B | 0.5293 | 0.7843 | 0.8697 | **0.9258** | 0.9591 |
| (d) Final system | 0.5172 | 0.7813 | 0.8662 | 0.9136 | **0.9682** |

**Table 2: `candidate_recall@k` of four system variants (55 queries, macro, total).** (a) と (c) の @1 が一致するのは偶然ではない。§2.4.2 のとおり anchor をランキング B の 先頭に固定するため、**A/B 融合は候補1位を入れ替えない**。

各構成の内容は次のとおりである。

| Configuration | 構成 | (c) との差 |
| --- | --- | --- |
| (a) Ranking A only | (c) からランキング B を外す | 要素1つ（§2.4.2） |
| (b) No search agent | LLM を1回も呼ばない。生の質問1本で検索し、ランキング B の起点は候補1位のみ | 要素3つ（分解・反復・Verifier） |
| (c) Base A+B | 2索引、chunk-level RRF、Seed Expansion なし、Reranker は順位を置換 | — |
| (d) Final system | 3索引、paper-level RRF、Seed Expansion、Reranker の順位融合 | 要素4つ（下表） |

(c) と (d) の差は次の4点だけである。エージェント段（ランキング B と A/B 融合を含む）、 埋め込みモデル、Reranker のモデル、`per_index_k`、`pool_k`、属性フィルタはすべて同一である。

| 項目 | (c) Base A+B | (d) Final system |
| --- | --- | --- |
| Indexes | `bm25s` + dense（2本） | + `bm25s_paper`（3本、§2.2.1） |
| Fusion unit | chunk-level RRF | **paper-level RRF**（§2.2.2） |
| Seed Expansion | 無効 | 有効（§2.2.3） |
| Reranker | 順位を完全に置き換える | **順位融合**（§2.3.1） |

**(a) を除き、いずれの構成もランキング B を含む。**

`multi_paper` に限ると構成間の差はさらに大きい。

| Configuration | @5 | @10 | @20 | @50 |
| --- | --- | --- | --- | --- |
| (a) Ranking A only | 0.4253 | 0.4866 | 0.5862 | 0.6466 |
| (c) Base A+B | 0.5910 | 0.7529 | **0.8592** | 0.9224 |
| (d) Final system | 0.5852 | 0.7462 | 0.8362 | **0.9397** |

**Table 3: `candidate_recall@k` on `multi_paper` queries (29 queries, macro).**

**主張 (1)：ランキング B の効果（単一要素の対照）.** (a) と (c) は、**同一の走行結果から 候補列のみを組み直した対照**であり、サブクエリ・検索結果・Verifier の判定はすべて同一である。 したがって差はランキング B の有無だけに由来し、LLM の非決定性も含まない。その差は `multi_paper` の @20 で **+0.2730**（0.5862 → 0.8592）に達し、@50 でも +0.2758（0.6466 → 0.9224） である。**これは本稿で唯一の単一要素の比較であり、A/B 融合の効果はこの対照のみを根拠とする。**

**他の行は構成全体の比較である.** (b) は LLM を用いない構成であり、分解・反復・Verifier の 3つが同時に無効になる。Verifier が無いためランキング B の起点も候補1位のみに退化する。 したがって (b) と (c) の差を分解や反復の寄与として読むことはできない。

**とくに (c) と (d) の差を「A/B 融合の効果」と読んではならない。** 両者ともランキング B を 含んでおり、異なるのは上表の4要素である。この差は A/B 融合の寄与ではなく、 **ランキング A の作り方を変えたことによる差**である。

**主張 (2)：最終構成の到達点.** 最終構成 (d) は、A/B 融合に加えて paper-level fusion、 Seed Expansion、reranker の順位融合を含む構成であり、**@50（total 0.9682、multi 0.9397）で 4構成中の最良**である。一方 @20 では (c) を下回る。§3.2 のとおり Reader が候補50本を集合 として利用するため、最終提出には @50 で優る (d) を選んだ。 **この選択は4要素それぞれの寄与を主張するものではなく、構成全体としての到達点に基づく。**

## **3.6 Test Results**

test 71問の gold は公開されていないため、性能は公式オンライン評価器が返すスコアで報告する。 **列は評価器が実際に返した指標名をそのまま用いる。** test の answer type は multiple choice 50問と table 21問であり freeform を含まないため、freeform の列は設けない。

| Component | Metric | Score |
| --- | --- | --- |
| Paper retrieval | `paper_precision_macro` | — |
| Paper retrieval | `paper_recall_macro` | — |
| Paper retrieval | `paper_f1_macro` | — |
| Evidence grounding | `evidence_precision_macro` | — |
| Evidence grounding | `evidence_recall_macro` | — |
| Evidence grounding | `evidence_f1_macro` | — |
| Answer (multiple choice) | `multiple_choice_accuracy` | — |
| Answer (table) | `table_row_f1_macro` | — |
| Answer (table) | `table_cell_accuracy_macro` | — |
| Answer (table) | `table_cell_accuracy_micro` | — |
| Overall | （評価器が返す場合のみ、その公式名称で記載） | — |

**Table 5: Official leaderboard scores on the test set (71 queries).** スコアと leaderboard 上の順位は提出結果が確定した時点で記載する。

# **4. Conclusions**

本稿は、LitTraceQA に提出した学習不要の retrieve–rerank–read システムを報告した。 Retriever は**質問起点のランキング A と論文間展開のランキング B を別々に構成し、順位融合 する**。A は語彙一致と密ベクトルを論文単位で融合して質問への適合を担い、B は SPECTER2・ 書誌結合・全文 MLT の近傍によって、質問文が直接言及しない論文を補う。cross-encoder reranker はチャンクの並べ替えを担い、LLM verifier は反復の制御に限定して用いる。

**同一の検索結果から候補列のみを再構成した対照では、ランキング A にランキング B を加える ことで、`multi_paper` 問題の `candidate_recall@20` は 0.5862 から 0.8592 へ上昇した。** これは、質問起点のランキング A を論文間展開によるランキング B で補うことが、複数論文問題の 候補被覆に有効であることを示す。

最終構成は、この A/B 融合に加えて paper-level fusion、Seed Expansion、reranker の順位融合を 含み、validation の `candidate_recall@50` で4構成中の最良（total 0.9682）であった。構成は validation 55問の候補被覆に基づいて選んでおり、**Reader が候補50本を集合として利用すること から `candidate_recall@50` を選択基準とした**。最終的な性能は test 71問に対する公式スコア（Table 5）で報告する。

## **Limitations**

validation は55問と小規模であり、開発用と最終評価用に分割していない。同じ55問で誤り分析と 構成選択を行っているため、validation 上の数値には過適合の可能性がある。test の gold は 公開されていないため、我々の手元では test 上の誤り分析ができず、どの設計要素が test で 寄与したかも分離できない。

§3.5 のとおり、構成間の比較の多くは複数の要素が同時に異なるため、単一要素の寄与を 主張できるのは (a) と (c) の対照だけである。また LLM の decoding は非決定的で seed を 固定できず、LLM を含む比較は各構成1回の実行である。数ポイントの差は実行間のばらつきと 区別できない。

コーパスは2024〜2025年の主要会議に限られる。書誌結合は参考文献の arXiv ID に依存するため、 参照が解決できない論文では近傍が得られない。

## **Ethics and Reproducibility**

本システムは公開された学術論文と、公開モデル（Qwen3-Embedding-8B、Qwen3-Reranker-8B、 SPECTER2）および商用 LLM API（GPT-5.4）を用いる。個人情報を含むデータは扱わない。 コード、設定ファイル、prompt、索引構築手順、モデルのバージョン、外部資源の一覧は Supplementary Material として公開する。

---

## **References**

> 投稿版では ACL 形式の完全な書誌に置き換える。少なくとも次を引用する。 **LitTraceQA のタスク論文**、**BM25**（Robertson and Zaragoza）、**RRF**（Cormack et al.）、 **書誌結合**（Kessler）、**Qwen3-Embedding / Qwen3-Reranker**、**SPECTER2**（Singh et al.）、 **FAISS**（Douze et al.）、**`bm25s`**、**MinerU**、**GPT-5.4 / Azure OpenAI API**。 SPECTER は使用していないため、SPECTER2 のみを引用する。
>

---

# **Appendix A. Reader: Execution and Reproducibility**

## **A.1 Run Manifest**

各 run の manifest に次を保存する。query、candidate、paper metadata、reader config の hash、
runtime source file の hash、model deployment と API version、token 上限と evidence 上限、
image root と chunk index、paper-set policy。

## **A.2 Checkpoint and Resume**

Stage 1 の論文判定と Stage 2 の回答を**質問単位で checkpoint** し、prompt version と
few-shot ID を各 checkpoint および cache key に記録する。provider 呼び出しごとに
prepare / finalize event、request ID、token usage、error outcome を ledger へ記録する。
resume 時は input、config、runtime hash を照合し、**現在の構成と一致する checkpoint だけ**
を再利用する。

## **A.3 Throughput Control**

長い context と画像を含むため、worker 数だけでなく **token-per-minute を考慮した
global launch pacer** を用いる。429 応答を受けた場合は該当 job だけを再 queue し、
共有の並列度を下げる。成功が続く区間では上限まで段階的に回復させる。

## **A.4 Visual Inputs and PDF Fallback**

画像パスは明示的な image root の下へ再配置し、コーパスが持つ任意の絶対パスを直接開かない。
preflight は path traversal、root 外の symlink、破損画像、過大画像、未対応形式を拒否する。
OCR と画像表示が矛盾する場合は、**添付画像を primary evidence として扱い**、その選択を
trace に残す。PDF fallback を用いる場合は、取得元、PDF hash、page alignment、render 設定、
crop 範囲を記録する。**自動 fallback と対象を限定した手動復旧は別の実験条件として扱う。**

## **A.5 Context Construction**

MinerU 出力を title/abstract・本文 span・table・figure・equation/algorithm・citation context の
source type へ正規化する。各 record は paper ID、chunk ID、本文を持ち、source type と抽出
状況に応じて page、section、object ID、画像パスを metadata として保持する。長い論文は、
質問との語彙的一致・source type・caption・object ID・周辺文脈に基づいて**決定論的に圧縮**
する。1つの論文を通常処理で複数の LLM request へ分割せず、**1つの bounded paper context**
として Stage 1 へ渡す。

# **Appendix B. Reader: Planned Ablations**

読解側の各要素の寄与は、次の対照で評価できる。**本稿では未実施である。**

| Ablation | 何を切り分けるか |
| --- | --- |
| 候補選別モード / 外部選定モード | 提出論文の決定主体 |
| 質問型 routing あり / なし | routing の寄与 |
| 合成 few-shot あり / なし | few-shot の寄与 |
| text-only / visual input あり | 画像入力の寄与 |
| deterministic validator あり / なし | 決定論的検証の寄与 |
| supplemental context あり / なし | Stage 2 での追加 context の寄与 |
| raw table generation / source-verified table selection | 表回答の検証工程の寄与 |

報告する際は、paper selection、evidence grounding、answer correctness、invalid output 率、
provider call 数、token usage を**別々に**示す。

# **Appendix C. Reader: Implementation Map**

| Component | Main implementation |
| --- | --- |
| query / candidate loading | `src/littraceqa/candidate_handoff.py` |
| Stage 1 / Stage 2 reader | `src/littraceqa/aoai_pairwise_reader.py` |
| prompt routing and examples | `src/littraceqa/pairwise_prompts.py` |
| derivation validation | `src/littraceqa/answer_derivation.py` |
| MinerU normalization | `src/littraceqa/mineru_record.py` |
| visual preflight | `src/littraceqa/corpus_preflight.py` |
| evidence locator | `src/littraceqa/citation_locator.py` |
| run orchestration | `scripts/run_aoai_pairwise_reader.py` |
| checkpoint ledger | `src/littraceqa/pairwise_run_store.py` |
| table output contract | `src/littraceqa/query_requirements.py` |
| source-reviewed table composition | `src/littraceqa/table_adjudication.py` |

# **Appendix D. Reader: Known Constraints**

- OCR 誤りや欠落 figure は、正しい論文が選定されても回答を阻害する。
- 複数の同義な根拠位置がある場合、最小 locator の選択は一意でない。
- 表の row key と string cell は表面形に敏感であり、意味的同値性だけでは一致しない。
- 固定論文集合に必要な owner が含まれない場合、Reader だけでは paper recall を回復できない。
- PDF fallback と画像 crop の再現には、取得元と変換 manifest が必要である。

