# 検索上位に入った「gold でない論文」の分析

候補上位5本のうち gold として登録されていない論文について、
(1) 質問とどんな関係があるのか (2) なぜ gold より上位に来たのか を LLM に判定させた。

- 判定: `scripts/audit_nongold.py` → `audits/nongold_audit.jsonl`（1クエリ1呼び出し。gold の
  タイトル・abstract・根拠の所在を同じプロンプトに入れて比較させている）
- 集計: `scripts/nongold_report.py`（判定ロジックは持たない）
- 予測: `predictions_8b_chunk_expand_fused_offline.jsonl`
- 対象: **199件**（55クエリ × 上位5本 − gold 本数）。
  うち **95件**が少なくとも1本の gold より上位
  （single 1件 /
  multi 94件）

## 質問との関係

| 区分 | 件数 | 割合 |
|---|---|---|
| 用語が重なるだけ | 58 | 29.1% |
| gold と同クラスタのピア | 51 | 25.6% |
| 同じ手法・別タスク | 34 | 17.1% |
| 同じ主題・別の知見 | 28 | 14.1% |
| 無関係 | 20 | 10.1% |
| 判定できず | 3 | 1.5% |
| gold になりうる（付与漏れ疑い） | 3 | 1.5% |
| gold の引用/ベースライン | 2 | 1.0% |

task_family 別の内訳:

- single_paper 104件: 用語が重なるだけ 36, gold と同クラスタのピア 27, 同じ手法・別タスク 20, 同じ主題・別の知見 11
- multi_paper 95件: gold と同クラスタのピア 24, 用語が重なるだけ 22, 同じ主題・別の知見 17, 同じ手法・別タスク 14

## gold より上位に来た理由

**gold を追い越した 95件だけ**を分母にした内訳
（追い越していない論文は `not_applicable` になるので分母から外す）:

| 区分 | 件数 | 割合 |
|---|---|---|
| 質問文の語がそのまま出る | 37 | 38.9% |
| その主題の代表論文 | 35 | 36.8% |
| gold の根拠が本文の奥にある | 13 | 13.7% |
| gold が名指しされないピア | 5 | 5.3% |
| 質問が gold を一意に指せない | 5 | 5.3% |

## 関係 × gold を追い越したか

| 関係 | gold を追い越した | 追い越していない | 計 |
|---|---|---|---|
| 用語が重なるだけ | 22 | 36 | 58 |
| gold と同クラスタのピア | 24 | 27 | 51 |
| 同じ手法・別タスク | 14 | 20 | 34 |
| 同じ主題・別の知見 | 17 | 11 | 28 |
| 無関係 | 13 | 7 | 20 |
| 判定できず | 1 | 2 | 3 |
| gold になりうる（付与漏れ疑い） | 3 | 0 | 3 |
| gold の引用/ベースライン | 1 | 1 | 2 |

## gold 付与漏れの疑い

`could_be_gold = true`（その論文だけで質問に答えられると判定された）は **3件**、
`relation = possible_gold` は **3件**。

- **q_029 / 4位** `neurips2025_04957` Uni-Instruct: One-step Diffusion Model through Unified Diffu
  - This is a broad one-step diffusion benchmark/unification paper that explicitly mentions SiD and reports CIFAR-10 unconditional FID, so it is highly likely to include comparison-table entries for sever
- **q_045 / 1位** `acl2025_02980` Visual Evidence Prompting Mitigates Hallucinations in Large 
  - LVLM hallucination mitigation の同系統論文で、POPE/LLaVA-1.5の比較表に VTI や MoD をベースラインとして載せている可能性が高い。そうであれば、この論文単体から質問の数値を拾える。
- **q_045 / 4位** `iclr2025_00910` Do You Keep an Eye on What I Ask? Mitigating Multimodal Hall
  - 同じ LVLM hallucination mitigation のデコーディング系手法で、POPE のような hallucination benchmark 上で LLaVA-1.5 と既存法を比較している可能性が高い。比較表に VTI と MoD の数値が載っていれば、この論文だけで質問に答えられる。

## 事例（gold を追い越したもの、順位の高い順）

**q_021 / 1位 — SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Act**（CVPR 2025, `cvpr2025_02317`）

- 質問: Across all venues, which VLM-based driving paper trained only on the Bench2Drive Base dataset (1000 clips) achieves the highest driving score on the Bench2Drive
- 関係: **gold と同クラスタのピア** — Bench2Drive での閉ループ自動運転を扱う VLM 系の近接論文で、質問の主題には強く関係します。 ただし abstract だけでは Bench2Drive Base 1000 clips のみで学習した条件や、その条件下で全会場合算の最高 driving score かどうかは確定できません。
- 上位化の理由: **gold の根拠が本文の奥にある** — SimLingo 側は title/abstract に "Vision-Language", "autonomous driving", "Bench2Drive benchmark", "state-of-the-art performance" が表に出ています。 一方 ORION の決め手である "Bench2Drive Base dataset (1000 clips) のみで学習" とその順位付けは Table 1 と本文奥の記述に依存するため、表層一致では不利だったと考えられます。

**q_022 / 1位 — HPS: Hard Preference Sampling for Human Preference Alignment**（ICML 2025, `icml2025_01272`）

- 質問: Among ICML 2025 papers that propose a reference-free preference-optimization objective for LLM alignment without using a frozen reference model, what is each pr
- 関係: **gold と同クラスタのピア** — ICML 2025 の LLM の human preference alignment 論文で、選好に基づく新しい学習損失を提案している点では近いです。 ただし abstract からは質問の核である「frozen reference model を使わない reference-free preference-optimization objective」とは読めず、gold の 3 手法にも含まれません。
- 上位化の理由: **その主題の代表論文** — タイトルと abstract にある「Hard Preference」「Human Preference Alignment」「LLM」が、質問の大きな話題中心に強く一致したため上位化したと考えられます。 一方で gold 側の決め手である reference-free / no frozen reference は abstract よりも手法説明や式に依存しやすく、表層一致で不利だった可能性があります。

**q_025 / 1位 — G en P ilot: A Multi-Agent System for Test-Time Prompt Optimization in**（EMNLP 2025, `emnlp2025_01201`）

- 質問: Across all venues, among 2025 inference-time / test-time scaling methods for text-to-image generation evaluated on the GenEval benchmark, what base model does e
- 関係: **gold と同クラスタのピア** — GenPilot is a 2025 test-time method for image generation and it is evaluated on GenEval, so it sits very close to the query. However, it is prompt-optimization and model-agnostic rather than one of the gold papers that report a specific text-to-image base model a scaling method builds on.
- 上位化の理由: **質問文の語がそのまま出る** — The title and abstract match the query very literally with terms like "test-time," "image generation," and "Geneval," and the abstract even mentions "test-time scaling methods." That strong lexical overlap can outrank gold papers whose crucial base-model evidence is not in the title and is sometimes deeper in the paper.

**q_029 / 1位 — Simple Distillation for One-Step Diffusion Models**（NeurIPS 2025, `neurips2025_04218`）

- 質問: What are the 1-step FID scores on unconditional CIFAR-10 for TCM, ECM-XL (with 102.4M training budget), iCT-deep, and SiD as reported in their respective papers
- 関係: **gold と同クラスタのピア** — This is a peer paper on one-step diffusion/distillation for image generation, so it is topically close to the query. However, it is not the source paper for TCM, ECM-XL, iCT-deep, or SiD, and the abstract does not show those exact baseline values.
- 上位化の理由: **gold の根拠が本文の奥にある** — The gold papers expose the asked baseline names and numbers mainly in Table 1 rather than in their abstracts. By contrast, this paper's title and abstract strongly emphasize "one-step diffusion models" and distillation, which gives it stronger surface-level relevance for retrieval.

**q_042 / 1位 — Improved Training Technique for Shortcut Models**（NeurIPS 2025, `neurips2025_02194`）

- 質問: What VAE latent channel mean normalization values do sCM and IMM use for their ImageNet experiments, and do they match?
- 関係: **gold と同クラスタのピア** — This is a peer paper in the same broad area of fast image generation on ImageNet: one-step/few-step generative models trained with specialized objectives. However, it is about shortcut models/iSM, not sCM or IMM, and it does not provide the VAE latent channel mean normalization values asked for.
- 上位化の理由: **その主題の代表論文** — Its title and abstract strongly match the same semantic neighborhood as the query and gold papers: ImageNet 256x256, one-step/few-step sampling, training techniques, and generative models. Those high-level topical cues are very visible in the abstract, whereas the gold answer itself lives in appendix-level details about VAE latent normalization.

**q_045 / 1位 — Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Lan**（ACL 2025, `acl2025_02980`）

- 質問: What POPE adversarial accuracy does VTI achieve on LLaVA-1.5, and what adversarial accuracy does MoD achieve on LLaVA-1.5?
- 関係: **gold になりうる（付与漏れ疑い）** — LVLM hallucination mitigation の同系統論文で、POPE/LLaVA-1.5の比較表に VTI や MoD をベースラインとして載せている可能性が高い。そうであれば、この論文単体から質問の数値を拾える。
- 上位化の理由: **gold の根拠が本文の奥にある** — 質問の鍵である POPE adversarial accuracy と LLaVA-1.5 上の数値は gold 側では表にあり、abstract からは見えにくい。一方この論文はタイトルと abstract の『Mitigates Hallucinations』『Large Vision-Language Models』が強く効き、同トピック論文として上位化された。

**q_020 / 2位 — MASTER : A Multi-Agent System with LLM Specialized MCTS**（NAACL 2025, `naacl2025_00722`）

- 質問: Which NAACL 2025 papers explicitly mention or reference MCTS (Monte Carlo Tree Search)  in their primary method/framework figure?
- 関係: **同じ手法・別タスク** — This is a NAACL 2025 paper centered on an MCTS-based LLM framework for multi-agent problem solving. It is highly related to the query by method, but the provided evidence does not establish that its primary method/framework figure explicitly labels MCTS, which is the gold criterion.
- 上位化の理由: **質問文の語がそのまま出る** — It has the exact query term "MCTS" in the title and the full phrase "Monte Carlo Tree Search (MCTS)" in the abstract, along with "framework" language. Those strong surface-form matches can rank above gold papers whose relevance depends specifically on figure evidence.

**q_022 / 2位 — BOPO: Neural Combinatorial Optimization via Best-anchored and Objectiv**（ICML 2025, `icml2025_00363`）

- 質問: Among ICML 2025 papers that propose a reference-free preference-optimization objective for LLM alignment without using a frozen reference model, what is each pr
- 関係: **同じ手法・別タスク** — 「preference optimization」を使う新手法という点は似ていますが、対象は LLM alignment ではなく neural combinatorial optimization です。 質問が求める LLM の reference-free alignment objective とはタスクが違います。
- 上位化の理由: **質問文の語がそのまま出る** — タイトルの「Preference Optimization」に加え、abstract の「removing reliance on reward models or reference policies」が質問中の「reference-free」「without using a frozen reference model」と強く表層一致しています。 そのため、LLM ではないという重要な不一致より語句一致が優先されて gold を追い越したと考えられます。

