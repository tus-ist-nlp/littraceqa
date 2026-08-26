"""Qwen3-Reranker（causal LM ベースの yes/no 判定型 reranker）によるチャンクの再ランク。

索引構築ではなくクエリ時の推論のみなので、Qwen3-Embedding の索引構築
（faiss_qwen3.py）のようなマルチGPU分散・memmap は不要で、GPUでバッチ推論すれば足りる。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from littraceqa.di_pipeline.accel import load_with_best_attn, maybe_compile
from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register

_DEFAULT_INSTRUCTION = (
    "Given a scientific question, retrieve passages from research papers that "
    "help identify or support the answer"
)
# Qwen3-Reranker 公式の prompt フォーマット。<Document> の後ろに続けて渡す本文だけを
# 別途トークナイズし、prefix/suffix の分だけ予算を引いてから切り詰める
# (単純にフォーマット済み文字列をまとめて max_length で切ると、末尾の suffix
#  <think></think> ごと吹き飛んで yes/no 判定の形式が壊れるため)。
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@register("reranker", "qwen3")
class Qwen3Reranker:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 16,
        max_tokens: int = 2048,
        instruction: str = _DEFAULT_INSTRUCTION,
        # rerank は 1 run で候補を数千件スコアリングするので torch.compile が
        # 回収できる。可変長入力なので dynamic=True でコンパイルする。
        # ただし **マルチGPU（devices 指定）時は自動で無効化される**（下記）。
        compile: bool = True,
        # **マルチGPU。** "cuda:1,cuda:2,cuda:3" のようにカンマ区切りで指定すると、
        # 各GPUにモデル複製を載せてスレッド並列でスコアリングする。reranker は
        # クエリのたびに pool_k 件を推論するので、pool_k を大きくすると1GPUでは
        # 実行時間が破綻する（8B・998件で1GPU 152.6秒 vs 3GPU 56.6秒 = 2.7倍）。
        # PyTorch の CUDA forward は GIL を解放するのでスレッドで実並列になる
        # （クエリのたびにプロセスを起こす索引ビルド方式のオーバーヘッドを避ける）。
        # 省略時は device 1枚（従来どおり）。
        devices: str | None = None,
        # **1バッチのパディング後トークン数の上限。** 件数固定(batch_size)だと、
        # 文書の長さがばらつくとき長い外れ値で「batch_size × 最長」がVRAMを食う。
        # 8B・論文単位の実測では batch_size=4 でピーク22GB、8/16 は即OOMした。
        # 件数ではなくトークン量で区切れば短い文書を多数詰めつつVRAMを一定に保てる
        # （索引構築 index/faiss_qwen3.py と同じ対策）。None なら batch_size 固定。
        max_batch_tokens: int | None = None,
    ):
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.compile = compile
        if devices:
            self.devices = [d.strip() for d in devices.split(",") if d.strip()]
        else:
            self.devices = [device]
        self.fp16 = fp16 and any(str(d).startswith("cuda") for d in self.devices)
        self.max_batch_tokens = max_batch_tokens

        # device -> (tokenizer, model)。_ensure_loaded で埋める。
        self._replicas: dict[str, tuple] = {}
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._no_id: int | None = None
        self._yes_id: int | None = None

    def _ensure_loaded(self) -> None:
        if self._replicas:
            return
        dtype = torch.float16 if self.fp16 else torch.float32
        # **マルチGPU（スレッド並列）のときは torch.compile を使わない。**
        # compile 済みモデルを複数スレッドから同時に呼ぶと dynamo が
        # 「FX symbolic trace of a dynamo-optimized function is not supported」
        # で落ちる。compile は実測でほぼ無効果（8B論文単位で188 vs 212ms）なので、
        # 並列化の利益と引き換えにしても損失はない。1GPU 構成でのみ compile する。
        use_compile = self.compile and len(self.devices) == 1
        for device in self.devices:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            model = (
                load_with_best_attn(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_name,
                    device,
                    dtype=dtype,
                )
                .to(device)
                .eval()
            )
            model = maybe_compile(model, use_compile)
            self._replicas[device] = (tokenizer, model)
        tokenizer, _ = self._replicas[self.devices[0]]
        self._prefix_ids = tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix_ids = tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._no_id = tokenizer.convert_tokens_to_ids("no")
        self._yes_id = tokenizer.convert_tokens_to_ids("yes")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        self._ensure_loaded()

        scores = self._score_all(query, [c.text for c in candidates])

        # score を rerank スコア（yes 確率）で**上書きして**返す。
        # 返り値の並び順だけに順位を乗せると、下流で消える。ReadingAgent は
        # 検索結果を chunk_id の dict に貯めてから r.score で並べ直すので
        # (agent/reading.py の _candidate_papers)、元の RRF スコアを残したままだと
        # rerank の順位が 100% 捨てられ、効果が「100件のプールから20件を選ぶ」
        # フィルタ効果だけになってしまう。
        # RRF スコアと yes 確率はスケールが違うが、reranker を設定した run では
        # 全チャンクが必ずここを通るので、1 run の中でスケールが混ざることはない。
        ranked = sorted(
            (replace(c, score=float(s)) for c, s in zip(candidates, scores)),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:top_k]

    # ---- スコアリング（マルチGPU・トークン予算バッチ）----------------------

    def _score_all(self, query: str, texts: list[str]) -> list[float]:
        """全文書をスコアリングする（元の texts の順で返す）。

        長さ順にソートしてバッチを組み、各GPU複製に配ってスレッド並列で流す。
        書き戻しは元インデックスで行うので、処理順が変わっても結果は変わらない。
        """
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches = self._make_batches(order, texts)
        scores = [0.0] * len(texts)

        def run(device: str, batch: list[int]) -> None:
            for local_score, idx in zip(
                self._score_on(device, query, [texts[i] for i in batch]), batch
            ):
                scores[idx] = local_score

        if len(self.devices) == 1:
            for batch in batches:
                run(self.devices[0], batch)
            return scores

        # バッチをラウンドロビンでGPUに割り当て、スレッドで並列実行する。
        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            futures = [
                pool.submit(run, self.devices[i % len(self.devices)], batch)
                for i, batch in enumerate(batches)
            ]
            for future in futures:
                future.result()
        return scores

    def _make_batches(self, order: list[int], texts: list[str]) -> list[list[int]]:
        """長さ昇順の order を、件数またはトークン予算でバッチに割る。

        max_batch_tokens 指定時は「(件数+1) x 新要素のトークン数 <= 予算」で切る。
        order は昇順なので後から入る要素が常にバッチ内最長になり、この1条件で
        パディング後トークン数を予算以下に保てる（索引構築の _token_budget_batches
        と同じ理屈）。トークン数はモデルの tokenizer で正確に測る。
        """
        if not self.max_batch_tokens:
            return [
                order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)
            ]

        tokenizer, _ = self._replicas[self.devices[0]]
        overhead = len(self._prefix_ids) + len(self._suffix_ids)
        batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            tokens = (
                min(len(tokenizer.encode(texts[index], add_special_tokens=False)), self.max_tokens)
                + overhead
            )
            if current and (len(current) + 1) * tokens > self.max_batch_tokens:
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        return batches

    def _score_on(self, device: str, query: str, texts: list[str]) -> list[float]:
        """指定デバイスの複製で1バッチをスコアリングする。"""
        tokenizer, model = self._replicas[device]
        pairs = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {text}"
            for text in texts
        ]
        budget = self.max_tokens - len(self._prefix_ids) - len(self._suffix_ids)
        encoded = tokenizer(
            pairs,
            padding=False,
            truncation=True,
            max_length=budget,
            add_special_tokens=False,
        )
        input_ids = [self._prefix_ids + ids + self._suffix_ids for ids in encoded["input_ids"]]
        # 各系列は budget で既に max_tokens 以下に切り詰め済みなので、ここでの
        # max_length 指定は不要 (指定すると padding=True 時に警告が出るだけ)。
        padded = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**padded).logits[:, -1, :]
        pair_logits = logits[:, [self._no_id, self._yes_id]]
        log_probs = torch.nn.functional.log_softmax(pair_logits, dim=-1)
        return log_probs[:, 1].exp().float().cpu().tolist()
