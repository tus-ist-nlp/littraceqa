"""Qwen3-Reranker: a causal-LM yes/no reranker over chunks.

This is query-time inference only, not index building, so it needs none of the
multi-GPU sharding or memmap machinery of the embedding build (faiss_qwen3.py) —
batched GPU inference is enough.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from littraceqa.di_pipeline.accel import load_with_best_attn, maybe_compile
from littraceqa.di_pipeline.contracts import RetrievalResult

_DEFAULT_INSTRUCTION = (
    "Given a scientific question, retrieve passages from research papers that "
    "help identify or support the answer"
)
# Qwen3-Reranker's official prompt format. The document text that follows
# <Document> is tokenised separately and truncated against a budget that already
# subtracts the prefix and suffix. Truncating the whole formatted string at
# max_length instead would cut off the trailing <think></think> and break the
# yes/no answer format.
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3Reranker:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 16,
        max_tokens: int = 2048,
        instruction: str = _DEFAULT_INSTRUCTION,
        # One run scores thousands of candidates, so torch.compile pays for itself.
        # Inputs are variable length, hence dynamic=True. **Automatically disabled
        # under multi-GPU (see below).**
        compile: bool = True,
        # **Multi-GPU.** A comma-separated list ("cuda:1,cuda:2,cuda:3") places a
        # replica on each GPU and scores in parallel threads. The reranker infers
        # over pool_k chunks on every query, so a large pool_k breaks run time on a
        # single GPU (8B, 998 chunks: 152.6s on one GPU vs 56.6s on three, 2.7x).
        # PyTorch's CUDA forward releases the GIL, so threads run genuinely in
        # parallel, avoiding the per-query process spawn the index builder uses.
        # Omitted means the single `device`.
        devices: str | None = None,
        # **Cap on padded tokens per batch.** With a fixed count, varying document
        # lengths mean one long outlier makes batch_size x longest eat the VRAM: at
        # 8B, batch_size=4 peaked at 22GB and 8 or 16 went straight to OOM. Budgeting
        # tokens instead packs many short documents while keeping VRAM flat (the
        # same fix as in index/faiss_qwen3.py). None keeps the fixed batch_size.
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

        # device -> (tokenizer, model), filled in by _ensure_loaded.
        self._replicas: dict[str, tuple] = {}
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._no_id: int | None = None
        self._yes_id: int | None = None

    def _ensure_loaded(self) -> None:
        if self._replicas:
            return
        dtype = torch.float16 if self.fp16 else torch.float32
        # **No torch.compile under multi-GPU (thread parallelism).** Calling a
        # compiled model from several threads at once makes dynamo fail with
        # 「FX symbolic trace of a dynamo-optimized function is not supported」
        # "FX symbolic trace of a dynamo-optimized function". compile measured
        # almost no benefit anyway (188 vs 212ms), so trading it for parallelism
        # costs nothing. Only single-GPU setups compile.
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

        # **Overwrite** score with the rerank score (the yes probability). A ranking
        # carried only by list order is lost downstream: ReadingAgent accumulates
        # results into a dict keyed by chunk_id and re-sorts by r.score
        # (_candidate_papers in agent/reading.py), so keeping the original RRF score
        # would discard 100% of the reranker's ordering and reduce it to a filter
        # that picks 20 out of a pool of 100.
        # RRF scores and yes probabilities are on different scales, but with a
        # reranker configured every chunk passes through here, so the scales never
        # mix within one run.
        ranked = sorted(
            (replace(c, score=float(s)) for c, s in zip(candidates, scores)),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:top_k]

    # ---- scoring (multi-GPU, token-budget batches) ------------------------------

    def _score_all(self, query: str, texts: list[str]) -> list[float]:
        """Score every document, returned in the original order of `texts`.

        Batches are formed after sorting by length and handed to the GPU replicas in
        parallel threads. Results are written back by original index, so a different
        processing order does not change the output.
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

        # Assign batches to GPUs round-robin and run the threads in parallel.
        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            futures = [
                pool.submit(run, self.devices[i % len(self.devices)], batch)
                for i, batch in enumerate(batches)
            ]
            for future in futures:
                future.result()
        return scores

    def _make_batches(self, order: list[int], texts: list[str]) -> list[list[int]]:
        """Split a length-ascending order into batches by count or token budget.

        With max_batch_tokens, a batch grows while (count + 1) x tokens(new element)
        stays within budget. Since the order ascends, the newest element is always the
        longest in the batch, which keeps padded tokens under the budget (the same
        reasoning as the index builder's token budgeting). Token counts come from the
        model's own tokenizer.
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
        """Score one batch on the replica of the given device."""
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
        # Each sequence is already truncated below max_tokens by the budget, so no
        # max_length here (specifying it only warns when padding=True).
        padded = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**padded).logits[:, -1, :]
        pair_logits = logits[:, [self._no_id, self._yes_id]]
        log_probs = torch.nn.functional.log_softmax(pair_logits, dim=-1)
        return log_probs[:, 1].exp().float().cpu().tolist()
