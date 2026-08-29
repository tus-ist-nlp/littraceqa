"""Dense retrieval with Qwen3-Embedding, over FAISS.

Embeds a Chunk's text with Qwen3-Embedding into a FAISS inner-product index
(`IndexFlatIP`). The embeddings are L2-normalised, so the inner product *is* cosine
similarity.

Qwen3-Embedding tells the document side from the query side by prefix
("passage: " / "query: ") and pools on the last non-padding token.

**At this scale the build works differently from SPECTER2's** (110M params, 768
dimensions):

* **fp16 is not optional.** Qwen3-Embedding-8B is about 32GB at fp32 and does not
  fit an RTX 3090 (24GB). At fp16 the weights are 15.1GB and the peak is 19.3GB.
* **The embeddings are written straight to a memmap on disk, never held in RAM.**
  4096 dimensions x 2.56M chunks is 42GB. Collecting them in a list, then
  np.concatenate, then copying into faiss peaks at 126GB and dies on a 125GB
  machine. They go to the memmap and enter faiss a slice at a time.
* **The work is split across GPUs, data-parallel.** At the measured 5.7 chunks/s
  one GPU needs about 124 hours for 2.56M chunks. Each GPU holds the whole model
  (15GB) and takes a share of the chunks; splitting the model itself
  (device_map="auto") does not raise throughput, so it is not used.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from littraceqa.search.accel import load_with_best_attn, maybe_compile
from littraceqa.search.contracts import (
    Chunk,
    RetrievalResult,
    filter_chunk_types,
    read_chunks_jsonl,
    write_chunks_jsonl,
)

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"
_EMBEDDINGS_FILENAME = "_embeddings.memmap"  # scratch file, only during a build
# Per-row "this row is embedded" flags (uint8), so an interrupted build can resume.
_DONE_FILENAME = "_embeddings.done"

# How many rows enter faiss at a time — the whole 42GB is never touched at once.
_ADD_ROWS = 100_000


# The production embedding settings, shared by `search.pipeline` and by
# `scripts/build_faiss_qwen3_shard.py` (the distributed build). **A rebuild has to
# use exactly the values search uses**, so they are written once, here, rather than
# in both places — that is what makes a model or prefix mismatch impossible.
# `devices` is deliberately not among them: a build takes every free GPU, while
# search uses devices[0] only and leaves the rest to the reranker.
INDEX_NAME = "faiss_qwen3_8b"
PRODUCTION_PARAMS: dict = {
    "model": "Qwen/Qwen3-Embedding-8B",
    "fp16": True,
    "max_tokens": 8192,
    "doc_prefix": "passage: ",
    "query_prefix": "query: ",
    "batch_size": 8,
}


class Qwen3FAISSIndex:
    name = "faiss_qwen3"

    def __init__(
        self,
        index_dir: str,
        model: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 8,
        device: str = "cuda",
        devices: str | None = None,
        fp16: bool = True,
        # torch.compile the build's hot loop. Worth it over millions of chunks; a
        # single query-time embedding never earns back the warmup, so it is not
        # compiled there.
        compile: bool = True,
        chunk_types: list[str] | None = None,
        # Sampling the corpus, a 1024-token cut truncates table chunks badly
        # (p99 = 3497 tokens, max 6310); text_span / equation / figure are almost
        # all under 1024, so they barely notice. Qwen3-Embedding has a native 32K
        # context, so this sits at 8192 with room to spare and **the back half of a
        # table stops going missing from the evidence**. The long outliers are under
        # 1.5% of all chunks, and _embed_shard sorts by length before batching, so
        # the cost lands on their batch alone.
        max_tokens: int = 8192,
        doc_prefix: str = "passage: ",
        query_prefix: str = "query: ",
        # **This is what actually prevents OOM.** Chunks are sorted by length before
        # batching, so a fixed count collects the long outliers into the last
        # batches, whose padded token count is batch_size x max_tokens (65,536 at
        # batch_size=8, max_tokens=8192). Measured, 99% of chunks are 738 tokens or
        # fewer and only 0.002% reach 8192 — so budgeting "rows x longest row in the
        # batch" instead of rows alone holds peak VRAM constant *and* lets the vast
        # majority of batches grow larger, which is faster. None keeps the old
        # fixed batch_size.
        max_batch_tokens: int | None = None,
        # How many times a batch that OOMs is halved and retried. Insurance against
        # a 30-hour build being lost to one batch near the end.
        oom_retries: int = 4,
        # Reuse a partly written memmap and carry on. Per-row completion flags live
        # in _embeddings.done, and finished rows are skipped.
        resume: bool = True,
        # How often (in batches) the completion flags — _embeddings.done, a few
        # hundred KB — are written out.
        #
        # **The embeddings themselves (14GB+) are NOT msynced here, only once at the
        # end.** np.memmap.flush() msyncs the *entire* mapping, and because the rows
        # were sorted by length the written ones are scattered through the whole
        # file. On a spinning disk the head crosses 14GB and blk-wbt's writeback
        # throttling (rq_qos_wait) **blocked for over 21 minutes** (nlp02's /data is
        # a TOSHIBA MQ04ABD2, rotational=1). Several workers on one shard make it
        # worse still, msyncing the same file at once.
        #
        # A write to an mmap survives the process dying — it stays in the page cache
        # and the kernel writes it back. **msync only protects against power loss**,
        # not against a kill or a crash, so resuming never needs it, and on an HDD
        # that insurance is not worth its price.
        flush_every: int = 200,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        # Comma-separated, e.g. "cuda:0,cuda:1,cuda:2"; unset means the single `device`.
        if devices:
            self.devices = [d.strip() for d in devices.split(",") if d.strip()]
        else:
            self.devices = [device]
        self.fp16 = fp16 and any(d.startswith("cuda") for d in self.devices)
        self.compile = compile
        # Which chunk_types go into the index (None takes them all) — for embedding
        # body text alone, one passage at a time.
        self.chunk_types = chunk_types
        self.max_tokens = max_tokens
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.max_batch_tokens = max_batch_tokens
        self.oom_retries = oom_retries
        self.resume = resume
        self.flush_every = flush_every

        self._tokenizer = None
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    # ---- building the index -----------------------------------------------

    def _prepare_memmap(
        self, memmap_path: Path, done_path: Path, n: int, dim: int
    ) -> bool:
        """Make the embedding memmap and the completion flags; True if resuming.

        With resume=True, a memmap and flag file left from last time are picked up
        **only when the row count and dimension match**. A different count (the
        chunks were rebuilt, say) means starting over.
        """
        expected_bytes = n * dim * 4
        can_resume = (
            self.resume
            and memmap_path.exists()
            and done_path.exists()
            and memmap_path.stat().st_size == expected_bytes
            and done_path.stat().st_size == n
        )
        if can_resume:
            return True

        if memmap_path.exists() or done_path.exists():
            print(
                f"  {self.name}: the scratch files from last time have the wrong "
                f"row count, rebuilding them (expected {expected_bytes:,} bytes)"
            )
        # Allocate the 42GB on disk. It never goes into RAM.
        np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n, dim)).flush()
        np.memmap(done_path, dtype="uint8", mode="w+", shape=(n,)).flush()
        return False

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = filter_chunk_types(chunks, self.chunk_types)
        n = len(self._chunks)
        if n == 0:
            raise ValueError(
                f"no chunk matches chunk_types={self.chunk_types}"
            )

        # Each worker reads its own share from this file. Pickling 2.5M texts across
        # processes would move several GB, so it goes through the filesystem instead.
        self._save_chunks()

        dim = AutoConfig.from_pretrained(self.model_name).hidden_size
        memmap_path = self.index_dir / _EMBEDDINGS_FILENAME
        done_path = self.index_dir / _DONE_FILENAME

        resumed = self._prepare_memmap(memmap_path, done_path, n, dim)
        already = int(np.memmap(done_path, dtype="uint8", mode="r", shape=(n,)).sum())

        print(
            f"  {self.name}: embedding {n:,} chunks x {dim} dims "
            f"({n * dim * 4 / 1e9:.1f}GB) across {len(self.devices)} GPU(s)"
        )
        if resumed:
            print(
                f"  {self.name}: resuming an interrupted build "
                f"({already:,} / {n:,} = {already / n * 100:.1f}% already done, skipping)"
            )
        self._embed_to_memmap(memmap_path, n, dim)

        index = faiss.IndexFlatIP(dim)
        embeddings = np.memmap(memmap_path, dtype="float32", mode="r", shape=(n, dim))
        for start in tqdm(
            range(0, n, _ADD_ROWS), desc=f"{self.name} faiss add", unit="slice"
        ):
            index.add(np.ascontiguousarray(embeddings[start : start + _ADD_ROWS]))
        del embeddings

        faiss.write_index(index, str(self.index_dir / _INDEX_FILENAME))
        memmap_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        self._index = index

    def _embed_to_memmap(self, memmap_path: Path, n: int, dim: int) -> None:
        """Split the chunks across the GPUs; each writes into the memmap directly."""
        args = [
            (
                rank,
                len(self.devices),
                device,
                str(self.index_dir / _CHUNKS_FILENAME),
                str(memmap_path),
                n,
                dim,
                self.model_name,
                self.batch_size,
                self.max_tokens,
                self.doc_prefix,
                self.fp16,
                self.compile,
                str(self.index_dir / _DONE_FILENAME),
                self.max_batch_tokens,
                self.oom_retries,
                self.flush_every,
            )
            for rank, device in enumerate(self.devices)
        ]

        if len(self.devices) == 1:
            _embed_shard(*args[0])
            return

        context = mp.get_context("spawn")  # CUDA cannot survive a fork
        processes = [context.Process(target=_embed_shard, args=a) for a in args]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"an embedding worker died (exitcode={failed})")

    # ---- search -----------------------------------------------------------

    def load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / _INDEX_FILENAME))
        self._chunks = self._load_chunks()

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        query_embedding = self._embed_query(query)
        scores, indices = self._index.search(query_embedding, k)
        results: list[RetrievalResult] = []
        for score, doc_index in zip(scores[0], indices[0]):
            if doc_index < 0:
                continue
            chunk = self._chunks[int(doc_index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(score),
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata,
                    source=self.name,
                )
            )
        return results

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            self._tokenizer, self._model = _load_model(
                self.model_name, self.devices[0], self.fp16
            )
        return _embed_texts(
            [self.query_prefix + query],
            self._tokenizer,
            self._model,
            self.devices[0],
            batch_size=1,
            max_tokens=self.max_tokens,
        )

    # ---- saving and loading the chunks -------------------------------------

    def _save_chunks(self) -> None:
        write_chunks_jsonl(self.index_dir / _CHUNKS_FILENAME, self._chunks)

    def _load_chunks(self) -> list[Chunk]:
        return read_chunks_jsonl(self.index_dir / _CHUNKS_FILENAME)


# ---- the worker side (module level, because it runs in another process) --------


def _load_model(model_name: str, device: str, fp16: bool, compile_model: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if fp16 else torch.float32
    model = (
        load_with_best_attn(AutoModel.from_pretrained, model_name, device, dtype=dtype)
        .to(device)
        .eval()
    )
    model = maybe_compile(model, compile_model)
    return tokenizer, model


def _embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_tokens: int,
) -> np.ndarray:
    """Embed the texts and return L2-normalised float32 vectors."""
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        pooled = _last_token_pool(hidden, encoded["attention_mask"])
        outputs.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(outputs, axis=0).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


# Characters per token, for estimating a batch's token count. Measured on English
# paper chunks it is about 4, but **over-estimating tokens is the safe direction**
# when dividing a budget, so the value used is deliberately lower.
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str, max_tokens: int) -> int:
    return min(int(len(text) / _CHARS_PER_TOKEN) + 1, max_tokens)


def _token_budget_batches(
    order: list[int], texts: list[str], max_batch_tokens: int, max_tokens: int
) -> list[list[int]]:
    """Cut `order` (ascending by length) into batches under a padded-token budget.

    Padding goes to the longest sequence in the batch, so what actually costs VRAM
    is **rows x longest row**. Because `order` ascends, whatever is added last is
    always the longest, which makes the test simply
    ``(current rows + 1) * the new row's tokens <= budget``.

    The long outliers therefore end up alone in their own batch, while the short
    chunks — 99% are 738 tokens or fewer — group into batches *larger* than a fixed
    count would allow.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        tokens = _estimate_tokens(texts[index], max_tokens)
        if current and (len(current) + 1) * tokens > max_batch_tokens:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def _embed_with_oom_retry(
    texts: list[str], tokenizer, model, device: str, max_tokens: int, retries: int
) -> np.ndarray:
    """On OOM, halve the batch and try again.

    Insurance so that a build of tens of hours is not lost to a single batch near
    the end. With the token-budget batching it normally never fires, but the token
    count is estimated from characters and the estimate can be wrong.
    """
    try:
        return _embed_texts(texts, tokenizer, model, device, len(texts), max_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if retries <= 0 or len(texts) == 1:
            raise
        middle = len(texts) // 2
        print(
            f"  OOM: splitting a batch of {len(texts)} into {middle} + {len(texts) - middle} and retrying",
            flush=True,
        )
        first = _embed_with_oom_retry(
            texts[:middle], tokenizer, model, device, max_tokens, retries - 1
        )
        second = _embed_with_oom_retry(
            texts[middle:], tokenizer, model, device, max_tokens, retries - 1
        )
        return np.concatenate([first, second], axis=0)


def _embed_shard(
    rank: int,
    world: int,
    device: str,
    chunks_path: str,
    memmap_path: str,
    n: int,
    dim: int,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    doc_prefix: str,
    fp16: bool,
    compile_model: bool = False,
    done_path: str | None = None,
    max_batch_tokens: int | None = None,
    oom_retries: int = 4,
    flush_every: int = 200,
) -> None:
    """Embed this worker's share alone, writing into its rows of the memmap."""
    # Work out this worker's range (a contiguous slice).
    start = n * rank // world
    end = n * (rank + 1) // world

    texts: list[str] = []
    with open(chunks_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            texts.append(doc_prefix + json.loads(line)["text"])

    tokenizer, model = _load_model(model_name, device, fp16, compile_model)
    embeddings = np.memmap(memmap_path, dtype="float32", mode="r+", shape=(n, dim))

    # The completion flags that let an interrupted build resume; finished rows are skipped.
    done = None
    if done_path is not None:
        done = np.memmap(done_path, dtype="uint8", mode="r+", shape=(n,))

    # Sorting by length before batching cuts the wasted padding and is much faster.
    # Writes go by row number, so processing order does not change the result.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    if done is not None:
        remaining = [i for i in order if not done[start + i]]
        if len(remaining) != len(order):
            print(
                f"  {device}: {len(order) - len(remaining):,} rows already done, skipping",
                flush=True,
            )
        order = remaining

    if max_batch_tokens:
        # Divide on padded tokens, not on row count (this is what prevents OOM).
        batches = _token_budget_batches(order, texts, max_batch_tokens, max_tokens)
    else:
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

    def commit(pending: list[int]) -> None:
        """Write out the completion flags only — a few hundred KB, so it is cheap.

        **The 14GB of embeddings are not msynced here.** That would msync the whole
        mapping, and since the written rows are scattered through the file, on an HDD
        blk-wbt catches it and blocks for a long time (21 minutes, measured). A write
        to an mmap stays in the page cache even if the process dies, so resuming
        never needs it.
        """
        if not pending:
            return
        for local_index in pending:
            done[start + local_index] = 1
        done.flush()

    pending: list[int] = []
    progress = tqdm(batches, desc=f"qwen3 emb {device}", unit="batch", position=rank)
    for batch_number, local_indices in enumerate(progress, start=1):
        vectors = _embed_with_oom_retry(
            [texts[i] for i in local_indices],
            tokenizer,
            model,
            device,
            max_tokens=max_tokens,
            retries=oom_retries,
        )
        for vector, local_index in zip(vectors, local_indices):
            embeddings[start + local_index] = vector
        if done is not None:
            pending.extend(local_indices)
            if batch_number % flush_every == 0:
                commit(pending)
                pending = []

    if done is not None:
        commit(pending)
    # The one and only msync of the embeddings. Syncing the whole 14GB mapping is
    # slow on an HDD, but this worker's share is finished, so nothing waits on it.
    embeddings.flush()
    del embeddings
    if done is not None:
        del done


def _last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the last non-padding token, whichever side the padding is on.

    This is the pooling Qwen3-Embedding officially uses.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, sequence_lengths]


# So a child process can initialise CUDA (spawn, never fork).
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
