"""FAISS + Qwen3-Embedding による dense 検索インデックス。

Chunk のテキストを Qwen3-Embedding でベクトル化し、FAISS の内積 (IndexFlatIP)
インデックスを構築する。埋め込みは L2 正規化してあるため、内積がそのまま
コサイン類似度になる。

Qwen3-Embedding は文書側/クエリ側を prefix (「passage: 」/「query: 」) で
区別し、プーリングは最後の非パディングトークンを使う。

SPECTER2 (110M, 768次元) とは規模が桁違いなので、索引構築の実装が違う:

* **fp16 必須。** Qwen3-Embedding-8B は fp32 だと約32GBで、RTX 3090 (24GB) に載らない。
  fp16 なら重み15.1GB・ピーク19.3GB で載る。
* **埋め込みは RAM に貯めずディスクの memmap に直接書く。** 4096次元 × 256万件 = 42GB。
  「全部リストに貯める → np.concatenate → faiss にコピー」だとピーク126GBになり、
  実RAM 125GB を超えて落ちる。memmap に書いてから faiss へスライス投入する。
* **複数GPUにデータ並列で分散する。** 実測 5.7 chunks/s（1GPU）なので、256万件は
  1GPUで約124時間かかる。各GPUがモデル全体（15GB）を持ち、チャンクを分担する。
  モデル分割（device_map="auto"）ではスループットは上がらないので使わない。
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

from littraceqa.di_pipeline.accel import load_with_best_attn, maybe_compile
from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.index.chunk_filter import filter_chunk_types

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"
_EMBEDDINGS_FILENAME = "_embeddings.memmap"  # 構築中の一時ファイル
# 行ごとの埋め込み完了フラグ(uint8)。中断したビルドを続きから再開するために使う。
_DONE_FILENAME = "_embeddings.done"

# faiss に投入するときのスライス幅。42GB を一度に触らないための刻み。
_ADD_ROWS = 100_000


# 本番構成の埋め込み設定。`di_pipeline.pipeline` と
# `scripts/build_faiss_qwen3_shard.py`（分散ビルド）が共有する。**索引を作り直すときは
# 検索時と同じ値でなければならない**ので、2箇所に書かない。
# `devices` だけは含めない——構築は空いている全GPUを使い、検索は devices[0] しか
# 使わないので1枚に絞る（残りを reranker に空ける）。
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
        # 索引構築のホットループを torch.compile する。数百万チャンクの埋め込みで
        # 効く。クエリ時の単発埋め込みは warmup が回収できないので compile しない。
        compile: bool = True,
        chunk_types: list[str] | None = None,
        # 実測(サンプリング)では1024トークンで切ると表(table)チャンクの
        # p99=3497・最大6310トークンが打ち切られる(text_span/equation/figureは
        # ほぼ全て1024未満で影響が小さい)。Qwen3-Embeddingはネイティブ32Kコンテキスト
        # なので余裕を持って8192に上げ、表の後半が根拠から欠落するのを防ぐ。
        # 長い外れ値はチャンク全体の1.5%未満で、_embed_shard は長さ順ソートしてから
        # バッチ化するため、影響は該当バッチのみに限定される。
        max_tokens: int = 8192,
        doc_prefix: str = "passage: ",
        query_prefix: str = "query: ",
        # **OOM 対策の本命。** チャンクは長さ順にソートしてからバッチ化されるので、
        # 件数固定のバッチだと末尾に長い外れ値が集まり「batch_size × max_tokens」の
        # パディング後トークン数になる（batch_size=8, max_tokens=8192 なら 65,536）。
        # 実測では全チャンクの99%が738トークン以下・8192に張り付くのは0.002%なので、
        # 件数ではなく「バッチ件数 × バッチ内最長トークン数」を予算で抑えたほうが、
        # ピークVRAMが一定になるうえ大多数のバッチはむしろ大きくできて速い。
        # None なら従来どおり batch_size 固定。
        max_batch_tokens: int | None = None,
        # 1バッチが OOM したときに、バッチを半分に割って再試行する回数。
        # 30時間級のビルドが末尾の1バッチで全滅するのを防ぐ。
        oom_retries: int = 4,
        # 途中まで書けた memmap を再利用して続きから埋める。行ごとの完了フラグを
        # _embeddings.done に持ち、既に埋めた行は飛ばす。
        resume: bool = True,
        # 完了フラグ(_embeddings.done, 数百KB)を書き出す間隔（バッチ数）。
        #
        # **埋め込み本体(14GB級)の msync はここでは行わず、最後に1回だけにする。**
        # np.memmap.flush() はマッピング「全体」に msync を掛ける。書き込む行は
        # 長さ順ソートの結果ファイル全体に散っているので、HDD だとヘッドが14GBを
        # 飛び回り、blk-wbt のライトバックスロットリング(rq_qos_wait)で
        # **21分以上ブロックした**（nlp02 の /data は TOSHIBA MQ04ABD2、rotational=1）。
        # とくに1シャードを複数ワーカーで回すと同じファイルを同時に msync して詰まる。
        #
        # mmap への書き込みはプロセスが死んでもページキャッシュに残り、カーネルが
        # 書き戻す。つまり msync が守るのは**電源断だけ**で、プロセスの kill や
        # クラッシュからの再開には要らない。HDD 上でその保険に払うコストに見合わない。
        flush_every: int = 200,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        # "cuda:0,cuda:1,cuda:2" のようにカンマ区切りで指定する。未指定なら device 1枚。
        if devices:
            self.devices = [d.strip() for d in devices.split(",") if d.strip()]
        else:
            self.devices = [device]
        self.fp16 = fp16 and any(d.startswith("cuda") for d in self.devices)
        self.compile = compile
        # 索引に入れる chunk_type を絞る（None なら全部）。本文だけを passage 単位で
        # 埋め込みたい場合に使う。
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

    # ---- 索引構築 ---------------------------------------------------------

    def _prepare_memmap(
        self, memmap_path: Path, done_path: Path, n: int, dim: int
    ) -> bool:
        """埋め込み用 memmap と完了フラグを用意する。再開できたら True。

        resume=True で、前回の memmap と完了フラグが**同じ件数・次元**で残っていれば
        それを引き継ぐ。件数が違う（チャンクを作り直した等）場合は作り直す。
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
                f"  {self.name}: 前回の中間ファイルは件数が合わないので作り直します"
                f"（期待 {expected_bytes:,} バイト）"
            )
        # 42GB をディスク上に確保する。RAM には載せない。
        np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n, dim)).flush()
        np.memmap(done_path, dtype="uint8", mode="w+", shape=(n,)).flush()
        return False

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = filter_chunk_types(chunks, self.chunk_types)
        n = len(self._chunks)
        if n == 0:
            raise ValueError(
                f"chunk_types={self.chunk_types} に一致するチャンクが1件もありません"
            )

        # ワーカーはこのファイルから自分の担当分を読む。2.5M件のテキストを
        # プロセス間で pickle すると数GBの転送になるので、ファイル経由にする。
        self._save_chunks()

        dim = AutoConfig.from_pretrained(self.model_name).hidden_size
        memmap_path = self.index_dir / _EMBEDDINGS_FILENAME
        done_path = self.index_dir / _DONE_FILENAME

        resumed = self._prepare_memmap(memmap_path, done_path, n, dim)
        already = int(np.memmap(done_path, dtype="uint8", mode="r", shape=(n,)).sum())

        print(
            f"  {self.name}: {n:,} 件 x {dim}次元 "
            f"({n * dim * 4 / 1e9:.1f}GB) を {len(self.devices)} GPU で埋め込み"
        )
        if resumed:
            print(
                f"  {self.name}: 中断した構築を再開します"
                f"（{already:,} / {n:,} 件 = {already / n * 100:.1f}% は完了済みで飛ばします）"
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
        """チャンクを GPU に分担させ、埋め込みを memmap に直接書き込む。"""
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

        context = mp.get_context("spawn")  # CUDA と fork は両立しない
        processes = [context.Process(target=_embed_shard, args=a) for a in args]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"埋め込みワーカーが異常終了しました (exitcode={failed})")

    # ---- 検索 -------------------------------------------------------------

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

    # ---- チャンクの保存・読み込み ------------------------------------------

    def _save_chunks(self) -> None:
        path = self.index_dir / _CHUNKS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _load_chunks(self) -> list[Chunk]:
        path = self.index_dir / _CHUNKS_FILENAME
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                chunks.append(Chunk(**record))
        return chunks


# ---- ワーカー側（別プロセスで走るのでモジュール直下に置く） --------------------


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
    """テキストを埋め込んで L2 正規化した float32 配列を返す。"""
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


# 文字数からトークン数を見積もる係数。実測（英語論文チャンク）では概ね4文字/トークン
# だが、予算を割るときは**多めに見積もる**ほうが安全なので小さめの値を使う。
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str, max_tokens: int) -> int:
    return min(int(len(text) / _CHARS_PER_TOKEN) + 1, max_tokens)


def _token_budget_batches(
    order: list[int], texts: list[str], max_batch_tokens: int, max_tokens: int
) -> list[list[int]]:
    """長さ昇順の order を「パディング後トークン数の予算」でバッチに割る。

    パディングはバッチ内の最長系列に合わせて行われるので、実際にVRAMを食うのは
    「バッチ件数 × バッチ内最長トークン数」。order は昇順なので、後から入る要素が
    常にそのバッチの最長になる。よって条件は
    ``(現在の件数 + 1) * 新要素のトークン数 <= 予算`` で足りる。

    これにより長い外れ値は自動的に1件ずつになり、短いチャンク（実測で99%が738
    トークン以下）はむしろ件数固定より大きなバッチにまとまる。
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
    """OOM したらバッチを半分に割って再試行する。

    数十時間かけたビルドが末尾の1バッチで全滅するのを防ぐための保険。
    予算方式でバッチを組んでいれば普通は発動しないが、トークン数の見積もりは
    文字数からの近似なので外れることがある。
    """
    try:
        return _embed_texts(texts, tokenizer, model, device, len(texts), max_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if retries <= 0 or len(texts) == 1:
            raise
        middle = len(texts) // 2
        print(
            f"  OOM: {len(texts)} 件のバッチを {middle} + {len(texts) - middle} に割って再試行します",
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
    """自分の担当分だけを埋め込み、memmap の該当行に書き込む。"""
    # 担当範囲（連続スライス）を決める
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

    # 中断したビルドを続きから再開するための完了フラグ。既に埋めた行は飛ばす。
    done = None
    if done_path is not None:
        done = np.memmap(done_path, dtype="uint8", mode="r+", shape=(n,))

    # 長さ順に並べてからバッチにすると、パディングの無駄が減って大幅に速くなる。
    # 書き込みは行番号で行うので、処理順が変わっても結果は変わらない。
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    if done is not None:
        remaining = [i for i in order if not done[start + i]]
        if len(remaining) != len(order):
            print(
                f"  {device}: {len(order) - len(remaining):,} 件は完了済みなので飛ばします",
                flush=True,
            )
        order = remaining

    if max_batch_tokens:
        # 件数ではなくパディング後トークン数で割る（OOM 対策の本命）。
        batches = _token_budget_batches(order, texts, max_batch_tokens, max_tokens)
    else:
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

    def commit(pending: list[int]) -> None:
        """完了フラグだけを書き出す（数百KBなので安い）。

        埋め込み本体(14GB)の msync はここでは**やらない**。マッピング全体への
        msync になり、書き込み行がファイル全体に散っているため HDD では
        blk-wbt に捕まって長時間ブロックする（実測21分以上）。mmap 書き込みは
        プロセスが死んでもページキャッシュに残るので、再開のためには不要。
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
    # 埋め込み本体の msync はここ1回だけ。HDD では14GBマッピング全体の同期に
    # 時間がかかるが、担当分を全部終えたあとなので誰も待たない。
    embeddings.flush()
    del embeddings
    if done is not None:
        del done


def _last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """パディング方向(left/right)によらず最後の非パディングトークンを取り出す。

    Qwen3-Embedding が公式に採用しているプーリング方法。
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, sequence_lengths]


# 子プロセスが CUDA を初期化できるようにしておく（spawn 前提）
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
