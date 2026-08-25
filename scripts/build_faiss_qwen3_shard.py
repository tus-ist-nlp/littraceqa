#!/usr/bin/env python3
"""faiss_qwen3 索引を「コーパスの担当スライスぶんだけ」ビルドする分散ビルド用スクリプト。

Qwen3-Embedding-8B の索引構築は4GPUでも約31時間かかる。これを複数マシン
(nlp01 + nlp02 の 8GPU) に分けて各マシンが担当スライスだけを埋め込み、あとで
scripts/merge_faiss_qwen3.py で1索引に合体することで所要時間を半減させる。

このスクリプトは faiss_qwen3 indexer 「1つだけ」を担当スライスで build する。
bm25s は全コーパスを1ドキュメント順で舐める必要があり分割に向かないので、
分散対象にしない(bm25s は既存の run_search.py --build で全コーパス版が
既に作られている前提)。faiss_qwen3.py 本体には一切手を入れない。

パラメータ(model/devices/batch_size/max_tokens 等)は search_style の yaml から
そのまま読むので、yaml と手打ちの値がズレる事故が起きない。索引の出力先も
compose_config と同じ規約 {index_dir}/{process}/{index_name} で導出する。

スライスの決め方は faiss_qwen3.py の _embed_shard と同じ整数分割式:
    start = n * shard_index // num_shards
    end   = n * (shard_index + 1) // num_shards
全マシンが「同一の chunks.jsonl(行の並びが1バイトも違わない)」を見る前提なので、
各マシンで num_shards を揃え shard_index だけを変えれば、重複なく全件を覆える。

使い方(例: nlp01=前半, nlp02=後半 の2分割):
    # nlp01 で
    uv run python scripts/build_faiss_qwen3_shard.py \
      --paths configs/paths/default.yaml \
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/8b.yaml \
      --shard-index 0 --num-shards 2

    # nlp02 で(chunks.jsonl を転送済み、nlp02 用 paths を使う)
    uv run python scripts/build_faiss_qwen3_shard.py \
      --paths configs/paths/nlp02.yaml \
      --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/8b.yaml \
      --shard-index 1 --num-shards 2

OOM対策は3段構えになっている(詳細は index/faiss_qwen3.py の各パラメータのコメント)。

1. **max_batch_tokens**: 件数固定ではなく「バッチ件数 x バッチ内最長トークン数」を
   予算で抑える。チャンクは長さ順にソートしてからバッチ化されるので、件数固定だと
   末尾に長い外れ値が集まり batch_size x max_tokens (8x8192=65,536) になる。
   実測のトークン分布は中央値265・p99 738・8192張り付きは0.002%なので、予算方式なら
   長い外れ値だけ自動的に1件ずつになり、大多数のバッチはむしろ大きくできて速い。
2. **oom_retries**: それでもOOMしたらバッチを半分に割って再試行する。
3. **resume**: 途中で落ちても _embeddings.done の完了フラグから続きを再開する。
   同じコマンドを再実行すればよい(数十時間のビルドをやり直さずに済む)。

さらに実行時に
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
を付けると断片化が緩和される。

空きGPUはマシンごとに違うので --devices で上書きする。8Bはfp16でもピーク19.3GB
使うため、**空きが20GB未満のGPUは指定しない**こと。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.contracts import Chunk

# config.py 全体を import すると marker/docling/siglip/colbert 等(figures extra 等)の
# 重い任意依存まで芋づるで読み込まれ、それらを入れていないマシン(分散ビルド先の
# nlp02 など)で import に失敗する。ここで必要なのは faiss_qwen3 indexer の登録だけ
# なので、そのモジュールだけを import して @register("indexer","faiss_qwen3") を効かせる。
import littraceqa.di_pipeline.index.faiss_qwen3  # noqa: F401


def load_config(path: str) -> dict:
    """yaml を dict で読む(config.py の load_config と同じだが重い import を避ける)。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def shard_bounds(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """faiss_qwen3.py の _embed_shard と同じ整数分割で担当範囲[start, end)を返す。"""
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index={shard_index} は 0..{num_shards - 1} の範囲で指定してください"
        )
    start = n * shard_index // num_shards
    end = n * (shard_index + 1) // num_shards
    return start, end


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def find_faiss_qwen3_indexer(search_cfg: dict, index_name: str | None) -> dict:
    """search_style yaml から faiss_qwen3 の indexer エントリを1つ取り出す。"""
    candidates = [ix for ix in search_cfg["indexers"] if ix["name"] == "faiss_qwen3"]
    if not candidates:
        raise ValueError("この search_style に faiss_qwen3 indexer がありません")
    if index_name is not None:
        for ix in candidates:
            if ix.get("index_name", "faiss_qwen3") == index_name:
                return ix
        raise ValueError(f"index_name={index_name} の faiss_qwen3 エントリが見つかりません")
    if len(candidates) > 1:
        names = [ix.get("index_name", "faiss_qwen3") for ix in candidates]
        raise ValueError(
            "faiss_qwen3 が複数あります。--index-name で選んでください: " + ", ".join(names)
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--process-name", default="mineru", help="chunksファイル名の接頭辞")
    parser.add_argument("--shard-index", type=int, required=True, help="担当スライス番号(0始まり)")
    parser.add_argument("--num-shards", type=int, required=True, help="全マシン合計のスライス数")
    parser.add_argument(
        "--index-name",
        default=None,
        help="faiss_qwen3 が複数あるとき、どのエントリを使うか(yamlのindex_name)",
    )
    parser.add_argument("--chunks", default=None, help="chunks.jsonlのパス(既定は paths から導出)")
    parser.add_argument("--index-dir", default=None, help="出力先(既定は paths から導出)")
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "使うGPU(例: 'cuda:0,cuda:1')。マシンごとに空きGPUが違うので、"
            "search_style yaml の devices をここで上書きできる。"
            "8Bはfp16でもピーク19.3GB使うので、空きが20GB未満のGPUは指定しないこと"
        ),
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help=(
            "1バッチのパディング後トークン数の上限(OOM対策)。yaml の値を上書きする。"
            "VRAMに余裕がないマシンでは下げる"
        ),
    )
    args = parser.parse_args()

    paths = load_config(args.paths)
    search_cfg = load_config(args.search)
    indexer_entry = find_faiss_qwen3_indexer(search_cfg, args.index_name)
    index_name = indexer_entry.get("index_name", "faiss_qwen3")
    params = dict(indexer_entry.get("params", {}))
    # マシンごとに空きGPU・VRAMが違うので、yaml の値をコマンドラインで上書きできる。
    if args.devices:
        params["devices"] = args.devices
    if args.max_batch_tokens:
        params["max_batch_tokens"] = args.max_batch_tokens

    chunks_path = (
        Path(args.chunks)
        if args.chunks
        else Path(paths["chunks_dir"]) / f"{args.process_name}_chunks.jsonl"
    )
    index_dir = (
        args.index_dir
        if args.index_dir
        else f"{paths['index_dir']}/{args.process_name}/{index_name}"
    )
    # 分割ビルドでは各スライスを別ディレクトリに出す(あとで merge する)。
    # 誤って本番の索引パスをスライスで上書きしないよう、末尾に shard 情報を足す。
    index_dir = f"{index_dir}__shard{args.shard_index}of{args.num_shards}"

    print(f"chunks を読み込み中: {chunks_path}")
    chunks = load_chunks(chunks_path)
    n = len(chunks)
    start, end = shard_bounds(n, args.shard_index, args.num_shards)
    shard = chunks[start:end]
    print(
        f"全 {n:,} 件のうち、shard {args.shard_index}/{args.num_shards} = "
        f"[{start:,}, {end:,}) の {len(shard):,} 件を {index_dir} に構築します"
    )

    indexer = registry.build("indexer", "faiss_qwen3", index_dir=index_dir, **params)
    indexer.build(shard)
    print(f"完了: {index_dir}")
    print(
        "全マシンのスライスが揃ったら scripts/merge_faiss_qwen3.py で"
        f" shard0..{args.num_shards - 1} を順番にマージしてください。"
    )


if __name__ == "__main__":
    main()
