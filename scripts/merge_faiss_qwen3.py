#!/usr/bin/env python3
"""複数マシンで分割ビルドした faiss_qwen3 索引スライスを、1つの索引に合体する。

scripts/build_faiss_qwen3_shard.py が各マシンで作った
  {index_dir}__shard{i}of{N}/index.faiss   (IndexFlatIP、担当スライスのベクトル)
  {index_dir}__shard{i}of{N}/chunks.jsonl  (同じ並びのチャンク)
を shard 0,1,...,N-1 の順に連結して、通常の run_search.py が読める形の
  {output}/index.faiss
  {output}/chunks.jsonl
を作る。

faiss_qwen3.py の build() は「memmap の i 行目 = chunks の i 行目」の対応で
IndexFlatIP に順番に add しているので、index の i 番目のベクトルは chunks.jsonl の
i 行目に対応する。よって「各スライスのベクトルとチャンクを、同じ shard 順で
そのまま連結する」だけで、元の全件を1マシンで作ったのと同一の索引になる
(IndexFlatIP は全ベクトルを平坦に持つだけで、順序以外の内部状態を持たないため)。

ベクトルは faiss_qwen3.py の build() で L2 正規化済み。reconstruct で取り出しても
正規化は保たれるので、再正規化はしない。

メモリ安全のため、8B(全体41.6GB)を一度に RAM に載せず _ADD_ROWS 行ずつ
reconstruct して add する。

使い方(2分割の例):
    uv run python scripts/merge_faiss_qwen3.py \
      --parts /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b__shard0of2 \
              /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b__shard1of2 \
      --output /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b

--parts は shard 0,1,...,N-1 の順で並べること(順番が命)。
nlp02 で作ったスライスは、事前に scp/rsync で nlp01 に集めておく。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import faiss
import numpy as np

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"
_ADD_ROWS = 100_000  # faiss_qwen3.py と揃える


def count_lines(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="+",
        required=True,
        help="スライス索引ディレクトリを shard 0,1,...,N-1 の順に並べる",
    )
    parser.add_argument("--output", required=True, help="合体後の索引ディレクトリ")
    args = parser.parse_args()

    part_dirs = [Path(p) for p in args.parts]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # まず全スライスの健全性を確認する(index件数 == chunks行数)。
    # 途中まで書いてから食い違いに気づくと索引がずれて全て無駄になるので、先に全部見る。
    part_indexes = []
    dim = None
    total_vectors = 0
    total_chunks = 0
    for part in part_dirs:
        index_path = part / _INDEX_FILENAME
        chunks_path = part / _CHUNKS_FILENAME
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"{part} に index.faiss か chunks.jsonl がありません")
        index = faiss.read_index(str(index_path))
        n_chunks = count_lines(chunks_path)
        if index.ntotal != n_chunks:
            raise ValueError(
                f"{part}: index のベクトル数 {index.ntotal:,} と "
                f"chunks 行数 {n_chunks:,} が一致しません(索引がずれます)"
            )
        if dim is None:
            dim = index.d
        elif index.d != dim:
            raise ValueError(
                f"{part}: 次元 {index.d} が他スライス({dim})と違います"
            )
        part_indexes.append(index)
        total_vectors += index.ntotal
        total_chunks += n_chunks
        print(f"  {part.name}: {index.ntotal:,} ベクトル / {n_chunks:,} チャンク OK")

    print(
        f"合計 {total_vectors:,} ベクトル({dim}次元)を {output_dir} に合体します"
    )

    # ベクトルを shard 順に連結して新しい IndexFlatIP に add する。
    merged = faiss.IndexFlatIP(dim)
    for part, index in zip(part_dirs, part_indexes):
        n = index.ntotal
        for start in range(0, n, _ADD_ROWS):
            count = min(_ADD_ROWS, n - start)
            vectors = index.reconstruct_n(start, count)
            merged.add(np.ascontiguousarray(vectors))
        print(f"  {part.name}: ベクトル追加済み(累計 {merged.ntotal:,})")

    faiss.write_index(merged, str(output_dir / _INDEX_FILENAME))
    print(f"index.faiss を書き出しました({merged.ntotal:,} ベクトル)")

    # chunks.jsonl を shard 順にそのまま連結する。
    out_chunks = output_dir / _CHUNKS_FILENAME
    with out_chunks.open("w", encoding="utf-8") as out:
        for part in part_dirs:
            with (part / _CHUNKS_FILENAME).open(encoding="utf-8") as f:
                shutil.copyfileobj(f, out)
    written = count_lines(out_chunks)
    if written != merged.ntotal:
        raise ValueError(
            f"合体後の chunks 行数 {written:,} と ベクトル数 {merged.ntotal:,} が"
            "一致しません(マージが壊れています)"
        )
    print(f"chunks.jsonl を書き出しました({written:,} 行)")
    print("完了。run_search.py から通常の索引として読めます。")


if __name__ == "__main__":
    main()
