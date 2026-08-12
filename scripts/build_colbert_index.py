#!/usr/bin/env python3
"""colbert 索引「1つだけ」を既存 chunks から構築する専用スクリプト。

用途: 重い ColBERT 索引構築を別マシン(GPUが空いている nlp02 等)で回すため。
run_search.py --build と違い、
* 再チャンク化(MinerU の content_list.json 読み直し)をしない。転送済みの
  mineru_chunks.jsonl を読むだけなので、MinerU 成果物が無いマシンでも動く。
* bm25s 等ほかの indexer を巻き込まない。colbert 索引だけを作る
  (bm25s は検索を回すマシン側の既存索引を使う想定)。

パラメータ(model/document_length/build_batch_size 等)は search_style の yaml から
そのまま読むので手打ちズレが起きない。出力先も compose_config と同じ規約
{index_dir}/{process}/{index_name} で導出する。

PLAID 索引はフォルダを別マシンへ移しても load できる(実測確認済み)ので、
ここで作った {index_dir}/{process}/{index_name} をそのまま検索マシンへ rsync し、
run_search.py(--build なし)で bm25s と融合して検索・評価する。

使い方(nlp02 で):
    uv run python scripts/build_colbert_index.py \
      --paths configs/paths/nlp02.yaml \
      --search configs/search_style/bm25_colbert/gte_modern.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from littraceqa.di_pipeline import registry
from littraceqa.di_pipeline.contracts import Chunk

# config.py 全体は marker/docling 等の重い任意依存を芋づるで読むので避け、
# colbert indexer の登録だけを効かせる。
import littraceqa.di_pipeline.index.colbert_index  # noqa: F401


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def find_colbert_indexer(search_cfg: dict, index_name: str | None) -> dict:
    candidates = [ix for ix in search_cfg["indexers"] if ix["name"] == "colbert"]
    if not candidates:
        raise ValueError("この search_style に colbert indexer がありません")
    if index_name is not None:
        for ix in candidates:
            if ix.get("index_name", "colbert") == index_name:
                return ix
        raise ValueError(f"index_name={index_name} の colbert エントリが見つかりません")
    if len(candidates) > 1:
        names = [ix.get("index_name", "colbert") for ix in candidates]
        raise ValueError("colbert が複数あります。--index-name で選んでください: " + ", ".join(names))
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--process-name", default="mineru", help="chunksファイル名の接頭辞")
    parser.add_argument("--index-name", default=None, help="colbertが複数あるとき選ぶ(yamlのindex_name)")
    parser.add_argument("--chunks", default=None, help="chunks.jsonlのパス(既定は paths から導出)")
    parser.add_argument("--index-dir", default=None, help="出力先(既定は paths から導出)")
    args = parser.parse_args()

    paths = load_config(args.paths)
    search_cfg = load_config(args.search)
    entry = find_colbert_indexer(search_cfg, args.index_name)
    index_name = entry.get("index_name", "colbert")
    params = dict(entry.get("params", {}))

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

    print(f"chunks を読み込み中: {chunks_path}")
    chunks = load_chunks(chunks_path)
    print(f"{len(chunks):,} 件を {index_dir} に colbert 索引として構築します")

    indexer = registry.build("indexer", "colbert", index_dir=index_dir, **params)
    indexer.build(chunks)
    print(f"完了: {index_dir}")
    print(
        "この索引フォルダを検索マシンの同名パスへ rsync し、"
        "run_search.py(--build なし)で bm25s と融合して検索してください。"
    )


if __name__ == "__main__":
    main()
