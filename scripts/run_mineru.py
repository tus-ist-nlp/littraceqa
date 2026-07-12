#!/usr/bin/env python3
"""MinerU(pipeline バックエンド)で PDF 群を Markdown / content_list.json に変換する。

本体パイプラインとは依存が両立しないため、隔離 venv (.venv-mineru) で実行する
（詳細は requirements-mineru.txt）。ここで書き出した content_list.json を
本体側の MinerUChunker が読んで Chunk 化する。

27,000件規模を現実的な時間で終わらせるため、GPU ごとに1プロセス立てて
paper_id で分割(シャード)して並列に流す。既に content_list.json がある論文は
飛ばすので、中断しても同じコマンドで再開できる。

構築:
    bash scripts/setup_mineru_env.sh

実行(4GPU 全部使う):
    .venv-mineru/bin/python scripts/run_mineru.py \\
      --paths configs/paths/default.yaml --gpus 0,1,2,3

出力先は paths.yaml の pdf_dir の兄弟 `mineru/` を既定とする
（process_style yaml にはパスを書かない方針のため。--mineru-dir で上書き可）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# MinerU の import 前に設定する必要がある。
os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")


def default_mineru_dir(pdf_dir: Path) -> Path:
    """pdf_dir と衝突しない兄弟ディレクトリを既定の出力先にする。"""
    return pdf_dir.parent / "mineru"


def content_list_path(mineru_dir: Path, paper_id: str) -> Path:
    """do_parse が書き出す content_list.json の位置。"""
    return mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"


def select_pdfs(pdf_dir: Path, mineru_dir: Path, shard: int, num_shards: int, overwrite: bool) -> list[Path]:
    """このシャードが担当する、まだ未処理の PDF を返す。"""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    mine = [p for i, p in enumerate(pdfs) if i % num_shards == shard]
    if overwrite:
        return mine
    return [p for p in mine if not content_list_path(mineru_dir, p.stem).exists()]


def parse_batch(pdfs: list[Path], mineru_dir: Path) -> None:
    """PDF をまとめて do_parse に渡す。まとめるほどモデル呼び出しが効率的になる。"""
    from mineru.cli.common import do_parse

    do_parse(
        str(mineru_dir),
        [p.stem for p in pdfs],
        [p.read_bytes() for p in pdfs],
        ["en"] * len(pdfs),
        backend="pipeline",
        # 索引には Markdown と content_list.json しか使わない。
        # bbox 描画済み PDF や model 出力は容量を食うだけなので落とす。
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_middle_json=False,
        f_dump_md=True,
        f_dump_content_list=True,
    )


def run_shard(args: argparse.Namespace, pdf_dir: Path, mineru_dir: Path) -> int:
    """1シャード分を処理する。戻り値は失敗件数。"""
    pdfs = select_pdfs(pdf_dir, mineru_dir, args.shard, args.num_shards, args.overwrite)
    if args.limit:
        pdfs = pdfs[: args.limit]

    log_dir = mineru_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_path = log_dir / f"failures_shard{args.shard}.jsonl"

    print(f"[shard {args.shard}] 未処理 {len(pdfs)} 件", flush=True)
    failures = 0
    done = 0
    start = time.time()

    for i in range(0, len(pdfs), args.batch_size):
        batch = pdfs[i : i + args.batch_size]
        try:
            parse_batch(batch, mineru_dir)
        except Exception as exc:
            # バッチのどれか1件が壊れていても他を巻き添えにしないよう、1件ずつ再試行する。
            print(f"[shard {args.shard}] バッチ失敗、1件ずつ再試行します: {exc}", file=sys.stderr, flush=True)
            for pdf in batch:
                try:
                    parse_batch([pdf], mineru_dir)
                except Exception as inner:
                    failures += 1
                    with failure_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"paper_id": pdf.stem, "error": str(inner)}, ensure_ascii=False) + "\n")
                    print(f"[shard {args.shard}] 失敗 {pdf.stem}: {inner}", file=sys.stderr, flush=True)

        done += len(batch)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0.0
        remaining = (len(pdfs) - done) / rate / 3600 if rate else float("nan")
        print(
            f"[shard {args.shard}] {done}/{len(pdfs)} 件 "
            f"({rate * 3600:.0f} 件/時, 残り {remaining:.1f} 時間, 失敗 {failures})",
            flush=True,
        )

    print(f"[shard {args.shard}] 完了: {done - failures} 件成功 / {failures} 件失敗", flush=True)
    return failures


def orchestrate(args: argparse.Namespace, pdf_dir: Path, mineru_dir: Path) -> int:
    """GPU ごとに自分自身を --shard 付きで起動して並列に流す。

    親プロセスでは CUDA を初期化しない(子が CUDA_VISIBLE_DEVICES を見て掴む)。
    """
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    todo = len(select_pdfs(pdf_dir, mineru_dir, 0, 1, args.overwrite))
    print(f"未処理 {todo} 件を GPU {gpus} の {len(gpus)} シャードで処理します", flush=True)

    log_dir = mineru_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    for shard, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--paths", args.paths,
            "--shard", str(shard),
            "--num-shards", str(len(gpus)),
            "--batch-size", str(args.batch_size),
            "--mineru-dir", str(mineru_dir),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        log_path = log_dir / f"shard{shard}.log"
        log_file = log_path.open("w", encoding="utf-8")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)))
        print(f"  GPU {gpu} -> shard {shard} (ログ: {log_path})", flush=True)

    exit_codes = [(gpu, proc.wait()) for gpu, proc in procs]
    failed = [gpu for gpu, code in exit_codes if code != 0]
    if failed:
        print(f"エラー: GPU {failed} のシャードが異常終了しました", file=sys.stderr)
        return 1

    remaining = len(select_pdfs(pdf_dir, mineru_dir, 0, 1, overwrite=False))
    print(f"全シャード完了。未処理として残っているのは {remaining} 件です")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--gpus", default="0", help="使う GPU の番号をカンマ区切りで（例: 0,1,2,3）")
    parser.add_argument("--batch-size", type=int, default=8, help="1回の do_parse に渡す PDF 数")
    parser.add_argument("--mineru-dir", help="出力先（既定: pdf_dir の兄弟 mineru/）")
    parser.add_argument("--limit", type=int, help="各シャードの処理件数を制限する（動作確認用）")
    parser.add_argument("--overwrite", action="store_true", help="既に変換済みの論文も再処理する")
    # 以下は orchestrate() が子プロセスを起動するときに使う。手動指定も可。
    parser.add_argument("--shard", type=int, help="このプロセスが担当するシャード番号")
    parser.add_argument("--num-shards", type=int, default=1, help="シャード総数")
    args = parser.parse_args()

    with open(args.paths, encoding="utf-8") as f:
        paths = yaml.safe_load(f)
    pdf_dir = Path(paths["pdf_dir"])
    mineru_dir = Path(args.mineru_dir) if args.mineru_dir else default_mineru_dir(pdf_dir)
    mineru_dir.mkdir(parents=True, exist_ok=True)

    if args.shard is None:
        return orchestrate(args, pdf_dir, mineru_dir)
    return 1 if run_shard(args, pdf_dir, mineru_dir) else 0


# MinerU は PDF レンダリング用の ProcessPoolExecutor を spawn で起動する。
# spawn の子は __main__ を再読み込みするため、このガードが無いと変換処理が
# 子プロセス内で再帰的に走り BrokenProcessPool で落ちる。
if __name__ == "__main__":
    sys.exit(main())
