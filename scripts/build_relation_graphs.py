#!/usr/bin/env python3
"""コーパスを走査して「どの論文がどの論文を名指ししているか」の索引を作る。

``title_mention`` と ``method_comention`` の2 expander が読む索引を1本作る
（解釈が違うだけで中身は同じなので、走査は1回で済む）。expander 側も
キャッシュが無ければ自分で作るが、クエリ実行の最中に数分待たされるのを
避けたいときはこれで先に作っておく。

    uv run python scripts/build_relation_graphs.py \
      --paths configs/paths/default.yaml --process configs/process_style/mineru.yaml

索引パスは configs/paths と process の名前から導出する
（agent_style に絶対パスを書かない方針に合わせる）:

    {index_dir}/{process名}/relations/mentions.pkl

GPU 不要。``--force`` を付けると既存キャッシュを作り直す。
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from littraceqa.di_pipeline.config import load_config
from littraceqa.di_pipeline.retrieve.relation_graph import build_relations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--force", action="store_true", help="既存キャッシュを作り直す")
    parser.add_argument("--chunks", help="chunks.jsonl を直接指定する（スモークテスト用）")
    parser.add_argument("--out", help="出力先を直接指定する（スモークテスト用）")
    parser.add_argument(
        "--max-key-degree",
        type=int,
        default=20,
        help="この本数を超える論文から名指しされた名前は捨てる（ALLCAPS の英単語対策）",
    )
    args = parser.parse_args()

    paths = load_config(args.paths)
    process_name = Path(args.process).stem

    chunks_path = Path(args.chunks or f"{paths['chunks_dir']}/{process_name}_chunks.jsonl")
    cache_path = Path(
        args.out or f"{paths['index_dir']}/{process_name}/relations/mentions.pkl"
    )

    if cache_path.exists() and not args.force:
        payload = pickle.loads(cache_path.read_bytes())
        print(f"既存: {cache_path}（--force で作り直す）", file=sys.stderr)
    else:
        if not chunks_path.exists():
            raise SystemExit(f"chunks が無い: {chunks_path}")
        print(f"走査中: {chunks_path}", file=sys.stderr)
        started = time.time()
        payload = build_relations(chunks_path, max_key_degree=args.max_key_degree)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(payload))
        print(f"{time.time() - started:.1f}秒 -> {cache_path}", file=sys.stderr)

    mentions = payload["mentions"]
    degrees = sorted((len(v) for v in mentions.values()), reverse=True)
    total = sum(degrees)
    print(
        f"一意な識別子 {payload['n_keys']:,} / "
        f"名指しをした論文 {len(mentions):,} / "
        f"名指しされた論文 {len(payload['mentioned_by']):,} / "
        f"辺 {total:,}",
        file=sys.stderr,
    )
    if degrees:
        middle = degrees[len(degrees) // 2]
        print(
            f"1論文あたりの名指し数: 中央 {middle} / 最大 {degrees[0]} / 平均 "
            f"{total / len(mentions):.1f}",
            file=sys.stderr,
        )
    hubs = payload.get("dropped_hubs") or []
    if hubs:
        shown = ", ".join(f"{key}({count})" for key, count in hubs[:15])
        print(
            f"ハブとして捨てた名前 {len(hubs)}件以上（>{payload.get('max_key_degree')}本）: {shown}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
