"""paper_id から MinerU コーパスの全チャンクを引くローダー。

検索パイプライン（di_pipeline）は「検索がヒットしたチャンク」しか返さないので、
「候補論文の本文をまるごと読む」タイプの読解エージェントには本文を引く経路が無い。
ここが唯一の入口になる。

コーパス（mineru_chunks.jsonl, 3.8GB / 2,564,545 chunks / 27,487 papers）は
**paper_id ごとに行が連続している**（全走査して確認済み、途中で他論文が割り込む
切り替わりは0回）。つまり1論文＝ファイル上の1つの連続バイト範囲なので、
paper_id -> (開始オフセット, バイト長) の dict だけ持てば seek 一発で読める。
3.8GB をメモリに載せる必要も DB を立てる必要もない。

実測: 索引の構築 23秒（初回のみ）/ 索引サイズ 1.0MB / 1論文のロード 0.7ms。

使い方:
    from littraceqa.chunk_store import ChunkStore

    store = ChunkStore("/path/to/mineru_chunks.jsonl")
    for chunk in store.load_paper("acl2025_00005"):
        print(chunk["chunk_id"], chunk["chunk_type"], chunk["text"][:80])

    # 図表の画像は metadata["image_path"] にある（table/figure のみ）
    for chunk in store.figures("acl2025_00005"):
        print(chunk["metadata"]["image_path"])

コーパスを別マシンへ転送した場合は image_path が壊れる（絶対パスで焼き込まれて
いるため）。転送先の mineru 出力ディレクトリを image_root に渡すと差し替わる。
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

Record = dict[str, Any]

# 画像を持つのはこの2種だけ（equation_algorithm は image_path を持たない）。
IMAGE_CHUNK_TYPES = ("table", "figure")

# 索引ファイルのフォーマット版。構造を変えたら上げる（古い索引は自動で作り直す）。
_INDEX_VERSION = 1


class ChunkStore:
    """MinerU チャンク JSONL を paper_id で引くための読み取り専用ストア。

    索引はコーパスの隣に `{chunks_path}.offsets.json` として置き、無ければ
    初回アクセス時に作る。コーパスのサイズか mtime が索引作成時と変わっていたら
    作り直す（前処理をやり直したのに古い索引を使い続ける事故を防ぐ）。
    """

    def __init__(
        self,
        chunks_path: str | Path,
        index_path: str | Path | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"コーパスが見つからない: {self.chunks_path}")
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.chunks_path.with_suffix(self.chunks_path.suffix + ".offsets.json")
        )
        self.image_root = Path(image_root) if image_root is not None else None
        self._offsets: dict[str, tuple[int, int]] | None = None
        self._offset_source: dict[str, int] | None = None

    # ---- 索引 ---------------------------------------------------------------

    @property
    def offsets(self) -> dict[str, tuple[int, int]]:
        # A long-running reader may outlive a corpus refresh.  Do not keep
        # seeking with byte ranges that were validated against an older file.
        if self._offsets is not None and self._offset_source != self._stat():
            self._offsets = None
            self._offset_source = None
        if self._offsets is None:
            self._offsets = self._load_or_build_index()
        return self._offsets

    def _stat(self) -> dict[str, int]:
        info = self.chunks_path.stat()
        return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}

    def _load_or_build_index(self) -> dict[str, tuple[int, int]]:
        source = self._stat()
        if self.index_path.exists():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None  # 壊れた索引は黙って作り直す
            if payload is not None and _index_is_fresh(payload, source):
                decoded = _decode_offsets(payload, source["size"])
                if decoded is not None:
                    self._offset_source = source
                    return decoded
        offsets, indexed_source = self._build_index()
        self._write_index(offsets, indexed_source)
        self._offset_source = indexed_source
        return offsets

    def _build_index(self) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
        """全走査して paper_id ごとのバイト範囲を求める（約23秒）。"""
        source_before = self._stat()
        offsets: dict[str, list[int]] = {}
        previous: str | None = None
        position = 0
        with self.chunks_path.open("rb") as handle:
            for line in handle:
                line_start = position
                position += len(line)
                # paper_id だけ要るが、部分パースは text 中の改行やエスケープで
                # 壊れるので素直に json.loads する（走査は1回きりなので割に合う）。
                if not line.strip():
                    raise ValueError(
                        f"空行があるため索引を構築できない: byte offset {line_start}"
                    )
                try:
                    record = json.loads(line)
                    paper_id = record["paper_id"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"不正なchunk JSONL: byte offset {line_start}"
                    ) from exc
                if not isinstance(paper_id, str) or not paper_id:
                    raise ValueError(
                        f"paper_idが空または文字列でない: byte offset {line_start}"
                    )
                if paper_id != previous:
                    if paper_id in offsets:
                        # 連続でなくなった＝この設計の前提が崩れている。黙って
                        # 壊れた索引を返すより落とす。
                        raise ValueError(
                            f"paper_id {paper_id!r} の行が連続していない。"
                            "ChunkStore は1論文=1連続ブロックを前提にしている。"
                        )
                    offsets[paper_id] = [line_start, 0]
                    previous = paper_id
                offsets[paper_id][1] += len(line)
        source_after = self._stat()
        if source_before != source_after or position != source_after["size"]:
            raise RuntimeError(
                "コーパスが索引構築中に更新されたため、offset索引を破棄します"
            )
        decoded = {k: (v[0], v[1]) for k, v in offsets.items()}
        return decoded, source_after

    def _write_index(
        self, offsets: dict[str, tuple[int, int]], source: dict[str, int]
    ) -> None:
        payload = {
            "version": _INDEX_VERSION,
            "source": source,
            "offsets": {k: [v[0], v[1]] for k, v in offsets.items()},
        }
        tmp = self.index_path.with_name(f".{self.index_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            if self._stat() != source:
                raise RuntimeError(
                    "コーパスが索引保存前に更新されたため、offset索引を破棄します"
                )
            os.replace(tmp, self.index_path)
        except RuntimeError:
            tmp.unlink(missing_ok=True)
            raise
        except OSError:
            # コーパスが読み取り専用の場所にあるだけなら、索引を残せなくても
            # そのセッションでは動く。次回また23秒かかるだけ。
            tmp.unlink(missing_ok=True)
            warnings.warn(
                f"offset索引を保存できないため次回は再構築します: {self.index_path}",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---- 取り出し -----------------------------------------------------------

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def paper_ids(self) -> list[str]:
        return list(self.offsets)

    def load_paper(self, paper_id: str) -> list[Record]:
        """1論文の全チャンクをコーパスJSONL上の順序で返す。

        存在しない paper_id は空リスト。読解エージェントは検索結果から
        paper_id を受け取るので、欠損で落ちるより空で進めるほうが扱いやすい。
        """
        return self._load_paper(paper_id, rebuild_on_mismatch=True)

    def _load_paper(
        self, paper_id: str, *, rebuild_on_mismatch: bool
    ) -> list[Record]:
        location = self.offsets.get(paper_id)
        if location is None:
            return []
        start, length = location
        with self.chunks_path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(length)
        invalid_reason: Exception | None = None
        try:
            chunks = [json.loads(line) for line in raw.splitlines() if line]
        except json.JSONDecodeError as exc:
            chunks = []
            invalid_reason = exc
        if not chunks or any(chunk.get("paper_id") != paper_id for chunk in chunks):
            if rebuild_on_mismatch:
                self._rebuild_cached_index()
                return self._load_paper(paper_id, rebuild_on_mismatch=False)
            raise ValueError(
                f"offset索引とコーパスのpaper_idが一致しない: paper_id={paper_id!r}"
            ) from invalid_reason
        if self.image_root is not None:
            for chunk in chunks:
                _rebase_image_path(chunk, self.image_root)
        return chunks

    def _rebuild_cached_index(self) -> None:
        """Rebuild after a structurally valid index points at the wrong paper."""

        offsets, source = self._build_index()
        self._write_index(offsets, source)
        self._offsets = offsets
        self._offset_source = source

    def load_papers(self, paper_ids: list[str]) -> dict[str, list[Record]]:
        return {paper_id: self.load_paper(paper_id) for paper_id in paper_ids}

    def iter_chunks(
        self, paper_id: str, chunk_types: tuple[str, ...] | None = None
    ) -> Iterator[Record]:
        for chunk in self.load_paper(paper_id):
            if chunk_types is None or chunk.get("chunk_type") in chunk_types:
                yield chunk

    def figures(self, paper_id: str) -> list[Record]:
        """画像ファイルが実在する table/figure チャンクだけを返す。

        実測では候補論文の table/figure の 99.3% が image_path を持ち、
        指す先の欠損は0件だった（残り0.7%はテキスト表として抽出され画像の
        切り出しが無かった table）。存在チェックを通すのは、転送や
        image_root の指定ミスに気づかず空画像を VLM に渡さないため。
        """
        found: list[Record] = []
        for chunk in self.iter_chunks(paper_id, IMAGE_CHUNK_TYPES):
            path = (chunk.get("metadata") or {}).get("image_path")
            if path and Path(path).exists():
                found.append(chunk)
        return found

    def paper_text(self, paper_id: str, separator: str = "\n\n") -> str:
        """全チャンクの text を連結した論文全文。

        1論文の実測は中央値78チャンク / 114KB（約24k tok）。候補50本を
        まとめて渡すと 1.1M tok になり、どのモデルにも入らない点に注意。
        """
        return separator.join(
            chunk["text"] for chunk in self.load_paper(paper_id) if chunk.get("text")
        )


def _index_is_fresh(payload: Any, source: dict[str, int]) -> bool:
    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        return False
    return payload.get("source") == source


def _decode_offsets(
    payload: Any, source_size: int
) -> dict[str, tuple[int, int]] | None:
    """Validate an on-disk index before trusting byte ranges from it."""

    if not isinstance(payload, dict) or not isinstance(payload.get("offsets"), dict):
        return None
    decoded: dict[str, tuple[int, int]] = {}
    ranges: list[tuple[int, int]] = []
    for paper_id, raw in payload["offsets"].items():
        if not isinstance(paper_id, str) or not paper_id:
            return None
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        start, length = raw
        if (
            isinstance(start, bool)
            or isinstance(length, bool)
            or not isinstance(start, int)
            or not isinstance(length, int)
            or start < 0
            or length <= 0
            or start + length > source_size
        ):
            return None
        decoded[paper_id] = (start, length)
        ranges.append((start, start + length))
    ranges.sort()
    expected_start = 0
    for start, end in ranges:
        if start != expected_start:
            return None
        expected_start = end
    if expected_start != source_size:
        return None
    return decoded


def _rebase_image_path(chunk: Record, image_root: Path) -> None:
    """image_path の `{mineru出力}/{paper_id}/...` の前半を差し替える。

    パスは `{root}/{paper_id}/auto/images/{sha256}.jpg` の形をしているので、
    paper_id のディレクトリ成分を探して、その手前を丸ごと入れ替える。
    root の文字列を前方一致で削るより、転送先の階層が違っても効く。
    """
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return
    raw = metadata.get("image_path")
    if not raw:
        return
    parts = Path(raw).parts
    paper_id = chunk.get("paper_id")
    if paper_id not in parts:
        return
    tail = parts[parts.index(paper_id) :]
    metadata["image_path"] = str(image_root.joinpath(*tail))
