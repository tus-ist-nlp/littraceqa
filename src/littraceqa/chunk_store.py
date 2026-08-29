"""Every chunk of a paper, by paper_id.

The retrieval pipeline (di_pipeline) hands back only the chunks a search hit, so a
reading agent that wants to read a candidate paper **whole** has no way to get at
the text. This is that way in.

The corpus (mineru_chunks.jsonl, 3.8GB / 2,564,545 chunks / 27,487 papers) keeps
**each paper's lines contiguous** — verified by walking the whole file; another
paper interrupts zero times. One paper is therefore one contiguous byte range, and
a dict of paper_id -> (offset, length) is enough to read it with a single seek.
Neither loading 3.8GB into memory nor standing up a database is necessary.

Measured: 23s to build the index (once), 1.0MB of index, 0.7ms to load a paper.

Usage:
    from littraceqa.chunk_store import ChunkStore

    store = ChunkStore("/data2/iseakira/pdfs/chunks/mineru_chunks.jsonl")
    for chunk in store.load_paper("acl2025_00005"):
        print(chunk["chunk_id"], chunk["chunk_type"], chunk["text"][:80])

    # A figure's image is at metadata["image_path"] (table/figure only)
    for chunk in store.figures("acl2025_00005"):
        print(chunk["metadata"]["image_path"])

**Moving the corpus to another machine breaks image_path**, which is stored
absolute. Passing that machine's MinerU output directory as image_root rewrites it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

Record = dict[str, Any]

# The only two chunk types with an image (equation_algorithm has no image_path).
IMAGE_CHUNK_TYPES = ("table", "figure")

# Format version of the index file. Raise it when the structure changes; an index
# written by an older version is rebuilt automatically.
_INDEX_VERSION = 1


class ChunkStore:
    """Read-only access to the MinerU chunk JSONL, keyed by paper_id.

    The index lives beside the corpus as `{chunks_path}.offsets.json` and is built
    on first access if absent. **It is rebuilt whenever the corpus's size or mtime
    differs from what it was built against**, so re-running the preprocessing can
    never leave a stale index silently in use.
    """

    def __init__(
        self,
        chunks_path: str | Path,
        index_path: str | Path | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"corpus not found: {self.chunks_path}")
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.chunks_path.with_suffix(self.chunks_path.suffix + ".offsets.json")
        )
        self.image_root = Path(image_root) if image_root is not None else None
        self._offsets: dict[str, tuple[int, int]] | None = None

    # ---- the index ----------------------------------------------------------

    @property
    def offsets(self) -> dict[str, tuple[int, int]]:
        if self._offsets is None:
            self._offsets = self._load_or_build_index()
        return self._offsets

    def _stat(self) -> dict[str, int]:
        info = self.chunks_path.stat()
        return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}

    def _load_or_build_index(self) -> dict[str, tuple[int, int]]:
        if self.index_path.exists():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None  # a corrupt index is simply rebuilt
            if payload is not None and _index_is_fresh(payload, self._stat()):
                return {k: (v[0], v[1]) for k, v in payload["offsets"].items()}
        offsets = self._build_index()
        self._write_index(offsets)
        return offsets

    def _build_index(self) -> dict[str, tuple[int, int]]:
        """Walk the whole file for each paper's byte range (about 23 seconds)."""
        offsets: dict[str, list[int]] = {}
        previous: str | None = None
        position = 0
        with self.chunks_path.open("rb") as handle:
            for line in handle:
                # Only paper_id is needed, but parsing part of the line breaks on
                # newlines and escapes inside text, so the line is parsed properly.
                # The walk happens once, so it is worth it.
                paper_id = json.loads(line)["paper_id"]
                if paper_id != previous:
                    if paper_id in offsets:
                        # No longer contiguous, so the premise of this whole design
                        # is gone. Better to fail than to hand back a broken index.
                        raise ValueError(
                            f"the lines for paper_id {paper_id!r} are not "
                            "contiguous; ChunkStore requires one paper to be one "
                            "contiguous block"
                        )
                    offsets[paper_id] = [position, 0]
                    previous = paper_id
                offsets[paper_id][1] += len(line)
                position += len(line)
        return {k: (v[0], v[1]) for k, v in offsets.items()}

    def _write_index(self, offsets: dict[str, tuple[int, int]]) -> None:
        payload = {
            "version": _INDEX_VERSION,
            "source": self._stat(),
            "offsets": {k: [v[0], v[1]] for k, v in offsets.items()},
        }
        tmp = self.index_path.with_name(f".{self.index_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.index_path)
        except OSError:
            # The corpus simply sits somewhere read-only. Not being able to keep the
            # index costs nothing this session — only another 23 seconds next time.
            tmp.unlink(missing_ok=True)

    # ---- reading it out -----------------------------------------------------

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def paper_ids(self) -> list[str]:
        return list(self.offsets)

    def load_paper(self, paper_id: str) -> list[Record]:
        """A paper's chunks, in corpus order — which is the order of the paper.

        **An unknown paper_id gives an empty list, not an error.** A reading agent
        receives its paper_ids from retrieval, so carrying on with nothing is easier
        to handle there than an exception.
        """
        location = self.offsets.get(paper_id)
        if location is None:
            return []
        start, length = location
        with self.chunks_path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(length)
        chunks = [json.loads(line) for line in raw.splitlines() if line]
        if self.image_root is not None:
            for chunk in chunks:
                _rebase_image_path(chunk, self.image_root)
        return chunks

    def load_papers(self, paper_ids: list[str]) -> dict[str, list[Record]]:
        return {paper_id: self.load_paper(paper_id) for paper_id in paper_ids}

    def iter_chunks(
        self, paper_id: str, chunk_types: tuple[str, ...] | None = None
    ) -> Iterator[Record]:
        for chunk in self.load_paper(paper_id):
            if chunk_types is None or chunk.get("chunk_type") in chunk_types:
                yield chunk

    def figures(self, paper_id: str) -> list[Record]:
        """The table/figure chunks whose image file actually exists.

        Measured over the candidate papers, 99.3% of table/figure chunks carry an
        image_path and none of those paths were missing; the other 0.7% are tables
        extracted as text, with no image cut out. **The existence check is there so
        that a bad transfer or a wrong image_root does not go unnoticed** and end up
        handing a VLM an image that is not there.
        """
        found: list[Record] = []
        for chunk in self.iter_chunks(paper_id, IMAGE_CHUNK_TYPES):
            path = (chunk.get("metadata") or {}).get("image_path")
            if path and Path(path).exists():
                found.append(chunk)
        return found

    def paper_text(self, paper_id: str, separator: str = "\n\n") -> str:
        """The whole paper: every chunk's text, joined.

        A paper is a median of 78 chunks and 114KB, about 24k tokens. **Fifty
        candidates at once is 1.1M tokens** and fits in no model, so this is for one
        paper at a time.
        """
        return separator.join(
            chunk["text"] for chunk in self.load_paper(paper_id) if chunk.get("text")
        )


def _index_is_fresh(payload: Any, source: dict[str, int]) -> bool:
    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        return False
    return payload.get("source") == source


def _rebase_image_path(chunk: Record, image_root: Path) -> None:
    """Rewrite the `{mineru output}/` half of an image_path.

    The path has the shape `{root}/{paper_id}/auto/images/{sha256}.jpg`, so the
    paper_id component is located and everything before it is replaced. **That works
    even when the directory layout on the new machine differs**, which stripping a
    known root prefix would not.
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
