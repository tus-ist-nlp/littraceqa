"""Paper-to-paper expansion: extend the candidate list with papers near the anchors.

Three ways of measuring "near", used together:

- ``specter2``: neighbours in SPECTER2(proximity) embedding space. Semantic.
- ``bib_coupling``: bibliographic coupling — Jaccard over the sets of arXiv IDs
  each paper cites.
- ``bm25_mlt``: more-like-this over full paper text, querying the prebuilt
  ``bm25s_paper`` index with the anchor's title+abstract. Lexical.

The result enters the candidate list only through the **RRF fusion of ranking A
and ranking B**. ``rank()`` returns a proximity ordering that still contains
papers already in the candidate list — overlapping papers are exactly what the
fusion rewards, so they must not be filtered out. The formula and the reasoning
live in ``_combine_rrf`` in agent.py.

An earlier design spliced the extra papers into fixed positions of the candidate
list. It lost to rank fusion on every metric and was removed.

**All three are worth keeping because they recover different gold papers**: of 37
gold papers outside the candidate list, SPECTER2 recovered 15, bibliographic
coupling 11 and full-text MLT 16; 2 were reachable only through MLT and 6 only
through the other two. ``fused`` fuses the sources' neighbours with RRF.

Bibliographic coupling works here *because* the corpus only spans 2024-2025.
Contemporaries cannot cite each other (TCM cites neither sCT nor IMM), so a
citation graph barely connects at all — only one in-corpus citation resolved from
an anchor in practice. But they do **cite the same older work**, which coupling
picks up (TCM against its three peers scores 0.19-0.24, while 30 random papers
score a median of 0.000 and at most 0.054).

Multi-paper gold sets reuse "the main papers of a topic cluster", and the peer
papers the question never names are close to impossible to reach by
**question->paper** retrieval (17 evidence-backed gold papers fell outside the top
50). They are, however, **close to the correct paper**. SPECTER2's proximity
adapter is a paper-level similarity embedding trained on citation proximity, so
anchoring on the top candidate (usually the supporting paper itself) pulls in the
rest of the cluster.

Implementation notes:

- **The prebuilt faiss_specter2_abstract index is read as is.** The anchor's
  vector comes out of the index via reconstruct(), so query time needs neither the
  SPECTER2 model nor a GPU — one CPU faiss search.
- Where the expansion lands is the caller's decision (ReadingAgent). This module
  only returns a ranked list of paper IDs.
- Submission is unaffected: expanded papers sit outside the `max_candidates` the
  reading LLM sees.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import bm25s
import faiss

from littraceqa.search.contracts import (
    CHUNKS_FILENAME,
    INDEX_FILENAME,
    PAPERS_FILENAME,
)

# Pulls the paper_id out of a papers.jsonl line without parsing the whole thing.
_PAPER_ID_RE = re.compile(r'"paper_id":\s*"([^"]+)"')

# Pull arXiv IDs out of the references. MinerU output can break the characters
# apart ("ar X iv : 2403.06807"), so whitespace is tolerated between them.
_ARXIV_RE = re.compile(r"ar\s*X\s*iv[:\s]*(\d{4}\.\d{4,5})")

class PaperExpander(Protocol):
    """Rank papers by proximity to the anchors (ranking B).

    **The caller chooses the anchors and passes them in.** Which papers anchor the
    expansion is a fusion decision (`_anchor_papers` in agent.py), so an
    expander is responsible for nothing but returning their neighbours.

    **Papers already in the candidate list are not excluded** from the result:
    overlapping papers are exactly what the RRF fusion rewards, so dropping them
    here would defeat it (see `_combine_rrf`).
    """

    def rank(self, anchors: list[str]) -> list[str]: ...


def _interleave(
    pools: list[list[str]], limit: int, exclude: set[str] | None = None
) -> list[str]:
    """Interleave several anchors' neighbour lists by rank into one list.

    A near neighbour of one anchor comes before a far neighbour of another.
    Papers in ``exclude`` are skipped (``rank()`` excludes nothing).
    """
    seen = set(exclude or ())
    merged: list[str] = []
    for rank in range(limit):
        for pool in pools:
            if rank < len(pool) and pool[rank] not in seen:
                seen.add(pool[rank])
                merged.append(pool[rank])
    return merged[:limit]


class _AnchorExpander:
    """The shape every expander shares: load once, map each anchor, interleave.

    **A subclass writes two methods and nothing else** — `_load()` to read whatever
    it needs off disk, and `_neighbors(paper_id)` to return one paper's neighbours,
    nearest first. Loading is deferred to the first `rank()` so that a --build-only
    run, or a test, can construct one before its index exists.

    An anchor the subclass cannot resolve returns an empty list, which contributes
    nothing at any rank; `rank()` therefore never fails a query over a missing anchor.
    """

    neighbors: int

    def __init__(self) -> None:
        self._is_loaded = False

    def _load(self) -> None:  # pragma: no cover - subclasses override
        raise NotImplementedError

    def _neighbors(self, paper_id: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _pools(self, anchors: list[str]) -> list[list[str]]:
        """Each anchor's neighbours, nearest first."""
        if not anchors:
            return []
        if not self._is_loaded:
            self._load()
            self._is_loaded = True
        return [self._neighbors(anchor) for anchor in anchors]

    def rank(self, anchors: list[str]) -> list[str]:
        return _interleave(self._pools(anchors), self.neighbors)


class Specter2PaperExpander(_AnchorExpander):
    """Neighbours in SPECTER2(proximity) space, read from the prebuilt faiss index."""

    def __init__(self, index_dir: str, neighbors: int = 20):
        super().__init__()
        self.index_dir = Path(index_dir)
        self.neighbors = neighbors
        self._index: faiss.Index | None = None
        self._row_of: dict[str, int] = {}
        self._pid_of: dict[int, str] = {}
        # Keep the title+abstract text too; it lives in this index's chunks.jsonl,
        # so no separate chunk store is needed.
        self._chunk_of: dict[str, dict] = {}

    def _load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / INDEX_FILENAME))
        with open(self.index_dir / CHUNKS_FILENAME, encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                chunk = json.loads(line)
                paper_id = chunk.get("paper_id", "")
                # The abstract index has one row per paper; if more, take the first.
                if paper_id and paper_id not in self._row_of:
                    self._row_of[paper_id] = row
                    self._pid_of[row] = paper_id
                    self._chunk_of[paper_id] = chunk

    def _neighbors(self, paper_id: str) -> list[str]:
        row = self._row_of.get(paper_id)
        if row is None:
            return []
        assert self._index is not None
        vector = self._index.reconstruct(row).reshape(1, -1)
        _, ids = self._index.search(vector, self.neighbors + 1)
        return [
            self._pid_of[i]
            for i in ids[0]
            if i >= 0 and i in self._pid_of and self._pid_of[i] != paper_id
        ]

class BibCouplingExpander(_AnchorExpander):
    """Papers that are near by bibliographic coupling (shared references).

    Pulls the arXiv IDs each paper cites out of its full text and measures Jaccard
    over those sets. **This is not a citation graph** (A cites B): the corpus only
    spans 2024-2025, contemporaries cannot cite each other, and in practice exactly
    one in-corpus citation resolved from an anchor. What it can measure is whether
    two papers cite the same older work.

    The index is built by one full pass over the corpus and pickled to
    ``cache_path`` (47 seconds, 25,012 papers, 68,418 IDs). Later runs just read the
    cache. No GPU.
    """

    def __init__(
        self,
        chunks: str,
        cache_path: str,
        neighbors: int = 20,
        min_shared: int = 2,
    ):
        super().__init__()
        self.chunks_path = Path(chunks)
        self.cache_path = Path(cache_path)
        self.neighbors = neighbors
        # Drop pairs sharing fewer references than this. A single shared reference
        # is usually a generic citation (Adam, ResNet) and is pure noise.
        self.min_shared = min_shared
        self._refs: dict[str, set[str]] | None = None
        self._inv: dict[str, set[str]] = {}

    def _load(self) -> None:
        if self.cache_path.exists():
            payload = pickle.loads(self.cache_path.read_bytes())
        else:
            payload = self._build()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(payload))
        self._refs = payload["refs"]
        self._inv = payload["inv"]

    def _build(self) -> dict:
        """One pass over chunks.jsonl to build {paper -> cited arXiv IDs} and its inverse."""
        refs: dict[str, set[str]] = {}
        current: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            if current is None:
                return
            ids = set(_ARXIV_RE.findall(" ".join(buffer)))
            if ids:
                refs[current] = ids

        with open(self.chunks_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                paper_id = chunk.get("paper_id", "")
                if paper_id != current:
                    flush()
                    current, buffer = paper_id, []
                buffer.append(chunk.get("text", ""))
        flush()

        inv: dict[str, set[str]] = {}
        for paper_id, ids in refs.items():
            for arxiv_id in ids:
                inv.setdefault(arxiv_id, set()).add(paper_id)
        return {"refs": refs, "inv": inv}

    def _neighbors(self, paper_id: str) -> list[str]:
        assert self._refs is not None
        own = self._refs.get(paper_id)
        if not own:
            return []
        shared: Counter[str] = Counter()
        for arxiv_id in own:
            for other in self._inv.get(arxiv_id, ()):
                if other != paper_id:
                    shared[other] += 1
        scores = {
            other: count / len(own | self._refs[other])
            for other, count in shared.items()
            if count >= self.min_shared
        }
        # Break ties on paper_id. scores is built by iterating a set, so without
        # this the order of ties **varies per process** (string hash randomisation).
        # Measured: 40%+ of queries reordered between runs, moving cr@20 by 0.4pt.
        return sorted(scores, key=lambda p: (-scores[p], p))[: self.neighbors]


def _json_string_prefix(line: str, start: int, max_chars: int) -> str:
    """Decode the first ``max_chars`` of the JSON string value starting at ``start``.

    A line of papers.jsonl runs from hundreds of KB to several MB, so ``json.loads``
    on whole lines takes minutes across 27,489 of them. This walks only as far as
    needed, unescaping by hand, instead of reading to the closing quote.
    """
    out: list[str] = []
    i = start
    while i < len(line) and len(out) < max_chars:
        char = line[i]
        if char == '"':  # end of the string value (a paper with very little text)
            break
        if char == "\\":
            escape = line[i : i + 6] if line[i + 1 : i + 2] == "u" else line[i : i + 2]
            try:
                out.append(json.loads(f'"{escape}"'))
            except json.JSONDecodeError:
                break
            i += len(escape)
            continue
        out.append(char)
        i += 1
    return "".join(out)


class BM25MLTExpander(_AnchorExpander):
    """More-like-this over full paper text: query `bm25s_paper` with the anchor's title+abstract.

    Independent of both SPECTER2 (semantic proximity of abstracts) and bibliographic
    coupling (shared references), this measures **lexical overlap across the whole
    body**. No LLM calls and nothing extra to build: it reuses the prebuilt
    `bm25s_paper` index (BM25 with one document per paper).

    ``papers.jsonl`` is 2.5GB but is **never read at query time**:

    - BM25 itself opens with ``mmap=True``, so the 490MB npy never enters RAM.
    - The row -> paper_id map and the anchors' title+abstract are streamed once and
      pickled (the same approach as BibCouplingExpander's refs.pkl). Each line's text
      runs "[venue year] title\n" + abstract + body, so the first ``query_chars``
      characters are exactly title+abstract.
    """

    def __init__(
        self,
        index_dir: str,
        cache_path: str,
        neighbors: int = 20,
        # How many leading characters of the anchor to use as the query — enough to
        # cover title+abstract. With a short abstract this reaches into the body,
        # which does no harm as an MLT query.
        query_chars: int = 1200,
    ):
        super().__init__()
        self.index_dir = Path(index_dir)
        self.cache_path = Path(cache_path)
        self.neighbors = neighbors
        self.query_chars = query_chars
        self._bm25 = None
        self._pids: list[str] = []
        self._text: dict[str, str] = {}

    def _load(self) -> None:
        if self.cache_path.exists():
            payload = pickle.loads(self.cache_path.read_bytes())
        else:
            payload = self._build()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(payload))
        self._pids = payload["pids"]
        self._text = payload["text"]
        self._bm25 = bm25s.BM25.load(str(self.index_dir), load_corpus=False, mmap=True)

    def _build(self) -> dict:
        """One streaming pass over papers.jsonl for {row -> paper_id} and title+abstract."""
        pids: list[str] = []
        text: dict[str, str] = {}
        with open(self.index_dir / PAPERS_FILENAME, encoding="utf-8") as handle:
            for line in handle:
                match = _PAPER_ID_RE.search(line, 0, 200)
                if match is None:
                    continue
                paper_id = match.group(1)
                pids.append(paper_id)
                marker = line.find('"text": "', match.end())
                if marker >= 0:
                    text[paper_id] = _json_string_prefix(
                        line, marker + len('"text": "'), self.query_chars
                    )
        return {"pids": pids, "text": text}

    def _neighbors(self, paper_id: str) -> list[str]:
        query = self._text.get(paper_id)
        if not query:
            return []
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(self.neighbors + 1, len(self._pids))
        if k <= 0:
            return []
        doc_indices, _ = self._bm25.retrieve(tokens, k=k, show_progress=False)
        return [
            self._pids[int(i)]
            for i in doc_indices[0]
            if 0 <= int(i) < len(self._pids) and self._pids[int(i)] != paper_id
        ]


class FusedPaperExpander:
    """Fuse several expanders' neighbours with RRF.

    The sources recover different gold papers, so neither is favoured; their
    rankings are fused. There is no principled basis for weighting them, hence RRF,
    which uses ranks only and is insensitive to score scale — the same reasoning as
    the retrieval-side fuser.
    """

    def __init__(
        self,
        sources: list[PaperExpander],
        neighbors: int = 20,
        rrf_k: int = 60,
    ):
        self.sources = sources
        self.neighbors = neighbors
        self.rrf_k = rrf_k

    def _fuse(self, per_source: list[list[str]]) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in per_source:
            for rank, paper_id in enumerate(ranking):
                scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ordered = sorted(scores, key=lambda p: -scores[p])
        return ordered[: self.neighbors]

    def rank(self, anchors: list[str]) -> list[str]:
        """RRF-fuse each source's neighbour ranking."""
        return self._fuse([source.rank(anchors) for source in self.sources])
