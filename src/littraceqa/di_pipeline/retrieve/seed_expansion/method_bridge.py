"""Method bridge exploration.

Spends the last output slot on a paper connected to the current results through
a shared method, so the stable prefix is never disturbed.
"""

from __future__ import annotations

from dataclasses import dataclass

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Retriever
from littraceqa.di_pipeline.retrieve.seed_expansion.method_owners import (
    METHOD_PROVIDER_METHODS,
    find_method_owner_records,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    without_method_hints,
)


@dataclass(frozen=True)
class MethodBridgeExploration:
    """Follows a topic-supported method edge from an existing paper."""

    retriever: Retriever
    seed_text_chars: int
    stable_prefix_k: int | None
    topic_seed_chars: int
    topic_seed_k: int
    topic_max_results: int
    bridge_topic_max_rank: int
    relation_max_results: int

    def replace_last(
        self,
        results: list[RetrievalResult],
        hints: SearchHints | None,
        indexers,
    ) -> list[RetrievalResult]:
        """Spend the final slot on a bridged paper, or return ``results`` as is."""

        if (
            self.bridge_topic_max_rank <= 0
            or hints is None
            or not hints.methods
            or len(results) < 2
        ):
            return results

        paper_index = find_paper_index(indexers, *METHOD_PROVIDER_METHODS)
        if paper_index is None:
            return results

        try:
            topic_by_id = self._collect_topic_support(
                paper_index,
                results,
                hints,
            )
            if not topic_by_id:
                return results

            bridge_by_id = self._collect_bridges(
                paper_index,
                results,
                topic_by_id,
            )
            ranked = sorted(
                bridge_by_id.values(),
                key=lambda state: (
                    state["best_topic_rank"],
                    state["best_bridge_rank"],
                    -state["strength"],
                    state["best_owner_rank"],
                    state["paper_id"],
                ),
            )
            selected_state = None
            selected_document = None
            for state in ranked:
                document = paper_index.get_document(state["paper_id"])
                if (
                    not isinstance(document, Chunk)
                    or document.paper_id != state["paper_id"]
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
                    continue
                selected_state = state
                selected_document = document
                break
        except Exception:
            return results

        if selected_state is None or selected_document is None:
            return results

        metadata = (
            dict(selected_document.metadata)
            if isinstance(selected_document.metadata, dict)
            else {}
        )
        metadata.update(
            {
                "method_bridge_topic_rank": selected_state["best_topic_rank"],
                "method_bridge_owner_papers": sorted(
                    selected_state["owner_papers"]
                ),
                "method_bridge_via_papers": sorted(
                    selected_state["bridge_papers"]
                ),
                "method_bridge_aliases": sorted(selected_state["aliases"]),
                "method_bridge_strength": selected_state["strength"],
                "method_bridge_replaced_paper_id": results[-1].paper_id,
                "method_bridge_is_new": True,
            }
        )
        replacement = RetrievalResult(
            chunk_id=selected_document.chunk_id,
            paper_id=selected_state["paper_id"],
            score=results[-1].score,
            text=selected_document.text[: self.seed_text_chars],
            chunk_type=selected_document.chunk_type,
            metadata=metadata,
            source="method_bridge_exploration",
        )
        return [*results[:-1], replacement]

    def _collect_topic_support(
        self,
        paper_index,
        results: list[RetrievalResult],
        hints: SearchHints,
    ) -> dict[str, dict]:
        """Rank papers by topical closeness to the method owners in the output."""

        owner_top_k = min(
            len(results),
            self.stable_prefix_k if self.stable_prefix_k is not None else 10,
        )
        owner_candidates = results[:owner_top_k]
        owner_candidate_ids = {result.paper_id for result in owner_candidates}
        owner_records = find_method_owner_records(
            paper_index,
            hints.methods,
            owner_candidates,
            limit=max(self.relation_max_results, self.topic_seed_k),
        )
        owner_records = tuple(
            record
            for record in owner_records
            if record["paper_id"] in owner_candidate_ids
        )
        if not owner_records:
            return {}

        retrieval_hints = without_method_hints(hints)
        topic_by_id: dict[str, dict] = {}
        for owner_rank, record in enumerate(
            owner_records[: self.topic_seed_k],
            start=1,
        ):
            owner_id = record["paper_id"]
            document = paper_index.get_document(owner_id)
            if (
                not isinstance(document, Chunk)
                or document.paper_id != owner_id
                or not isinstance(document.text, str)
            ):
                continue
            topic_query = document.text[: self.topic_seed_chars].strip()
            if not topic_query:
                continue
            topic_results = self.retriever.retrieve(
                topic_query,
                self.topic_max_results,
                hints=retrieval_hints,
            )[: self.topic_max_results]
            seen_topic_papers: set[str] = set()
            for topic_rank, result in enumerate(topic_results, start=1):
                paper_id = getattr(result, "paper_id", None)
                if (
                    not isinstance(paper_id, str)
                    or not paper_id
                    or paper_id in seen_topic_papers
                ):
                    continue
                seen_topic_papers.add(paper_id)
                state = topic_by_id.setdefault(
                    paper_id,
                    {
                        "best_topic_rank": topic_rank,
                        "best_owner_rank": owner_rank,
                        "owner_papers": set(),
                    },
                )
                state["best_topic_rank"] = min(
                    state["best_topic_rank"],
                    topic_rank,
                )
                state["best_owner_rank"] = min(
                    state["best_owner_rank"],
                    owner_rank,
                )
                state["owner_papers"].add(owner_id)
        return topic_by_id

    def _collect_bridges(
        self,
        paper_index,
        results: list[RetrievalResult],
        topic_by_id: dict[str, dict],
    ) -> dict[str, dict]:
        """Find new papers linked by method edges and backed by topic evidence."""

        result_ids = {result.paper_id for result in results}
        bridge_by_id: dict[str, dict] = {}
        for bridge_rank, bridge in enumerate(results[:-1], start=1):
            records = paper_index.get_method_neighbors(
                bridge.paper_id,
                limit=self.relation_max_results,
            )
            for record in records:
                if not isinstance(record, dict):
                    continue
                paper_id = record.get("paper_id")
                topic_state = topic_by_id.get(paper_id)
                if (
                    not isinstance(paper_id, str)
                    or paper_id in result_ids
                    or topic_state is None
                    or topic_state["best_topic_rank"]
                    > self.bridge_topic_max_rank
                ):
                    continue
                state = bridge_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "best_topic_rank": topic_state["best_topic_rank"],
                        "best_owner_rank": topic_state["best_owner_rank"],
                        "best_bridge_rank": bridge_rank,
                        "strength": 0,
                        "owner_papers": set(topic_state["owner_papers"]),
                        "bridge_papers": set(),
                        "aliases": set(),
                    },
                )
                state["best_bridge_rank"] = min(
                    state["best_bridge_rank"],
                    bridge_rank,
                )
                strength = record.get("strength")
                if isinstance(strength, int) and not isinstance(strength, bool):
                    state["strength"] += max(strength, 0)
                state["bridge_papers"].add(bridge.paper_id)
                aliases = record.get("aliases")
                if isinstance(aliases, (list, tuple)):
                    state["aliases"].update(
                        alias for alias in aliases if isinstance(alias, str)
                    )
        return bridge_by_id
