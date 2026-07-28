"""Relation expansion.

Three lanes that reach papers the plain query missed by following explicit
links: cross-paper mentions, method ownership plus method-to-method edges, and
one bounded method bridge that spends the last output slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult, SearchHints
from littraceqa.di_pipeline.retrieve.base import Retriever
from littraceqa.di_pipeline.retrieve.paper_neighborhood import (
    PaperNeighborhoodReranker,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.paper_index import (
    find_paper_index,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    without_method_hints,
)


METHOD_PROVIDER_METHODS = (
    "get_document",
    "get_method_neighbors",
    "find_method_owners",
)


def find_method_owner_records(
    provider,
    methods: tuple[str, ...],
    candidates: list[RetrievalResult],
    *,
    limit: int,
) -> tuple[dict, ...]:
    """Merge a prebuilt owner index with bounded live extraction."""

    try:
        indexed_records = tuple(provider.find_method_owners(methods, limit=limit))
    except Exception:
        indexed_records = ()

    live_records: tuple = ()
    live_finder = getattr(provider, "find_method_owners_in_papers", None)
    if not indexed_records and callable(live_finder):
        try:
            live_records = tuple(
                live_finder(
                    methods,
                    (candidate.paper_id for candidate in candidates[:10]),
                    limit=limit,
                )
            )
        except Exception:
            live_records = ()

    merged: dict[str, dict] = {}
    for record in (*indexed_records, *live_records):
        if not isinstance(record, dict):
            continue
        paper_id = record.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        state = merged.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "aliases": set(),
                "strength": 0,
            },
        )
        aliases = record.get("aliases")
        if isinstance(aliases, (list, tuple)):
            state["aliases"].update(
                alias for alias in aliases if isinstance(alias, str)
            )
        strength = record.get("strength")
        if isinstance(strength, int) and not isinstance(strength, bool):
            state["strength"] = max(state["strength"], max(strength, 0))

    records = [
        {
            "paper_id": state["paper_id"],
            "aliases": sorted(state["aliases"]),
            "strength": state["strength"],
        }
        for state in merged.values()
    ]
    records.sort(key=lambda record: (-record["strength"], record["paper_id"]))
    return tuple(records[:limit])


@dataclass(frozen=True)
class PaperNeighborhoodExpansion:
    """Uses explicit cross-paper mentions to rerank a bounded candidate pool."""

    rrf_k: int
    candidate_k: int
    relation_weight: float
    two_hop_weight: float
    max_hub_degree: int

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        indexers,
    ) -> list[RetrievalResult]:
        """Rerank the head of the pool and leave the tail untouched."""

        if (
            self.relation_weight <= 0 and self.two_hop_weight <= 0
        ) or len(candidates) < 2:
            return candidates

        paper_index = find_paper_index(indexers, "get_document")
        if paper_index is None:
            return candidates

        pool_size = min(self.candidate_k, len(candidates))
        selector = PaperNeighborhoodReranker(
            get_document=paper_index.get_document,
            rrf_k=self.rrf_k,
            relation_weight=self.relation_weight,
            two_hop_weight=self.two_hop_weight,
            max_hub_degree=self.max_hub_degree,
        )
        reranked = selector.rerank(query, candidates[:pool_size], pool_size)
        return [*reranked, *candidates[pool_size:]]


@dataclass(frozen=True)
class _Lanes:
    """Which method lanes the configuration and the hints jointly enable."""

    owners: bool
    relations: bool
    topics: bool

    @property
    def any_enabled(self) -> bool:
        return self.owners or self.relations or self.topics


@dataclass
class _RelationPlan:
    """Everything gathered before scoring; discarded wholesale on failure."""

    valid_owners: list[dict] = field(default_factory=list)
    relation_records: list[dict] = field(default_factory=list)
    topic_by_id: dict[str, dict] = field(default_factory=dict)
    topic_search_succeeded: bool = False
    added_by_id: dict[str, RetrievalResult] = field(default_factory=dict)
    protected_count: int = 0
    protected_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class MethodRelationExpansion:
    """Ranks method-linked papers and reports the normal results to preserve."""

    retriever: Retriever
    rrf_k: int
    candidate_k: int
    seed_text_chars: int
    owner_weight: float
    relation_weight: float
    topic_weight: float
    topic_seed_chars: int
    topic_seed_k: int
    topic_max_results: int
    relation_seed_k: int
    relation_max_results: int
    relation_max_new_papers: int
    relation_protected_top_k: int

    def expand(
        self,
        candidates: list[RetrievalResult],
        hints: SearchHints | None,
        indexers,
        *,
        output_k: int,
    ) -> tuple[list[RetrievalResult], tuple[RetrievalResult, ...]]:
        """Rank method-linked papers and return normal results to preserve."""

        has_method_hints = hints is not None and bool(hints.methods)
        lanes = _Lanes(
            owners=self.owner_weight > 0 and has_method_hints,
            relations=self.relation_weight > 0 and has_method_hints,
            topics=self.topic_weight > 0 and has_method_hints,
        )
        if not candidates or not lanes.any_enabled:
            return candidates, ()

        provider = find_paper_index(indexers, *METHOD_PROVIDER_METHODS)
        if provider is None:
            return candidates, ()

        try:
            plan = self._build_plan(
                provider,
                candidates,
                hints,
                lanes,
                output_k=output_k,
            )
        except Exception:
            return candidates, ()

        topic_records = sorted(
            plan.topic_by_id.values(),
            key=lambda record: (
                record["best_result_rank"],
                record["best_seed_rank"],
                record["paper_id"],
            ),
        )
        if not (
            (lanes.owners and plan.valid_owners)
            or (lanes.relations and plan.relation_records)
            or topic_records
        ):
            return candidates, ()

        ranked = self._score_and_rank(candidates, plan, topic_records, lanes)
        protected_candidates = tuple(candidates[: plan.protected_count])
        if not plan.protected_ids or output_k <= 0:
            return ranked, protected_candidates
        return (
            self._restore_protected(ranked, candidates, plan, output_k),
            protected_candidates,
        )

    def _build_plan(
        self,
        provider,
        candidates: list[RetrievalResult],
        hints: SearchHints,
        lanes: _Lanes,
        *,
        output_k: int,
    ) -> _RelationPlan:
        """Collect owners, topics and relations; any failure aborts the stage."""

        owner_records = find_method_owner_records(
            provider,
            hints.methods,
            candidates,
            limit=max(
                self.relation_max_results,
                self.topic_seed_k if lanes.topics else 0,
            ),
        )
        original_by_id = {
            candidate.paper_id: candidate for candidate in candidates
        }
        normal_candidate_ids = {
            candidate.paper_id for candidate in candidates[: self.candidate_k]
        }
        protected_count = min(
            self.relation_protected_top_k,
            output_k,
            len(candidates),
        )
        protected_ids = {
            candidate.paper_id for candidate in candidates[:protected_count]
        }

        valid_owners: list[dict] = []
        for record in owner_records:
            if not isinstance(record, dict):
                continue
            paper_id = record.get("paper_id")
            if isinstance(paper_id, str) and paper_id in normal_candidate_ids:
                valid_owners.append(record)

        seed_ids: list[str] = []
        for record in valid_owners[: self.relation_seed_k]:
            paper_id = record["paper_id"]
            if paper_id not in seed_ids:
                seed_ids.append(paper_id)

        owner_ids = {record["paper_id"] for record in valid_owners}

        plan = _RelationPlan(
            valid_owners=valid_owners,
            protected_count=protected_count,
            protected_ids=protected_ids,
        )

        if lanes.topics:
            try:
                plan.topic_by_id = self._collect_topic_lane(
                    provider,
                    hints,
                    valid_owners,
                    normal_candidate_ids=normal_candidate_ids,
                    owner_ids=owner_ids,
                    protected_ids=protected_ids,
                )
                plan.topic_search_succeeded = True
            except Exception:
                # Topic expansion is optional. Keep exact owner and
                # relation evidence when the additional search fails.
                plan.topic_by_id = {}

        relation_by_id: dict[str, dict] = {}
        if lanes.relations and seed_ids:
            relation_by_id = self._collect_relation_lane(provider, seed_ids)

        ranked_relation_records = sorted(
            relation_by_id.values(),
            key=lambda record: (
                -record["strength"],
                record["best_seed_rank"],
                record["paper_id"],
            ),
        )[: self.relation_max_results]
        for record in ranked_relation_records:
            paper_id = record["paper_id"]
            if paper_id not in original_by_id:
                if len(plan.added_by_id) >= self.relation_max_new_papers:
                    continue
                document = provider.get_document(paper_id)
                if (
                    not hasattr(document, "paper_id")
                    or document.paper_id != paper_id
                    or not isinstance(document.text, str)
                    or not document.text.strip()
                ):
                    continue
                metadata = (
                    dict(document.metadata)
                    if isinstance(document.metadata, dict)
                    else {}
                )
                plan.added_by_id[paper_id] = RetrievalResult(
                    chunk_id=document.chunk_id,
                    paper_id=paper_id,
                    score=0.0,
                    text=document.text[: self.seed_text_chars],
                    chunk_type=document.chunk_type,
                    metadata=metadata,
                    source="method_relation",
                )
            plan.relation_records.append(record)
        return plan

    def _collect_topic_lane(
        self,
        provider,
        hints: SearchHints,
        valid_owners: list[dict],
        *,
        normal_candidate_ids: set[str],
        owner_ids: set[str],
        protected_ids: set[str],
    ) -> dict[str, dict]:
        """Search each owner's own text to surface topically adjacent papers."""

        topic_seed_ids: list[str] = []
        for record in valid_owners[: self.topic_seed_k]:
            paper_id = record["paper_id"]
            if paper_id not in topic_seed_ids:
                topic_seed_ids.append(paper_id)

        retrieval_hints = without_method_hints(hints)
        topic_by_id: dict[str, dict] = {}
        for seed_rank, seed_id in enumerate(topic_seed_ids, start=1):
            document = provider.get_document(seed_id)
            if (
                not hasattr(document, "paper_id")
                or document.paper_id != seed_id
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
            for result_rank, result in enumerate(topic_results, start=1):
                paper_id = getattr(result, "paper_id", None)
                if (
                    not isinstance(paper_id, str)
                    or paper_id in seen_topic_papers
                ):
                    continue
                seen_topic_papers.add(paper_id)
                if (
                    paper_id not in normal_candidate_ids
                    or paper_id in owner_ids
                    or paper_id in protected_ids
                ):
                    continue
                state = topic_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "best_result_rank": result_rank,
                        "best_seed_rank": seed_rank,
                        "via_papers": set(),
                    },
                )
                state["best_result_rank"] = min(
                    state["best_result_rank"],
                    result_rank,
                )
                state["best_seed_rank"] = min(
                    state["best_seed_rank"],
                    seed_rank,
                )
                state["via_papers"].add(seed_id)
        return topic_by_id

    def _collect_relation_lane(
        self,
        provider,
        seed_ids: list[str],
    ) -> dict[str, dict]:
        """Follow method-to-method edges out of each owner seed."""

        relation_by_id: dict[str, dict] = {}
        for seed_rank, seed_id in enumerate(seed_ids, start=1):
            records = provider.get_method_neighbors(
                seed_id,
                limit=self.relation_max_results,
            )
            for record in records:
                if not isinstance(record, dict):
                    continue
                paper_id = record.get("paper_id")
                if not isinstance(paper_id, str) or paper_id == seed_id:
                    continue
                state = relation_by_id.setdefault(
                    paper_id,
                    {
                        "paper_id": paper_id,
                        "aliases": set(),
                        "via_papers": set(),
                        "strength": 0,
                        "best_seed_rank": seed_rank,
                    },
                )
                aliases = record.get("aliases")
                if isinstance(aliases, (list, tuple)):
                    state["aliases"].update(
                        alias for alias in aliases if isinstance(alias, str)
                    )
                strength = record.get("strength")
                if isinstance(strength, int) and not isinstance(strength, bool):
                    state["strength"] += max(strength, 0)
                state["via_papers"].add(seed_id)
                state["best_seed_rank"] = min(
                    state["best_seed_rank"],
                    seed_rank,
                )
        return relation_by_id

    def _score_and_rank(
        self,
        candidates: list[RetrievalResult],
        plan: _RelationPlan,
        topic_records: list[dict],
        lanes: _Lanes,
    ) -> list[RetrievalResult]:
        """Fuse the baseline order with the owner, relation and topic lanes."""

        owner_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(plan.valid_owners, start=1)
        }
        owner_by_id = {
            record["paper_id"]: record for record in plan.valid_owners
        }
        relation_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(plan.relation_records, start=1)
        }
        relation_by_id = {
            record["paper_id"]: record for record in plan.relation_records
        }
        topic_rank_by_id = {
            record["paper_id"]: rank
            for rank, record in enumerate(topic_records, start=1)
        }
        topic_record_by_id = {
            record["paper_id"]: record for record in topic_records
        }
        combined = [*candidates, *plan.added_by_id.values()]
        scored: list[tuple[float, int, RetrievalResult]] = []
        for baseline_rank, candidate in enumerate(combined, start=1):
            owner_rank = owner_rank_by_id.get(candidate.paper_id)
            relation_rank = relation_rank_by_id.get(candidate.paper_id)
            topic_rank = topic_rank_by_id.get(candidate.paper_id)
            score = 1.0 / (self.rrf_k + baseline_rank)
            if owner_rank is not None:
                score += self.owner_weight / (self.rrf_k + owner_rank)
            if relation_rank is not None:
                score += self.relation_weight / (self.rrf_k + relation_rank)
            if topic_rank is not None:
                score += self.topic_weight / (self.rrf_k + topic_rank)

            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "method_relation_baseline_rank": baseline_rank,
                    "method_owner_rank": owner_rank,
                    "method_relation_rank": relation_rank,
                }
            )
            if lanes.topics and plan.topic_search_succeeded:
                metadata["method_topic_rank"] = topic_rank
            owner_record = owner_by_id.get(candidate.paper_id)
            if owner_record is not None:
                metadata["method_owner_aliases"] = list(
                    owner_record.get("aliases") or ()
                )
            relation_record = relation_by_id.get(candidate.paper_id)
            if relation_record is not None:
                metadata.update(
                    {
                        "method_relation_aliases": sorted(
                            relation_record["aliases"]
                        ),
                        "method_relation_via_papers": sorted(
                            relation_record["via_papers"]
                        ),
                        "method_relation_strength": relation_record["strength"],
                    }
                )
            topic_record = topic_record_by_id.get(candidate.paper_id)
            if topic_record is not None:
                metadata.update(
                    {
                        "method_topic_via_papers": sorted(
                            topic_record["via_papers"]
                        ),
                        "method_topic_search_rank": topic_record[
                            "best_result_rank"
                        ],
                    }
                )
            scored.append(
                (
                    score,
                    baseline_rank,
                    replace(
                        candidate,
                        score=score,
                        metadata=metadata,
                        source="method_relation_rrf",
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1], item[2].paper_id))
        return [result for _, _, result in scored]

    @staticmethod
    def _restore_protected(
        ranked: list[RetrievalResult],
        candidates: list[RetrievalResult],
        plan: _RelationPlan,
        output_k: int,
    ) -> list[RetrievalResult]:
        """Put baseline papers back, evicting non-owners before owners."""

        selected = ranked[:output_k]
        selected_ids = {candidate.paper_id for candidate in selected}
        ranked_by_id = {candidate.paper_id: candidate for candidate in ranked}
        owner_ids = {record["paper_id"] for record in plan.valid_owners}
        protected_ids = plan.protected_ids
        missing = [
            ranked_by_id[candidate.paper_id]
            for candidate in candidates[: plan.protected_count]
            if candidate.paper_id not in selected_ids
        ]
        for candidate in missing:
            replacement_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index].paper_id not in protected_ids
                    and selected[index].paper_id not in owner_ids
                ),
                None,
            )
            if replacement_index is None:
                replacement_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index].paper_id not in protected_ids
                    ),
                    None,
                )
            if replacement_index is None:
                break
            del selected[replacement_index]
            selected.append(candidate)

        selected_ids = {candidate.paper_id for candidate in selected}
        return [
            *selected,
            *(
                candidate
                for candidate in ranked
                if candidate.paper_id not in selected_ids
            ),
        ]


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
