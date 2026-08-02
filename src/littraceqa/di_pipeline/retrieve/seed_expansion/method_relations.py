"""Method relation expansion.

Follows method ownership and method-to-method edges to reach papers the plain
query missed, then rescores the candidate list. Papers promoted by this lane
are protected so later stages cannot silently drop them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints
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
class _Lanes:
    """Which method lanes the configuration and the hints jointly enable."""

    owners: bool
    relations: bool
    topics: bool

    @property
    def any_enabled(self) -> bool:
        return self.owners or self.relations or self.topics


@dataclass(frozen=True)
class _LaneRanks:
    """Rank and record of each paper, per lane, for one fusion pass."""

    owner_rank: dict[str, int]
    owner_record: dict[str, dict]
    relation_rank: dict[str, int]
    relation_record: dict[str, dict]
    topic_rank: dict[str, int]
    topic_record: dict[str, dict]

    @classmethod
    def from_plan(
        cls,
        plan: _RelationPlan,
        topic_records: list[dict],
    ) -> _LaneRanks:
        def indexed(records: list[dict]) -> tuple[dict[str, int], dict[str, dict]]:
            ranks = {
                record["paper_id"]: rank
                for rank, record in enumerate(records, start=1)
            }
            return ranks, {record["paper_id"]: record for record in records}

        owner_rank, owner_record = indexed(plan.valid_owners)
        relation_rank, relation_record = indexed(plan.relation_records)
        topic_rank, topic_record = indexed(topic_records)
        return cls(
            owner_rank=owner_rank,
            owner_record=owner_record,
            relation_rank=relation_rank,
            relation_record=relation_record,
            topic_rank=topic_rank,
            topic_record=topic_record,
        )


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
        self._admit_relation_records(
            plan,
            provider,
            relation_by_id,
            original_by_id,
        )
        return plan

    def _admit_relation_records(
        self,
        plan: _RelationPlan,
        provider,
        relation_by_id: dict[str, dict],
        original_by_id: dict[str, RetrievalResult],
    ) -> None:
        """Accept the strongest relations, materialising a bounded number of new papers."""

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

        ranks = _LaneRanks.from_plan(plan, topic_records)
        combined = [*candidates, *plan.added_by_id.values()]
        scored: list[tuple[float, int, RetrievalResult]] = []
        for baseline_rank, candidate in enumerate(combined, start=1):
            score = self._fused_score(ranks, candidate.paper_id, baseline_rank)
            metadata = self._lane_metadata(
                candidate,
                ranks,
                baseline_rank,
                report_topic_rank=lanes.topics and plan.topic_search_succeeded,
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

    def _fused_score(
        self,
        ranks: _LaneRanks,
        paper_id: str,
        baseline_rank: int,
    ) -> float:
        """Weight the baseline rank by each lane that also found this paper."""

        score = 1.0 / (self.rrf_k + baseline_rank)
        for weight, rank in (
            (self.owner_weight, ranks.owner_rank.get(paper_id)),
            (self.relation_weight, ranks.relation_rank.get(paper_id)),
            (self.topic_weight, ranks.topic_rank.get(paper_id)),
        ):
            if rank is not None:
                score += weight / (self.rrf_k + rank)
        return score

    @staticmethod
    def _lane_metadata(
        candidate: RetrievalResult,
        ranks: _LaneRanks,
        baseline_rank: int,
        *,
        report_topic_rank: bool,
    ) -> dict:
        """Record which lane contributed the paper so evaluation can trace it."""

        paper_id = candidate.paper_id
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "method_relation_baseline_rank": baseline_rank,
                "method_owner_rank": ranks.owner_rank.get(paper_id),
                "method_relation_rank": ranks.relation_rank.get(paper_id),
            }
        )
        if report_topic_rank:
            metadata["method_topic_rank"] = ranks.topic_rank.get(paper_id)
        owner_record = ranks.owner_record.get(paper_id)
        if owner_record is not None:
            metadata["method_owner_aliases"] = list(
                owner_record.get("aliases") or ()
            )
        relation_record = ranks.relation_record.get(paper_id)
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
        topic_record = ranks.topic_record.get(paper_id)
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
        return metadata

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
