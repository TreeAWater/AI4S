from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from evolab.backends.skills.base import SkillBackend
from evolab.backends.skills.evolution import (
    SkillEvolutionAnalyzer,
    SkillEvolutionPolicy,
    SkillEvolutionRecordSet,
    SkillEvolutionStore,
    SkillUpdateDecision,
    SkillUpdateProposal,
)
from evolab.backends.skills.graph_indexer import (
    CategoryIndex,
    GraphTreeIndexer,
    _build_category_index,
    _get_ancestors,
    _get_descendants,
    _get_root_for_category,
    _get_subtree_category_ids,
)
from evolab.backends.skills.graph_schema import MissingSkillReport, SkillUpdateSummary
from evolab.backends.skills.searcher import (
    QueryInfo,
    RetrievalCandidate,
    RetrievalPath,
    _build_retrieval_paths,
    _category_path,
    _coverage_report,
    _expand_relationships,
    _match_root_capabilities,
    _match_scientific_tasks,
    _parse_query_info,
    _retrieve_path_seed_candidates,
    GraphSkillSearcher,
)
from evolab.backends.skills.store import _EMPTY_GRAPH, GraphSkillStore
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest, SkillUpdateResult


class GraphSkillBackend(SkillBackend):
    backend_id = "graph_skill"

    def __init__(
        self,
        graph_path: Path | str,
        *,
        repo_root: Path | str | None = None,
        strict_packages: bool = False,
        evolution_root: Path | str | None = None,
        evolution_policy: SkillEvolutionPolicy | None = None,
    ):
        self.store = GraphSkillStore(graph_path, repo_root=repo_root, strict_packages=strict_packages)
        self.graph_path = self.store.graph_path
        self.evolution_root = Path(evolution_root) if evolution_root is not None else (
            self.graph_path.parent / "backend_state" / "skill_evolution"
        )
        self.evolution_policy = evolution_policy or SkillEvolutionPolicy()

    def _load_raw_graph(self) -> dict[str, Any]:
        return self.store.load_raw_graph()

    def _load_graph(self):
        loaded = self.store.load_graph()
        return loaded.graph, loaded.skipped_skills

    def get(self, request: RetrievalRequest) -> SkillBundle:
        loaded = self.store.load_graph()
        indexer = GraphTreeIndexer(loaded.graph)
        search_result = GraphSkillSearcher(loaded.graph, indexer).search(request)
        metadata = dict(search_result.metadata)
        graph_context_summary = metadata["graph_context_summary"]
        if loaded.warnings:
            graph_context_summary["warnings"].extend(loaded.warnings)
            metadata["package_warnings"] = loaded.warnings
        if loaded.skipped_skills:
            metadata["skipped_skills"] = loaded.skipped_skills

        coverage = graph_context_summary["coverage_report"]
        if search_result.query.tokens and not search_result.ranked_candidates:
            metadata["missing_skill_report"] = MissingSkillReport(
                missing_capability=request.query,
                reason="No CandidateSkill matched the retrieval query.",
                can_be_solved_by_existing_tools=False,
                risk_level="medium",
                on_demand_synthesis_allowed=False,
            ).model_dump(mode="json")
        elif search_result.ranked_candidates and not coverage["sufficient"]:
            missing = ", ".join(coverage["missing"])
            metadata["missing_skill_report"] = MissingSkillReport(
                missing_capability=request.query,
                reason=f"Matched CandidateSkill coverage is insufficient: {missing}.",
                can_be_solved_by_existing_tools=False,
                risk_level="medium",
                on_demand_synthesis_allowed=False,
                metadata={"coverage_report": coverage},
            ).model_dump(mode="json")

        return SkillBundle(
            skills=search_result.skills,
            required_tools=search_result.required_tools,
            backend_id=self.backend_id,
            graph_version_ref=loaded.graph.version,
            skill_state_ref=loaded.graph.version,
            metadata=metadata,
        )

    def look_at(self, event: dict[str, Any] | SkillObservationRequest) -> SkillUpdateResult:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        raw_graph = self._load_raw_graph()
        graph_version = raw_graph.get("version")
        graph_version_ref = graph_version if isinstance(graph_version, str) else None
        before_state_hash = _graph_state_hash(raw_graph)
        records = SkillEvolutionAnalyzer().analyze(
            payload,
            backend_id=self.backend_id,
            graph_version_ref=graph_version_ref,
            policy=self.evolution_policy,
        )
        applied_updates, changed_library = _apply_safe_updates(
            raw_graph,
            records=records,
            policy=self.evolution_policy,
        )
        if changed_library:
            _write_graph_atomic(self.graph_path, raw_graph)
        after_state_hash = _graph_state_hash(raw_graph)
        output_paths = SkillEvolutionStore(self.evolution_root).persist(records)

        update_log = self.graph_path.with_suffix(".updates.jsonl")
        summary = SkillUpdateSummary(
            source_run_id=payload.get("source_run_id") or payload.get("run_ref"),
            candidate_skill_id=payload.get("candidate_skill_id") or payload.get("skill_id"),
            update_type=payload.get("update_type", "post_run_evolution_v1"),
            affected_skill_ids=_dedupe(
                [
                    *payload.get("affected_skill_ids", []),
                    *records.observation_summary.get("selected_skill_ids", []),
                ]
            ),
            affected_edges=payload.get("affected_edges", []),
            decision_rationale=payload.get("decision_rationale"),
            validation_signals=payload.get("validation_signals", []),
            graph_version_before=graph_version_ref,
            graph_version_after=graph_version_ref,
            provenance={
                **payload.get("provenance", {}),
                "observation_id": records.observation_summary["observation_id"],
                "proposal_ids": [proposal.proposal_id for proposal in records.proposals],
            },
        )
        with update_log.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(summary.model_dump(mode="json"), sort_keys=True) + "\n")
        output_paths["legacy_update_log"] = str(update_log)
        decisions_by_status = _decision_counts(records.decisions)
        staged_proposals = [
            proposal
            for proposal in records.proposals
            if _decision_for_proposal(records.decisions, proposal.proposal_id) == "staged"
        ]
        status = "updated" if changed_library else ("staged" if staged_proposals else "recorded")
        update_summary = {
            "observation_id": records.observation_summary["observation_id"],
            "proposal_count": len(records.proposals),
            "decision_count": len(records.decisions),
            "applied_count": decisions_by_status.get("applied", 0),
            "staged_count": decisions_by_status.get("staged", 0),
            "rejected_count": decisions_by_status.get("rejected", 0),
            "changed_library": changed_library,
            "staged_updates": bool(staged_proposals),
            "before_graph_version": graph_version_ref,
            "after_graph_version": graph_version_ref,
            "graph_version_before": graph_version_ref,
            "graph_version_after": graph_version_ref,
            "before_state_hash": before_state_hash,
            "after_state_hash": after_state_hash,
            "output_paths": output_paths,
            "proposal_summaries": [
                {
                    "proposal_id": proposal.proposal_id,
                    "proposal_type": proposal.proposal_type,
                    "related_skill_ids": proposal.related_skill_ids,
                }
                for proposal in records.proposals
            ],
        }
        return SkillUpdateResult(
            status=status,
            update_summary=update_summary,
            graph_version_ref=graph_version_ref,
            skill_state_ref=graph_version_ref,
            metadata={
                "update_log": str(update_log),
                "graph_version_before": graph_version_ref,
                "graph_version_after": graph_version_ref,
                "observation_summary": records.observation_summary,
                "proposals": [proposal.model_dump(mode="json") for proposal in records.proposals],
                "decisions": [decision.model_dump(mode="json") for decision in records.decisions],
                "applied_updates": applied_updates,
                "staged_updates": [proposal.model_dump(mode="json") for proposal in staged_proposals],
                "candidate_records": [record.model_dump(mode="json") for record in records.candidate_records],
                "changed_library": changed_library,
                "output_paths": output_paths,
            },
        )

    def mine_resources(self) -> None:
        raise NotImplementedError("mine_resources is not implemented for GraphSkillBackend")

    def rewire_edges(self) -> None:
        raise NotImplementedError("rewire_edges is not implemented for GraphSkillBackend")


def _apply_safe_updates(
    raw_graph: dict[str, Any],
    *,
    records: SkillEvolutionRecordSet,
    policy: SkillEvolutionPolicy,
) -> tuple[list[dict[str, Any]], bool]:
    decisions_by_proposal_id = {decision.proposal_id: decision for decision in records.decisions}
    applied_updates: list[dict[str, Any]] = []
    changed = False
    for proposal in records.proposals:
        decision = decisions_by_proposal_id.get(proposal.proposal_id)
        if decision is None or decision.decision != "applied":
            continue
        if proposal.proposal_type not in {"usage_stats_update", "failure_note_update"}:
            continue
        for raw_skill in _raw_skill_entries(raw_graph, proposal.related_skill_ids):
            metadata = raw_skill.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                raw_skill["metadata"] = metadata
            stats = metadata.setdefault("evolution_stats", {})
            if not isinstance(stats, dict):
                stats = {}
                metadata["evolution_stats"] = stats
            before = json.dumps(stats, sort_keys=True)
            if proposal.proposal_type == "usage_stats_update":
                _apply_usage_stats(stats, proposal)
            elif proposal.proposal_type == "failure_note_update":
                _apply_failure_note(stats, proposal, policy)
            after = json.dumps(stats, sort_keys=True)
            if before != after:
                changed = True
                applied_updates.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "proposal_type": proposal.proposal_type,
                        "skill_id": _raw_skill_id(raw_skill),
                        "metadata_path": "metadata.evolution_stats",
                    }
                )
    return applied_updates, changed


def _apply_usage_stats(stats: dict[str, Any], proposal: SkillUpdateProposal) -> None:
    payload = proposal.payload
    stats["usage_count"] = _int(stats.get("usage_count")) + _int(payload.get("increment_usage_count"))
    stats["success_count"] = _int(stats.get("success_count")) + _int(payload.get("increment_success_count"))
    stats["failure_count"] = _int(stats.get("failure_count")) + _int(payload.get("increment_failure_count"))
    stats["last_observed_at"] = proposal.created_at
    stats["last_status"] = str(payload.get("status") or "unknown")


def _apply_failure_note(
    stats: dict[str, Any],
    proposal: SkillUpdateProposal,
    policy: SkillEvolutionPolicy,
) -> None:
    reason = proposal.payload.get("failure_reason")
    if not isinstance(reason, str) or not reason:
        return
    bounded = reason[: policy.max_failure_reason_chars]
    current = stats.get("recent_failure_reasons")
    reasons = current if isinstance(current, list) else []
    deduped = _dedupe([bounded, *(item for item in reasons if isinstance(item, str))])
    stats["recent_failure_reasons"] = deduped[: policy.max_recent_failure_reasons]


def _raw_skill_entries(raw_graph: dict[str, Any], skill_ids: list[str]) -> list[dict[str, Any]]:
    skill_id_set = set(skill_ids)
    return [
        raw_skill
        for raw_skill in raw_graph.get("skills", [])
        if isinstance(raw_skill, dict) and _raw_skill_id(raw_skill) in skill_id_set
    ]


def _raw_skill_id(raw_skill: dict[str, Any]) -> str | None:
    value = raw_skill.get("id") or raw_skill.get("skill_id")
    return value if isinstance(value, str) else None


def _write_graph_atomic(path: Path, raw_graph: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(raw_graph, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _graph_state_hash(raw_graph: dict[str, Any]) -> str:
    payload = json.dumps(raw_graph, sort_keys=True, separators=(",", ":"))
    return f"skill-graph-sha256-{sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _decision_counts(decisions: list[SkillUpdateDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.decision] = counts.get(decision.decision, 0) + 1
    return counts


def _decision_for_proposal(decisions: list[SkillUpdateDecision], proposal_id: str) -> str | None:
    for decision in decisions:
        if decision.proposal_id == proposal_id:
            return decision.decision
    return None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    deduped: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


__all__ = [
    "CategoryIndex",
    "GraphSkillBackend",
    "GraphSkillSearcher",
    "GraphTreeIndexer",
    "QueryInfo",
    "RetrievalCandidate",
    "RetrievalPath",
    "_EMPTY_GRAPH",
    "_build_category_index",
    "_build_retrieval_paths",
    "_category_path",
    "_coverage_report",
    "_expand_relationships",
    "_get_ancestors",
    "_get_descendants",
    "_get_root_for_category",
    "_get_subtree_category_ids",
    "_match_root_capabilities",
    "_match_scientific_tasks",
    "_parse_query_info",
    "_retrieve_path_seed_candidates",
]
