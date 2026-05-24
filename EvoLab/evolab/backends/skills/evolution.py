from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


SkillUpdateProposalType = Literal[
    "metadata_update",
    "usage_stats_update",
    "failure_note_update",
    "example_trace_memory_update",
    "required_tools_update_proposal",
    "candidate_skill_creation",
    "relationship_update_proposal",
]

SkillUpdateDecisionStatus = Literal["applied", "staged", "rejected"]
SkillRunStatus = Literal["success", "failure", "partial", "unknown"]


class SkillUpdateProposal(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    proposal_id: str
    proposal_type: SkillUpdateProposalType
    observation_id: str
    backend_id: str
    graph_version_ref: str | None = None
    related_skill_ids: list[str] = Field(default_factory=list)
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillUpdateDecision(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    decision_id: str
    proposal_id: str
    decision: SkillUpdateDecisionStatus
    reason: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillEvolutionPolicy(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    auto_apply_proposal_types: list[SkillUpdateProposalType] = Field(
        default_factory=lambda: ["usage_stats_update", "failure_note_update"]
    )
    staged_proposal_types: list[SkillUpdateProposalType] = Field(
        default_factory=lambda: [
            "metadata_update",
            "example_trace_memory_update",
            "required_tools_update_proposal",
            "candidate_skill_creation",
            "relationship_update_proposal",
        ]
    )
    max_failure_reason_chars: int = Field(default=240, ge=1)
    max_recent_failure_reasons: int = Field(default=5, ge=1)
    max_example_summary_chars: int = Field(default=600, ge=1)
    auto_apply_valid_candidates: bool = False
    auto_apply_relationship_updates: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def decide(self, proposal: SkillUpdateProposal, *, now: str | None = None) -> SkillUpdateDecision:
        timestamp = now or _utc_now()
        if (
            proposal.proposal_type == "candidate_skill_creation"
            and self.auto_apply_valid_candidates
            and _is_bounded_payload(proposal, self)
        ):
            decision = "applied"
            reason = "auto_apply_valid_candidates policy enabled"
        elif (
            proposal.proposal_type == "relationship_update_proposal"
            and self.auto_apply_relationship_updates
            and _is_bounded_payload(proposal, self)
        ):
            decision = "applied"
            reason = "auto_apply_relationship_updates policy enabled"
        elif proposal.proposal_type in self.auto_apply_proposal_types and _is_bounded_payload(proposal, self):
            decision = "applied"
            reason = "safe append-only metadata update"
        elif proposal.proposal_type in self.staged_proposal_types:
            decision = "staged"
            reason = "requires review before modifying stable skill library"
        else:
            decision = "rejected"
            reason = "proposal type is not allowed by policy"
        return SkillUpdateDecision(
            decision_id=_stable_id("skill-decision", [proposal.proposal_id, decision, reason]),
            proposal_id=proposal.proposal_id,
            decision=decision,
            reason=reason,
            created_at=timestamp,
        )


class CandidateSkillRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    candidate_id: str
    proposed_name: str
    proposed_task: str | None = None
    proposed_category: str | None = None
    source_observation_id: str
    evidence_summary: str
    missing_capability_description: str
    suggested_required_inputs: list[str] = Field(default_factory=list)
    suggested_expected_outputs: list[str] = Field(default_factory=list)
    suggested_required_tools: list[str] = Field(default_factory=list)
    status: Literal["staged"] = "staged"
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillEvolutionRecordSet(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    observation_summary: dict[str, Any]
    proposals: list[SkillUpdateProposal] = Field(default_factory=list)
    decisions: list[SkillUpdateDecision] = Field(default_factory=list)
    candidate_records: list[CandidateSkillRecord] = Field(default_factory=list)


class SkillEvolutionAnalyzer:
    def analyze(
        self,
        observation: dict[str, Any],
        *,
        backend_id: str,
        graph_version_ref: str | None,
        policy: SkillEvolutionPolicy | None = None,
    ) -> SkillEvolutionRecordSet:
        policy = policy or SkillEvolutionPolicy()
        timestamp = _utc_now()
        summary = _observation_summary(observation, backend_id=backend_id, graph_version_ref=graph_version_ref)
        proposals: list[SkillUpdateProposal] = []

        skill_ids = list(summary["selected_skill_ids"])
        status = summary["status"]
        if skill_ids:
            proposals.append(
                _proposal(
                    proposal_type="usage_stats_update",
                    observation_summary=summary,
                    related_skill_ids=skill_ids,
                    summary=f"Record {status} usage for {len(skill_ids)} retrieved skills.",
                    payload={
                        "status": status,
                        "increment_usage_count": 1,
                        "increment_success_count": 1 if status == "success" else 0,
                        "increment_failure_count": 1 if status == "failure" else 0,
                    },
                    timestamp=timestamp,
                )
            )

        if status == "failure" and skill_ids:
            proposals.append(
                _proposal(
                    proposal_type="failure_note_update",
                    observation_summary=summary,
                    related_skill_ids=skill_ids,
                    summary="Append bounded failure note to retrieved skill metadata.",
                    payload={"failure_reason": _bounded(summary["failure_reason"], policy.max_failure_reason_chars)},
                    timestamp=timestamp,
                )
            )

        if status == "success" and skill_ids:
            proposals.append(
                _proposal(
                    proposal_type="example_trace_memory_update",
                    observation_summary=summary,
                    related_skill_ids=skill_ids,
                    summary="Stage compact successful trace as future example memory.",
                    payload={"example_summary": _bounded(_example_summary(summary), policy.max_example_summary_chars)},
                    timestamp=timestamp,
                )
            )

        missing_tools = _missing_required_tools(observation)
        if missing_tools and skill_ids:
            proposals.append(
                _proposal(
                    proposal_type="required_tools_update_proposal",
                    observation_summary=summary,
                    related_skill_ids=skill_ids,
                    summary="Stage missing required tool update for review.",
                    payload={"missing_required_tools": missing_tools},
                    timestamp=timestamp,
                )
            )

        if _needs_candidate_skill(summary) and (summary["missing_capability"] or summary["query"]):
            missing_capability = summary["missing_capability"] or summary["query"]
            record = _candidate_record(summary, missing_capability=missing_capability, timestamp=timestamp)
            proposals.append(
                _proposal(
                    proposal_type="candidate_skill_creation",
                    observation_summary=summary,
                    related_skill_ids=[],
                    summary="Stage candidate skill for uncovered capability.",
                    payload={"candidate_record": record.model_dump(mode="json")},
                    timestamp=timestamp,
                )
            )
            candidate_records = [record]
        else:
            candidate_records = []

        decisions = [policy.decide(proposal, now=timestamp) for proposal in proposals]
        return SkillEvolutionRecordSet(
            observation_summary=summary,
            proposals=proposals,
            decisions=decisions,
            candidate_records=candidate_records,
        )


class SkillEvolutionStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def output_paths(self) -> dict[str, str]:
        return {
            "observations": str(self.root / "observations.jsonl"),
            "proposals": str(self.root / "proposals.jsonl"),
            "decisions": str(self.root / "decisions.jsonl"),
            "staged_candidates": str(self.root / "staged_candidates.jsonl"),
            "staged_updates": str(self.root / "staged_updates.jsonl"),
            "applied_updates": str(self.root / "applied_updates.jsonl"),
        }

    def persist(self, records: SkillEvolutionRecordSet) -> dict[str, str]:
        self._append("observations.jsonl", records.observation_summary)
        for proposal in records.proposals:
            self._append("proposals.jsonl", proposal.model_dump(mode="json"))
        decisions_by_proposal = {decision.proposal_id: decision for decision in records.decisions}
        proposals_by_id = {proposal.proposal_id: proposal for proposal in records.proposals}
        for decision in records.decisions:
            self._append("decisions.jsonl", decision.model_dump(mode="json"))
            proposal = proposals_by_id.get(decision.proposal_id)
            if proposal is None:
                continue
            if decision.decision == "applied":
                self._append(
                    "applied_updates.jsonl",
                    {
                        "proposal": proposal.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            elif decision.decision == "staged":
                self._append(
                    "staged_updates.jsonl",
                    {
                        "proposal": proposal.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                    },
                )
        staged_candidate_proposal_ids = {
            proposal.proposal_id
            for proposal in records.proposals
            if proposal.proposal_type == "candidate_skill_creation"
            and decisions_by_proposal.get(proposal.proposal_id) is not None
            and decisions_by_proposal[proposal.proposal_id].decision == "staged"
        }
        for candidate in records.candidate_records:
            if staged_candidate_proposal_ids:
                self._append("staged_candidates.jsonl", candidate.model_dump(mode="json"))
        return self.output_paths

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.root / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _proposal(
    *,
    proposal_type: SkillUpdateProposalType,
    observation_summary: dict[str, Any],
    related_skill_ids: list[str],
    summary: str,
    payload: dict[str, Any],
    timestamp: str,
) -> SkillUpdateProposal:
    proposal_id = _stable_id(
        "skill-proposal",
        [
            proposal_type,
            observation_summary["observation_id"],
            related_skill_ids,
            payload,
        ],
    )
    return SkillUpdateProposal(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        observation_id=observation_summary["observation_id"],
        backend_id=observation_summary["backend_id"],
        graph_version_ref=observation_summary.get("graph_version_ref"),
        related_skill_ids=related_skill_ids,
        summary=summary,
        payload=payload,
        created_at=timestamp,
        metadata={
            "task_id": observation_summary.get("task_id"),
            "run_ref": observation_summary.get("run_ref"),
            "status": observation_summary.get("status"),
        },
    )


def _candidate_record(
    observation_summary: dict[str, Any],
    *,
    missing_capability: str,
    timestamp: str,
) -> CandidateSkillRecord:
    candidate_id = _stable_id("candidate-skill", [observation_summary["observation_id"], missing_capability])
    proposed_name = _candidate_name(missing_capability)
    return CandidateSkillRecord(
        candidate_id=candidate_id,
        proposed_name=proposed_name,
        proposed_task=observation_summary.get("role"),
        proposed_category=observation_summary.get("target_category"),
        source_observation_id=observation_summary["observation_id"],
        evidence_summary=_bounded(_example_summary(observation_summary), 600),
        missing_capability_description=missing_capability,
        suggested_required_inputs=["task goal", "available evidence"],
        suggested_expected_outputs=["validated task output"],
        suggested_required_tools=list(observation_summary.get("missing_required_tools", [])),
        created_at=timestamp,
        metadata={
            "query": observation_summary.get("query"),
            "coverage_report": observation_summary.get("coverage_report"),
        },
    )


def _observation_summary(
    observation: dict[str, Any],
    *,
    backend_id: str,
    graph_version_ref: str | None,
) -> dict[str, Any]:
    retrieval_request = _dict(observation.get("retrieval_request"))
    skill_bundle = _dict(observation.get("skill_bundle"))
    metadata = _dict(observation.get("metadata"))
    tool_trace = _dict(observation.get("tool_trace"))
    skills = _list(skill_bundle.get("skills"))
    skill_ids = [skill["skill_id"] for skill in skills if isinstance(skill, dict) and isinstance(skill.get("skill_id"), str)]
    tool_calls = _tool_call_summaries(tool_trace)
    artifact_refs = _artifact_refs(observation, metadata, tool_trace)
    coverage_report = _coverage_report(skill_bundle)
    missing_report = _dict(skill_bundle.get("metadata")).get("missing_skill_report")
    observation_id = _stable_id(
        "skill-observation",
        [
            observation.get("task_id"),
            observation.get("run_ref"),
            observation.get("role"),
            retrieval_request.get("query"),
        ],
    )
    missing_tools = _missing_required_tools(observation)
    status = _run_status(observation, tool_calls)
    return {
        "observation_id": observation_id,
        "backend_id": backend_id,
        "task_id": observation.get("task_id"),
        "run_ref": observation.get("run_ref"),
        "role": observation.get("role"),
        "query": retrieval_request.get("query"),
        "target_category": _first_string(
            retrieval_request.get("metadata", {}).get("target_category")
            if isinstance(retrieval_request.get("metadata"), dict)
            else None,
            retrieval_request.get("filters", {}).get("target_category")
            if isinstance(retrieval_request.get("filters"), dict)
            else None,
        ),
        "graph_version_ref": graph_version_ref or observation.get("graph_version_ref") or skill_bundle.get("graph_version_ref"),
        "skill_state_ref": observation.get("skill_state_ref") or skill_bundle.get("skill_state_ref"),
        "selected_skill_ids": skill_ids,
        "required_tools": _dedupe(_list_of_strings(skill_bundle.get("required_tools"))),
        "missing_required_tools": missing_tools,
        "status": status,
        "failure_reason": _failure_reason(observation, tool_calls, status),
        "final_answer_present": bool(observation.get("final_answer")),
        "final_answer_preview": _bounded(str(observation.get("final_answer") or ""), 240),
        "tool_calls": tool_calls,
        "tool_names": _dedupe([call["name"] for call in tool_calls if call.get("name")]),
        "artifact_refs": artifact_refs,
        "coverage_report": coverage_report,
        "missing_skill_report": missing_report,
        "missing_capability": _missing_capability(observation, skill_bundle),
    }


def _run_status(observation: dict[str, Any], tool_calls: list[dict[str, Any]]) -> SkillRunStatus:
    metadata = _dict(observation.get("metadata"))
    explicit = _first_string(metadata.get("status"), metadata.get("run_status"), observation.get("status"))
    if explicit:
        normalized = explicit.casefold()
        if normalized in {"success", "succeeded", "completed", "ok"}:
            return "success"
        if normalized in {"failure", "failed", "error"}:
            return "failure"
        if normalized == "partial":
            return "partial"
    plan_status = _dict(metadata.get("plan_execution_trace")).get("status")
    if isinstance(plan_status, str):
        if plan_status == "completed":
            return "success"
        if plan_status == "failed":
            return "failure"
        if plan_status == "partial":
            return "partial"
    if any(call.get("status") == "error" for call in tool_calls):
        return "failure"
    if observation.get("final_answer"):
        return "success"
    return "unknown"


def _failure_reason(
    observation: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    status: SkillRunStatus,
) -> str:
    metadata = _dict(observation.get("metadata"))
    explicit = _first_string(metadata.get("failure_reason"), metadata.get("error"), observation.get("failure_reason"))
    if explicit:
        return explicit
    for call in tool_calls:
        if call.get("status") == "error":
            return str(call.get("content") or f"tool {call.get('name')} failed")
    plan = _dict(metadata.get("plan_execution_trace"))
    for record in _list(plan.get("node_records")):
        if isinstance(record, dict) and record.get("status") == "failed":
            return str(record.get("output_summary") or "workflow node failed")
    if status == "failure":
        return "run failed"
    return ""


def _tool_call_summaries(tool_trace: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for raw_call in _list(tool_trace.get("calls")):
        if not isinstance(raw_call, dict):
            continue
        call = _dict(raw_call.get("tool_call"))
        result = _dict(raw_call.get("result"))
        name = call.get("name")
        summaries.append(
            {
                "call_id": call.get("call_id") or result.get("call_id"),
                "name": name if isinstance(name, str) else None,
                "status": result.get("status"),
                "content": _bounded(str(result.get("content") or ""), 240),
                "error_type": _dict(result.get("metadata")).get("error_type"),
                "artifact_refs": _list(result.get("artifact_refs")),
            }
        )
    return summaries


def _artifact_refs(observation: dict[str, Any], metadata: dict[str, Any], tool_trace: dict[str, Any]) -> list[Any]:
    refs: list[Any] = []
    refs.extend(_list(metadata.get("artifact_refs")))
    for call in _list(tool_trace.get("calls")):
        if isinstance(call, dict):
            refs.extend(_list(_dict(call.get("result")).get("artifact_refs")))
    return refs[:20]


def _coverage_report(skill_bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = _dict(skill_bundle.get("metadata"))
    graph_summary = _dict(metadata.get("graph_context_summary"))
    retrieval_trace = _dict(metadata.get("retrieval_trace"))
    coverage = graph_summary.get("coverage_report") or retrieval_trace.get("coverage_report")
    return coverage if isinstance(coverage, dict) else {}


def _missing_capability(observation: dict[str, Any], skill_bundle: dict[str, Any]) -> str | None:
    metadata = _dict(skill_bundle.get("metadata"))
    report = _dict(metadata.get("missing_skill_report"))
    capability = report.get("missing_capability")
    if isinstance(capability, str) and capability:
        return capability
    retrieval_request = _dict(observation.get("retrieval_request"))
    query = retrieval_request.get("query")
    return query if isinstance(query, str) else None


def _missing_required_tools(observation: dict[str, Any]) -> list[str]:
    metadata = _dict(observation.get("metadata"))
    candidates: list[Any] = [
        metadata.get("missing_required_tools"),
        metadata.get("missing_tools"),
        _dict(metadata.get("error")).get("missing_required_tools"),
        _dict(metadata.get("error")).get("missing_tools"),
    ]
    if metadata.get("exception_type") == "MissingRequiredToolError":
        candidates.append(metadata.get("missing_tools"))
    tool_trace = _dict(observation.get("tool_trace"))
    for raw_call in _list(tool_trace.get("calls")):
        if not isinstance(raw_call, dict):
            continue
        result = _dict(raw_call.get("result"))
        result_metadata = _dict(result.get("metadata"))
        if result_metadata.get("error_type") in {"missing_required_tool", "missing_required_tools"}:
            candidates.append(result_metadata.get("missing_tools"))
        if result_metadata.get("error_type") == "unprepared_tool":
            tool_name = _dict(raw_call.get("tool_call")).get("name")
            if isinstance(tool_name, str):
                candidates.append([tool_name])
    tools: list[str] = []
    for value in candidates:
        tools.extend(_list_of_strings(value))
    return _dedupe(tools)


def _needs_candidate_skill(summary: dict[str, Any]) -> bool:
    if not summary.get("selected_skill_ids"):
        return True
    coverage = summary.get("coverage_report")
    if isinstance(coverage, dict) and coverage.get("sufficient") is False:
        return True
    if summary.get("missing_skill_report"):
        return True
    return False


def _example_summary(summary: dict[str, Any]) -> str:
    parts = [
        f"query={summary.get('query') or ''}",
        f"status={summary.get('status')}",
        f"skills={','.join(summary.get('selected_skill_ids') or [])}",
        f"tools={','.join(summary.get('tool_names') or [])}",
        f"artifacts={len(summary.get('artifact_refs') or [])}",
    ]
    if summary.get("failure_reason"):
        parts.append(f"failure={summary['failure_reason']}")
    return "; ".join(parts)


def _candidate_name(missing_capability: str) -> str:
    tokens = [token for token in missing_capability.replace("_", " ").split() if token]
    words = tokens[:8] or ["Uncovered", "Capability"]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _is_bounded_payload(proposal: SkillUpdateProposal, policy: SkillEvolutionPolicy) -> bool:
    if proposal.proposal_type == "failure_note_update":
        reason = proposal.payload.get("failure_reason")
        return isinstance(reason, str) and len(reason) <= policy.max_failure_reason_chars
    return True


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _list_of_strings(value: Any) -> list[str]:
    return [item for item in _list(value) if isinstance(item, str) and item]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
