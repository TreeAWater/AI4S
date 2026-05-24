from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from evolab.backends.skills.evolution import CandidateSkillRecord, SkillUpdateProposal
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch


class Trace2SkillSkillBackendAdapter:
    def __init__(self, *, backend_id: str = "graph_skill", graph_version_ref: str | None = None):
        self.backend_id = backend_id
        self.graph_version_ref = graph_version_ref

    def to_skill_update_proposals(self, patches: Iterable[ConsolidatedSkillPatch]) -> list[SkillUpdateProposal]:
        proposals = [self._convert(patch) for patch in patches]
        return [proposal for proposal in proposals if proposal is not None]

    def to_required_tools_update(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal:
        tools = _strings(patch.merged_content.get("required_tools"))
        return self._proposal(
            patch,
            proposal_type="required_tools_update_proposal",
            summary=f"Stage required tool update for {patch.target_skill_id}.",
            payload={"missing_required_tools": tools, "required_tools": tools},
            related_skill_ids=[patch.target_skill_id] if patch.target_skill_id else [],
        )

    def to_example_memory_update(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal:
        return self._proposal(
            patch,
            proposal_type="example_trace_memory_update",
            summary=f"Stage example trace memory for {patch.target_skill_id}.",
            payload={"example_summary": str(patch.merged_content.get("example_summary") or patch.merged_content)},
            related_skill_ids=[patch.target_skill_id] if patch.target_skill_id else [],
        )

    def to_candidate_skill_creation(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal:
        content = patch.merged_content
        candidate = CandidateSkillRecord(
            candidate_id=str(patch.candidate_skill_id or content.get("candidate_id")),
            proposed_name=str(content.get("proposed_name") or "Uncovered Capability"),
            proposed_task=_optional_str(content.get("proposed_task")),
            proposed_category=_optional_str(content.get("proposed_category")),
            source_observation_id=_observation_id_for_patch(patch),
            evidence_summary=patch.validation_hints[0] if patch.validation_hints else "Trace2Skill staged candidate.",
            missing_capability_description=str(content.get("missing_capability_description") or "uncovered capability"),
            suggested_required_inputs=_strings(content.get("suggested_required_inputs")),
            suggested_expected_outputs=_strings(content.get("suggested_expected_outputs")),
            suggested_required_tools=_strings(content.get("suggested_required_tools")),
            created_at=_utc_now(),
            metadata={
                "source_patch_ids": patch.source_patch_ids,
                "source_lesson_ids": patch.source_lesson_ids,
                "source_trace_ids": patch.source_trace_ids,
            },
        )
        return self._proposal(
            patch,
            proposal_type="candidate_skill_creation",
            summary="Stage candidate skill discovered from trace coverage gap.",
            payload={"candidate_record": candidate.model_dump(mode="json")},
            related_skill_ids=[],
        )

    def to_relationship_update(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal:
        return self._proposal(
            patch,
            proposal_type="relationship_update_proposal",
            summary="Stage skill relationship update discovered from traces.",
            payload={"relationship_update": patch.merged_content},
            related_skill_ids=[skill_id for skill_id in [patch.target_skill_id] if skill_id],
        )

    def to_metadata_update(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal:
        return self._proposal(
            patch,
            proposal_type="metadata_update",
            summary=f"Stage Trace2Skill metadata patch for {patch.target_skill_id}.",
            payload={"trace2skill_patch": patch.model_dump(mode="json")},
            related_skill_ids=[patch.target_skill_id] if patch.target_skill_id else [],
        )

    def validate_conversion(self, proposal: SkillUpdateProposal) -> bool:
        if not proposal.proposal_id or not proposal.proposal_type:
            return False
        if proposal.proposal_type != "candidate_skill_creation" and not proposal.related_skill_ids:
            return False
        return True

    def _convert(self, patch: ConsolidatedSkillPatch) -> SkillUpdateProposal | None:
        if patch.patch_type == "required_tools_patch":
            return self.to_required_tools_update(patch)
        if patch.patch_type == "example_memory_patch":
            return self.to_example_memory_update(patch)
        if patch.patch_type == "skill_create_patch":
            return self.to_candidate_skill_creation(patch)
        if patch.patch_type == "relationship_patch":
            return self.to_relationship_update(patch)
        if patch.patch_type == "failure_case_patch":
            return self._proposal(
                patch,
                proposal_type="failure_note_update",
                summary=f"Append bounded failure note for {patch.target_skill_id}.",
                payload={"failure_reason": str(patch.merged_content.get("failure_reason") or patch.merged_content)[:240]},
                related_skill_ids=[patch.target_skill_id] if patch.target_skill_id else [],
            )
        if patch.patch_type in {
            "metadata_patch",
            "procedure_step_patch",
            "precondition_patch",
            "validation_rule_patch",
            "skill_deepen_patch",
        }:
            return self.to_metadata_update(patch)
        return None

    def _proposal(
        self,
        patch: ConsolidatedSkillPatch,
        *,
        proposal_type: str,
        summary: str,
        payload: dict[str, Any],
        related_skill_ids: list[str],
    ) -> SkillUpdateProposal:
        return SkillUpdateProposal(
            proposal_id=_stable_id(
                "trace2skill-skill-proposal",
                [patch.consolidated_patch_id, proposal_type, related_skill_ids, payload],
            ),
            proposal_type=proposal_type,  # type: ignore[arg-type]
            observation_id=_observation_id_for_patch(patch),
            backend_id=self.backend_id,
            graph_version_ref=self.graph_version_ref,
            related_skill_ids=related_skill_ids,
            summary=summary,
            payload=payload,
            created_at=_utc_now(),
            metadata={
                "trace2skill_consolidated_patch_id": patch.consolidated_patch_id,
                "source_patch_ids": patch.source_patch_ids,
                "source_lesson_ids": patch.source_lesson_ids,
                "source_trace_ids": patch.source_trace_ids,
                "risk_level": patch.risk_level,
                "support_count": patch.support_count,
                "confidence": patch.confidence,
            },
        )


def _observation_id_for_patch(patch: ConsolidatedSkillPatch) -> str:
    if patch.source_trace_ids:
        return _stable_id("trace2skill-observation", patch.source_trace_ids)
    return _stable_id("trace2skill-observation", [patch.consolidated_patch_id])


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
