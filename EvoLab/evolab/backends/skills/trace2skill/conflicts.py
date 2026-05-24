from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json
from typing import Any

from evolab.backends.skills.trace2skill.schema import (
    ConsolidatedSkillPatch,
    PatchConflict,
    SkillPatchProposal,
)


SUPPORTED_RELATIONS = {
    "depends_on",
    "requires",
    "consumes",
    "prerequisite",
    "related_to",
    "complements",
    "validates",
    "produces",
    "alternative_to",
    "specializes",
    "conflicts_with",
    "replaces",
    "deprecated_by",
}


class ConflictChecker:
    def __init__(self, *, tool_registry: Any | None = None, skill_ids: Iterable[str] | None = None):
        self.tool_registry = tool_registry
        self.skill_ids = set(skill_ids or [])

    def check_patch_conflicts(
        self,
        patches: Iterable[SkillPatchProposal | ConsolidatedSkillPatch],
    ) -> list[PatchConflict]:
        patch_list = list(patches)
        conflicts: list[PatchConflict] = []
        conflicts.extend(self._candidate_collisions(patch_list))
        conflicts.extend(self._conflicting_required_tool_sets(patch_list))
        conflicts.extend(self._duplicate_relations(patch_list))
        return conflicts

    def check_against_skill_library(
        self,
        patches: Iterable[SkillPatchProposal | ConsolidatedSkillPatch],
        *,
        existing_skill_ids: Iterable[str] | None = None,
        existing_edges: Iterable[dict[str, Any]] | None = None,
    ) -> list[PatchConflict]:
        skill_ids = set(existing_skill_ids or self.skill_ids)
        existing_edge_keys = {
            (edge.get("source_id"), edge.get("target_id"), edge.get("relation"))
            for edge in existing_edges or []
            if isinstance(edge, dict)
        }
        conflicts: list[PatchConflict] = []
        for patch in patches:
            content = _content(patch)
            patch_id = _patch_id(patch)
            if _patch_type(patch) == "required_tools_patch":
                for tool in _strings(content.get("required_tools")):
                    if self.tool_registry is not None and self.tool_registry.get_spec(tool) is None:
                        conflicts.append(
                            _conflict(
                                "missing_tool",
                                [patch_id],
                                [_target_skill_id(patch)],
                                f"Required tool {tool!r} is not registered.",
                                "defer",
                                "medium",
                            )
                        )
            if _patch_type(patch) == "skill_create_patch":
                candidate_id = _candidate_skill_id(patch)
                if candidate_id and candidate_id in skill_ids:
                    conflicts.append(
                        _conflict(
                            "candidate_skill_id_collision",
                            [patch_id],
                            [candidate_id],
                            f"Candidate skill id {candidate_id!r} already exists.",
                            "reject",
                            "high",
                        )
                    )
            if _patch_type(patch) == "relationship_patch":
                for relation in _relations(content):
                    relation_type = relation.get("relation")
                    source_id = relation.get("source_skill_id") or relation.get("source_id")
                    target_id = relation.get("target_skill_id") or relation.get("target_id")
                    if relation_type not in SUPPORTED_RELATIONS:
                        conflicts.append(
                            _conflict(
                                "unsupported_relation",
                                [patch_id],
                                _strings([source_id, target_id]),
                                f"Unsupported relation type {relation_type!r}.",
                                "reject",
                                "high",
                            )
                        )
                    missing = [skill_id for skill_id in (source_id, target_id) if skill_id and skill_id not in skill_ids]
                    if missing:
                        conflicts.append(
                            _conflict(
                                "relation_missing_node",
                                [patch_id],
                                _strings([source_id, target_id]),
                                f"Relation references missing skill ids: {', '.join(missing)}.",
                                "reject",
                                "high",
                            )
                        )
                    if (source_id, target_id, relation_type) in existing_edge_keys:
                        conflicts.append(
                            _conflict(
                                "duplicate_relation",
                                [patch_id],
                                _strings([source_id, target_id]),
                                "Relationship already exists in the graph.",
                                "merge",
                                "low",
                            )
                        )
            if _patch_type(patch) == "example_memory_patch":
                example = content.get("example_summary")
                if isinstance(example, str) and len(example) > 5000:
                    conflicts.append(
                        _conflict(
                            "oversized_example_memory",
                            [patch_id],
                            [_target_skill_id(patch)],
                            "Example memory patch is too large for append-only storage.",
                            "defer",
                            "medium",
                        )
                    )
        return conflicts

    def resolve_conflicts(self, conflicts: Iterable[PatchConflict]) -> list[PatchConflict]:
        resolved = []
        for conflict in conflicts:
            if conflict.conflict_type in {"duplicate_relation"}:
                resolved.append(conflict.model_copy(update={"resolution": "merge"}))
            elif conflict.severity == "high":
                resolved.append(conflict.model_copy(update={"resolution": "reject"}))
            else:
                resolved.append(conflict.model_copy(update={"resolution": "defer"}))
        return resolved

    def is_conflict_free(self, conflicts: Iterable[PatchConflict]) -> bool:
        return not any(conflict.resolution in {"reject", "defer"} for conflict in conflicts)

    def explain_conflicts(self, conflicts: Iterable[PatchConflict]) -> list[str]:
        return [f"{conflict.conflict_type}: {conflict.description}" for conflict in conflicts]

    def _candidate_collisions(self, patches: list[SkillPatchProposal | ConsolidatedSkillPatch]) -> list[PatchConflict]:
        by_candidate: dict[str, list[str]] = {}
        for patch in patches:
            candidate_id = _candidate_skill_id(patch)
            if candidate_id:
                by_candidate.setdefault(candidate_id, []).append(_patch_id(patch))
        return [
            _conflict(
                "duplicate_candidate_patch",
                patch_ids,
                [candidate_id],
                f"Multiple patches propose candidate skill id {candidate_id!r}.",
                "merge",
                "low",
            )
            for candidate_id, patch_ids in by_candidate.items()
            if len(patch_ids) > 1
        ]

    def _conflicting_required_tool_sets(
        self,
        patches: list[SkillPatchProposal | ConsolidatedSkillPatch],
    ) -> list[PatchConflict]:
        conflicts = []
        for patch in patches:
            if _patch_type(patch) != "required_tools_patch":
                continue
            tools = _strings(_content(patch).get("required_tools"))
            if len(tools) != len(set(tools)):
                conflicts.append(
                    _conflict(
                        "duplicate_required_tool",
                        [_patch_id(patch)],
                        [_target_skill_id(patch)],
                        "Required tool patch includes duplicates.",
                        "merge",
                        "low",
                    )
                )
        return conflicts

    def _duplicate_relations(self, patches: list[SkillPatchProposal | ConsolidatedSkillPatch]) -> list[PatchConflict]:
        seen: dict[tuple[str, str, str], str] = {}
        conflicts = []
        for patch in patches:
            if _patch_type(patch) != "relationship_patch":
                continue
            for relation in _relations(_content(patch)):
                key = (
                    str(relation.get("source_skill_id") or relation.get("source_id") or ""),
                    str(relation.get("target_skill_id") or relation.get("target_id") or ""),
                    str(relation.get("relation") or ""),
                )
                if key in seen:
                    conflicts.append(
                        _conflict(
                            "duplicate_relation_patch",
                            [seen[key], _patch_id(patch)],
                            [key[0], key[1]],
                            "Multiple patches propose the same relationship.",
                            "merge",
                            "low",
                        )
                    )
                else:
                    seen[key] = _patch_id(patch)
        return conflicts


def _conflict(
    conflict_type: str,
    patch_ids: list[str],
    target_skill_ids: list[str],
    description: str,
    resolution: str,
    severity: str,
) -> PatchConflict:
    return PatchConflict(
        conflict_id=_stable_id("patch-conflict", [conflict_type, patch_ids, target_skill_ids, description]),
        conflict_type=conflict_type,
        patch_ids=[patch_id for patch_id in patch_ids if patch_id],
        target_skill_ids=[skill_id for skill_id in target_skill_ids if skill_id],
        description=description,
        resolution=resolution,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
    )


def _patch_id(patch: SkillPatchProposal | ConsolidatedSkillPatch) -> str:
    return getattr(patch, "patch_id", None) or getattr(patch, "consolidated_patch_id")


def _patch_type(patch: SkillPatchProposal | ConsolidatedSkillPatch) -> str:
    return str(getattr(patch, "patch_type"))


def _target_skill_id(patch: SkillPatchProposal | ConsolidatedSkillPatch) -> str:
    return str(getattr(patch, "target_skill_id") or "")


def _candidate_skill_id(patch: SkillPatchProposal | ConsolidatedSkillPatch) -> str | None:
    candidate = getattr(patch, "candidate_skill_id", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    content = _content(patch)
    candidate = content.get("candidate_id") or content.get("skill_id")
    return candidate if isinstance(candidate, str) and candidate else None


def _content(patch: SkillPatchProposal | ConsolidatedSkillPatch) -> dict[str, Any]:
    if hasattr(patch, "patch_content"):
        return getattr(patch, "patch_content")
    return getattr(patch, "merged_content")


def _relations(content: dict[str, Any]) -> list[dict[str, Any]]:
    relation = content.get("relation")
    relation_updates = content.get("relations") or content.get("relation_updates")
    values = []
    if isinstance(relation, dict):
        values.append(relation)
    if isinstance(relation_updates, list):
        values.extend(item for item in relation_updates if isinstance(item, dict))
    return values


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


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
