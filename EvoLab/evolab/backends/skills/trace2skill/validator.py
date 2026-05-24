from __future__ import annotations

from typing import Any

from evolab.backends.skills.trace2skill.conflicts import ConflictChecker, SUPPORTED_RELATIONS
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch, PatchValidationResult
from evolab.contracts.retrieval import RetrievalRequest


class SkillPatchValidator:
    def __init__(self, *, graph_backend: Any | None = None, tool_registry: Any | None = None):
        self.graph_backend = graph_backend
        self.tool_registry = tool_registry

    def validate_consolidated_patch(self, patch: ConsolidatedSkillPatch) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if patch.patch_type == "required_tools_patch":
            errors.extend(self.validate_required_tools_patch(patch))
        elif patch.patch_type == "skill_create_patch":
            errors.extend(self.validate_candidate_skill(patch))
        elif patch.patch_type == "relationship_patch":
            errors.extend(self.validate_relationship_patch(patch))
        elif patch.patch_type == "example_memory_patch":
            errors.extend(self.validate_example_patch(patch))
        elif patch.patch_type in {
            "procedure_step_patch",
            "precondition_patch",
            "failure_case_patch",
            "validation_rule_patch",
            "metadata_patch",
            "skill_deepen_patch",
        }:
            if not patch.target_skill_id:
                errors.append(f"{patch.patch_type} requires target_skill_id")
        else:
            errors.append(f"unsupported patch type {patch.patch_type}")
        return not errors, errors

    def validate_patch_bundle(self, patches: list[ConsolidatedSkillPatch]) -> PatchValidationResult:
        valid: list[ConsolidatedSkillPatch] = []
        invalid: list[ConsolidatedSkillPatch] = []
        warnings: list[str] = []
        skill_ids = self._skill_ids()
        edges = self._edges()
        checker = ConflictChecker(tool_registry=self.tool_registry, skill_ids=skill_ids)
        conflicts = checker.resolve_conflicts(
            [
                *checker.check_patch_conflicts(patches),
                *checker.check_against_skill_library(patches, existing_skill_ids=skill_ids, existing_edges=edges),
            ]
        )
        conflict_patch_ids = {
            patch_id
            for conflict in conflicts
            if conflict.resolution == "reject"
            for patch_id in conflict.patch_ids
        }
        for patch in patches:
            ok, errors = self.validate_consolidated_patch(patch)
            if patch.consolidated_patch_id in conflict_patch_ids:
                ok = False
                errors.append("patch has rejecting conflict")
            if ok:
                valid.append(patch)
            else:
                invalid.append(patch)
                warnings.extend(f"{patch.consolidated_patch_id}: {error}" for error in errors)
        return PatchValidationResult(
            valid_patches=valid,
            invalid_patches=invalid,
            conflicts=conflicts,
            warnings=warnings,
            stats={
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "conflict_count": len(conflicts),
            },
        )

    def validate_candidate_skill(self, patch: ConsolidatedSkillPatch) -> list[str]:
        content = patch.merged_content
        errors = []
        candidate_id = patch.candidate_skill_id or content.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append("candidate skill patch requires candidate_id")
        if not content.get("proposed_name"):
            errors.append("candidate skill patch requires proposed_name")
        if not content.get("missing_capability_description"):
            errors.append("candidate skill patch requires missing_capability_description")
        if candidate_id in self._skill_ids():
            errors.append(f"candidate skill id already exists: {candidate_id}")
        return errors

    def validate_required_tools_patch(self, patch: ConsolidatedSkillPatch) -> list[str]:
        tools = _strings(patch.merged_content.get("required_tools"))
        errors = []
        if not patch.target_skill_id:
            errors.append("required tools patch requires target_skill_id")
        if not tools:
            errors.append("required tools patch requires at least one tool")
        if self.tool_registry is not None:
            missing = [tool for tool in tools if self.tool_registry.get_spec(tool) is None]
            if missing:
                errors.append(f"unregistered tools: {', '.join(missing)}")
        return errors

    def validate_relationship_patch(self, patch: ConsolidatedSkillPatch) -> list[str]:
        errors = []
        skill_ids = self._skill_ids()
        relations = _relations(patch.merged_content)
        if not relations:
            errors.append("relationship patch requires relation payload")
        for relation in relations:
            relation_type = relation.get("relation")
            source_id = relation.get("source_skill_id") or relation.get("source_id")
            target_id = relation.get("target_skill_id") or relation.get("target_id")
            if relation_type not in SUPPORTED_RELATIONS:
                errors.append(f"unsupported relation type: {relation_type}")
            for skill_id in (source_id, target_id):
                if isinstance(skill_id, str) and skill_ids and skill_id not in skill_ids:
                    errors.append(f"relation references missing skill id: {skill_id}")
        return errors

    def validate_example_patch(self, patch: ConsolidatedSkillPatch) -> list[str]:
        example = patch.merged_content.get("example_summary")
        if not patch.target_skill_id:
            return ["example memory patch requires target_skill_id"]
        if isinstance(example, str) and len(example) <= 5000:
            return []
        return ["example memory patch requires bounded example_summary"]

    def run_retrieval_sanity_check(self, *, query: str = "generic skill retrieval sanity check") -> dict[str, Any]:
        if self.graph_backend is None:
            return {"status": "skipped", "reason": "no graph backend configured"}
        try:
            bundle = self.graph_backend.get(RetrievalRequest(task_id="trace2skill-sanity", role="validator", query=query))
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {"status": "failed", "error": str(exc)}
        return {
            "status": "ok",
            "returned_skill_count": len(bundle.skills),
            "graph_version_ref": bundle.graph_version_ref,
        }

    def run_regression_gate(self, before_metrics: dict[str, Any] | None = None, after_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        from evolab.backends.skills.trace2skill.regression import SkillEvolutionRegressionGate

        return SkillEvolutionRegressionGate(graph_backend=self.graph_backend).evaluate(
            [],
            before_snapshot_ref="before",
            after_snapshot_ref="after",
        ).model_dump(mode="json")

    def _skill_ids(self) -> set[str]:
        if self.graph_backend is None:
            return set()
        try:
            loaded = self.graph_backend.store.load_graph()
            return {skill.skill_id for skill in loaded.graph.skills}
        except Exception:
            return set()

    def _edges(self) -> list[dict[str, Any]]:
        if self.graph_backend is None:
            return []
        try:
            raw = self.graph_backend._load_raw_graph()
            return [edge for edge in raw.get("edges", []) if isinstance(edge, dict)]
        except Exception:
            return []


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _relations(content: dict[str, Any]) -> list[dict[str, Any]]:
    relation = content.get("relation")
    relation_updates = content.get("relations") or content.get("relation_updates")
    values = []
    if isinstance(relation, dict):
        values.append(relation)
    if isinstance(relation_updates, list):
        values.extend(item for item in relation_updates if isinstance(item, dict))
    return values
