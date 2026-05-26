from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from evolab.backends.skills.trace2skill.conflicts import ConflictChecker
from evolab.backends.skills.trace2skill.patches import merge_patch_content
from evolab.backends.skills.trace2skill.schema import (
    ConsolidatedSkillPatch,
    PatchConsolidationResult,
    PatchConflict,
    SkillPatchProposal,
)


class HierarchicalPatchConsolidator:
    def __init__(
        self,
        *,
        min_support_count: int = 1,
        min_confidence: float = 0.25,
        max_patches_per_skill: int = 20,
        max_auto_risk_level: str = "medium",
        conflict_checker: ConflictChecker | None = None,
    ):
        self.min_support_count = min_support_count
        self.min_confidence = min_confidence
        self.max_patches_per_skill = max_patches_per_skill
        self.max_auto_risk_level = max_auto_risk_level
        self.conflict_checker = conflict_checker or ConflictChecker()

    def consolidate(self, patches: Iterable[SkillPatchProposal]) -> PatchConsolidationResult:
        patch_list = sorted(patches, key=lambda patch: patch.patch_id)
        grouped = self.group_patches(patch_list)
        consolidated: list[ConsolidatedSkillPatch] = []
        rejected: list[SkillPatchProposal] = []
        deferred: list[SkillPatchProposal] = []
        conflicts: list[PatchConflict] = []

        for group_key in sorted(grouped):
            group = grouped[group_key]
            low_support, low_confidence = self.reject_low_support_patches(group)
            rejected.extend(low_confidence)
            deferred.extend(low_support)
            eligible = [patch for patch in group if patch not in low_support and patch not in low_confidence]
            if not eligible:
                continue
            merged = self.merge_group(eligible)
            group_conflicts = self.conflict_checker.resolve_conflicts(self.conflict_checker.check_patch_conflicts([merged]))
            conflicts.extend(group_conflicts)
            if any(conflict.resolution == "reject" for conflict in group_conflicts):
                rejected.extend(eligible)
                continue
            if any(conflict.resolution == "defer" for conflict in group_conflicts):
                deferred.extend(eligible)
                continue
            consolidated.append(merged.model_copy(update={"conflicts_resolved": group_conflicts}))

        consolidated = self._limit_patches_per_skill(consolidated)
        return self.produce_consolidation_result(
            consolidated_patches=consolidated,
            rejected_patches=rejected,
            deferred_patches=deferred,
            conflicts=conflicts,
            original_count=len(patch_list),
        )

    def group_patches(self, patches: Iterable[SkillPatchProposal]) -> dict[tuple[str, str], list[SkillPatchProposal]]:
        groups: dict[tuple[str, str], list[SkillPatchProposal]] = defaultdict(list)
        for patch in patches:
            subject = patch.target_skill_id or patch.candidate_skill_id or str(
                patch.patch_content.get("missing_capability_description")
                or patch.patch_content.get("proposed_name")
                or "global"
            )
            groups[(patch.patch_type, subject)].append(patch)
        return dict(groups)

    def merge_group(self, patches: Iterable[SkillPatchProposal]) -> ConsolidatedSkillPatch:
        group = sorted(patches, key=lambda patch: patch.patch_id)
        first = group[0]
        support = self.summarize_support(group)
        confidence = max(patch.confidence for patch in group)
        risk = _max_risk(patch.risk_level for patch in group)
        content = merge_patch_content(group)
        return ConsolidatedSkillPatch(
            consolidated_patch_id=_stable_id("consolidated-patch", [patch.patch_id for patch in group]),
            patch_type=first.patch_type,
            target_skill_id=first.target_skill_id,
            candidate_skill_id=first.candidate_skill_id or content.get("candidate_id"),
            merged_content=content,
            source_patch_ids=[patch.patch_id for patch in group],
            source_lesson_ids=_dedupe(lesson_id for patch in group for lesson_id in patch.source_lesson_ids),
            source_trace_ids=_dedupe(trace_id for patch in group for trace_id in patch.source_trace_ids),
            support_count=support["support_count"],
            confidence=confidence,
            validation_hints=support["validation_hints"],
            risk_level=risk,
            created_at=_utc_now(),
        )

    def hierarchical_merge(self, patches: Iterable[SkillPatchProposal]) -> list[ConsolidatedSkillPatch]:
        return self.consolidate(patches).consolidated_patches

    def summarize_support(self, patches: Iterable[SkillPatchProposal]) -> dict[str, Any]:
        group = list(patches)
        traces = _dedupe(trace_id for patch in group for trace_id in patch.source_trace_ids)
        lessons = _dedupe(lesson_id for patch in group for lesson_id in patch.source_lesson_ids)
        return {
            "support_count": max(sum(patch.support_count for patch in group), len(traces), 1),
            "trace_count": len(traces),
            "lesson_count": len(lessons),
            "validation_hints": [
                f"supported_by_traces={len(traces)}",
                f"supported_by_lessons={len(lessons)}",
            ],
        }

    def reject_low_support_patches(
        self,
        patches: Iterable[SkillPatchProposal],
    ) -> tuple[list[SkillPatchProposal], list[SkillPatchProposal]]:
        low_support = []
        low_confidence = []
        for patch in patches:
            if patch.confidence < self.min_confidence:
                low_confidence.append(patch)
            elif patch.support_count < self.min_support_count and patch.risk_level != "low":
                low_support.append(patch)
        return low_support, low_confidence

    def produce_consolidation_result(
        self,
        *,
        consolidated_patches: list[ConsolidatedSkillPatch],
        rejected_patches: list[SkillPatchProposal],
        deferred_patches: list[SkillPatchProposal],
        conflicts: list[PatchConflict],
        original_count: int,
    ) -> PatchConsolidationResult:
        return PatchConsolidationResult(
            result_id=_stable_id(
                "consolidation-result",
                [
                    [patch.consolidated_patch_id for patch in consolidated_patches],
                    [patch.patch_id for patch in rejected_patches],
                    [patch.patch_id for patch in deferred_patches],
                ],
            ),
            consolidated_patches=consolidated_patches,
            rejected_patches=rejected_patches,
            deferred_patches=deferred_patches,
            conflicts=conflicts,
            stats={
                "input_patch_count": original_count,
                "consolidated_patch_count": len(consolidated_patches),
                "rejected_patch_count": len(rejected_patches),
                "deferred_patch_count": len(deferred_patches),
                "conflict_count": len(conflicts),
            },
        )

    def _limit_patches_per_skill(self, patches: list[ConsolidatedSkillPatch]) -> list[ConsolidatedSkillPatch]:
        counts: dict[str, int] = defaultdict(int)
        result = []
        for patch in sorted(patches, key=lambda item: (-item.confidence, item.consolidated_patch_id)):
            skill_key = patch.target_skill_id or patch.candidate_skill_id or "global"
            if counts[skill_key] >= self.max_patches_per_skill:
                continue
            counts[skill_key] += 1
            result.append(patch)
        return sorted(result, key=lambda item: item.consolidated_patch_id)


def _max_risk(values: Iterable[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    max_value = max(values, key=lambda value: order.get(value, 1))
    return max_value


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
