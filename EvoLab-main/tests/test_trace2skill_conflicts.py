from evolab.backends.skills.trace2skill.conflicts import ConflictChecker
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry


def _patch(**overrides) -> ConsolidatedSkillPatch:
    data = {
        "consolidated_patch_id": "patch-1",
        "patch_type": "required_tools_patch",
        "target_skill_id": "skill.generic.v1",
        "merged_content": {"required_tools": ["read_text"]},
        "source_patch_ids": ["local-1"],
        "source_lesson_ids": ["lesson-1"],
        "source_trace_ids": ["trace-1"],
        "confidence": 0.8,
        "risk_level": "medium",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return ConsolidatedSkillPatch(**data)


def test_missing_tool_is_deferred_against_tool_registry():
    registry = ToolRegistry()
    conflicts = ConflictChecker(tool_registry=registry).check_against_skill_library([_patch()])

    assert conflicts[0].conflict_type == "missing_tool"
    assert conflicts[0].resolution == "defer"


def test_registered_tool_passes_without_conflict():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="read", parameters_schema={}), lambda args: "ok")

    assert ConflictChecker(tool_registry=registry).check_against_skill_library([_patch()]) == []


def test_candidate_skill_id_collision_is_rejected():
    conflict = ConflictChecker(skill_ids=["skill.new.v1"]).check_against_skill_library(
        [
            _patch(
                patch_type="skill_create_patch",
                candidate_skill_id="skill.new.v1",
                target_skill_id=None,
                merged_content={"candidate_id": "skill.new.v1", "proposed_name": "New"},
                risk_level="high",
            )
        ],
        existing_skill_ids=["skill.new.v1"],
    )[0]

    assert conflict.conflict_type == "candidate_skill_id_collision"
    assert conflict.resolution == "reject"


def test_invalid_relation_and_duplicate_relation_are_reported():
    patch = _patch(
        patch_type="relationship_patch",
        merged_content={
            "relations": [
                {
                    "source_skill_id": "skill.a.v1",
                    "target_skill_id": "skill.b.v1",
                    "relation": "invalid_relation",
                }
            ]
        },
    )

    invalid = ConflictChecker(skill_ids=["skill.a.v1", "skill.b.v1"]).check_against_skill_library([patch])
    duplicate = ConflictChecker(skill_ids=["skill.a.v1", "skill.b.v1"]).check_against_skill_library(
        [
            patch.model_copy(
                update={
                    "merged_content": {
                        "relations": [
                            {
                                "source_skill_id": "skill.a.v1",
                                "target_skill_id": "skill.b.v1",
                                "relation": "related_to",
                            }
                        ]
                    }
                }
            )
        ],
        existing_edges=[{"source_id": "skill.a.v1", "target_id": "skill.b.v1", "relation": "related_to"}],
    )

    assert invalid[0].conflict_type == "unsupported_relation"
    assert duplicate[0].conflict_type == "duplicate_relation"


def test_duplicate_relation_patch_conflict_is_mergeable():
    patch_a = _patch(
        consolidated_patch_id="patch-a",
        patch_type="relationship_patch",
        merged_content={
            "relations": [{"source_skill_id": "skill.a.v1", "target_skill_id": "skill.b.v1", "relation": "related_to"}]
        },
    )
    patch_b = patch_a.model_copy(update={"consolidated_patch_id": "patch-b"})

    conflicts = ConflictChecker().check_patch_conflicts([patch_a, patch_b])

    assert conflicts[0].conflict_type == "duplicate_relation_patch"
    assert conflicts[0].resolution == "merge"
