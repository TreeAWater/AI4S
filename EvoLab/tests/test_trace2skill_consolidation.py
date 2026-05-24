from evolab.backends.skills.trace2skill.consolidator import HierarchicalPatchConsolidator
from evolab.backends.skills.trace2skill.schema import SkillPatchProposal


def _patch(patch_id: str, **overrides) -> SkillPatchProposal:
    data = {
        "patch_id": patch_id,
        "patch_type": "example_memory_patch",
        "target_skill_id": "skill.generic.v1",
        "source_lesson_ids": [f"lesson-{patch_id}"],
        "source_trace_ids": [f"trace-{patch_id}"],
        "patch_content": {"example_summary": "worked"},
        "evidence_summary": "worked",
        "confidence": 0.8,
        "support_count": 1,
        "risk_level": "low",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return SkillPatchProposal(**data)


def test_consolidator_merges_similar_patches_and_keeps_evidence_trace_ids():
    result = HierarchicalPatchConsolidator().consolidate([_patch("a"), _patch("b")])

    assert len(result.consolidated_patches) == 1
    consolidated = result.consolidated_patches[0]
    assert consolidated.support_count == 2
    assert consolidated.source_trace_ids == ["trace-a", "trace-b"]
    assert result.stats["input_patch_count"] == 2


def test_consolidator_defers_low_support_risky_patches():
    result = HierarchicalPatchConsolidator(min_support_count=2).consolidate(
        [
            _patch(
                "tool",
                patch_type="required_tools_patch",
                patch_content={"required_tools": ["inspect_table"]},
                risk_level="medium",
            )
        ]
    )

    assert result.consolidated_patches == []
    assert [patch.patch_id for patch in result.deferred_patches] == ["tool"]


def test_consolidator_rejects_low_confidence_patches():
    result = HierarchicalPatchConsolidator(min_confidence=0.5).consolidate([_patch("low", confidence=0.2)])

    assert result.consolidated_patches == []
    assert [patch.patch_id for patch in result.rejected_patches] == ["low"]


def test_consolidator_detects_duplicate_required_tool_conflict_and_merges():
    result = HierarchicalPatchConsolidator().consolidate(
        [
            _patch(
                "dup",
                patch_type="required_tools_patch",
                patch_content={"required_tools": ["read_text", "read_text"]},
                risk_level="medium",
            )
        ]
    )

    assert result.consolidated_patches
    assert result.conflicts or result.consolidated_patches[0].merged_content["required_tools"] == ["read_text"]
