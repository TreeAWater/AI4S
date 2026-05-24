from evolab.backends.skills.trace2skill.adapter import Trace2SkillSkillBackendAdapter
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch


def _patch(patch_type: str, **overrides) -> ConsolidatedSkillPatch:
    data = {
        "consolidated_patch_id": f"consolidated-{patch_type}",
        "patch_type": patch_type,
        "target_skill_id": "skill.generic.v1",
        "merged_content": {},
        "source_patch_ids": ["patch-1"],
        "source_lesson_ids": ["lesson-1"],
        "source_trace_ids": ["trace-1"],
        "confidence": 0.8,
        "risk_level": "medium",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return ConsolidatedSkillPatch(**data)


def test_required_tools_patch_converts_to_required_tools_update_proposal():
    proposal = Trace2SkillSkillBackendAdapter().to_skill_update_proposals(
        [_patch("required_tools_patch", merged_content={"required_tools": ["inspect_table"]})]
    )[0]

    assert proposal.proposal_type == "required_tools_update_proposal"
    assert proposal.payload["required_tools"] == ["inspect_table"]


def test_example_memory_patch_converts_to_example_trace_memory_update():
    proposal = Trace2SkillSkillBackendAdapter().to_skill_update_proposals(
        [_patch("example_memory_patch", merged_content={"example_summary": "worked"})]
    )[0]

    assert proposal.proposal_type == "example_trace_memory_update"
    assert proposal.payload["example_summary"] == "worked"


def test_skill_create_patch_converts_to_candidate_skill_creation():
    proposal = Trace2SkillSkillBackendAdapter().to_skill_update_proposals(
        [
            _patch(
                "skill_create_patch",
                target_skill_id=None,
                candidate_skill_id="candidate-1",
                merged_content={
                    "candidate_id": "candidate-1",
                    "proposed_name": "Uncovered Capability",
                    "missing_capability_description": "uncovered capability",
                },
                risk_level="high",
            )
        ]
    )[0]

    assert proposal.proposal_type == "candidate_skill_creation"
    assert proposal.payload["candidate_record"]["candidate_id"] == "candidate-1"


def test_relationship_and_invalid_patch_conversion_are_safe():
    adapter = Trace2SkillSkillBackendAdapter()
    relationship = adapter.to_skill_update_proposals(
        [_patch("relationship_patch", merged_content={"relations": [{"relation": "related_to"}]})]
    )[0]
    unsupported = _patch("metadata_patch", target_skill_id=None, merged_content={"note": "no target"})
    metadata = adapter.to_skill_update_proposals([unsupported])[0]

    assert relationship.proposal_type == "relationship_update_proposal"
    assert adapter.validate_conversion(metadata) is False
