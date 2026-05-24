import json

from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch
from evolab.backends.skills.trace2skill.validator import SkillPatchValidator
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry


def _graph(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": [_candidate("skill.generic.v1"), _candidate("skill.other.v1")],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _candidate(skill_id: str):
    return {
        "skill_id": skill_id,
        "name": skill_id,
        "description": "Generic skill.",
        "source_type": "human",
        "source_uri": "seed://test",
        "scope": "generic",
        "applicability": [],
        "limitations": [],
        "required_inputs": [],
        "expected_outputs": [],
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": [],
        "required_tools": [],
        "scripts": [],
        "resources": [],
        "examples": [],
        "smoke_tests": [],
        "synthetic_tests": [],
        "system_tests": [],
        "benchmark_tests": [],
        "validation_signals": [],
        "confidence": 0.8,
        "metadata": {},
    }


def _patch(patch_type: str, **overrides) -> ConsolidatedSkillPatch:
    data = {
        "consolidated_patch_id": f"patch-{patch_type}",
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


def test_required_tools_patch_validation_checks_registry(tmp_path):
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="read", parameters_schema={}), lambda args: "ok")
    validator = SkillPatchValidator(
        graph_backend=GraphSkillBackend(_graph(tmp_path), evolution_root=tmp_path / "state"),
        tool_registry=registry,
    )

    valid, valid_errors = validator.validate_consolidated_patch(
        _patch("required_tools_patch", merged_content={"required_tools": ["read_text"]})
    )
    invalid, invalid_errors = validator.validate_consolidated_patch(
        _patch("required_tools_patch", merged_content={"required_tools": ["missing_tool"]})
    )

    assert valid is True
    assert valid_errors == []
    assert invalid is False
    assert "unregistered tools" in invalid_errors[0]


def test_candidate_and_relationship_validation(tmp_path):
    backend = GraphSkillBackend(_graph(tmp_path), evolution_root=tmp_path / "state")
    validator = SkillPatchValidator(graph_backend=backend)

    valid_candidate, _ = validator.validate_consolidated_patch(
        _patch(
            "skill_create_patch",
            target_skill_id=None,
            candidate_skill_id="skill.new.v1",
            merged_content={
                "candidate_id": "skill.new.v1",
                "proposed_name": "New Skill",
                "missing_capability_description": "new capability",
            },
            risk_level="high",
        )
    )
    invalid_candidate, candidate_errors = validator.validate_consolidated_patch(
        _patch("skill_create_patch", target_skill_id=None, candidate_skill_id="skill.generic.v1", risk_level="high")
    )
    valid_relation, _ = validator.validate_consolidated_patch(
        _patch(
            "relationship_patch",
            merged_content={
                "relations": [
                    {
                        "source_skill_id": "skill.generic.v1",
                        "target_skill_id": "skill.other.v1",
                        "relation": "related_to",
                    }
                ]
            },
        )
    )
    invalid_relation, relation_errors = validator.validate_consolidated_patch(
        _patch(
            "relationship_patch",
            merged_content={
                "relations": [
                    {
                        "source_skill_id": "skill.generic.v1",
                        "target_skill_id": "skill.missing.v1",
                        "relation": "related_to",
                    }
                ]
            },
        )
    )

    assert valid_candidate is True
    assert invalid_candidate is False
    assert "already exists" in candidate_errors[-1]
    assert valid_relation is True
    assert invalid_relation is False
    assert "missing skill id" in relation_errors[0]


def test_patch_bundle_validation_and_retrieval_sanity_check(tmp_path):
    backend = GraphSkillBackend(_graph(tmp_path), evolution_root=tmp_path / "state")
    validator = SkillPatchValidator(graph_backend=backend)
    result = validator.validate_patch_bundle(
        [_patch("example_memory_patch", merged_content={"example_summary": "worked"}, risk_level="low")]
    )
    sanity = validator.run_retrieval_sanity_check(query="generic")
    gate = validator.run_regression_gate()

    assert len(result.valid_patches) == 1
    assert sanity["status"] == "ok"
    assert gate["status"] == "skipped_no_benchmark"
