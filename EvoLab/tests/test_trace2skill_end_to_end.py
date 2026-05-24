import json
from pathlib import Path

from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.trace2skill import Trace2SkillEvolver, Trace2SkillRunConfig
from evolab.contracts.retrieval import RetrievalRequest, SkillObservationRequest
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry


def test_trace2skill_end_to_end_dry_run_writes_audit_reports(tmp_path: Path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": [_candidate_skill()],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state")
    skill_bundle = backend.get(RetrievalRequest(task_id="task-1", role="solver", query="generic extraction"))
    observation = SkillObservationRequest(
        task_id="task-1",
        run_ref="run-1",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="generic extraction"),
        skill_bundle=skill_bundle,
        final_answer="completed with artifact",
        metadata={"status": "success", "artifact_refs": [{"uri": "artifact://report", "type": "text"}]},
    )
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="read", parameters_schema={}), lambda args: "ok")

    result = Trace2SkillEvolver(graph_backend=backend, tool_registry=registry).run(
        Trace2SkillRunConfig(mode="combined", dry_run=True, output_dir=str(tmp_path / "trace2skill")),
        observations=[observation],
    )

    assert result.status == "dry_run"
    assert result.lessons
    assert result.local_patches
    assert Path(result.report_paths["trace2skill_run_summary"]).exists()
    assert Path(result.report_paths["lessons"]).read_text(encoding="utf-8").strip()
    assert "Trace2Skill Audit Report" in Path(result.report_paths["trace2skill_audit_report"]).read_text(
        encoding="utf-8"
    )
    assert not any("promoter" in skill.skill_id for skill in skill_bundle.skills)


def _candidate_skill():
    return {
        "skill_id": "skill.generic_extraction.v1",
        "name": "Generic Extraction",
        "description": "Extract generic records from evidence.",
        "source_type": "human",
        "source_uri": "seed://test",
        "scope": "generic extraction",
        "applicability": ["generic extraction tasks"],
        "limitations": [],
        "required_inputs": [],
        "expected_outputs": [],
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": [],
        "required_tools": ["read_text"],
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
