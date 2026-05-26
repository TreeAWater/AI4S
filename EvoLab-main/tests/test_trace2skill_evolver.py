import json

from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.evolution import SkillEvolutionPolicy
from evolab.backends.skills.trace2skill import Trace2SkillEvolver, Trace2SkillRunConfig
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest, SkillRef
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry


def _graph(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": [_candidate()],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _candidate():
    return {
        "skill_id": "skill.generic.v1",
        "name": "Generic Skill",
        "description": "Solve generic tasks.",
        "source_type": "human",
        "source_uri": "seed://test",
        "scope": "generic",
        "applicability": ["generic tasks"],
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


def _observation(status: str, *, skills: list[SkillRef] | None = None, missing_tools: list[str] | None = None):
    skills = [_skill_ref()] if skills is None else skills
    bundle = SkillBundle(
        backend_id="graph_skill",
        graph_version_ref="graph-v1",
        skills=skills,
        required_tools=sorted({tool for skill in skills for tool in skill.required_tools}),
        metadata={"graph_context_summary": {"coverage_report": {"sufficient": bool(skills)}}},
    )
    return SkillObservationRequest(
        task_id="task-1",
        run_ref=f"run-{status}-{len(skills)}",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="generic task"),
        skill_bundle=bundle,
        final_answer="done" if status == "success" else None,
        metadata={"status": status, "missing_tools": missing_tools or []},
    )


def _skill_ref(required_tools: list[str] | None = None) -> SkillRef:
    return SkillRef(
        skill_id="skill.generic.v1",
        name="Generic Skill",
        content="Generic skill.",
        required_tools=required_tools or ["read_text"],
    )


def _registry(*tool_names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return registry


def test_evolver_runs_error_success_and_combined_modes(tmp_path):
    backend = GraphSkillBackend(_graph(tmp_path), evolution_root=tmp_path / "state")
    registry = _registry("read_text", "inspect_table")

    for mode in ["error_only", "success_only", "combined", "skill_deepening"]:
        result = Trace2SkillEvolver(graph_backend=backend, tool_registry=registry).run(
            Trace2SkillRunConfig(mode=mode, dry_run=True, output_dir=str(tmp_path / mode)),
            observations=[_observation("failed", missing_tools=["inspect_table"]), _observation("success")],
        )
        assert result.status == "dry_run"
        assert result.report_paths["trace2skill_audit_report"].endswith("trace2skill_audit_report.md")


def test_skill_creation_from_scratch_stages_candidate_without_mutation(tmp_path):
    graph_path = _graph(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state")
    before = json.loads(graph_path.read_text(encoding="utf-8"))

    result = Trace2SkillEvolver(graph_backend=backend, tool_registry=_registry("read_text")).run(
        Trace2SkillRunConfig(mode="skill_creation_from_scratch", dry_run=False, output_dir=str(tmp_path / "creation")),
        observations=[_observation("failed", skills=[])],
    )
    after = json.loads(graph_path.read_text(encoding="utf-8"))

    assert result.changed_library is False
    assert any(update["proposal"]["proposal_type"] == "candidate_skill_creation" for update in result.staged_updates)
    assert before == after


def test_dry_run_does_not_mutate_even_for_auto_applied_failure_note(tmp_path):
    graph_path = _graph(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state")
    before = json.loads(graph_path.read_text(encoding="utf-8"))

    result = Trace2SkillEvolver(graph_backend=backend, tool_registry=_registry("read_text")).run(
        Trace2SkillRunConfig(mode="error_only", dry_run=True, output_dir=str(tmp_path / "dry")),
        observations=[_observation("failed")],
    )

    assert result.changed_library is False
    assert json.loads(graph_path.read_text(encoding="utf-8")) == before


def test_auto_apply_validated_required_tool_patch_mutates_embedded_skill_and_retrieval_reflects_it(tmp_path):
    graph_path = _graph(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state")
    policy = SkillEvolutionPolicy(
        auto_apply_proposal_types=["required_tools_update_proposal"],
        staged_proposal_types=[
            "metadata_update",
            "example_trace_memory_update",
            "candidate_skill_creation",
            "relationship_update_proposal",
        ],
    )

    result = Trace2SkillEvolver(
        graph_backend=backend,
        tool_registry=_registry("read_text", "inspect_table"),
        policy=policy,
    ).run(
        Trace2SkillRunConfig(mode="error_only", dry_run=False, output_dir=str(tmp_path / "apply-tools")),
        observations=[_observation("failed", missing_tools=["inspect_table"])],
    )

    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    bundle = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state").get(
        RetrievalRequest(task_id="task-1", role="solver", query="generic")
    )
    assert result.changed_library is True
    assert result.before_graph_hash != result.after_graph_hash
    assert raw_graph["skills"][0]["required_tools"] == ["read_text", "inspect_table"]
    assert "inspect_table" in bundle.skills[0].required_tools


def test_invalid_required_tool_patch_does_not_mutate_graph(tmp_path):
    graph_path = _graph(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "state")
    before = json.loads(graph_path.read_text(encoding="utf-8"))
    policy = SkillEvolutionPolicy(auto_apply_proposal_types=["required_tools_update_proposal"], staged_proposal_types=[])

    result = Trace2SkillEvolver(graph_backend=backend, tool_registry=_registry("read_text"), policy=policy).run(
        Trace2SkillRunConfig(mode="error_only", dry_run=False, output_dir=str(tmp_path / "invalid")),
        observations=[_observation("failed", missing_tools=["missing_unregistered_tool"])],
    )

    assert result.changed_library is False
    assert result.rejected_updates
    assert json.loads(graph_path.read_text(encoding="utf-8")) == before
