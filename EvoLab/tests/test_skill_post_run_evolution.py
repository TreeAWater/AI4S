import json
from pathlib import Path

from evolab.backends.skills import GraphSkillBackend
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest, SkillRef
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolTrace


def _candidate_skill(**overrides):
    data = {
        "schema_version": "v1",
        "skill_id": "skill.generic_record_review.v1",
        "name": "Generic Record Review",
        "description": "Review generic task records.",
        "source_type": "human",
        "source_uri": "seed://test/generic",
        "provenance": {},
        "domain_tags": ["generic"],
        "task_types": ["review"],
        "target_category": None,
        "scope": "Generic record review",
        "applicability": ["A task output needs review."],
        "limitations": ["Does not perform domain-specific validation."],
        "required_inputs": ["task output"],
        "expected_outputs": ["review summary"],
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": ["Read the task output.", "Summarize issues."],
        "required_tools": ["read_text"],
        "scripts": [],
        "resources": [],
        "examples": [],
        "smoke_tests": [],
        "synthetic_tests": [],
        "system_tests": [],
        "benchmark_tests": [],
        "validation_signals": ["observation"],
        "confidence": 0.8,
        "metadata": {},
    }
    data.update(overrides)
    return data


def _graph_path(tmp_path: Path, *, skills: list[dict] | None = None) -> Path:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": skills if skills is not None else [_candidate_skill()],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _request(query: str = "review generic records") -> RetrievalRequest:
    return RetrievalRequest(task_id="task-1", role="solver", query=query)


def _skill_ref(**overrides) -> SkillRef:
    data = {
        "skill_id": "skill.generic_record_review.v1",
        "name": "Generic Record Review",
        "content": "Description:\nReview generic task records.",
        "required_tools": ["read_text"],
    }
    data.update(overrides)
    return SkillRef(**data)


def _observation(
    *,
    skill_bundle: SkillBundle,
    final_answer: str | None = "done",
    metadata: dict | None = None,
    tool_trace: ToolTrace | None = None,
) -> SkillObservationRequest:
    return SkillObservationRequest(
        task_id="task-1",
        run_ref="run-1",
        role="solver",
        retrieval_request=_request(),
        skill_bundle=skill_bundle,
        graph_version_ref=skill_bundle.graph_version_ref,
        skill_state_ref=skill_bundle.skill_state_ref,
        tool_trace=tool_trace,
        final_answer=final_answer,
        metadata=metadata or {},
    )


def _bundle(*skills: SkillRef, metadata: dict | None = None) -> SkillBundle:
    return SkillBundle(
        backend_id="graph_skill",
        graph_version_ref="graph-v1",
        skills=list(skills),
        required_tools=sorted({tool for skill in skills for tool in skill.required_tools}),
        metadata=metadata or {},
    )


def _tool_trace_error(content: str = "lookup failed") -> ToolTrace:
    call = ToolCall(call_id="tool-1", name="lookup", arguments={})
    result = ToolResult(call_id="tool-1", status="error", content=content)
    return ToolTrace(run_ref="run-1", calls=[ToolCallRecord(tool_call=call, result=result)])


def test_successful_observation_applies_usage_stats_and_stages_example_trace(tmp_path):
    graph_path = _graph_path(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution")

    result = backend.look_at(_observation(skill_bundle=_bundle(_skill_ref()), final_answer="completed"))

    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    stats = raw_graph["skills"][0]["metadata"]["evolution_stats"]
    assert result.status == "updated"
    assert stats["usage_count"] == 1
    assert stats["success_count"] == 1
    assert stats["failure_count"] == 0
    assert stats["last_status"] == "success"
    proposal_types = {proposal["proposal_type"] for proposal in result.metadata["proposals"]}
    assert {"usage_stats_update", "example_trace_memory_update"}.issubset(proposal_types)
    assert any(update["proposal_type"] == "example_trace_memory_update" for update in result.metadata["staged_updates"])
    assert (tmp_path / "skill_evolution" / "observations.jsonl").exists()
    assert (tmp_path / "skill_evolution" / "applied_updates.jsonl").exists()


def test_failed_observation_applies_failure_stats_and_failure_note(tmp_path):
    graph_path = _graph_path(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution")

    result = backend.look_at(
        _observation(
            skill_bundle=_bundle(_skill_ref()),
            final_answer=None,
            tool_trace=_tool_trace_error("tool failed because input was malformed"),
        )
    )

    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    stats = raw_graph["skills"][0]["metadata"]["evolution_stats"]
    assert result.status == "updated"
    assert stats["usage_count"] == 1
    assert stats["failure_count"] == 1
    assert stats["last_status"] == "failure"
    assert stats["recent_failure_reasons"] == ["tool failed because input was malformed"]
    assert any(proposal["proposal_type"] == "failure_note_update" for proposal in result.metadata["proposals"])


def test_missing_required_tool_update_is_staged_without_changing_required_tools(tmp_path):
    graph_path = _graph_path(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution")

    result = backend.look_at(
        _observation(
            skill_bundle=_bundle(_skill_ref()),
            final_answer=None,
            metadata={"status": "failed", "missing_required_tools": ["inspect_table"]},
        )
    )

    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert raw_graph["skills"][0]["required_tools"] == ["read_text"]
    staged_types = {proposal["proposal_type"] for proposal in result.metadata["staged_updates"]}
    assert "required_tools_update_proposal" in staged_types
    assert result.update_summary["staged_updates"] is True
    reloaded = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution").get(_request("generic review"))
    assert reloaded.skills[0].required_tools == ["read_text"]


def test_candidate_skill_creation_is_staged_without_inserting_stable_skill(tmp_path):
    graph_path = _graph_path(tmp_path, skills=[])
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution")
    low_coverage_bundle = SkillBundle(
        backend_id="graph_skill",
        graph_version_ref="graph-v1",
        skills=[],
        required_tools=[],
        metadata={
            "missing_skill_report": {
                "missing_capability": "summarize unsupported instrument logs",
                "reason": "No CandidateSkill matched the retrieval query.",
            }
        },
    )

    result = backend.look_at(
        _observation(
            skill_bundle=low_coverage_bundle,
            final_answer=None,
            metadata={"status": "failed"},
        )
    )

    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert result.status == "staged"
    assert raw_graph["skills"] == []
    assert result.metadata["candidate_records"][0]["status"] == "staged"
    assert "unsupported instrument logs" in result.metadata["candidate_records"][0]["missing_capability_description"]
    staged_candidates = (tmp_path / "skill_evolution" / "staged_candidates.jsonl").read_text(encoding="utf-8")
    assert "unsupported instrument logs" in staged_candidates
    assert GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution").get(_request()).skills == []


def test_staged_candidate_and_required_tool_proposals_do_not_change_stable_retrieval(tmp_path):
    graph_path = _graph_path(tmp_path)
    backend = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution")
    before = backend.get(_request("generic record review"))

    backend.look_at(
        _observation(
            skill_bundle=_bundle(_skill_ref()),
            final_answer=None,
            metadata={
                "status": "failed",
                "missing_required_tools": ["write_report"],
                "missing_skill_report": {"missing_capability": "new generic capability"},
            },
        )
    )
    after = GraphSkillBackend(graph_path, evolution_root=tmp_path / "skill_evolution").get(
        _request("generic record review")
    )

    assert [skill.skill_id for skill in after.skills] == [skill.skill_id for skill in before.skills]
    assert after.skills[0].required_tools == before.skills[0].required_tools
