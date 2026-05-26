from evolab.backends.skills.trace2skill.trace_pool import TracePoolBuilder
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest, SkillRef
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolTrace


def _bundle(*skills: SkillRef, metadata: dict | None = None) -> SkillBundle:
    return SkillBundle(
        backend_id="graph_skill",
        graph_version_ref="graph-v1",
        skills=list(skills),
        required_tools=[tool for skill in skills for tool in skill.required_tools],
        metadata=metadata or {"graph_context_summary": {"coverage_report": {"sufficient": True}}},
    )


def _skill() -> SkillRef:
    return SkillRef(
        skill_id="skill.generic.v1",
        name="Generic Skill",
        content="Generic skill.",
        required_tools=["read_text"],
    )


def _observation(status: str, *, missing_tools: list[str] | None = None, empty_bundle: bool = False):
    tool_trace = None
    if status == "failed":
        call = ToolCall(call_id="call-1", name="missing_tool", arguments={})
        result = ToolResult(
            call_id="call-1",
            status="error",
            content="not prepared",
            metadata={"error_type": "unprepared_tool"},
        )
        tool_trace = ToolTrace(run_ref="run-1", calls=[ToolCallRecord(tool_call=call, result=result)])
    return SkillObservationRequest(
        task_id="task-1",
        run_ref=f"run-{status}-{'empty' if empty_bundle else 'skills'}",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="generic task"),
        skill_bundle=_bundle() if empty_bundle else _bundle(_skill()),
        tool_trace=tool_trace,
        final_answer="done" if status == "success" else None,
        metadata={"status": status, "missing_tools": missing_tools or []},
    )


def test_build_trace_pool_from_observations_separates_outcomes_and_missing_tools():
    pool = TracePoolBuilder().build_from_observations(
        [
            _observation("success"),
            _observation("failed", missing_tools=["inspect_table"]),
            _observation("failed", empty_bundle=True),
        ]
    )

    assert pool.stats["trace_count"] == 3
    assert pool.stats["success_count"] == 1
    assert pool.stats["failure_count"] == 2
    failure_missing = sorted({tool for trace in pool.failure_traces for tool in trace.missing_tools})
    assert failure_missing == ["inspect_table", "missing_tool"]
    assert any(trace.metadata["low_retrieval_coverage"] for trace in pool.failure_traces)


def test_trace_pool_deduplicates_and_applies_max_trace_limit():
    builder = TracePoolBuilder()
    trace = builder.normalize_trace(
        {
            "trace_id": "trace-1",
            "task_id": "task-1",
            "task_summary": "generic",
            "selected_skill_ids": ["skill.generic.v1"],
            "final_status": "runtime_success",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )

    pool = builder.build_trace_pool([trace, trace], max_traces=1)

    assert [item.trace_id for item in pool.traces] == ["trace-1"]


def test_build_trace_pool_from_run_records_filters_by_skill_and_task_type():
    record = {
        "run_ref": "run-1",
        "task_id": "task-1",
        "role": "solver",
        "instruction": "generic",
        "retrieval_request": {"query": "generic"},
        "skill_bundle": {
            "skills": [{"skill_id": "skill.generic.v1", "name": "Generic", "required_tools": []}],
            "metadata": {},
        },
        "output_messages": [{"content": "done"}],
        "metadata": {"status": "success"},
    }

    pool = TracePoolBuilder().build_from_run_records(
        [record],
        task_type="solver",
        target_skill_ids=["skill.generic.v1"],
    )

    assert len(pool.traces) == 1
    assert pool.traces[0].run_record_ref == "run-1"
