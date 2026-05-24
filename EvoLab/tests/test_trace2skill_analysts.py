from evolab.backends.skills.trace2skill.analysts import (
    CoverageAnalyst,
    ErrorAnalyst,
    PatchProposalAnalyst,
    SuccessAnalyst,
)
from evolab.backends.skills.trace2skill.schema import TraceRecord


def _trace(status: str, **overrides) -> TraceRecord:
    data = {
        "trace_id": f"trace-{status}",
        "task_id": "task-1",
        "task_summary": "generic task",
        "selected_skill_ids": ["skill.generic.v1"],
        "tools_used": ["read_text"],
        "final_status": status,
        "compact_execution_summary": "done" if "success" in status else None,
        "error_summary": "missing required tool" if "failure" in status else None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return TraceRecord(**data)


def test_error_analyst_produces_failure_and_tool_lessons():
    lessons = ErrorAnalyst().analyze(
        [_trace("runtime_failure", missing_tools=["inspect_table"], error_summary="tool missing")]
    )

    assert {lesson.lesson_type for lesson in lessons} == {"tool_lesson"}
    assert lessons[0].proposed_delta["missing_tools"] == ["inspect_table"]


def test_success_analyst_produces_example_and_tool_sequence_lessons():
    lessons = SuccessAnalyst().analyze([_trace("runtime_success")])

    assert [lesson.lesson_type for lesson in lessons] == ["success_lesson", "success_lesson"]
    assert {lesson.metadata["lesson_subtype"] for lesson in lessons} == {"example_memory", "tool_sequence"}


def test_coverage_analyst_and_patch_proposal_analyst_create_candidate_patch():
    coverage_trace = _trace(
        "runtime_failure",
        selected_skill_ids=[],
        metadata={"coverage_report": {"sufficient": False}},
    )
    lessons = CoverageAnalyst().analyze([coverage_trace])
    patches = PatchProposalAnalyst().analyze(lessons)

    assert lessons[0].lesson_type == "coverage_lesson"
    assert patches[0].patch_type == "skill_create_patch"
    assert patches[0].candidate_skill_id


def test_patch_proposal_analyst_maps_missing_tool_to_required_tools_patch():
    lessons = ErrorAnalyst().analyze([_trace("runtime_failure", missing_tools=["search_text"])])
    patches = PatchProposalAnalyst().analyze(lessons)

    assert patches[0].patch_type == "required_tools_patch"
    assert patches[0].patch_content["required_tools"] == ["search_text"]
