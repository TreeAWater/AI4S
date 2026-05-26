from evolab.backends.skills.trace2skill.outcome import (
    classify_trace_outcome,
    detect_low_retrieval_coverage,
    detect_missing_tools,
    summarize_failure,
    summarize_success,
)


def test_classify_trace_outcome_uses_runtime_and_evaluation_signals():
    assert classify_trace_outcome({"evaluation_metrics": {"passed": True}}) == "evaluation_success"
    assert classify_trace_outcome({"evaluation_metrics": {"score": 0.2, "threshold": 0.8}}) == "evaluation_failure"
    assert classify_trace_outcome({"metadata": {"status": "success"}}) == "runtime_success"
    assert classify_trace_outcome({"error_summary": "failed"}) == "runtime_failure"
    assert classify_trace_outcome({}) == "unknown"


def test_detect_missing_tools_from_trace_metadata_and_tool_calls():
    missing = detect_missing_tools(
        {
            "missing_tools": ["read_text"],
            "metadata": {
                "tool_calls": [
                    {
                        "tool_call": {"name": "inspect_table"},
                        "result": {"metadata": {"error_type": "unprepared_tool"}},
                    }
                ]
            },
        }
    )

    assert missing == ["read_text", "inspect_table"]


def test_low_coverage_and_summaries_are_generic():
    trace = {
        "task_summary": "unsupported task",
        "selected_skill_ids": [],
        "metadata": {"coverage_report": {"sufficient": False}},
    }

    assert detect_low_retrieval_coverage(trace) is True
    assert "Low retrieval coverage" in summarize_failure(trace)
    assert "Tools used: read_text" in summarize_success({"compact_execution_summary": "done", "tools_used": ["read_text"]})
