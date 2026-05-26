import json

import pytest

from evolab.contracts.common import ArtifactRef
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult
from evolab.runtime.task_runtime import (
    TaskRuntime,
    _meta_agent_routing_state,
    _meta_dispatch_repair_guidance,
    _parse_dispatch_decision,
    _subagent_completion_contract,
    _validate_meta_dispatch_decision,
    _work_item_lifecycle_status_for_run,
)


POLICY = {
    "work_item_routing": {
        "enabled": True,
        "executor_roles": ["ExecAgent"],
        "reviewer_roles": ["CriticAgent"],
        "finalizer_roles": ["WriteAgent"],
        "work_item_id_field": "work_item_id",
    }
}
REQUIRED_POLICY = {
    "work_item_routing": {
        "enabled": True,
        "executor_roles": ["ExecAgent"],
        "reviewer_roles": ["CriticAgent"],
        "finalizer_roles": ["WriteAgent"],
        "work_item_id_field": "work_item_id",
        "required_work_item_ids": ["item-1", "item-2"],
    }
}
ROLES = ["SurveyAgent", "DesignAgent", "ExecAgent", "CriticAgent", "WriteAgent"]


def _route(route: str, metadata: dict | None = None):
    return _parse_dispatch_decision(
        json.dumps(
            {
                "route": route,
                "instruction": f"Run {route}.",
                "metadata": metadata or {},
            }
        )
    )


def _finish():
    return _parse_dispatch_decision(
        json.dumps({"route": "END", "instruction": "Done.", "metadata": {"final_answer": "Done."}})
    )


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="task-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Extract scientific records.",
    )


def _completed(role: str, work_item_id: str | None = None) -> dict:
    dispatch_metadata = {}
    if work_item_id is not None:
        dispatch_metadata["work_item_id"] = work_item_id
    return {
        "status": "completed",
        "role": role,
        "run_ref": f"{role}-{work_item_id or 'unscoped'}",
        "dispatch_metadata": dispatch_metadata,
    }


def _failed(role: str, work_item_id: str | None = None) -> dict:
    run = _completed(role, work_item_id)
    run["status"] = "guard_failed"
    run["failure_reason"] = "incomplete completed subagent outputs"
    run["completion_contract"] = {
        "assigned_task_complete": False,
        "ready_for_task_end": False,
        "blocking_issues": [{"type": "internal_dag_node_failed"}],
    }
    return run


def _budget_exceeded_with_artifact(role: str, work_item_id: str, filename: str, artifact_kind: str) -> dict:
    run = _completed(role, work_item_id)
    run["status"] = "budget_exceeded"
    run["failure_reason"] = f"budget_exceeded after producing {filename}"
    run["artifact_refs"] = [
        {
            "uri": f"/tmp/{work_item_id}/{filename}",
            "type": "dataset",
            "metadata": {
                "filename": filename,
                "artifact_kind": artifact_kind,
                "work_item_id": work_item_id,
            },
        }
    ]
    return run


def _guard_failed_with_artifact(role: str, work_item_id: str, filename: str, artifact_kind: str) -> dict:
    run = _budget_exceeded_with_artifact(role, work_item_id, filename, artifact_kind)
    run["status"] = "guard_failed"
    run["failure_reason"] = f"guard_failed after producing {filename}"
    return run


def _tool_record(
    *,
    call_id: str,
    tool_name: str,
    status: str,
    content: str,
    arguments: dict | None = None,
):
    return ToolCallRecord(
        tool_call=ToolCall(call_id=call_id, name=tool_name, arguments=arguments or {}),
        result=ToolResult(call_id=call_id, status=status, content=content),
    )


def test_work_item_policy_requires_executor_dispatch_to_name_one_work_item():
    decision = _route("ExecAgent")

    with pytest.raises(RuntimeError, match="work_item_id"):
        _validate_meta_dispatch_decision(decision, ROLES, [], runtime_metadata=POLICY)


def test_work_item_policy_allows_non_executor_dispatch_without_work_item():
    decision = _route("DesignAgent")

    _validate_meta_dispatch_decision(decision, ROLES, [], runtime_metadata=POLICY)


def test_work_item_policy_requires_review_after_executor_before_next_execution_or_finish():
    completed_runs = [_completed("ExecAgent", "item-1")]

    with pytest.raises(RuntimeError, match="requires reviewer"):
        _validate_meta_dispatch_decision(
            _route("ExecAgent", {"work_item_id": "item-2"}),
            ROLES,
            completed_runs,
            runtime_metadata=POLICY,
        )

    with pytest.raises(RuntimeError, match="requires reviewer"):
        _validate_meta_dispatch_decision(_finish(), ROLES, completed_runs, runtime_metadata=POLICY)


def test_work_item_policy_requires_reviewer_to_match_pending_work_item():
    completed_runs = [_completed("ExecAgent", "item-1")]

    with pytest.raises(RuntimeError, match="item-1"):
        _validate_meta_dispatch_decision(
            _route("CriticAgent", {"work_item_id": "item-2"}),
            ROLES,
            completed_runs,
            runtime_metadata=POLICY,
        )

    _validate_meta_dispatch_decision(
        _route("CriticAgent", {"work_item_id": "item-1"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_policy_allows_next_work_item_after_review():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _completed("CriticAgent", "item-1"),
    ]

    _validate_meta_dispatch_decision(
        _route("ExecAgent", {"work_item_id": "item-2"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_policy_failed_reviewer_does_not_clear_pending_review():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _failed("CriticAgent", "item-1"),
    ]

    with pytest.raises(RuntimeError, match="requires reviewer"):
        _validate_meta_dispatch_decision(
            _route("ExecAgent", {"work_item_id": "item-2"}),
            ROLES,
            completed_runs,
            runtime_metadata=POLICY,
        )

    _validate_meta_dispatch_decision(
        _route("CriticAgent", {"work_item_id": "item-1"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_policy_rejects_reviewer_after_failed_executor():
    completed_runs = [_failed("ExecAgent", "item-1")]

    with pytest.raises(RuntimeError, match="no pending completed executor"):
        _validate_meta_dispatch_decision(
            _route("CriticAgent", {"work_item_id": "item-1"}),
            ROLES,
            completed_runs,
            runtime_metadata=POLICY,
        )

    _validate_meta_dispatch_decision(
        _route("ExecAgent", {"work_item_id": "item-1"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_policy_allows_review_after_budget_exceeded_executor_with_candidate_artifact():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records")
    ]

    _validate_meta_dispatch_decision(
        _route("CriticAgent", {"work_item_id": "item-1"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_policy_allows_review_after_guard_failed_executor_with_candidate_artifact():
    completed_runs = [_guard_failed_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records")]

    _validate_meta_dispatch_decision(
        _route("CriticAgent", {"work_item_id": "item-1"}),
        ROLES,
        completed_runs,
        runtime_metadata=POLICY,
    )


def test_work_item_routing_state_reports_pending_review_after_budget_exceeded_candidate_handoff():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records")
    ]

    state = _meta_agent_routing_state(completed_runs, REQUIRED_POLICY)

    work_item_state = state["work_item_routing"]
    assert work_item_state["pending_reviews"] == [
        {"work_item_id": "item-1", "run_ref": "ExecAgent-item-1", "role": "ExecAgent"}
    ]
    assert work_item_state["required_next_action"] == {
        "route_one_of": ["CriticAgent"],
        "metadata": {"work_item_id": "item-1"},
        "reason": "review pending work item before any executor, finalizer, or END route",
    }


def test_work_item_policy_allows_finalizer_after_budget_exceeded_review_with_validated_artifact():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _budget_exceeded_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
        _completed("ExecAgent", "item-2"),
        _completed("CriticAgent", "item-2"),
    ]

    _validate_meta_dispatch_decision(
        _route("WriteAgent"),
        ROLES,
        completed_runs,
        runtime_metadata=REQUIRED_POLICY,
    )


def test_work_item_policy_allows_finalizer_after_failed_review_with_validated_artifact():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _guard_failed_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
        _completed("ExecAgent", "item-2"),
        _completed("CriticAgent", "item-2"),
    ]
    completed_runs[1]["status"] = "failed"
    completed_runs[1]["failure_reason"] = "failed after producing validated records"

    _validate_meta_dispatch_decision(
        _route("WriteAgent"),
        ROLES,
        completed_runs,
        runtime_metadata=REQUIRED_POLICY,
    )


def test_work_item_routing_state_requires_finalizer_after_review_handoff_before_more_execution():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _guard_failed_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
    ]
    policy = {
        "work_item_routing": {
            **REQUIRED_POLICY["work_item_routing"],
            "required_work_item_ids": ["item-1"],
        }
    }

    state = _meta_agent_routing_state(completed_runs, policy)

    work_item_state = state["work_item_routing"]
    assert work_item_state["pending_reviews"] == []
    assert work_item_state["reviewed_work_item_ids"] == ["item-1"]
    assert work_item_state["missing_required_unresolved_work_item_ids"] == []
    assert work_item_state["required_next_action"] == {
        "route_one_of": ["WriteAgent"],
        "reason": "write final artifacts after resolved work items before any executor, reviewer, or END route",
    }


def test_work_item_policy_rejects_executor_and_end_after_review_until_finalizer_writes_records():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _guard_failed_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
    ]
    policy = {
        "work_item_routing": {
            **REQUIRED_POLICY["work_item_routing"],
            "required_work_item_ids": ["item-1"],
        }
    }

    with pytest.raises(RuntimeError, match="requires finalizer dispatch before continuing"):
        _validate_meta_dispatch_decision(
            _route("ExecAgent", {"work_item_id": "item-1"}),
            ROLES,
            completed_runs,
            runtime_metadata=policy,
        )
    with pytest.raises(RuntimeError, match="requires finalizer dispatch before finish"):
        _validate_meta_dispatch_decision(_finish(), ROLES, completed_runs, runtime_metadata=policy)

    _validate_meta_dispatch_decision(
        _route("WriteAgent"),
        ROLES,
        completed_runs,
        runtime_metadata=policy,
    )


def test_meta_dispatch_repair_guidance_routes_finalizer_when_final_records_are_missing():
    guidance = _meta_dispatch_repair_guidance(
        RuntimeError(
            "work-item routing requires finalizer dispatch before continuing: "
            "all required work items are resolved and final records are not written"
        )
    )

    assert guidance["required_next_action"] == "route_finalizer_for_resolved_work_items"
    assert "finalizer" in guidance["instruction"]


def test_work_item_policy_allows_finish_after_budget_exceeded_handoff_and_final_artifact():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _budget_exceeded_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
        {
            **_completed("WriteAgent"),
            "artifact_refs": [
                {
                    "uri": "/tmp/biology_component_records.jsonl",
                    "type": "dataset",
                    "metadata": {
                        "filename": "biology_component_records.jsonl",
                        "artifact_kind": "final_records",
                    },
                }
            ],
        },
    ]
    policy = {
        **POLICY,
        "required_final_artifacts": ["biology_component_records.jsonl"],
    }

    _validate_meta_dispatch_decision(_finish(), ROLES, completed_runs, runtime_metadata=policy)


def test_work_item_policy_rejects_subagent_route_after_final_records_are_written():
    completed_runs = [
        _budget_exceeded_with_artifact("ExecAgent", "item-1", "candidate_records.json", "candidate_records"),
        _budget_exceeded_with_artifact("CriticAgent", "item-1", "validated_records.json", "validated_records"),
        {
            **_completed("WriteAgent", "item-1"),
            "artifact_refs": [
                {
                    "uri": "/tmp/final_records.jsonl",
                    "type": "dataset",
                    "metadata": {
                        "filename": "final_records.jsonl",
                        "artifact_kind": "final_records",
                    },
                }
            ],
        },
    ]
    policy = POLICY

    with pytest.raises(RuntimeError, match="requires END after final records are written"):
        _validate_meta_dispatch_decision(_route("WriteAgent"), ROLES, completed_runs, runtime_metadata=policy)

    _validate_meta_dispatch_decision(_finish(), ROLES, completed_runs, runtime_metadata=policy)


def test_meta_agent_result_accepts_non_completed_finalizer_with_final_records():
    runtime = TaskRuntime()
    finalizer_result = {
        **_completed("WriteAgent", "item-1"),
        "status": "failed",
        "failure_reason": "repeated_tool_call_suppressed after final records were written",
        "artifact_refs": [
            {
                "uri": "/tmp/final_records.jsonl",
                "type": "dataset",
                "metadata": {
                    "filename": "final_records.jsonl",
                    "artifact_kind": "final_records",
                },
            }
        ],
    }

    result = runtime._meta_agent_result(
        _request(),
        [finalizer_result],
        ["meta-1"],
        DispatchDecision(
            action=DispatchAction.FINISH_TASK,
            instruction="Done.",
            metadata={"final_answer": "Done."},
        ),
    )

    assert result["status"] == "completed"
    assert result["run_ref"] == "WriteAgent-item-1"


def test_work_item_lifecycle_marks_successful_finalizer_completed():
    run = {
        **_completed("WriteAgent", "item-1"),
        "artifact_refs": [
            {
                "uri": "/tmp/final_records.jsonl",
                "type": "dataset",
                "metadata": {"filename": "final_records.jsonl", "artifact_kind": "final_records"},
            }
        ],
    }

    status = _work_item_lifecycle_status_for_run(run, "WriteAgent", REQUIRED_POLICY["work_item_routing"])

    assert status == "completed"


def test_work_item_lifecycle_finalizer_budget_exceeded_with_final_artifacts_is_completed():
    run = _budget_exceeded_with_artifact("WriteAgent", "item-1", "final_records.jsonl", "final_records")

    status = _work_item_lifecycle_status_for_run(run, "WriteAgent", REQUIRED_POLICY["work_item_routing"])

    assert status == "completed"


def test_work_item_lifecycle_finalizer_failures_do_not_remain_claimed():
    failed = _failed("WriteAgent", "item-1")
    budget_exceeded = _completed("WriteAgent", "item-1")
    budget_exceeded["status"] = "budget_exceeded"
    budget_exceeded["failure_reason"] = "budget exceeded before final records"
    interrupted = _completed("WriteAgent", "item-1")
    interrupted["status"] = "interrupted"

    assert _work_item_lifecycle_status_for_run(failed, "WriteAgent", REQUIRED_POLICY["work_item_routing"]) == "failed"
    assert (
        _work_item_lifecycle_status_for_run(
            budget_exceeded,
            "WriteAgent",
            REQUIRED_POLICY["work_item_routing"],
        )
        == "budget_exceeded"
    )
    assert (
        _work_item_lifecycle_status_for_run(interrupted, "WriteAgent", REQUIRED_POLICY["work_item_routing"])
        == "interrupted"
    )


def test_work_item_policy_is_inactive_by_default():
    decision = _route("ExecAgent")

    _validate_meta_dispatch_decision(decision, ROLES, [], runtime_metadata={})


def test_work_item_policy_rejects_finalizer_until_required_work_items_are_reviewed():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _completed("CriticAgent", "item-1"),
    ]

    with pytest.raises(RuntimeError, match="item-2"):
        _validate_meta_dispatch_decision(
            _route("WriteAgent"),
            ROLES,
            completed_runs,
            runtime_metadata=REQUIRED_POLICY,
        )


def test_work_item_policy_rejects_finish_until_required_work_items_are_reviewed():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _completed("CriticAgent", "item-1"),
    ]

    with pytest.raises(RuntimeError, match="item-2"):
        _validate_meta_dispatch_decision(_finish(), ROLES, completed_runs, runtime_metadata=REQUIRED_POLICY)


def test_work_item_policy_allows_finalizer_after_required_work_items_are_reviewed():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _completed("CriticAgent", "item-1"),
        _completed("ExecAgent", "item-2"),
        _completed("CriticAgent", "item-2"),
    ]

    _validate_meta_dispatch_decision(
        _route("WriteAgent"),
        ROLES,
        completed_runs,
        runtime_metadata=REQUIRED_POLICY,
    )


def test_work_item_policy_allows_finalizer_after_failed_item_exhausts_retry_budget():
    completed_runs = [
        _failed("ExecAgent", "item-1"),
        _completed("ExecAgent", "item-2"),
        _completed("CriticAgent", "item-2"),
    ]
    policy = {
        "work_item_routing": {
            **REQUIRED_POLICY["work_item_routing"],
            "max_failed_executor_attempts_per_work_item": 1,
        }
    }

    _validate_meta_dispatch_decision(
        _route("WriteAgent"),
        ROLES,
        completed_runs,
        runtime_metadata=policy,
    )


def test_work_item_routing_state_reports_pending_review_for_meta_agent():
    completed_runs = [_completed("ExecAgent", "item-1")]

    state = _meta_agent_routing_state(completed_runs, REQUIRED_POLICY)

    work_item_state = state["work_item_routing"]
    assert work_item_state["pending_reviews"] == [
        {"work_item_id": "item-1", "run_ref": "ExecAgent-item-1", "role": "ExecAgent"}
    ]
    assert work_item_state["missing_required_reviewed_work_item_ids"] == ["item-1", "item-2"]
    assert work_item_state["required_next_action"] == {
        "route_one_of": ["CriticAgent"],
        "metadata": {"work_item_id": "item-1"},
        "reason": "review pending work item before any executor, finalizer, or END route",
    }


def test_work_item_routing_state_keeps_pending_review_after_failed_reviewer():
    completed_runs = [
        _completed("ExecAgent", "item-1"),
        _failed("CriticAgent", "item-1"),
    ]

    state = _meta_agent_routing_state(completed_runs, REQUIRED_POLICY)

    work_item_state = state["work_item_routing"]
    assert work_item_state["pending_reviews"] == [
        {"work_item_id": "item-1", "run_ref": "ExecAgent-item-1", "role": "ExecAgent"}
    ]
    assert work_item_state["reviewed_work_item_ids"] == []
    assert work_item_state["missing_required_reviewed_work_item_ids"] == ["item-1", "item-2"]
    assert work_item_state["required_next_action"] == {
        "route_one_of": ["CriticAgent"],
        "metadata": {"work_item_id": "item-1"},
        "reason": "review pending work item before any executor, finalizer, or END route",
    }


def test_work_item_routing_state_guides_next_item_after_retry_exhaustion():
    completed_runs = [_failed("ExecAgent", "item-1")]
    policy = {
        "work_item_routing": {
            **REQUIRED_POLICY["work_item_routing"],
            "max_failed_executor_attempts_per_work_item": 1,
        }
    }

    state = _meta_agent_routing_state(completed_runs, policy)

    work_item_state = state["work_item_routing"]
    assert work_item_state["failed_work_item_ids"] == ["item-1"]
    assert work_item_state["retry_budget_exhausted_work_item_ids"] == ["item-1"]
    assert work_item_state["missing_required_unresolved_work_item_ids"] == ["item-2"]
    assert work_item_state["required_next_action"] == {
        "route_one_of": ["ExecAgent"],
        "metadata": {"work_item_id": "item-2"},
        "reason": "process the next unresolved required work item before finalizer or END",
    }


def test_meta_dispatch_repair_guidance_names_pending_work_item_review():
    guidance = _meta_dispatch_repair_guidance(
        RuntimeError("work-item routing requires reviewer dispatch before continuing: item-2")
    )

    assert guidance == {
        "required_next_action": "route_reviewer_for_pending_work_item",
        "work_item_ids": ["item-2"],
        "instruction": "Route one configured reviewer role for the listed work_item_id before routing any executor, finalizer, or END.",
    }


def test_meta_dispatch_repair_guidance_rejects_review_without_successful_execution():
    guidance = _meta_dispatch_repair_guidance(
        RuntimeError(
            "work-item routing reviewer rejected because no pending completed executor work item matches "
            "work_item_id='item-2'"
        )
    )

    assert guidance == {
        "required_next_action": "route_executor_for_uncompleted_work_item",
        "work_item_id": "item-2",
        "instruction": (
            "Route one configured executor role for this work_item_id. Do not route a reviewer until that "
            "executor run completes successfully."
        ),
    }


def test_subagent_completion_contract_blocks_missing_required_successful_tool_call():
    contract = _subagent_completion_contract(
        status="completed",
        failure_reason=None,
        artifact_refs=[],
        node_records=[],
        role="WriteAgent",
        tool_trace_records=[],
        policy_metadata={
            "completion_guards_by_role": {
                "WriteAgent": {"required_tool_calls_before_final": ["write_jsonl", "write_report"]}
            }
        },
    )

    assert contract["blocking_issues"] == [
        {
            "type": "missing_required_tool_call",
            "severity": "blocking",
            "message": "missing successful required tool call before completion: write_jsonl. Continue by calling write_jsonl successfully.",
            "tool_name": "write_jsonl",
        },
        {
            "type": "missing_required_tool_call",
            "severity": "blocking",
            "message": "missing successful required tool call before completion: write_report. Continue by calling write_report successfully.",
            "tool_name": "write_report",
        },
    ]


def test_subagent_completion_contract_does_not_clear_tool_error_with_different_target_success():
    contract = _subagent_completion_contract(
        status="completed",
        failure_reason=None,
        artifact_refs=[ArtifactRef(uri="/tmp/review.md", type="log")],
        node_records=[],
        role="CriticAgent",
        tool_trace_records=[
            _tool_record(
                call_id="missing-schema",
                tool_name="read_text",
                status="error",
                content="[Errno 2] No such file or directory: '/wrong/schema.json'",
                arguments={"path": "/wrong/schema.json"},
            ),
            _tool_record(
                call_id="read-records",
                tool_name="read_text",
                status="ok",
                content="records",
                arguments={"path": "/tmp/records.jsonl"},
            ),
        ],
        policy_metadata={
            "completion_guards_by_role": {
                "CriticAgent": {"required_tool_calls_before_final": ["write_report"]}
            }
        },
    )

    blocking = contract["blocking_issues"]
    assert any(issue["type"] == "tool_call_error" and issue["call_id"] == "missing-schema" for issue in blocking)
    assert contract["assigned_task_complete"] is False
