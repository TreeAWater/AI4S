from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evolab.contracts.common import EvolutionBudget, RuntimePolicy
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.task import TaskJob, TaskOrigin, TaskPurpose, TaskRequest


def test_dispatch_decision_accepts_run_subagent():
    decision = DispatchDecision(
        action=DispatchAction.RUN_SUBAGENT,
        target_role="solver",
        instruction="Solve the problem.",
        retrieval_query="relevant prior work",
    )
    assert decision.action == DispatchAction.RUN_SUBAGENT


def test_proposer_task_requires_relation():
    with pytest.raises(ValidationError):
        TaskRequest(
            task_id="task-proposed-1",
            origin=TaskOrigin.PROPOSER,
            purpose=TaskPurpose.TRAINING_ROLLOUT,
            goal="Variant task",
        )


def test_human_task_does_not_require_relation():
    request = TaskRequest(
        task_id="task-human-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Find a biomarker.",
    )
    assert request.proposed_task_relation is None


def test_schema_version_rejects_unknown_version():
    with pytest.raises(ValidationError):
        TaskRequest(
            schema_version="v2",
            task_id="task-human-1",
            origin=TaskOrigin.HUMAN,
            purpose=TaskPurpose.SCIENCE,
            goal="Find a biomarker.",
        )


def test_runtime_policy_rejects_string_max_tool_steps():
    with pytest.raises(ValidationError):
        RuntimePolicy(max_tool_steps="20")


def test_evolution_budget_rejects_negative_values():
    with pytest.raises(ValidationError):
        EvolutionBudget(max_wall_clock_s=-1)

    with pytest.raises(ValidationError):
        EvolutionBudget(max_train_samples=-1)

    with pytest.raises(ValidationError):
        EvolutionBudget(max_cost_usd=-0.01)


def test_task_job_enqueued_at_requires_datetime():
    enqueued_at = datetime(2026, 5, 2, 8, 45, tzinfo=timezone.utc)

    job = TaskJob(
        job_id="job-1",
        request_payload_uri="file:///tmp/request.json",
        enqueued_at=enqueued_at,
    )
    assert job.enqueued_at == enqueued_at

    with pytest.raises(ValidationError):
        TaskJob(
            job_id="job-1",
            request_payload_uri="file:///tmp/request.json",
            enqueued_at="not-a-timestamp",
        )
