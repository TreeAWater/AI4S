from pathlib import Path

from evolab.contracts.lab_state import SubagentReportRecord
from evolab.contracts.records import TrajectoryEventRecord
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.lab_state import LabStateBuilder


def test_lab_state_builder_scopes_index_and_requested_details_to_task(tmp_path: Path):
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    request = TaskRequest(
        task_id="task-current",
        goal="Audit current task only.",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
    )
    lab_state_registry.save_subagent_report(
        SubagentReportRecord(
            report_ref="report-current",
            task_id=request.task_id,
            run_ref="run-current",
            role="SurveyAgent",
            status="completed",
            assigned_task="Inspect current task.",
            summary="Current task report.",
        )
    )
    lab_state_registry.save_subagent_report(
        SubagentReportRecord(
            report_ref="report-other",
            task_id="task-other",
            run_ref="run-other",
            role="SurveyAgent",
            status="completed",
            assigned_task="Inspect other task.",
            summary="Other task report.",
        )
    )
    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref="state-current",
            backend_id="memory-local",
            backend_type="memory",
            created_from_task_id=request.task_id,
            created_from_run_ref="run-current",
        )
    )
    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref="state-other",
            backend_id="memory-local",
            backend_type="memory",
            created_from_task_id="task-other",
            created_from_run_ref="run-other",
        )
    )
    trajectory_registry.save_event(
        TrajectoryEventRecord(
            event_ref="event-current",
            event_type="task_started",
            subject_type="task",
            subject_ref=request.task_id,
            task_id=request.task_id,
        )
    )
    trajectory_registry.save_event(
        TrajectoryEventRecord(
            event_ref="event-other",
            event_type="task_started",
            subject_type="task",
            subject_ref="task-other",
            task_id="task-other",
        )
    )

    lab_state = LabStateBuilder(
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        lab_state_registry=lab_state_registry,
    ).build_for_meta_agent(
        request=request,
        requested_detail_refs={
            "subagent_reports": ["report-current", "report-other"],
            "backend_states": ["state-current", "state-other"],
        },
    )

    assert [report["report_ref"] for report in lab_state["index"]["subagent_reports"]] == ["report-current"]
    assert [state["state_ref"] for state in lab_state["index"]["backend_states"]] == ["state-current"]
    assert lab_state["index"]["trajectory"]["event_count"] == 1
    assert [report["report_ref"] for report in lab_state["requested_details"]["subagent_reports"]] == [
        "report-current"
    ]
    assert [state["state_ref"] for state in lab_state["requested_details"]["backend_states"]] == ["state-current"]
