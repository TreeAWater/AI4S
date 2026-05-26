import warnings as py_warnings
from pathlib import Path

import pytest

from evolab.contracts.common import ArtifactRef
from evolab.contracts.lab_state import (
    ArtifactIndexRecord,
    LabStateDigest,
    LabStateIndex,
    LabStateIndexTaskSummary,
    LabStateTrajectorySummary,
    RunLedgerRecord,
    SubagentReportRecord,
    TrainingIndexRecord,
)
from evolab.registries.lab_state import FileLabStateRegistry, LabStateRegistryLoadWarning


def test_lab_state_registry_round_trips_new_lab_objects(tmp_path: Path):
    registry = FileLabStateRegistry(tmp_path)
    ledger = RunLedgerRecord(
        task_run_id="task-1",
        task_id="task-1",
        task_goal="Extract components.",
        status="running",
        config_ref="configs/biology_component_extraction_v1_generic_subagents.yaml",
    )
    report = SubagentReportRecord(
        report_ref="report-1",
        task_id="task-1",
        run_ref="subagent-1",
        role="SurveyAgent",
        status="completed",
        assigned_task="Survey dataset.",
        summary="Surveyed inputs.",
        artifact_refs=[ArtifactRef(uri="/tmp/inventory.json", type="dataset")],
        coverage={"processed_article_count": 2},
    )
    artifact = ArtifactIndexRecord(
        artifact_ref="artifact-1",
        task_id="task-1",
        producer_run_ref="subagent-1",
        uri="/tmp/records.jsonl",
        artifact_type="dataset",
        role="final_records",
        status="final",
        metadata={"record_count": 3},
    )
    training = TrainingIndexRecord(
        sample_ref="training-1",
        task_id="task-1",
        source_run_ref="subagent-1",
        source_llm_call_refs=["llm-1"],
        sample_kind="subagent_trace",
        quality_label="accepted",
    )
    index = LabStateIndex(
        index_ref="index-1",
        task=LabStateIndexTaskSummary(
            task_id="task-1",
            task_goal="Extract components.",
            task_request_ref="registries/task/task-1.json",
        ),
        trajectory=LabStateTrajectorySummary(
            meta_agent_run_count=1,
            subagent_run_count=1,
            llm_call_count=2,
            tool_call_count=3,
            event_count=4,
            evolution_run_count=0,
        ),
        subagent_reports=[{"report_ref": "report-1", "role": "SurveyAgent"}],
        artifacts=[{"artifact_ref": "artifact-1", "status": "final"}],
        training_samples=[{"sample_ref": "training-1"}],
    )
    digest = LabStateDigest(
        digest_ref="digest-1",
        index_ref=index.index_ref,
        task_id="task-1",
        summary="Survey completed; records pending.",
        sections={"recent_reports": ["SurveyAgent completed."]},
    )

    registry.save_run_ledger(ledger)
    registry.save_subagent_report(report)
    registry.save_artifact_index_record(artifact)
    registry.save_training_index_record(training)
    registry.save_index(index)
    registry.save_digest(digest)

    assert registry.get_run_ledger("task-1") == ledger
    assert registry.list_subagent_reports("task-1") == [report]
    assert registry.list_artifacts("task-1") == [artifact]
    assert registry.list_training_samples("task-1") == [training]
    assert registry.latest_index("task-1") == index
    assert registry.latest_digest("task-1") == digest


def test_lab_state_registry_loads_valid_jsonl_records_without_warning(tmp_path: Path):
    registry = FileLabStateRegistry(tmp_path)
    report = SubagentReportRecord(
        report_ref="report-1",
        task_id="task-1",
        run_ref="subagent-1",
        role="ExecAgent",
        status="completed",
        assigned_task="Extract records.",
        summary="Done.",
    )
    registry.save_subagent_report(report)

    with py_warnings.catch_warnings(record=True) as warnings:
        py_warnings.simplefilter("always")
        records = registry.list_subagent_reports("task-1")

    assert records == [report]
    assert len(warnings) == 0


def test_lab_state_registry_skips_malformed_jsonl_records_with_warning(tmp_path: Path):
    registry = FileLabStateRegistry(tmp_path)
    first = SubagentReportRecord(
        report_ref="report-1",
        task_id="task-1",
        run_ref="subagent-1",
        role="ExecAgent",
        status="completed",
        assigned_task="Extract records.",
        summary="Done.",
    )
    second = first.model_copy(update={"report_ref": "report-2", "run_ref": "subagent-2"})
    (tmp_path / "subagent_reports.jsonl").write_text(
        "\n".join(
            [
                first.model_dump_json(),
                '{"report_ref": "bad", "task_id": ',
                second.model_dump_json(),
                '{"report_ref": 123}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(LabStateRegistryLoadWarning) as warnings:
        records = registry.list_subagent_reports("task-1")

    assert records == [first, second]
    assert len(warnings) == 2
    first_message = str(warnings[0].message)
    assert "path=" in first_message
    assert "subagent_reports.jsonl" in first_message
    assert "line=2" in first_message
    assert "record_type=SubagentReportRecord" in first_message
    assert "error=" in first_message
    assert "prefix=" in first_message
