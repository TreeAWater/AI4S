import os
import warnings as py_warnings
from pathlib import Path

import pytest

from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionResult
from evolab.contracts.records import (
    EvolutionRunRecord,
    LLMCallRecord,
    MetaAgentRunRecord,
    SubagentRunRecord,
    ToolCallTrajectoryRecord,
    TrajectoryEventRecord,
)
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.registries.backend_state import BackendStateRegistry, FileBackendStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult
from evolab.registries.trajectory import (
    FileTrajectoryRegistry,
    TrajectoryRegistry,
    TrajectoryRegistryLoadWarning,
)


def _task_request(task_id: str) -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Find target.",
    )


def _subagent_run(run_ref: str, human_anchor_task_refs: list[str]) -> SubagentRunRecord:
    return SubagentRunRecord(
        run_ref=run_ref,
        task_id="task-1",
        task_origin=TaskOrigin.HUMAN,
        task_purpose=TaskPurpose.SCIENCE,
        stage_index=0,
        role="solver",
        instruction="Solve it.",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="prior work"),
        memory_bundle=MemoryBundle(backend_id="memory-local"),
        skill_bundle=SkillBundle(backend_id="skill-local"),
        prompt_messages=[Message(role="user", content="Solve it.")],
        llm_backend_id="llm-api",
        human_anchor_task_refs=human_anchor_task_refs,
    )


def test_trajectory_registry_round_trips_shared_records(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    meta = MetaAgentRunRecord(
        run_ref="meta-1",
        task_id="task-1",
        decision=DispatchDecision(
            action=DispatchAction.RUN_SUBAGENT,
            target_role="solver",
            instruction="Solve it.",
        ),
    )
    subagent = _subagent_run("subagent-1", ["human-1"])
    llm_call = LLMCallRecord(
        call_ref="call-1",
        run_ref=subagent.run_ref,
        backend_id="llm-fake",
        model="fake-model",
        input_messages=[Message(role="user", content="hello")],
        output_messages=[Message(role="assistant", content="world")],
    )
    tool_call = ToolCallTrajectoryRecord(
        record_ref="tool-record-1",
        run_ref=subagent.run_ref,
        task_id="task-1",
        tool_call_id="tool-call-1",
        tool_name="lookup",
        role="solver",
        runtime_stage="subagent_flat",
        step_index=0,
        record=ToolCallRecord(
            tool_call=ToolCall(call_id="tool-call-1", name="lookup", arguments={"query": "x"}),
            result=ToolResult(call_id="tool-call-1", status="ok", content="found x"),
        ),
    )
    event = TrajectoryEventRecord(
        event_ref="event-1",
        event_type="subagent_started",
        subject_type="subagent",
        subject_ref=subagent.run_ref,
        task_id="task-1",
        run_ref=subagent.run_ref,
        metadata={"role": "solver"},
    )
    evolution = EvolutionRunRecord(
        run_ref="evo-1",
        mode=LLMEvolutionMode.BASICS,
        backend_id="llm-fake",
        result_status="not_recommended",
        result=LLMEvolutionResult(status="not_recommended", recommend_for_promotion=False),
        training_trajectory_refs=[subagent.run_ref],
    )

    assert isinstance(registry, TrajectoryRegistry)
    assert registry.save_meta_agent_run(meta) == meta.run_ref
    assert registry.save_subagent_run(subagent) == subagent.run_ref
    assert registry.save_llm_call(llm_call) == llm_call.call_ref
    assert registry.save_tool_call_record(tool_call) == tool_call.record_ref
    assert registry.save_event(event) == event.event_ref
    assert registry.save_evolution_run(evolution) == evolution.run_ref

    assert registry.get_meta_agent_run(meta.run_ref) == meta
    assert registry.get_subagent_run(subagent.run_ref) == subagent
    assert registry.get_llm_call(llm_call.call_ref) == llm_call
    assert registry.get_tool_call_record(tool_call.record_ref) == tool_call
    assert registry.get_event(event.event_ref) == event
    assert registry.get_evolution_run(evolution.run_ref) == evolution
    assert registry.list_meta_agent_runs() == [meta]
    assert registry.list_subagent_runs() == [subagent]
    assert registry.list_llm_calls() == [llm_call]
    assert registry.list_tool_call_records() == [tool_call]
    assert registry.list_events() == [event]
    assert registry.list_evolution_runs() == [evolution]
    assert registry.get_subagent_run("missing") is None


def test_trajectory_registry_loads_valid_records_without_warning(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    subagent = _subagent_run("subagent-1", ["human-1"])
    registry.save_subagent_run(subagent)

    with py_warnings.catch_warnings(record=True) as warnings:
        py_warnings.simplefilter("always")
        records = registry.list_subagent_runs()

    assert records == [subagent]
    assert len(warnings) == 0


def test_trajectory_registry_skips_malformed_jsonl_line_with_warning(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    valid = _subagent_run("subagent-1", ["human-1"])
    (tmp_path / "subagent.jsonl").write_text(
        valid.model_dump_json() + "\n" + '{"run_ref": "bad", "task_id": ' + "\n",
        encoding="utf-8",
    )

    with pytest.warns(TrajectoryRegistryLoadWarning) as warnings:
        records = registry.list_subagent_runs()

    assert records == [valid]
    message = str(warnings[0].message)
    assert "path=" in message
    assert "subagent.jsonl" in message
    assert "line=2" in message
    assert "record_type=SubagentRunRecord" in message
    assert "prefix=" in message


def test_trajectory_registry_skips_truncated_final_line(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    valid = _subagent_run("subagent-1", ["human-1"])
    (tmp_path / "subagent.jsonl").write_text(
        valid.model_dump_json() + "\n" + '{"run_ref": "truncated',
        encoding="utf-8",
    )

    with pytest.warns(TrajectoryRegistryLoadWarning, match="line=2"):
        records = registry.list_subagent_runs()

    assert records == [valid]


def test_trajectory_registry_mixed_valid_and_invalid_lines_returns_valid_records(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    first = _subagent_run("subagent-1", ["human-1"])
    second = _subagent_run("subagent-2", ["human-2"])
    (tmp_path / "subagent.jsonl").write_text(
        "\n".join(
            [
                first.model_dump_json(),
                '{"run_ref": "bad", "task_id": ',
                second.model_dump_json(),
                '{"run_ref": 123}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(TrajectoryRegistryLoadWarning) as warnings:
        records = registry.list_subagent_runs()

    assert records == [first, second]
    assert len(warnings) == 2
    assert "line=2" in str(warnings[0].message)
    assert "line=4" in str(warnings[1].message)


def test_backend_state_registry_round_trips_records(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path)
    first = BackendStateRecord(
        state_ref="state-1",
        backend_id="llm-fake",
        backend_type="llm",
        artifact_refs=[ArtifactRef(uri="file:///tmp/adapter.bin", type="model_adapter")],
    )
    second = BackendStateRecord(state_ref="state-2", backend_id="memory-local", backend_type="memory")

    assert isinstance(registry, BackendStateRegistry)
    registry.register_candidate(first)
    registry.register_candidate(second)

    assert registry.get_state("state-1") == first
    assert registry.get_state("missing") is None
    assert registry.list_states() == [first, second]
    assert registry.list_states(backend_id="llm-fake") == [first]


def test_backend_state_promote_and_resolve(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path)
    record = BackendStateRecord(state_ref="state-1", backend_id="llm-api", backend_type="llm")
    registry.register_candidate(record)
    registry.promote("llm-api", "state-1", "evo-1")
    assert registry.resolve_active_state("llm-api") == "state-1"


def test_backend_state_promote_uses_atomic_replace(tmp_path: Path, monkeypatch):
    registry = FileBackendStateRegistry(tmp_path)
    record = BackendStateRecord(state_ref="state-1", backend_id="llm-api", backend_type="llm")
    registry.register_candidate(record)
    calls = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("evolab.registries.backend_state.os.replace", recording_replace)
    registry.promote("llm-api", "state-1", "evo-1")

    assert calls
    assert calls[-1][0].name == "active.json.tmp"
    assert calls[-1][1] == registry.active_path


def test_task_registry_query_by_origin(tmp_path: Path):
    registry = FileTaskRegistry(tmp_path)
    request = _task_request("task-1")
    registry.save_task_request(request)
    assert [item.task_id for item in registry.query_by_origin(TaskOrigin.HUMAN)] == ["task-1"]


def test_task_registry_rejects_path_traversal_task_id(tmp_path: Path):
    registry = FileTaskRegistry(tmp_path)

    with pytest.raises(ValueError, match="unsafe task_id"):
        registry.save_task_request(_task_request("../escape"))

    assert not (tmp_path.parent / "escape.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_backend_state_promote_requires_registered_state(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path)

    with pytest.raises(ValueError, match="registered backend state"):
        registry.promote("llm-api", "state-1", "evo-1")

    assert registry.resolve_active_state("llm-api") is None


def test_backend_state_promote_rejects_state_registered_to_different_backend(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path)
    registry.register_candidate(
        BackendStateRecord(state_ref="state-1", backend_id="other-backend", backend_type="llm")
    )

    with pytest.raises(ValueError, match="registered backend state"):
        registry.promote("llm-api", "state-1", "evo-1")

    assert registry.resolve_active_state("llm-api") is None


def test_trajectory_registry_filters_list_fields_by_membership(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    registry.save_subagent_run(_subagent_run("run-1", ["human-1", "human-2"]))
    registry.save_subagent_run(_subagent_run("run-2", ["human-3"]))

    results = registry.query_subagent_runs({"human_anchor_task_refs": "human-1"})

    assert [record.run_ref for record in results] == ["run-1"]


def test_trajectory_registry_rejects_unknown_filter_keys(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path)
    registry.save_subagent_run(_subagent_run("run-1", ["human-1"]))

    with pytest.raises(ValueError, match="Unknown subagent run filter"):
        registry.query_subagent_runs({"unknown": "value"})
