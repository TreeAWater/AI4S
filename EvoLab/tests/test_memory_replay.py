from pathlib import Path

from evolab.cli import run_clean_demo
from evolab.contracts.common import Message
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.lab.resolver import LabResolver
from evolab.runtime.memory_replay import replay_memory_trace


def test_replay_memory_trace_from_clean_lab_validates_memory_lineage(tmp_path: Path):
    lab_root = tmp_path / "demo-v0-lab"

    run_clean_demo(Path("configs/demo_v0.yaml"), lab_root)
    report = replay_memory_trace(lab_root, task_id="demo-v0")

    assert report.ok is True
    assert report.issues == []
    assert len(report.records) == 4
    task_records = [record for record in report.records if record.memory_scope == "task"]
    assert [record.pre_state_ref for record in task_records] == [
        "fake-memory://fake-task-memory/12:task:demo-v0/v0",
        "fake-memory://fake-task-memory/12:task:demo-v0/v1",
    ]
    assert [record.update_state_ref for record in task_records] == [
        "fake-memory://fake-task-memory/12:task:demo-v0/v1",
        "fake-memory://fake-task-memory/12:task:demo-v0/v2",
    ]
    assert task_records[1].parent_state_refs == ["fake-memory://fake-task-memory/12:task:demo-v0/v1"]


def test_replay_memory_trace_from_clean_native_mem0_lab_validates_backend_scoped_refs(tmp_path: Path):
    lab_root = tmp_path / "demo-v1-lab"

    run_clean_demo(Path("configs/demo_v1_ci.yaml"), lab_root)
    report = replay_memory_trace(lab_root, task_id="demo-v1")

    assert report.ok is True
    assert report.issues == []
    assert [(record.backend_id, record.memory_scope, record.memory_scope_id) for record in report.records] == [
        ("mem0-agent-memory", "agent", "agent:solver"),
        ("mem0-task-memory", "task", "task:demo-v1"),
    ]
    assert all(record.pre_state_ref and record.pre_state_ref.startswith("method://mem0/") for record in report.records)
    assert all(
        record.update_state_ref and record.update_state_ref.startswith("method://mem0/")
        for record in report.records
    )
    assert all(record.previous_state_ref in record.parent_state_refs for record in report.records)


def test_replay_memory_trace_reports_missing_backend_state_record(tmp_path: Path):
    lab_root = tmp_path / "demo-v1-lab"

    run_clean_demo(Path("configs/demo_v1_ci.yaml"), lab_root)
    (lab_root / "registries" / "backend_state" / "states.jsonl").write_text("", encoding="utf-8")

    report = replay_memory_trace(lab_root, task_id="demo-v1")

    assert report.ok is False
    assert any("missing BackendStateRecord" in issue for issue in report.issues)


def test_replay_memory_trace_keeps_same_scope_id_lineage_separate_by_memory_scope(tmp_path: Path):
    lab_root = tmp_path / "scope-collision-lab"
    resolver = LabResolver(lab_root)
    trajectory_registry = resolver.trajectory_registry()
    backend_state_registry = resolver.backend_state_registry()

    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref="memory://agent/v1",
            backend_id="agent-memory",
            backend_type="memory",
            created_from_task_id="scope-collision",
            created_from_run_ref="subagent-1",
            parent_state_refs=["memory://agent/v0"],
            metadata={"memory_scope": "agent", "memory_scope_id": "shared"},
        )
    )
    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref="memory://task/v1",
            backend_id="task-memory",
            backend_type="memory",
            created_from_task_id="scope-collision",
            created_from_run_ref="subagent-1",
            parent_state_refs=["memory://task/v0"],
            metadata={"memory_scope": "task", "memory_scope_id": "shared"},
        )
    )
    trajectory_registry.save_subagent_run(
        SubagentRunRecord(
            run_ref="subagent-1",
            task_id="scope-collision",
            task_origin=TaskOrigin.HUMAN,
            task_purpose=TaskPurpose.REGRESSION,
            stage_index=0,
            role="solver",
            instruction="Exercise replay lineage.",
            retrieval_request=RetrievalRequest(task_id="scope-collision", role="solver", query="prior work"),
            memory_bundle=MemoryBundle(backend_id="combined-memory"),
            skill_bundle=SkillBundle(backend_id="skill-local"),
            prompt_messages=[Message(role="user", content="Solve it.")],
            llm_backend_id="llm-api",
            metadata={
                "agent_memory_bundle": {
                    "backend_id": "agent-memory",
                    "state_ref": "memory://agent/v0",
                    "metadata": {"memory_scope": "agent", "memory_scope_id": "shared"},
                },
                "agent_memory_update_result": {
                    "status": "updated",
                    "state_ref": "memory://agent/v1",
                    "previous_state_ref": "memory://agent/v0",
                    "metadata": {"memory_scope": "agent", "memory_scope_id": "shared"},
                },
                "task_memory_bundle": {
                    "backend_id": "task-memory",
                    "state_ref": "memory://task/v0",
                    "metadata": {"memory_scope": "task", "memory_scope_id": "shared"},
                },
                "task_memory_update_result": {
                    "status": "updated",
                    "state_ref": "memory://task/v1",
                    "previous_state_ref": "memory://task/v0",
                    "metadata": {"memory_scope": "task", "memory_scope_id": "shared"},
                },
            },
        )
    )

    report = replay_memory_trace(lab_root, task_id="scope-collision")

    assert report.ok is True
    assert report.issues == []


def test_replay_memory_trace_resolves_backend_bound_native_mem0_refs(tmp_path: Path):
    lab_root = tmp_path / "backend-bound-mem0-lab"
    resolver = LabResolver(lab_root)
    trajectory_registry = resolver.trajectory_registry()
    backend_state_registry = resolver.backend_state_registry()
    first_ref = "method://mem0/8:memory-a/5:agent/12:agent:solver/v1"
    second_ref = "method://mem0/8:memory-b/5:agent/12:agent:solver/v1"

    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref=first_ref,
            backend_id="memory-a",
            backend_type="memory",
            created_from_task_id="mem0-collision",
            created_from_run_ref="subagent-1",
            parent_state_refs=["method://mem0/8:memory-a/5:agent/12:agent:solver/v0"],
            metadata={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )
    backend_state_registry.register_candidate(
        BackendStateRecord(
            state_ref=second_ref,
            backend_id="memory-b",
            backend_type="memory",
            created_from_task_id="mem0-collision",
            created_from_run_ref="subagent-1",
            parent_state_refs=["method://mem0/8:memory-b/5:agent/12:agent:solver/v0"],
            metadata={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )
    trajectory_registry.save_subagent_run(
        SubagentRunRecord(
            run_ref="subagent-1",
            task_id="mem0-collision",
            task_origin=TaskOrigin.HUMAN,
            task_purpose=TaskPurpose.REGRESSION,
            stage_index=0,
            role="solver",
            instruction="Exercise backend-bound mem0 replay.",
            retrieval_request=RetrievalRequest(task_id="mem0-collision", role="solver", query="prior work"),
            memory_bundle=MemoryBundle(backend_id="memory-a"),
            skill_bundle=SkillBundle(backend_id="skill-local"),
            prompt_messages=[Message(role="user", content="Solve it.")],
            llm_backend_id="llm-api",
            metadata={
                "agent_memory_bundle": {
                    "backend_id": "memory-a",
                    "state_ref": "method://mem0/8:memory-a/5:agent/12:agent:solver/v0",
                    "metadata": {"memory_scope": "agent", "memory_scope_id": "agent:solver"},
                },
                "agent_memory_update_result": {
                    "status": "updated",
                    "state_ref": first_ref,
                    "previous_state_ref": "method://mem0/8:memory-a/5:agent/12:agent:solver/v0",
                    "metadata": {"memory_scope": "agent", "memory_scope_id": "agent:solver"},
                },
                "task_memory_bundle": {
                    "backend_id": "memory-b",
                    "state_ref": second_ref,
                    "metadata": {"memory_scope": "task", "memory_scope_id": "task:mem0-collision"},
                },
                "task_memory_update_result": {
                    "status": "skipped",
                    "state_ref": second_ref,
                    "previous_state_ref": second_ref,
                    "metadata": {"memory_scope": "task", "memory_scope_id": "task:mem0-collision"},
                },
            },
        )
    )

    report = replay_memory_trace(lab_root, task_id="mem0-collision")

    assert report.ok is True
    assert report.issues == []
    assert len(backend_state_registry.list_states()) == 2
