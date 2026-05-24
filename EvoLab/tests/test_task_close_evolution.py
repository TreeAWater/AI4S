import json
from pathlib import Path
from typing import Callable

import pytest

from evolab.backends.llm import LocalTrainableLLMBackend
from evolab.backends.evolution import FakeSAGETrainer
from evolab.backends.trainers import SFTTrainer, SFTTrainerConfig
from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.evolution import (
    LLMEvolutionMode,
    LLMEvolutionRequest,
    LLMEvolutionResult,
    StandardEvolutionMetrics,
)
from evolab.contracts.llm import LLMGenerationConfig
from evolab.contracts.records import EvolutionRunRecord, LLMCallRecord, SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.evolve_worker import EvolveWorker
from evolab.runtime.sft_exporter import SFTExportConfig
from evolab.runtime.task_close_evolution import TaskCloseEvolutionScheduler
from evolab.runtime.task_worker import TaskWorker


class FakeRuntime:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.requests: list[TaskRequest] = []

    def run(self, request: TaskRequest) -> dict[str, object]:
        self.requests.append(request)
        return dict(self.result)


class FactoryEvolutionBackend:
    trainer_id = "factory"

    def __init__(self, factory: Callable[[LLMEvolutionRequest], LLMEvolutionResult]):
        self.factory = factory
        self.requests: list[LLMEvolutionRequest] = []

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        self.requests.append(request)
        return self.factory(request)


class RaisingEvolutionSaveRegistry(FileTrajectoryRegistry):
    def save_evolution_run(self, record: EvolutionRunRecord) -> str:
        raise RuntimeError("evolution record save failed")


def _request(task_id: str = "task-1") -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Solve the task.",
    )


def _enqueue_task(layout: LabLayout, request: TaskRequest) -> None:
    path = layout.root / "requests" / f"{request.task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(request.model_dump_json(), encoding="utf-8")
    FileWorkQueue(layout.tasks_queue_dir).enqueue(
        "job-1",
        {"request_payload_uri": str(path)},
    )


def _subagent_run(
    *,
    run_ref: str = "subagent-1",
    task_id: str = "task-1",
    role: str = "solver",
    llm_backend_id: str = "fake-llm",
    llm_backend_state_ref: str | None = "state-0",
) -> SubagentRunRecord:
    return SubagentRunRecord(
        run_ref=run_ref,
        task_id=task_id,
        task_origin=TaskOrigin.HUMAN,
        task_purpose=TaskPurpose.SCIENCE,
        stage_index=0,
        role=role,
        instruction="Solve it.",
        retrieval_request=RetrievalRequest(task_id=task_id, role=role, query="prior work"),
        memory_bundle=MemoryBundle(backend_id="memory-local"),
        skill_bundle=SkillBundle(backend_id="skill-local"),
        prompt_messages=[Message(role="user", content="Solve it.")],
        output_messages=[Message(role="assistant", content="Done.")],
        llm_backend_id=llm_backend_id,
        llm_backend_state_ref=llm_backend_state_ref,
    )


def _seed_sft_final_call(registry: FileTrajectoryRegistry, *, backend_id: str = "local-llm") -> None:
    registry.save_subagent_run(
        _subagent_run(
            llm_backend_id=backend_id,
            llm_backend_state_ref="local-trainable://local-llm/state/base",
        )
    )
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-final",
            run_ref="subagent-1",
            backend_id=backend_id,
            model="local-teacher",
            input_messages=[Message(role="user", content="Solve it.")],
            output_messages=[
                Message(role="assistant", content="Done.", metadata={"action": "final_answer"})
            ],
            metadata={"runtime_stage": "subagent_flat", "action": "final_answer", "role": "solver"},
        )
    )


def test_task_worker_enqueues_task_close_evolution_before_marking_task_done(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    request = _request()
    _enqueue_task(layout, request)
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    backend_state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    trajectory_registry.save_subagent_run(_subagent_run())
    runtime = FakeRuntime({"task_id": request.task_id, "run_ref": "subagent-1"})
    evolution_backend = FakeSAGETrainer()
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_runtime=runtime,
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        evolution_backends={"fake-llm": evolution_backend},
    )
    worker.startup()

    result = worker.run_once()

    assert result is not None
    queued_jobs = sorted((layout.evolve_queue_dir / "queued").glob("*.json"))
    assert len(queued_jobs) == 1
    assert result["evolution_run_refs"] == [queued_jobs[0].stem]
    payload = json.loads(queued_jobs[0].read_text(encoding="utf-8"))
    evolution_request = LLMEvolutionRequest.model_validate_json(
        Path(payload["request_payload_uri"]).read_text(encoding="utf-8")
    )
    assert evolution_request.backend_id == "fake-llm"
    assert evolution_request.previous_state_ref == "state-0"
    assert evolution_request.trigger_trajectory_ref == "subagent-1"
    assert evolution_request.proposer_input_refs[0].ref_id == "subagent-1"
    assert evolution_request.metadata["trigger"] == "task_close"
    assert trajectory_registry.list_evolution_runs() == []
    assert backend_state_registry.resolve_active_state("fake-llm") is None
    assert sorted((layout.tasks_queue_dir / "done").glob("*.json"))
    assert not list((layout.tasks_queue_dir / "failed").glob("*.json"))


@pytest.mark.parametrize("scenario", ["skipped", "failed", "not_recommended"])
def test_task_close_evolution_records_non_promotions_without_state_update(
    tmp_path: Path,
    scenario: str,
):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    request = _request()
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    backend_state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    trajectory_registry.save_subagent_run(_subagent_run())
    scheduler = TaskCloseEvolutionScheduler(
        layout=layout,
        evolve_queue=FileWorkQueue(layout.evolve_queue_dir),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        worker_id="worker-1",
        evolution_backend_ids={"fake-llm"},
    )
    run_refs = scheduler.run(request, {"run_ref": "subagent-1"})

    EvolveWorker(
        FileWorkQueue(layout.evolve_queue_dir),
        {"fake-llm": FakeSAGETrainer(scenario=scenario)},
        backend_state_registry,
        worker_id="evolve-1",
        trajectory_registry=trajectory_registry,
    ).run_once()

    records = trajectory_registry.list_evolution_runs()
    assert [record.run_ref for record in records] == run_refs
    assert [record.result_status for record in records] == [scenario]
    assert backend_state_registry.resolve_active_state("fake-llm") is None
    assert backend_state_registry.list_states() == []


@pytest.mark.parametrize(
    ("factory", "expected_error"),
    [
        (
            lambda request: LLMEvolutionResult.model_construct(
                schema_version="v1",
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="",
                lora_role="solver",
                standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
                artifact_refs=[
                    ArtifactRef(
                        uri=str(Path(request.artifact_root_uri) / "adapter.json"),
                        type="model_adapter",
                    )
                ],
                metadata={},
            ),
            "new_state_ref",
        ),
        (
            lambda request: LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
                lora_role="solver",
                standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
                artifact_refs=[
                    ArtifactRef(
                        uri=str(Path(request.artifact_root_uri).parent / "outside" / "adapter.json"),
                        type="model_adapter",
                    )
                ],
            ),
            "artifact_root_uri",
        ),
        (
            lambda request: LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
                lora_role="solver",
                artifact_refs=[
                    ArtifactRef(
                        uri=str(Path(request.artifact_root_uri) / "adapter.json"),
                        type="model_adapter",
                    )
                ],
            ),
            "eval_score_after",
        ),
        (
            lambda request: LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
                lora_role="composed",
                standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
                artifact_refs=[
                    ArtifactRef(
                        uri=str(Path(request.artifact_root_uri) / "adapter.json"),
                        type="model_adapter",
                    )
                ],
            ),
            "lora_role",
        ),
    ],
)
def test_task_close_evolution_records_failed_guard_and_does_not_promote(
    tmp_path: Path,
    factory: Callable[[LLMEvolutionRequest], LLMEvolutionResult],
    expected_error: str,
):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    request = _request()
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    backend_state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    trajectory_registry.save_subagent_run(_subagent_run())
    scheduler = TaskCloseEvolutionScheduler(
        layout=layout,
        evolve_queue=FileWorkQueue(layout.evolve_queue_dir),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        worker_id="worker-1",
        evolution_backend_ids={"fake-llm"},
    )
    scheduler.run(request, {"run_ref": "subagent-1"})

    EvolveWorker(
        FileWorkQueue(layout.evolve_queue_dir),
        {"fake-llm": FactoryEvolutionBackend(factory)},
        backend_state_registry,
        worker_id="evolve-1",
        trajectory_registry=trajectory_registry,
    ).run_once()

    records = trajectory_registry.list_evolution_runs()
    assert len(records) == 1
    record = records[0]
    assert record.result_status == "failed"
    assert any(expected_error in error for error in record.result.metadata["promotion_errors"])
    assert record.result.metadata["original_result"]["status"] == "promoted_candidate"
    assert backend_state_registry.resolve_active_state("fake-llm") is None
    assert backend_state_registry.list_states() == []


def test_task_close_evolution_saves_record_before_promoting_state(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    request = _request()
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    backend_state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    trajectory_registry.save_subagent_run(_subagent_run())
    scheduler = TaskCloseEvolutionScheduler(
        layout=layout,
        evolve_queue=FileWorkQueue(layout.evolve_queue_dir),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        worker_id="worker-1",
        evolution_backend_ids={"fake-llm"},
    )
    scheduler.run(request, {"run_ref": "subagent-1"})

    EvolveWorker(
        FileWorkQueue(layout.evolve_queue_dir),
        {"fake-llm": FakeSAGETrainer()},
        backend_state_registry,
        worker_id="evolve-1",
        trajectory_registry=RaisingEvolutionSaveRegistry(layout.registries_dir / "trajectory"),
    ).run_once()

    assert backend_state_registry.resolve_active_state("fake-llm") is None
    assert backend_state_registry.list_states() == []
    failed_jobs = sorted((layout.evolve_queue_dir / "failed").glob("*.json"))
    assert len(failed_jobs) == 1
    assert "evolution record save failed" in failed_jobs[0].read_text(encoding="utf-8")


def test_evolve_worker_promotes_sft_state_that_local_trainable_can_instantiate(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    request = _request()
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    backend_state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    _seed_sft_final_call(trajectory_registry, backend_id="local-llm")
    scheduler = TaskCloseEvolutionScheduler(
        layout=layout,
        evolve_queue=FileWorkQueue(layout.evolve_queue_dir),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        worker_id="worker-1",
        evolution_backend_ids={"local-llm"},
    )
    scheduler.run(request, {"run_ref": "subagent-1"})
    trainer = SFTTrainer(
        trajectory_registry=trajectory_registry,
        config=SFTTrainerConfig(
            promote_dry_run=True,
            export=SFTExportConfig(source_run_refs=["subagent-1"]),
        ),
    )

    EvolveWorker(
        FileWorkQueue(layout.evolve_queue_dir),
        {"local-llm": trainer},
        backend_state_registry,
        worker_id="evolve-1",
        trajectory_registry=trajectory_registry,
    ).run_once()

    active_state_ref = backend_state_registry.resolve_active_state("local-llm")
    assert active_state_ref is not None
    assert active_state_ref.startswith("local-trainable://local-llm/state/")
    backend = LocalTrainableLLMBackend(
        backend_id="local-llm",
        state_registry=backend_state_registry,
    )
    runtime = backend.instantiate(active_state_ref)
    response = runtime.generate(
        messages=[Message(role="user", content="Use promoted state.")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="local"),
    )

    assert response.action.action == "final_answer"
    assert response.raw_response["state_ref"] == active_state_ref
    assert response.raw_response["state_manifest"]["created_by_trainer"] == "sft"
    assert response.raw_response["state_manifest"]["parent_state_ref"] == "local-trainable://local-llm/state/base"
