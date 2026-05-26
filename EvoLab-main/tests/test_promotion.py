import inspect
import json
from pathlib import Path

from evolab.backends.trainers import BlankTrainer, LLMTrainer
from evolab.backends.trainers.base import LLMTrainer as BaseLLMTrainer
from evolab.backends.trainers.blank import BlankTrainer as PackageBlankTrainer
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import (
    InstanceSnapshot,
    LLMEvolutionMode,
    LLMEvolutionRequest,
    LLMEvolutionResult,
    StandardEvolutionMetrics,
)
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.evolve_worker import EvolveWorker
from evolab.runtime.promotion import expected_roles_for, validate_promotion


class FakeTrainer:
    trainer_id = "fake"

    def __init__(self, result: LLMEvolutionResult):
        self.result = result
        self.requests: list[LLMEvolutionRequest] = []

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        self.requests.append(request)
        return self.result


def test_blank_trainer_inherits_from_llm_trainer():
    trainer = BlankTrainer()

    assert issubclass(BlankTrainer, LLMTrainer)
    assert isinstance(trainer, LLMTrainer)
    assert inspect.isabstract(LLMTrainer)
    assert LLMTrainer.__abstractmethods__ == {"train"}
    assert LLMTrainer is BaseLLMTrainer
    assert BlankTrainer is PackageBlankTrainer


def _request(tmp_path: Path, **overrides) -> LLMEvolutionRequest:
    data = {
        "mode": LLMEvolutionMode.BASICS,
        "backend_id": "fake",
        "previous_state_ref": "state-0",
        "artifact_root_uri": str(tmp_path / "artifacts"),
    }
    data.update(overrides)
    return LLMEvolutionRequest(**data)


def _write_request(tmp_path: Path, request: LLMEvolutionRequest) -> Path:
    path = tmp_path / "request.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    return path


def _enqueue(tmp_path: Path, request: LLMEvolutionRequest) -> FileWorkQueue:
    queue = FileWorkQueue(tmp_path / "queue")
    queue.enqueue("evolve-1", {"request_payload_uri": str(_write_request(tmp_path, request))})
    return queue


def _worker(
    tmp_path: Path,
    queue: FileWorkQueue,
    trainers: dict[str, object],
    registry: FileBackendStateRegistry,
    *,
    worker_id: str = "worker-1",
    trajectory_registry: FileTrajectoryRegistry | None = None,
) -> EvolveWorker:
    return EvolveWorker(
        queue,
        trainers,  # type: ignore[arg-type]
        registry,
        worker_id=worker_id,
        trajectory_registry=trajectory_registry or FileTrajectoryRegistry(tmp_path / "trajectory"),
    )


def test_expected_roles_for_mode():
    assert expected_roles_for(LLMEvolutionMode.BASICS) == {"solver"}
    assert expected_roles_for(LLMEvolutionMode.CONSOLIDATION) == {"skill_distilled"}


def test_validate_promotion_rejects_path_prefix_false_positive(tmp_path: Path):
    request = _request(tmp_path, artifact_root_uri=str(tmp_path / "evo"))
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
        artifact_refs=[ArtifactRef(uri=str(tmp_path / "evo2" / "adapter.bin"), type="model_adapter")],
    )

    assert "no artifact under artifact_root_uri" in validate_promotion(result, request)


def test_validate_promotion_accepts_remote_artifact_under_root(tmp_path: Path):
    request = _request(tmp_path, artifact_root_uri="s3://bucket/evo/root")
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
        artifact_refs=[ArtifactRef(uri="s3://bucket/evo/root/adapter.bin", type="model_adapter")],
    )

    assert validate_promotion(result, request) == []


def test_validate_promotion_accepts_remote_artifact_under_bucket_root(tmp_path: Path):
    request = _request(tmp_path, artifact_root_uri="s3://bucket")
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
        artifact_refs=[ArtifactRef(uri="s3://bucket/adapter.bin", type="model_adapter")],
    )

    assert validate_promotion(result, request) == []


def test_validate_promotion_rejects_remote_artifact_parent_traversal(tmp_path: Path):
    request = _request(tmp_path, artifact_root_uri="s3://bucket/evo/root")
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
        artifact_refs=[ArtifactRef(uri="s3://bucket/evo/root/../outside.bin", type="model_adapter")],
    )

    assert "no artifact under artifact_root_uri" in validate_promotion(result, request)


def test_validate_promotion_rejects_remote_artifact_percent_encoded_traversal(tmp_path: Path):
    request = _request(tmp_path, artifact_root_uri="s3://bucket/evo/root")
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
        artifact_refs=[ArtifactRef(uri="s3://bucket/evo/root/%2e%2e/outside.bin", type="model_adapter")],
    )

    assert "no artifact under artifact_root_uri" in validate_promotion(result, request)


def test_worker_marks_failed_and_does_not_promote_invalid_promotion(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
            artifact_refs=[
                ArtifactRef(uri=str(tmp_path / "artifacts" / "adapter.bin"), type="model_adapter")
            ],
        )
    )

    _worker(tmp_path, queue, {"fake": trainer}, registry).run_once()

    failed_files = list((tmp_path / "queue" / "failed").glob("*.json"))
    assert len(failed_files) == 1
    assert "eval_score_after" in failed_files[0].read_text(encoding="utf-8")
    assert registry.resolve_active_state("fake") is None


def test_worker_marks_failed_when_trainer_returns_failed_result(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="failed",
            recommend_for_promotion=False,
            metadata={"error": "optimizer diverged"},
        )
    )

    _worker(tmp_path, queue, {"fake": trainer}, registry).run_once()

    failed_files = list((tmp_path / "queue" / "failed").glob("*.json"))
    assert len(failed_files) == 1
    assert "optimizer diverged" in failed_files[0].read_text(encoding="utf-8")
    assert not list((tmp_path / "queue" / "done").glob("*.json"))
    assert registry.resolve_active_state("fake") is None


def test_worker_records_failed_result_for_invalid_non_promoted_result(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult.model_construct(
            schema_version="v1",
            status="not_recommended",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
            artifact_refs=[
                ArtifactRef(uri=str(tmp_path / "artifacts" / "adapter.bin"), type="model_adapter")
            ],
            metadata={},
        )
    )

    _worker(tmp_path, queue, {"fake": trainer}, registry).run_once()

    assert list((tmp_path / "queue" / "failed").glob("*.json"))
    record = FileTrajectoryRegistry(tmp_path / "trajectory").list_evolution_runs()[0]
    assert record.result_status == "failed"
    assert record.result.metadata["error"] == "evolution result validation failed"
    assert record.result.metadata["original_result"]["status"] == "not_recommended"
    assert registry.resolve_active_state("fake") is None


def test_worker_marks_skipped_when_trainer_returns_skipped_result(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="skipped",
            recommend_for_promotion=False,
            metadata={"reason": "not enough samples"},
        )
    )

    _worker(tmp_path, queue, {"fake": trainer}, registry).run_once()

    skipped_files = list((tmp_path / "queue" / "skipped").glob("*.json"))
    assert len(skipped_files) == 1
    assert "not enough samples" in skipped_files[0].read_text(encoding="utf-8")
    assert not list((tmp_path / "queue" / "done").glob("*.json"))
    assert registry.resolve_active_state("fake") is None


def test_worker_promotes_valid_candidate_and_marks_done(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
            artifact_refs=[
                ArtifactRef(uri=str(tmp_path / "artifacts" / "adapter.bin"), type="model_adapter")
            ],
        )
    )

    _worker(tmp_path, queue, {"fake": trainer}, registry).run_once()

    assert list((tmp_path / "queue" / "done").glob("*.json"))
    assert registry.resolve_active_state("fake") == "state-1"
    assert trainer.requests == [request]


def test_worker_fails_before_training_when_trajectory_registry_is_missing(tmp_path: Path):
    request = _request(tmp_path)
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
            artifact_refs=[
                ArtifactRef(uri=str(tmp_path / "artifacts" / "adapter.bin"), type="model_adapter")
            ],
        )
    )

    EvolveWorker(queue, {"fake": trainer}, registry, worker_id="worker-1").run_once()

    failed_files = list((tmp_path / "queue" / "failed").glob("*.json"))
    assert len(failed_files) == 1
    assert "trajectory registry is required" in failed_files[0].read_text(encoding="utf-8")
    assert trainer.requests == []
    assert registry.resolve_active_state("fake") is None


def test_worker_records_snapshot_refs_and_run_ledger_events(tmp_path: Path):
    request = _request(
        tmp_path,
        instance_snapshots=[
            InstanceSnapshot(snapshot_ref="toolset-before"),
            InstanceSnapshot(snapshot_ref="skill-before"),
        ],
    )
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    trainer = FakeTrainer(
        LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(eval_score_after=0.75),
            artifact_refs=[
                ArtifactRef(uri=str(tmp_path / "artifacts" / "adapter.bin"), type="model_adapter")
            ],
            metadata={"output_snapshot_refs": ["toolset-after", "skill-after"]},
        )
    )

    _worker(
        tmp_path,
        queue,
        {"fake": trainer},
        registry,
        worker_id="worker-1",
        trajectory_registry=trajectory_registry,
    ).run_once()

    record = trajectory_registry.list_evolution_runs()[0]
    assert record.input_snapshot_refs == ["toolset-before", "skill-before"]
    assert record.consumed_instance_snapshot_refs == ["toolset-before", "skill-before"]
    assert record.output_snapshot_refs == ["toolset-after", "skill-after"]
    events_path = Path(request.artifact_root_uri) / "events.jsonl"
    event_types = [
        json.loads(line)["event_type"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert event_types == [
        "run_started",
        "trainer_invoked",
        "trainer_completed",
        "evolution_record_saved",
        "promotion_decided",
        "run_finished",
    ]
    run_spec = json.loads((Path(request.artifact_root_uri) / "run_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(request.artifact_root_uri) / "manifest.latest.json").read_text(encoding="utf-8"))
    assert run_spec["input_snapshot_refs"] == ["toolset-before", "skill-before"]
    assert manifest["result_status"] == "promoted_candidate"
    assert manifest["promoted"] is True


def test_worker_skips_missing_trainer(tmp_path: Path):
    request = _request(tmp_path, backend_id="missing")
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")

    _worker(tmp_path, queue, {}, registry).run_once()

    skipped_files = list((tmp_path / "queue" / "skipped").glob("*.json"))
    assert len(skipped_files) == 1
    assert "no trainer for backend_id" in skipped_files[0].read_text(encoding="utf-8")
    assert registry.resolve_active_state("missing") is None


def test_worker_skips_blank_trainer(tmp_path: Path):
    request = _request(tmp_path, backend_id="blank")
    queue = _enqueue(tmp_path, request)
    registry = FileBackendStateRegistry(tmp_path / "states")

    _worker(tmp_path, queue, {"blank": BlankTrainer()}, registry).run_once()

    skipped_files = list((tmp_path / "queue" / "skipped").glob("*.json"))
    assert len(skipped_files) == 1
    assert "training algorithm is not implemented" in skipped_files[0].read_text(encoding="utf-8")
