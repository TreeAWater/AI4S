from pathlib import Path

from evolab.backends.evolution import FakeSAGETrainer
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import (
    LLMEvolutionMode,
    LLMEvolutionRequest,
    LLMEvolutionResult,
)
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.runtime.evolution_executor import EvolutionExecutor


def _request(tmp_path: Path, **overrides) -> LLMEvolutionRequest:
    data = {
        "mode": LLMEvolutionMode.BASICS,
        "backend_id": "fake-llm",
        "previous_state_ref": "state-0",
        "artifact_root_uri": str(tmp_path / "artifacts"),
        "trigger_trajectory_ref": "subagent-1",
    }
    data.update(overrides)
    return LLMEvolutionRequest(**data)


def test_evolution_executor_promotes_valid_trainer_result(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path / "states")
    executor = EvolutionExecutor(registry, worker_id="worker-1")
    request = _request(tmp_path)

    outcome = executor.run(
        request=request,
        trainer=FakeSAGETrainer(),
        run_ref="evo-1",
        task_id="task-1",
    )

    assert outcome.result.status == "promoted_candidate"
    assert outcome.promoted is True
    assert outcome.promotion_errors == []
    assert registry.resolve_active_state("fake-llm") == outcome.result.new_state_ref
    state = registry.get_state(outcome.result.new_state_ref)
    assert state is not None
    assert state.created_from_task_id == "task-1"
    assert state.created_from_run_ref == "evo-1"
    assert state.metadata["trainer_id"] == "fake_sage"


def test_evolution_executor_converts_guard_failure_to_failed_result_without_promotion(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path / "states")
    executor = EvolutionExecutor(registry, worker_id="worker-1")
    request = _request(tmp_path)
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
        lora_role="solver",
        artifact_refs=[
            ArtifactRef(uri=str(tmp_path / "outside" / "adapter.json"), type="model_adapter")
        ],
    )

    class InvalidTrainer:
        trainer_id = "invalid"

        def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
            return result

    outcome = executor.run(
        request=request,
        trainer=InvalidTrainer(),
        run_ref="evo-1",
    )

    assert outcome.result.status == "failed"
    assert outcome.promoted is False
    assert "no artifact under artifact_root_uri" in outcome.promotion_errors
    assert outcome.result.metadata["promotion_errors"] == outcome.promotion_errors
    assert registry.resolve_active_state("fake-llm") is None
