from pathlib import Path

import pytest

import evolab.backends.evolution as evolution
from evolab.backends.evolution import FakeAgent0Trainer, FakeEvolutionBackend, FakeSAGETrainer
from evolab.backends.trainers import LLMTrainer
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest


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


def test_fake_sage_trainer_promotes_candidate_with_artifact_under_root(tmp_path: Path):
    backend = FakeSAGETrainer()
    request = _request(tmp_path)

    result = backend.train(request)

    assert isinstance(backend, LLMTrainer)
    assert backend.requests == [request]
    assert result.status == "promoted_candidate"
    assert result.recommend_for_promotion is True
    assert result.new_state_ref == "fake-sage://fake-llm/state/1"
    assert result.lora_role == "solver"
    assert result.standard_metrics.eval_score_after == pytest.approx(0.75)
    assert result.metadata["trainer"] == "sage"
    assert len(result.artifact_refs) == 1
    artifact_path = Path(result.artifact_refs[0].uri)
    assert artifact_path.is_file()
    assert artifact_path.parent == tmp_path / "artifacts"


@pytest.mark.parametrize(
    ("scenario", "expected_status", "metadata_key"),
    [
        ("skipped", "skipped", "reason"),
        ("failed", "failed", "error"),
        ("not_recommended", "not_recommended", "reason"),
    ],
)
def test_fake_sage_trainer_non_promotion_scenarios(
    tmp_path: Path,
    scenario: str,
    expected_status: str,
    metadata_key: str,
):
    backend = FakeSAGETrainer(scenario=scenario)

    result = backend.train(_request(tmp_path))

    assert result.status == expected_status
    assert result.recommend_for_promotion is False
    assert result.new_state_ref is None
    assert metadata_key in result.metadata


def test_fake_sage_trainer_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown fake evolution scenario"):
        FakeSAGETrainer(scenario="unknown")


def test_fake_evolution_backend_is_compatible_name_for_fake_sage_trainer(tmp_path: Path):
    backend = FakeEvolutionBackend()

    result = backend.train(_request(tmp_path))

    assert isinstance(backend, FakeSAGETrainer)
    assert "FakeEvolutionBackend" in evolution.__all__
    assert result.status == "promoted_candidate"


def test_fake_agent0_sage_runtime_alias_is_not_exported():
    assert "FakeAgent0SAGERuntime" not in evolution.__all__
    assert not hasattr(evolution, "FakeAgent0SAGERuntime")


def test_fake_sage_trainer_is_standalone_parameter_update_trainer(tmp_path: Path):
    trainer = FakeSAGETrainer()
    request = _request(tmp_path)

    result = trainer.train(request)

    assert isinstance(trainer, LLMTrainer)
    assert trainer.trainer_id == "fake_sage"
    assert result.status == "promoted_candidate"
    assert result.recommend_for_promotion is True
    assert result.metadata["trainer"] == "sage"
    assert result.artifact_refs[0].metadata["backend"] == "fake_sage"
    assert (tmp_path / "artifacts" / "adapter.json").is_file()


def test_fake_agent0_trainer_is_outer_loop_and_delegates_to_sage(tmp_path: Path):
    sage = FakeSAGETrainer()
    trainer = FakeAgent0Trainer(sage_trainer=sage)
    request = _request(tmp_path)

    result = trainer.train(request)

    assert isinstance(trainer, LLMTrainer)
    assert trainer.trainer_id == "fake_agent0"
    assert trainer.requests == [request]
    assert sage.requests
    assert sage.requests[0].metadata["source_trainer_id"] == "fake_agent0"
    assert result.status == "promoted_candidate"
    assert result.metadata["agent0_sage"]["trainer_id"] == "fake_agent0"
    assert result.metadata["trainer"] == "sage"
    assert (tmp_path / "artifacts" / "agent0_sage_samples.jsonl").is_file()
    assert (tmp_path / "artifacts" / "solver" / "adapter.json").is_file()


def test_fake_agent0_trainer_generates_samples_and_delegates_solver_promotion(tmp_path: Path):
    runtime = FakeAgent0Trainer()
    request = _request(tmp_path)

    result = runtime.train(request)

    assert runtime.requests == [request]
    assert result.status == "promoted_candidate"
    assert result.recommend_for_promotion is True
    assert result.lora_role == "solver"
    assert result.metadata["agent0_sage"]["trainer_id"] == "fake_agent0"
    assert result.metadata["agent0_sage"]["accepted_count"] == 1
    artifact_roles = {artifact.metadata.get("role") for artifact in result.artifact_refs}
    assert {"accepted_samples", "rejections", "summary"} <= artifact_roles
    assert (tmp_path / "artifacts" / "agent0_sage_samples.jsonl").is_file()
    assert (tmp_path / "artifacts" / "solver" / "adapter.json").is_file()
