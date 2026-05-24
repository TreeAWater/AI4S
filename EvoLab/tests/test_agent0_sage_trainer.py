import json
from pathlib import Path

from evolab.backends.trainers import Agent0SAGETrainer, LLMTrainer
from evolab.backends.trainers.agent0_sage import (
    Agent0SAGECriticResult,
    Agent0SAGEPlan,
    Agent0SAGEProposal,
    Agent0SAGERolloutResult,
)
from evolab.contracts.common import ArtifactRef, EvolutionBudget
from evolab.contracts.evolution import (
    LLMEvolutionMode,
    LLMEvolutionRequest,
    LLMEvolutionResult,
    StandardEvolutionMetrics,
)
from evolab.contracts.task import TaskOrigin, TaskPurpose


class _RecordingSolverTrainer(LLMTrainer):
    trainer_id = "recording_solver"

    def __init__(self, result: LLMEvolutionResult | None = None) -> None:
        self.requests: list[LLMEvolutionRequest] = []
        self.result = result or LLMEvolutionResult(
            status="not_recommended",
            recommend_for_promotion=False,
            metadata={"reason": "recorded"},
        )

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        self.requests.append(request)
        return self.result


def test_agent0_sage_trainer_is_exported_llm_trainer():
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="task-1",
            goal="Solve a frontier problem.",
            human_anchor_task_refs=["human-1"],
            relation_rationale="Exercises a related skill.",
            expected_transfer="Improves solver robustness.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Reason step by step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.8,
            verifier_passed=True,
        ),
        solver_trainer=_RecordingSolverTrainer(),
    )

    assert isinstance(trainer, LLMTrainer)
    assert trainer.trainer_id == "agent0_sage"


def _evolution_request(tmp_path: Path, **overrides) -> LLMEvolutionRequest:
    data = {
        "mode": LLMEvolutionMode.BASICS,
        "backend_id": "solver-backend",
        "previous_state_ref": "state-0",
        "artifact_root_uri": str(tmp_path / "artifacts"),
        "budget": EvolutionBudget(max_train_samples=1),
    }
    data.update(overrides)
    return LLMEvolutionRequest(**data)


def test_generated_task_uses_shared_task_request_with_proposer_provenance(tmp_path: Path):
    rollout_calls = []

    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="generated-1",
            goal="Create a tool-aware arithmetic task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="A harder variant of the human task.",
            expected_transfer="Improves tool-aware arithmetic.",
            metadata={"required_tools": ["python"], "difficulty": 0.6},
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Use Python.", "Verify result."]),
        rollout_runner=lambda task_request, plan, context: rollout_calls.append(task_request)
        or Agent0SAGERolloutResult(status="completed", output="42"),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.9,
            verifier_passed=True,
        ),
        solver_trainer=_RecordingSolverTrainer(),
    )

    result = trainer.train(_evolution_request(tmp_path))

    generated = rollout_calls[0]
    assert generated.origin == TaskOrigin.PROPOSER
    assert generated.purpose == TaskPurpose.TRAINING_ROLLOUT
    assert generated.producer_ref == "agent0_sage:agent0_sage"
    assert generated.round_id == "round-1"
    assert generated.proposed_task_relation is not None
    assert generated.proposed_task_relation.human_anchor_task_refs == ["human-task-1"]
    assert generated.metadata["candidate_id"] == "candidate-1"
    assert generated.metadata["required_tools"] == ["python"]
    assert result.status in {"not_recommended", "promoted_candidate"}


def test_invalid_candidate_without_human_anchor_is_rejected_before_rollout(tmp_path: Path):
    rollout_calls = []

    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="generated-1",
            goal="Unanchored generated task.",
            human_anchor_task_refs=[],
            human_anchor_trajectory_refs=[],
            relation_rationale="No anchor.",
            expected_transfer="No transfer claim.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan."]),
        rollout_runner=lambda task_request, plan, context: rollout_calls.append(task_request)
        or Agent0SAGERolloutResult(status="completed", output="answer"),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=1.0,
            verifier_passed=True,
        ),
        solver_trainer=_RecordingSolverTrainer(),
    )

    result = trainer.train(_evolution_request(tmp_path))

    assert rollout_calls == []
    assert result.status == "skipped"
    assert result.metadata["accepted_count"] == 0
    assert result.metadata["rejected_count"] == 1


def test_accepted_samples_are_written_and_nested_solver_trainer_receives_manifest(tmp_path: Path):
    solver_trainer = _RecordingSolverTrainer()
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="generated-1",
            goal="Solve generated task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="Curriculum step.",
            expected_transfer="Improves solver.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
            trajectory_ref="traj-1",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.9,
            verifier_passed=True,
        ),
        solver_trainer=solver_trainer,
    )

    result = trainer.train(_evolution_request(tmp_path))

    artifact_root = tmp_path / "artifacts"
    samples_path = artifact_root / "agent0_sage_samples.jsonl"
    rejections_path = artifact_root / "agent0_sage_rejections.jsonl"
    summary_path = artifact_root / "agent0_sage_summary.json"
    assert samples_path.exists()
    assert rejections_path.exists()
    assert summary_path.exists()
    sample = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["candidate_id"] == "candidate-1"
    assert sample["task_request"]["origin"] == "proposer"
    assert len(solver_trainer.requests) == 1
    nested_request = solver_trainer.requests[0]
    assert nested_request.metadata["agent0_sage_manifest_uri"] == str(samples_path)
    assert nested_request.metadata["accepted_count"] == 1
    assert nested_request.artifact_root_uri == str(artifact_root / "solver")
    assert any(artifact.uri == str(samples_path) for artifact in result.artifact_refs)


def test_outer_result_preserves_nested_solver_promotion_fields(tmp_path: Path):
    nested = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="local-trainable://solver-backend/state/solver-state-1",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.88),
        artifact_refs=[
            ArtifactRef(
                uri=str(tmp_path / "artifacts" / "solver" / "adapter.bin"),
                type="model_adapter",
            )
        ],
        metadata={"nested": "ok"},
    )
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="generated-1",
            goal="Solve generated task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="Curriculum step.",
            expected_transfer="Improves solver.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.9,
            verifier_passed=True,
        ),
        solver_trainer=_RecordingSolverTrainer(nested),
    )

    result = trainer.train(_evolution_request(tmp_path))

    assert result.status == "promoted_candidate"
    assert result.recommend_for_promotion is True
    assert result.new_state_ref == "local-trainable://solver-backend/state/solver-state-1"
    assert result.lora_role == "solver"
    assert result.standard_metrics.eval_score_after == 0.88
    assert result.metadata["agent0_sage"]["accepted_count"] == 1
    assert result.metadata["nested"] == "ok"


def test_critic_threshold_rejects_low_score_and_writes_rejection(tmp_path: Path):
    solver_trainer = _RecordingSolverTrainer()
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-low",
            task_id="generated-low",
            goal="Low quality task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="Curriculum step.",
            expected_transfer="Weak transfer.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.2,
            verifier_passed=True,
            reason="below frontier",
        ),
        solver_trainer=solver_trainer,
        acceptance_threshold=0.5,
    )

    result = trainer.train(_evolution_request(tmp_path))

    assert result.status == "skipped"
    assert solver_trainer.requests == []
    rejections = (tmp_path / "artifacts" / "agent0_sage_rejections.jsonl").read_text(encoding="utf-8")
    assert "candidate-low" in rejections
    assert "below frontier" in rejections


def test_low_score_without_critic_reason_records_threshold_reason(tmp_path: Path):
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-low",
            task_id="generated-low",
            goal="Low quality task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="Curriculum step.",
            expected_transfer="Weak transfer.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.2,
            verifier_passed=True,
        ),
        solver_trainer=_RecordingSolverTrainer(),
        acceptance_threshold=0.5,
    )

    trainer.train(_evolution_request(tmp_path))

    rejections = (tmp_path / "artifacts" / "agent0_sage_rejections.jsonl").read_text(encoding="utf-8")
    assert "critic score below acceptance threshold" in rejections


class _RaisingSolverTrainer(LLMTrainer):
    trainer_id = "raising_solver"

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        raise RuntimeError("solver trainer failed")


def test_nested_solver_trainer_exception_returns_failed_result_with_artifacts(tmp_path: Path):
    trainer = Agent0SAGETrainer(
        proposer=lambda context: Agent0SAGEProposal(
            candidate_id="candidate-1",
            task_id="generated-1",
            goal="Solve generated task.",
            human_anchor_task_refs=["human-task-1"],
            relation_rationale="Curriculum step.",
            expected_transfer="Improves solver.",
        ),
        planner=lambda task_request, context: Agent0SAGEPlan(steps=["Plan step."]),
        rollout_runner=lambda task_request, plan, context: Agent0SAGERolloutResult(
            status="completed",
            output="answer",
        ),
        critic=lambda task_request, plan, rollout, context: Agent0SAGECriticResult(
            accepted=True,
            score=0.9,
            verifier_passed=True,
        ),
        solver_trainer=_RaisingSolverTrainer(),
    )

    result = trainer.train(_evolution_request(tmp_path))

    assert result.status == "failed"
    assert result.metadata["error"] == "solver trainer failed"
    assert any(artifact.metadata["role"] == "accepted_samples" for artifact in result.artifact_refs)
