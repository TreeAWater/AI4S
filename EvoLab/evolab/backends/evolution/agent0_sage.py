from __future__ import annotations

from typing import Any

from evolab.backends.evolution.fake import FakeEvolutionScenario, FakeSAGETrainer
from evolab.backends.trainers.agent0_sage import (
    Agent0SAGECriticResult,
    Agent0SAGEPlan,
    Agent0SAGEProposal,
    Agent0SAGERolloutResult,
    Agent0SAGETrainer,
)
from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.task import TaskRequest


class FakeAgent0Trainer(LLMTrainer):
    trainer_id = "fake_agent0"

    def __init__(
        self,
        *,
        sage_trainer: LLMTrainer | None = None,
        solver_scenario: FakeEvolutionScenario = "promoted_candidate",
        max_rounds: int = 1,
        trainer_id: str | None = None,
    ) -> None:
        if trainer_id is not None:
            self.trainer_id = trainer_id
        self.requests: list[LLMEvolutionRequest] = []
        self.sage_trainer = sage_trainer or FakeSAGETrainer(scenario=solver_scenario)
        self._trainer = Agent0SAGETrainer(
            proposer=self._propose,
            planner=self._plan,
            rollout_runner=self._rollout,
            critic=self._critic,
            solver_trainer=self.sage_trainer,
            trainer_id=self.trainer_id,
            max_rounds=max_rounds,
        )

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        self.requests.append(request)
        return self._trainer.train(request)

    def _propose(self, context: dict[str, Any]) -> Agent0SAGEProposal:
        request: LLMEvolutionRequest = context["request"]
        round_id = str(context["round_id"])
        trigger_ref = request.trigger_trajectory_ref or "cold-start"
        return Agent0SAGEProposal(
            candidate_id=f"{round_id}-candidate",
            task_id=f"agent0-sage-{round_id}",
            goal=f"Generate a training rollout derived from {trigger_ref}.",
            human_anchor_trajectory_refs=[trigger_ref],
            relation_rationale="Deterministic V0 fake Agent0 curriculum step.",
            expected_transfer="Exercises the same solver behavior before SAGE state promotion.",
            metadata={
                "backend_id": request.backend_id,
                "source": self.trainer_id,
            },
        )

    def _plan(self, task_request: TaskRequest, context: dict[str, Any]) -> Agent0SAGEPlan:
        return Agent0SAGEPlan(
            steps=[
                "Restate the anchored solver trajectory.",
                "Produce one deterministic accepted rollout sample.",
            ],
            metadata={"round_id": context["round_id"]},
        )

    def _rollout(
        self,
        task_request: TaskRequest,
        plan: Agent0SAGEPlan,
        context: dict[str, Any],
    ) -> Agent0SAGERolloutResult:
        return Agent0SAGERolloutResult(
            status="completed",
            output=f"Fake Agent0 rollout for {task_request.task_id}.",
            trajectory_ref=f"{task_request.task_id}:trajectory",
            metadata={"plan_steps": len(plan.steps)},
        )

    def _critic(
        self,
        task_request: TaskRequest,
        plan: Agent0SAGEPlan,
        rollout: Agent0SAGERolloutResult,
        context: dict[str, Any],
    ) -> Agent0SAGECriticResult:
        return Agent0SAGECriticResult(
            accepted=True,
            score=0.9,
            verifier_passed=True,
            reason="deterministic fake accepted sample",
            metadata={"round_id": context["round_id"]},
        )
