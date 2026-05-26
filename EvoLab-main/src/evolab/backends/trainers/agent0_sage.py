from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import Field
from pydantic import ValidationError

from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.common import ArtifactRef, StrictBaseModel
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.task import (
    ProposedTaskRelation,
    ProposedTaskRelationType,
    TaskOrigin,
    TaskPurpose,
    TaskRequest,
)


class Agent0SAGEProposal(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    candidate_id: str
    task_id: str
    goal: str
    human_anchor_task_refs: list[str] = Field(default_factory=list)
    human_anchor_trajectory_refs: list[str] = Field(default_factory=list)
    relation_rationale: str
    expected_transfer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent0SAGEPlan(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent0SAGERolloutResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: Literal["completed", "failed"]
    output: str | None = None
    trajectory_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent0SAGECriticResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    accepted: bool
    score: float = Field(ge=0, le=1)
    verifier_passed: bool
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


Agent0SAGEProposer = Callable[[dict[str, Any]], Agent0SAGEProposal]
Agent0SAGEPlanner = Callable[[TaskRequest, dict[str, Any]], Agent0SAGEPlan]
Agent0SAGERolloutRunner = Callable[[TaskRequest, Agent0SAGEPlan, dict[str, Any]], Agent0SAGERolloutResult]
Agent0SAGECritic = Callable[
    [TaskRequest, Agent0SAGEPlan, Agent0SAGERolloutResult, dict[str, Any]],
    Agent0SAGECriticResult,
]


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class Agent0SAGETrainer(LLMTrainer):
    trainer_id = "agent0_sage"

    def __init__(
        self,
        *,
        proposer: Agent0SAGEProposer,
        planner: Agent0SAGEPlanner,
        rollout_runner: Agent0SAGERolloutRunner,
        critic: Agent0SAGECritic,
        solver_trainer: LLMTrainer,
        trainer_id: str | None = None,
        max_rounds: int = 1,
        min_accepted_samples: int = 1,
        acceptance_threshold: float = 0.5,
    ) -> None:
        if trainer_id is not None:
            self.trainer_id = trainer_id
        self.proposer = proposer
        self.planner = planner
        self.rollout_runner = rollout_runner
        self.critic = critic
        self.solver_trainer = solver_trainer
        self.max_rounds = max_rounds
        self.min_accepted_samples = min_accepted_samples
        self.acceptance_threshold = acceptance_threshold

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for round_index in range(1, self.max_rounds + 1):
            context = {
                "request": request,
                "round_index": round_index,
                "round_id": f"round-{round_index}",
                "accepted": accepted,
            }
            proposal = self.proposer(context)
            try:
                task_request = self._task_request_from_proposal(proposal, context["round_id"])
            except (ValidationError, ValueError) as exc:
                rejected.append(
                    {
                        "candidate_id": proposal.candidate_id,
                        "round_id": context["round_id"],
                        "reason": str(exc),
                    }
                )
                continue

            plan = self.planner(task_request, context)
            try:
                rollout = self.rollout_runner(task_request, plan, context)
            except Exception as exc:
                rejected.append(
                    {
                        "candidate_id": proposal.candidate_id,
                        "task_id": task_request.task_id,
                        "round_id": context["round_id"],
                        "reason": f"rollout failed: {exc}",
                    }
                )
                continue

            critic = self.critic(task_request, plan, rollout, context)
            sample = self._sample_record(proposal, task_request, plan, rollout, critic, context["round_id"])
            if critic.accepted and critic.verifier_passed and critic.score >= self.acceptance_threshold:
                accepted.append(sample)
            else:
                rejection_reason = critic.reason
                if rejection_reason is None and not critic.verifier_passed:
                    rejection_reason = "verifier failed"
                if rejection_reason is None and critic.score < self.acceptance_threshold:
                    rejection_reason = "critic score below acceptance threshold"
                if rejection_reason is None:
                    rejection_reason = "critic rejected sample"
                rejected.append({**sample, "reason": rejection_reason})

        samples_path, _rejections_path, _summary_path, artifacts = self._write_artifacts(
            request,
            accepted,
            rejected,
        )
        if len(accepted) < self.min_accepted_samples:
            return LLMEvolutionResult(
                status="skipped",
                recommend_for_promotion=False,
                artifact_refs=artifacts,
                metadata={
                    "reason": "not enough accepted samples",
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "trainer_id": self.trainer_id,
                },
            )

        try:
            nested = self.solver_trainer.train(
                self._nested_solver_request(request, samples_path, len(accepted))
            )
        except Exception as exc:
            return LLMEvolutionResult(
                status="failed",
                recommend_for_promotion=False,
                artifact_refs=artifacts,
                metadata={
                    "error": str(exc),
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "trainer_id": self.trainer_id,
                },
            )
        return self._wrap_nested_result(nested, artifacts, len(accepted), len(rejected))

    def _task_request_from_proposal(self, proposal: Agent0SAGEProposal, round_id: str) -> TaskRequest:
        relation = ProposedTaskRelation(
            human_anchor_task_refs=proposal.human_anchor_task_refs,
            human_anchor_trajectory_refs=proposal.human_anchor_trajectory_refs,
            relation_type=ProposedTaskRelationType.CURRICULUM_STEP,
            relation_rationale=proposal.relation_rationale,
            expected_transfer=proposal.expected_transfer,
            metadata={"candidate_id": proposal.candidate_id},
        )
        return TaskRequest(
            task_id=proposal.task_id,
            origin=TaskOrigin.PROPOSER,
            purpose=TaskPurpose.TRAINING_ROLLOUT,
            goal=proposal.goal,
            producer_ref=f"agent0_sage:{self.trainer_id}",
            round_id=round_id,
            proposed_task_relation=relation,
            metadata={
                **proposal.metadata,
                "candidate_id": proposal.candidate_id,
                "agent0_sage_trainer_id": self.trainer_id,
            },
        )

    def _sample_record(
        self,
        proposal: Agent0SAGEProposal,
        task_request: TaskRequest,
        plan: Agent0SAGEPlan,
        rollout: Agent0SAGERolloutResult,
        critic: Agent0SAGECriticResult,
        round_id: str,
    ) -> dict[str, Any]:
        return {
            "candidate_id": proposal.candidate_id,
            "task_request": task_request.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "rollout": rollout.model_dump(mode="json"),
            "critic": critic.model_dump(mode="json"),
            "round_id": round_id,
        }

    def _write_artifacts(
        self,
        request: LLMEvolutionRequest,
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> tuple[Path, Path, Path, list[ArtifactRef]]:
        artifact_root = _local_path_from_uri(request.artifact_root_uri)
        if artifact_root is None:
            raise ValueError("Agent0SAGETrainer requires local artifact_root_uri")
        artifact_root.mkdir(parents=True, exist_ok=True)
        samples_path = artifact_root / "agent0_sage_samples.jsonl"
        rejections_path = artifact_root / "agent0_sage_rejections.jsonl"
        summary_path = artifact_root / "agent0_sage_summary.json"
        _write_jsonl(samples_path, accepted)
        _write_jsonl(rejections_path, rejected)
        summary = {
            "trainer_id": self.trainer_id,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "acceptance_threshold": self.acceptance_threshold,
            "min_accepted_samples": self.min_accepted_samples,
            "max_rounds": self.max_rounds,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        artifacts = [
            ArtifactRef(uri=str(samples_path), type="dataset", metadata={"role": "accepted_samples"}),
            ArtifactRef(uri=str(rejections_path), type="log", metadata={"role": "rejections"}),
            ArtifactRef(uri=str(summary_path), type="log", metadata={"role": "summary"}),
        ]
        return samples_path, rejections_path, summary_path, artifacts

    def _nested_solver_request(
        self,
        request: LLMEvolutionRequest,
        samples_path: Path,
        accepted_count: int,
    ) -> LLMEvolutionRequest:
        artifact_root = _local_path_from_uri(request.artifact_root_uri)
        solver_artifact_root = str((artifact_root or Path(request.artifact_root_uri)) / "solver")
        return LLMEvolutionRequest(
            mode=request.mode,
            backend_id=request.backend_id,
            previous_state_ref=request.previous_state_ref,
            artifact_root_uri=solver_artifact_root,
            budget=request.budget,
            trigger_trajectory_ref=request.trigger_trajectory_ref,
            proposer_input_refs=request.proposer_input_refs,
            lab_signals=request.lab_signals,
            instance_snapshots=request.instance_snapshots,
            metadata={
                **request.metadata,
                "agent0_sage_manifest_uri": str(samples_path),
                "accepted_count": accepted_count,
                "source_trainer_id": self.trainer_id,
            },
        )

    def _wrap_nested_result(
        self,
        nested: LLMEvolutionResult,
        artifacts: list[ArtifactRef],
        accepted_count: int,
        rejected_count: int,
    ) -> LLMEvolutionResult:
        return LLMEvolutionResult(
            status=nested.status,
            new_state_ref=nested.new_state_ref,
            recommend_for_promotion=nested.recommend_for_promotion,
            lora_role=nested.lora_role,
            standard_metrics=nested.standard_metrics,
            artifact_refs=[*artifacts, *nested.artifact_refs],
            metadata={
                **nested.metadata,
                "agent0_sage": {
                    "trainer_id": self.trainer_id,
                    "accepted_count": accepted_count,
                    "rejected_count": rejected_count,
                },
            },
        )
