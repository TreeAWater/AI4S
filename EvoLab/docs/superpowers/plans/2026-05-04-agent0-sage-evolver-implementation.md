# Agent0 SAGE Evolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Agent0SAGETrainer`, a concrete `LLMTrainer` that generates proposer tasks through `TaskRequest`, runs injected solver rollouts, filters samples with a critic, writes co-evolution artifacts, and delegates solver evolution to a nested trainer.

**Architecture:** Keep this as one trainer implementation in `evolab/backends/trainers/agent0_sage.py`. The trainer owns orchestration only; collaborator callables supply proposer, planner, rollout, critic, and nested solver training behavior. Generated tasks use existing task provenance contracts so proposer tasks share the human/benchmark interface while remaining distinguishable.

**Tech Stack:** Python 3.11, Pydantic v2 contracts already in `evolab.contracts`, pytest, existing `LLMTrainer` and `LLMEvolutionRequest` / `LLMEvolutionResult` contracts.

---

## File Structure

- Create `evolab/backends/trainers/agent0_sage.py`: concrete trainer, collaborator result models, artifact writing helpers, local path handling.
- Modify `evolab/backends/trainers/__init__.py`: export `Agent0SAGETrainer` and support models.
- Create `tests/test_agent0_sage_trainer.py`: TDD coverage for export, task provenance, rejection behavior, rollout calls, artifact writing, nested solver trainer invocation, and result wrapping.

## Task 1: Exports And Collaborator Models

**Files:**
- Create: `evolab/backends/trainers/agent0_sage.py`
- Modify: `evolab/backends/trainers/__init__.py`
- Test: `tests/test_agent0_sage_trainer.py`

- [ ] **Step 1: Write the failing export and inheritance test**

Add this test file:

```python
from evolab.backends.trainers import Agent0SAGETrainer, LLMTrainer
from evolab.backends.trainers.agent0_sage import (
    Agent0SAGECriticResult,
    Agent0SAGEPlan,
    Agent0SAGEProposal,
    Agent0SAGERolloutResult,
)


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
```

Add this helper to the same test file:

```python
from evolab.backends.trainers import LLMTrainer
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent0_sage_trainer.py::test_agent0_sage_trainer_is_exported_llm_trainer -q`

Expected: FAIL with `ImportError` or `ModuleNotFoundError` for `Agent0SAGETrainer`.

- [ ] **Step 3: Add minimal trainer module and exports**

Create `evolab/backends/trainers/agent0_sage.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field

from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.common import StrictBaseModel
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.task import TaskRequest


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
    ) -> None:
        if trainer_id is not None:
            self.trainer_id = trainer_id
        self.proposer = proposer
        self.planner = planner
        self.rollout_runner = rollout_runner
        self.critic = critic
        self.solver_trainer = solver_trainer

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        return LLMEvolutionResult(
            status="skipped",
            recommend_for_promotion=False,
            metadata={"reason": "no co-evolution rounds configured"},
        )
```

Modify `evolab/backends/trainers/__init__.py`:

```python
from evolab.backends.trainers.agent0_sage import (
    Agent0SAGECriticResult,
    Agent0SAGEPlan,
    Agent0SAGEProposal,
    Agent0SAGERolloutResult,
    Agent0SAGETrainer,
)
from evolab.backends.trainers.base import LLMTrainer
from evolab.backends.trainers.blank import BlankTrainer

__all__ = [
    "Agent0SAGECriticResult",
    "Agent0SAGEPlan",
    "Agent0SAGEProposal",
    "Agent0SAGERolloutResult",
    "Agent0SAGETrainer",
    "BlankTrainer",
    "LLMTrainer",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent0_sage_trainer.py::test_agent0_sage_trainer_is_exported_llm_trainer -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/trainers/agent0_sage.py evolab/backends/trainers/__init__.py tests/test_agent0_sage_trainer.py
git commit -m "feat: add agent0 sage trainer shell"
```

## Task 2: Candidate Task Provenance And Invalid Proposal Rejection

**Files:**
- Modify: `evolab/backends/trainers/agent0_sage.py`
- Modify: `tests/test_agent0_sage_trainer.py`

- [ ] **Step 1: Write failing tests for generated task provenance and invalid proposal rejection**

Append these tests:

```python
import json
from pathlib import Path

import pytest

from evolab.contracts.common import EvolutionBudget
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.task import TaskOrigin, TaskPurpose


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_generated_task_uses_shared_task_request_with_proposer_provenance tests/test_agent0_sage_trainer.py::test_invalid_candidate_without_human_anchor_is_rejected_before_rollout -q
```

Expected: FAIL because `train(...)` does not create proposer `TaskRequest`s or rejections.

- [ ] **Step 3: Implement proposal-to-task conversion and basic one-round orchestration**

Extend `evolab/backends/trainers/agent0_sage.py`:

```python
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from evolab.contracts.task import (
    ProposedTaskRelation,
    ProposedTaskRelationType,
    TaskOrigin,
    TaskPurpose,
    TaskRequest,
)
```

Add constructor options:

```python
        max_rounds: int = 1,
        min_accepted_samples: int = 1,
        acceptance_threshold: float = 0.5,
```

Store them:

```python
        self.max_rounds = max_rounds
        self.min_accepted_samples = min_accepted_samples
        self.acceptance_threshold = acceptance_threshold
```

Replace `train(...)` with:

```python
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
                rejected.append({**sample, "reason": critic.reason or "critic rejected sample"})

        if len(accepted) < self.min_accepted_samples:
            return LLMEvolutionResult(
                status="skipped",
                recommend_for_promotion=False,
                metadata={
                    "reason": "not enough accepted samples",
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "trainer_id": self.trainer_id,
                },
            )

        return LLMEvolutionResult(
            status="not_recommended",
            recommend_for_promotion=False,
            metadata={
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "trainer_id": self.trainer_id,
            },
        )
```

Add helpers:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_generated_task_uses_shared_task_request_with_proposer_provenance tests/test_agent0_sage_trainer.py::test_invalid_candidate_without_human_anchor_is_rejected_before_rollout -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/trainers/agent0_sage.py tests/test_agent0_sage_trainer.py
git commit -m "feat: generate proposer training tasks"
```

## Task 3: Artifacts And Nested Solver Trainer Delegation

**Files:**
- Modify: `evolab/backends/trainers/agent0_sage.py`
- Modify: `tests/test_agent0_sage_trainer.py`

- [ ] **Step 1: Write failing tests for artifact writing and nested solver trainer call**

Append these tests:

```python
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import LLMEvolutionResult, StandardEvolutionMetrics


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
        new_state_ref="solver-state-1",
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
    assert result.new_state_ref == "solver-state-1"
    assert result.lora_role == "solver"
    assert result.standard_metrics.eval_score_after == 0.88
    assert result.metadata["agent0_sage"]["accepted_count"] == 1
    assert result.metadata["nested"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_accepted_samples_are_written_and_nested_solver_trainer_receives_manifest tests/test_agent0_sage_trainer.py::test_outer_result_preserves_nested_solver_promotion_fields -q
```

Expected: FAIL because artifacts are not written and nested trainer is not called.

- [ ] **Step 3: Implement artifact writing, nested request, and result wrapping**

Add imports:

```python
from evolab.contracts.common import ArtifactRef
```

Add helpers:

```python
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
```

Add methods:

```python
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
        solver_artifact_root = str(Path(request.artifact_root_uri) / "solver")
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
```

Update the end of `train(...)`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_accepted_samples_are_written_and_nested_solver_trainer_receives_manifest tests/test_agent0_sage_trainer.py::test_outer_result_preserves_nested_solver_promotion_fields -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/trainers/agent0_sage.py tests/test_agent0_sage_trainer.py
git commit -m "feat: delegate agent0 sage solver evolution"
```

## Task 4: Critic Thresholds, Rejection Artifacts, And Full Regression

**Files:**
- Modify: `evolab/backends/trainers/agent0_sage.py`
- Modify: `tests/test_agent0_sage_trainer.py`

- [ ] **Step 1: Write failing tests for threshold rejection and nested trainer errors**

Append these tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail or expose incomplete rejection reasons**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_critic_threshold_rejects_low_score_and_writes_rejection tests/test_agent0_sage_trainer.py::test_nested_solver_trainer_exception_returns_failed_result_with_artifacts -q
```

Expected: FAIL if rejection reason is not preserved or nested exceptions are not converted to failed results.

- [ ] **Step 3: Refine rejection reason handling**

Update the critic rejection branch in `train(...)`:

```python
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
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_agent0_sage_trainer.py::test_critic_threshold_rejects_low_score_and_writes_rejection tests/test_agent0_sage_trainer.py::test_nested_solver_trainer_exception_returns_failed_result_with_artifacts -q
```

Expected: PASS.

- [ ] **Step 5: Run full relevant test file**

Run: `pytest tests/test_agent0_sage_trainer.py -q`

Expected: PASS.

- [ ] **Step 6: Run full suite and diff check**

Run:

```bash
pytest -q
git diff --check
```

Expected: `pytest -q` passes all tests. `git diff --check` produces no output.

- [ ] **Step 7: Commit**

```bash
git add evolab/backends/trainers/agent0_sage.py tests/test_agent0_sage_trainer.py
git commit -m "test: cover agent0 sage rejection paths"
```

## Self-Review Notes

- Spec coverage: Task 1 covers trainer boundary and exports. Task 2 covers shared `TaskRequest` provenance and invalid proposal rejection. Task 3 covers artifacts, nested solver trainer delegation, and final result wrapping. Task 4 covers thresholds, error handling, and full verification.
- Type consistency: all tests and implementation snippets use existing `LLMEvolutionRequest`, `LLMEvolutionResult`, `ArtifactRef`, `TaskRequest`, `ProposedTaskRelation`, and `LLMTrainer` names.
- Scope control: this plan does not change `TaskWorker`, queues, `TaskRuntime`, or real fine-tuning behavior.
