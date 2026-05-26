from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import (
    LLMEvolutionRequest,
    LLMEvolutionResult,
    StandardEvolutionMetrics,
)

FakeEvolutionScenario = Literal["promoted_candidate", "not_recommended", "skipped", "failed"]


class FakeSAGETrainer(LLMTrainer):
    trainer_id = "fake_sage"

    def __init__(self, scenario: FakeEvolutionScenario = "promoted_candidate") -> None:
        if scenario not in {"promoted_candidate", "not_recommended", "skipped", "failed"}:
            raise ValueError(f"unknown fake evolution scenario: {scenario!r}")
        self.scenario = scenario
        self.requests: list[LLMEvolutionRequest] = []

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        self.requests.append(request)
        if self.scenario == "skipped":
            return LLMEvolutionResult(
                status="skipped",
                metadata={"reason": "fake SAGE skipped", "trainer": "sage"},
            )
        if self.scenario == "failed":
            return LLMEvolutionResult(
                status="failed",
                metadata={"error": "fake SAGE failed", "trainer": "sage"},
            )
        if self.scenario == "not_recommended":
            return LLMEvolutionResult(
                status="not_recommended",
                standard_metrics=StandardEvolutionMetrics(
                    eval_score_before=0.5 if request.previous_state_ref else None,
                    eval_score_after=0.5,
                    eval_metric_name="fake_eval",
                ),
                metadata={
                    "reason": "fake SAGE did not improve eval score",
                    "trainer": "sage",
                },
            )
        return self._promoted_candidate(request)

    def _promoted_candidate(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        artifact_uri = _write_fake_adapter_artifact(
            request.artifact_root_uri,
            {
                "backend_id": request.backend_id,
                "previous_state_ref": request.previous_state_ref,
                "trigger_trajectory_ref": request.trigger_trajectory_ref,
                "request_index": len(self.requests),
                "trainer": "sage",
            },
        )
        return LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref=f"fake-sage://{request.backend_id}/state/{len(self.requests)}",
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(
                n_train_samples=1,
                eval_score_before=0.5 if request.previous_state_ref else None,
                eval_score_after=0.75,
                eval_metric_name="fake_eval",
                promotion_margin=0.25 if request.previous_state_ref else None,
            ),
            artifact_refs=[
                ArtifactRef(
                    uri=artifact_uri,
                    type="model_adapter",
                    metadata={"backend": self.trainer_id},
                )
            ],
            metadata={"trainer": "sage"},
        )


class FakeEvolutionBackend(FakeSAGETrainer):
    """Compatibility name for the V0 fake evolution backend."""


def _write_fake_adapter_artifact(root_uri: str, payload: dict[str, object]) -> str:
    root_path = _local_path_from_uri(root_uri)
    if root_path is None:
        return f"{root_uri.rstrip('/')}/adapter.json"
    root_path.mkdir(parents=True, exist_ok=True)
    artifact_path = root_path / "adapter.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(artifact_path)


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None
