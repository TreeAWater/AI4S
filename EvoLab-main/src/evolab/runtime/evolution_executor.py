from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.state import BackendStateRecord
from evolab.registries.backend_state import BackendStateRegistry
from evolab.runtime.promotion import validate_promotion


@dataclass(frozen=True)
class EvolutionExecutionOutcome:
    result: LLMEvolutionResult
    promotion_errors: list[str]
    promoted: bool


class EvolutionExecutor:
    def __init__(self, backend_state_registry: BackendStateRegistry, worker_id: str) -> None:
        self.backend_state_registry = backend_state_registry
        self.worker_id = worker_id

    def run(
        self,
        *,
        request: LLMEvolutionRequest,
        trainer: Any,
        run_ref: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        promote: bool = True,
    ) -> EvolutionExecutionOutcome:
        _ensure_local_dir(request.artifact_root_uri)
        result = self._train(trainer, request)
        promotion_errors = validate_promotion(result, request)
        if promotion_errors:
            result = _failed_promotion_result(result, promotion_errors)
        promoted = False
        if promote:
            promoted = self.promote(
                request=request,
                result=result,
                trainer=trainer,
                run_ref=run_ref,
                task_id=task_id,
                metadata=metadata,
            )
        return EvolutionExecutionOutcome(
            result=result,
            promotion_errors=promotion_errors,
            promoted=promoted,
        )

    def _train(self, trainer: Any, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        try:
            return trainer.train(request)
        except NotImplementedError as exc:
            return LLMEvolutionResult(status="skipped", metadata={"reason": str(exc)})
        except Exception as exc:
            return LLMEvolutionResult(status="failed", metadata={"error": str(exc)})

    def promote(
        self,
        *,
        request: LLMEvolutionRequest,
        result: LLMEvolutionResult,
        trainer: Any,
        run_ref: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return self._promote_if_recommended(
            request=request,
            result=result,
            trainer=trainer,
            run_ref=run_ref,
            task_id=task_id,
            metadata=metadata or {},
        )

    def _promote_if_recommended(
        self,
        *,
        request: LLMEvolutionRequest,
        result: LLMEvolutionResult,
        trainer: Any,
        run_ref: str,
        task_id: str | None,
        metadata: dict[str, Any],
    ) -> bool:
        if not _should_promote(result):
            return False
        assert result.new_state_ref is not None
        self.backend_state_registry.register_candidate(
            BackendStateRecord(
                state_ref=result.new_state_ref,
                backend_id=request.backend_id,
                backend_type="llm",
                created_from_task_id=task_id,
                created_from_run_ref=run_ref,
                parent_state_refs=[request.previous_state_ref] if request.previous_state_ref else [],
                artifact_refs=result.artifact_refs,
                metadata={
                    "evolution_mode": request.mode.value,
                    "worker_id": self.worker_id,
                    "trainer_id": getattr(trainer, "trainer_id", None),
                    **_state_metadata_from_result(result),
                    **metadata,
                },
            )
        )
        self.backend_state_registry.promote(request.backend_id, result.new_state_ref, run_ref)
        return True


def _failed_promotion_result(
    original: LLMEvolutionResult,
    promotion_errors: list[str],
) -> LLMEvolutionResult:
    return LLMEvolutionResult(
        status="failed",
        metadata={
            "error": "promotion guard failed",
            "promotion_errors": promotion_errors,
            "original_result": original.model_dump(mode="json"),
        },
    )


def _should_promote(result: LLMEvolutionResult) -> bool:
    return (
        result.status == "promoted_candidate"
        and result.recommend_for_promotion
        and result.new_state_ref is not None
    )


def _state_metadata_from_result(result: LLMEvolutionResult) -> dict[str, Any]:
    state_metadata: dict[str, Any] = {}
    prompt_overlay = result.metadata.get("prompt_overlay")
    if isinstance(prompt_overlay, dict):
        state_metadata["state_kind"] = "prompt_overlay"
        state_metadata["prompt_overlay"] = prompt_overlay
        role = prompt_overlay.get("role")
        if isinstance(role, str) and role:
            state_metadata["role"] = role
    return state_metadata


def _ensure_local_dir(uri: str) -> None:
    path = _local_path_from_uri(uri)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None
