from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from evolab.backends.trainers import LLMTrainer
from evolab.contracts.evolution import (
    EvolutionRunEvent,
    EvolutionRunEventType,
    LLMEvolutionRequest,
    LLMEvolutionResult,
)
from evolab.contracts.records import EvolutionRunRecord
from evolab.lab.queue import ClaimedJob, FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import TrajectoryRegistry
from evolab.runtime.evolution_executor import EvolutionExecutor
from evolab.runtime.trajectory_collector import TrajectoryCollector


class EvolveWorker:
    def __init__(
        self,
        queue: FileWorkQueue,
        trainers: dict[str, LLMTrainer],
        backend_state_registry: FileBackendStateRegistry,
        worker_id: str,
        trajectory_registry: TrajectoryRegistry | None = None,
    ):
        self.queue = queue
        self.trainers = trainers
        self.backend_state_registry = backend_state_registry
        self.worker_id = worker_id
        self.trajectory_registry = trajectory_registry
        self.trajectory_collector = TrajectoryCollector(trajectory_registry)

    def run_once(self) -> bool:
        job = self.queue.claim(self.worker_id)
        if job is None:
            return False

        request: LLMEvolutionRequest | None = None
        try:
            request = self._load_request(job)
            self._write_run_spec(job, request)
            self._append_event(job.job_id, request, EvolutionRunEventType.RUN_STARTED)
            trainer = self.trainers.get(request.backend_id)
            if trainer is None:
                reason = f"no trainer for backend_id {request.backend_id!r}"
                self._save_evolution_run(
                    job_id=job.job_id,
                    request=request,
                    result=LLMEvolutionResult(status="skipped", metadata={"reason": reason}),
                    trainer=None,
                )
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_SKIPPED, {"reason": reason})
                self._write_manifest_latest(
                    request,
                    {
                        "run_ref": job.job_id,
                        "queue_status": "skipped",
                        "result_status": "skipped",
                        "reason": reason,
                    },
                )
                self.queue.mark_skipped(job, reason)
                return True

            if self.trajectory_registry is None:
                reason = "trajectory registry is required to save evolution run before promotion"
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_FAILED, {"error": reason})
                self._write_manifest_latest(
                    request,
                    {
                        "run_ref": job.job_id,
                        "queue_status": "failed",
                        "result_status": "failed",
                        "error": reason,
                    },
                )
                self.queue.mark_failed(job, reason)
                return True

            executor = EvolutionExecutor(self.backend_state_registry, self.worker_id)
            self._append_event(
                job.job_id,
                request,
                EvolutionRunEventType.TRAINER_INVOKED,
                {"trainer_id": getattr(trainer, "trainer_id", None)},
            )
            outcome = executor.run(
                request=request,
                trainer=trainer,
                run_ref=job.job_id,
                task_id=_task_id_from_request(request),
                metadata=_promotion_metadata(request),
                promote=False,
            )
            result = _validated_result(outcome.result)
            self._append_event(
                job.job_id,
                request,
                EvolutionRunEventType.TRAINER_COMPLETED,
                {
                    "result_status": result.status,
                    "promotion_errors": outcome.promotion_errors,
                },
            )
            saved = self._save_evolution_run(
                job_id=job.job_id,
                request=request,
                result=result,
                trainer=trainer,
            )
            if not saved:
                reason = "evolution record was not saved"
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_FAILED, {"error": reason})
                self.queue.mark_failed(job, reason)
                return True
            self._append_event(job.job_id, request, EvolutionRunEventType.EVOLUTION_RECORD_SAVED, {"status": result.status})
            promoted = executor.promote(
                request=request,
                result=result,
                trainer=trainer,
                run_ref=job.job_id,
                task_id=_task_id_from_request(request),
                metadata=_promotion_metadata(request),
            )
            self._append_event(
                job.job_id,
                request,
                EvolutionRunEventType.PROMOTION_DECIDED,
                {"promoted": promoted, "new_state_ref": result.new_state_ref},
            )
            if result.status == "failed":
                message = _result_message(result, "trainer returned failed result")
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_FAILED, {"message": message})
                self._write_manifest_latest(
                    request,
                    _manifest_payload(job.job_id, request, result, promoted, "failed"),
                )
                self.queue.mark_failed(job, _result_message(result, "trainer returned failed result"))
                return True
            if result.status == "skipped":
                message = _result_message(result, "trainer returned skipped result")
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_SKIPPED, {"message": message})
                self._write_manifest_latest(
                    request,
                    _manifest_payload(job.job_id, request, result, promoted, "skipped"),
                )
                self.queue.mark_skipped(job, _result_message(result, "trainer returned skipped result"))
                return True

            self._append_event(job.job_id, request, EvolutionRunEventType.RUN_FINISHED, {"queue_status": "done"})
            self._write_manifest_latest(
                request,
                _manifest_payload(job.job_id, request, result, promoted, "done"),
            )
            self.queue.mark_done(job)
        except NotImplementedError as exc:
            if request is not None:
                self._append_event(job.job_id, request, EvolutionRunEventType.RUN_SKIPPED, {"reason": str(exc)})
                self._try_write_manifest_latest(
                    request,
                    {
                        "run_ref": job.job_id,
                        "queue_status": "skipped",
                        "result_status": "skipped",
                        "reason": str(exc),
                    },
                )
            self.queue.mark_skipped(job, str(exc))
        except Exception as exc:
            if request is not None:
                self._try_append_event(job.job_id, request, EvolutionRunEventType.RUN_FAILED, {"error": str(exc)})
                self._try_write_manifest_latest(
                    request,
                    {
                        "run_ref": job.job_id,
                        "queue_status": "failed",
                        "result_status": "failed",
                        "error": str(exc),
                    },
                )
            self.queue.mark_failed(job, str(exc))
        return True

    def _load_request(self, job: ClaimedJob) -> LLMEvolutionRequest:
        request_payload_uri = job.payload["request_payload_uri"]
        path = _local_path_from_uri(request_payload_uri)
        if path is None:
            raise ValueError(f"unsupported request_payload_uri {request_payload_uri!r}")
        return LLMEvolutionRequest.model_validate_json(path.read_text(encoding="utf-8"))

    def _save_evolution_run(
        self,
        *,
        job_id: str,
        request: LLMEvolutionRequest,
        result: LLMEvolutionResult,
        trainer: LLMTrainer | None,
    ) -> bool:
        if not self.trajectory_collector.enabled:
            return False
        self.trajectory_collector.save_evolution_run(
            EvolutionRunRecord(
                run_ref=job_id,
                mode=request.mode,
                backend_id=request.backend_id,
                result_status=result.status,
                result=result,
                training_trajectory_refs=_training_trajectory_refs(request),
                input_snapshot_refs=_input_snapshot_refs(request),
                consumed_instance_snapshot_refs=[
                    snapshot.snapshot_ref for snapshot in request.instance_snapshots
                ],
                output_snapshot_refs=_output_snapshot_refs(result),
                lora_role=result.lora_role,
                metadata={
                    "task_id": _task_id_from_request(request),
                    "worker_id": self.worker_id,
                    "trainer_id": getattr(trainer, "trainer_id", None),
                    "request": request.model_dump(mode="json"),
                },
            )
        )
        return True

    def _append_event(
        self,
        run_ref: str,
        request: LLMEvolutionRequest,
        event_type: EvolutionRunEventType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_path = _events_path(request.artifact_root_uri)
        if event_path is None:
            return
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event = EvolutionRunEvent(
            run_ref=run_ref,
            event_type=event_type,
            backend_id=request.backend_id,
            task_id=_task_id_from_request(request),
            metadata=metadata or {},
        )
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def _try_append_event(
        self,
        run_ref: str,
        request: LLMEvolutionRequest,
        event_type: EvolutionRunEventType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._append_event(run_ref, request, event_type, metadata)
        except Exception:
            return

    def _write_run_spec(self, job: ClaimedJob, request: LLMEvolutionRequest) -> None:
        path = _run_spec_path(request.artifact_root_uri)
        if path is None or path.exists():
            return
        _write_json_atomic(
            path,
            {
                "schema_version": "v1",
                "run_ref": job.job_id,
                "backend_id": request.backend_id,
                "mode": request.mode.value,
                "task_id": _task_id_from_request(request),
                "request_payload_uri": job.payload.get("request_payload_uri"),
                "request": request.model_dump(mode="json"),
                "input_snapshot_refs": _input_snapshot_refs(request),
            },
        )

    def _write_manifest_latest(self, request: LLMEvolutionRequest, payload: dict[str, Any]) -> None:
        path = _manifest_latest_path(request.artifact_root_uri)
        if path is None:
            return
        _write_json_atomic(path, {"schema_version": "v1", **payload})

    def _try_write_manifest_latest(self, request: LLMEvolutionRequest, payload: dict[str, Any]) -> None:
        try:
            self._write_manifest_latest(request, payload)
        except Exception:
            return


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        path = unquote(parsed.path if parsed.scheme == "file" else uri)
        return Path(path)
    return None


def _events_path(artifact_root_uri: str) -> Path | None:
    root = _local_path_from_uri(artifact_root_uri)
    if root is None:
        return None
    return root / "events.jsonl"


def _run_spec_path(artifact_root_uri: str) -> Path | None:
    root = _local_path_from_uri(artifact_root_uri)
    if root is None:
        return None
    return root / "run_spec.json"


def _manifest_latest_path(artifact_root_uri: str) -> Path | None:
    root = _local_path_from_uri(artifact_root_uri)
    if root is None:
        return None
    return root / "manifest.latest.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _result_message(result: LLMEvolutionResult, fallback: str) -> str:
    promotion_errors = result.metadata.get("promotion_errors")
    if isinstance(promotion_errors, list) and promotion_errors:
        return "; ".join(str(error) for error in promotion_errors)
    for key in ("error", "reason", "message"):
        value = result.metadata.get(key)
        if value:
            return str(value)
    return fallback


def _task_id_from_request(request: LLMEvolutionRequest) -> str | None:
    task_id = request.metadata.get("task_id")
    return task_id if isinstance(task_id, str) else None


def _promotion_metadata(request: LLMEvolutionRequest) -> dict[str, str | None]:
    return {
        "trigger_trajectory_ref": request.trigger_trajectory_ref,
    }


def _training_trajectory_refs(request: LLMEvolutionRequest) -> list[str]:
    refs: list[str] = []
    if request.trigger_trajectory_ref is not None:
        refs.append(request.trigger_trajectory_ref)
    for ref in request.proposer_input_refs:
        if ref.ref_type == "trajectory" and ref.ref_id not in refs:
            refs.append(ref.ref_id)
    return refs


def _input_snapshot_refs(request: LLMEvolutionRequest) -> list[str]:
    return [snapshot.snapshot_ref for snapshot in request.instance_snapshots]


def _output_snapshot_refs(result: LLMEvolutionResult) -> list[str]:
    return _metadata_string_list(result.metadata, "output_snapshot_refs")


def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _validated_result(result: LLMEvolutionResult) -> LLMEvolutionResult:
    try:
        return LLMEvolutionResult.model_validate(result.model_dump(mode="json"))
    except ValidationError as exc:
        return LLMEvolutionResult(
            status="failed",
            metadata={
                "error": "evolution result validation failed",
                "validation_errors": exc.errors(include_context=False),
                "original_result": result.model_dump(mode="json"),
            },
        )


def _manifest_payload(
    run_ref: str,
    request: LLMEvolutionRequest,
    result: LLMEvolutionResult,
    promoted: bool,
    queue_status: str,
) -> dict[str, Any]:
    return {
        "run_ref": run_ref,
        "backend_id": request.backend_id,
        "task_id": _task_id_from_request(request),
        "queue_status": queue_status,
        "result_status": result.status,
        "promoted": promoted,
        "training_trajectory_refs": _training_trajectory_refs(request),
        "input_snapshot_refs": _input_snapshot_refs(request),
        "output_snapshot_refs": _output_snapshot_refs(result),
        "result": result.model_dump(mode="json"),
    }
