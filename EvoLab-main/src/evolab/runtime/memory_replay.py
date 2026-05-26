from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel
from evolab.lab.resolver import LabResolver


class MemoryReplayScopeRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    run_ref: str
    role: str
    memory_scope: Literal["agent", "task"]
    memory_scope_id: str
    backend_id: str
    pre_state_ref: str | None = None
    update_status: str | None = None
    update_state_ref: str | None = None
    previous_state_ref: str | None = None
    parent_state_refs: list[str] = Field(default_factory=list)
    update_summary: dict[str, Any] = Field(default_factory=dict)


class MemoryReplayReport(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    lab_root: str
    task_id: str | None = None
    ok: bool
    records: list[MemoryReplayScopeRecord] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def replay_memory_trace(lab_root: Path | str, task_id: str | None = None) -> MemoryReplayReport:
    root = Path(lab_root)
    resolver = LabResolver(root)
    trajectory_registry = resolver.trajectory_registry()
    backend_state_registry = resolver.backend_state_registry()
    subagent_runs = trajectory_registry.list_subagent_runs()
    if task_id is not None:
        subagent_runs = [run for run in subagent_runs if run.task_id == task_id]

    records: list[MemoryReplayScopeRecord] = []
    issues: list[str] = []
    last_update_by_scope: dict[tuple[str, str, str], str] = {}

    for run in subagent_runs:
        for scope in ("agent", "task"):
            bundle_payload = run.metadata.get(f"{scope}_memory_bundle")
            update_payload = run.metadata.get(f"{scope}_memory_update_result")
            if scope == "agent" and _run_uses_task_only_memory(run.metadata):
                if bundle_payload is None and update_payload is None:
                    continue
            if not isinstance(bundle_payload, dict):
                issues.append(f"{run.run_ref}:{scope}: missing memory bundle metadata")
                continue
            if not isinstance(update_payload, dict):
                issues.append(f"{run.run_ref}:{scope}: missing memory update metadata")
                continue

            scope_metadata = _scope_metadata(scope, bundle_payload, update_payload)
            memory_scope_id = scope_metadata.get("memory_scope_id")
            backend_id = bundle_payload.get("backend_id")
            if not isinstance(memory_scope_id, str) or not memory_scope_id:
                issues.append(f"{run.run_ref}:{scope}: missing memory_scope_id")
                continue
            if not isinstance(backend_id, str) or not backend_id:
                issues.append(f"{run.run_ref}:{scope}: missing backend_id")
                continue

            pre_state_ref = _optional_string(bundle_payload.get("state_ref"))
            update_state_ref = _optional_string(update_payload.get("state_ref"))
            previous_state_ref = _optional_string(update_payload.get("previous_state_ref"))
            update_status = _optional_string(update_payload.get("status"))
            state_record = backend_state_registry.get_state(update_state_ref) if update_state_ref else None
            parent_state_refs = list(state_record.parent_state_refs) if state_record is not None else []

            if update_status == "updated":
                if not update_state_ref:
                    issues.append(f"{run.run_ref}:{scope}: updated memory missing state_ref")
                elif state_record is None:
                    issues.append(f"{run.run_ref}:{scope}: missing BackendStateRecord for {update_state_ref}")
                else:
                    expected_parent = previous_state_ref or pre_state_ref
                    if expected_parent and expected_parent not in state_record.parent_state_refs:
                        issues.append(
                            f"{run.run_ref}:{scope}: state {update_state_ref} missing parent {expected_parent}"
                        )
                    if state_record.backend_id != backend_id:
                        issues.append(f"{run.run_ref}:{scope}: backend_id mismatch for {update_state_ref}")
                    if state_record.metadata.get("memory_scope") != scope:
                        issues.append(f"{run.run_ref}:{scope}: memory_scope mismatch for {update_state_ref}")
                    if state_record.metadata.get("memory_scope_id") != memory_scope_id:
                        issues.append(f"{run.run_ref}:{scope}: memory_scope_id mismatch for {update_state_ref}")

            lineage_key = (backend_id, scope, memory_scope_id)
            last_state_ref = last_update_by_scope.get(lineage_key)
            if last_state_ref is not None and pre_state_ref != last_state_ref:
                issues.append(
                    f"{run.run_ref}:{scope}: pre_state_ref {pre_state_ref!r} does not continue {last_state_ref!r}"
                )
            if update_state_ref:
                last_update_by_scope[lineage_key] = update_state_ref

            records.append(
                MemoryReplayScopeRecord(
                    task_id=run.task_id,
                    run_ref=run.run_ref,
                    role=run.role,
                    memory_scope=scope,
                    memory_scope_id=memory_scope_id,
                    backend_id=backend_id,
                    pre_state_ref=pre_state_ref,
                    update_status=update_status,
                    update_state_ref=update_state_ref,
                    previous_state_ref=previous_state_ref,
                    parent_state_refs=parent_state_refs,
                    update_summary=_update_summary(update_payload),
                )
            )

    return MemoryReplayReport(
        lab_root=str(root),
        task_id=task_id,
        ok=not issues,
        records=records,
        issues=issues,
    )


def _run_uses_task_only_memory(metadata: dict[str, Any]) -> bool:
    dispatch_metadata = metadata.get("dispatch_metadata")
    if not isinstance(dispatch_metadata, dict):
        return False
    mode = dispatch_metadata.get("execution_mode") or dispatch_metadata.get("mode")
    return metadata.get("memory_mode") == "task_only" and isinstance(mode, str) and mode.casefold() == "dynamic"


def _scope_metadata(scope: str, bundle_payload: dict[str, Any], update_payload: dict[str, Any]) -> dict[str, Any]:
    bundle_metadata = bundle_payload.get("metadata")
    update_metadata = update_payload.get("metadata")
    metadata: dict[str, Any] = {"memory_scope": scope}
    if isinstance(bundle_metadata, dict):
        metadata.update(bundle_metadata)
    if isinstance(update_metadata, dict):
        metadata.update(update_metadata)
    return metadata


def _update_summary(update_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = update_payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    update_summary = metadata.get("update_summary")
    return update_summary if isinstance(update_summary, dict) else {}


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
