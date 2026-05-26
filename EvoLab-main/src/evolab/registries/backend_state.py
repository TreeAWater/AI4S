from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from evolab.contracts.state import BackendStateRecord


class BackendStateRegistry(ABC):
    @abstractmethod
    def register_candidate(self, record: BackendStateRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self, state_ref: str) -> BackendStateRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_states(self, backend_id: str | None = None) -> list[BackendStateRecord]:
        raise NotImplementedError

    @abstractmethod
    def promote(self, backend_id: str, new_state_ref: str, evolution_run_ref: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def resolve_active_state(self, backend_id: str, role: str | None = None) -> str | None:
        raise NotImplementedError


class FileBackendStateRegistry(BackendStateRegistry):
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "states.jsonl"
        self.active_path = self.root / "active.json"

    def register_candidate(self, record: BackendStateRecord) -> None:
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _load_records(self) -> list[BackendStateRecord]:
        if not self.records_path.exists():
            return []
        return [
            BackendStateRecord.model_validate_json(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def get_state(self, state_ref: str) -> BackendStateRecord | None:
        for record in self._load_records():
            if record.state_ref == state_ref:
                return record
        return None

    def list_states(self, backend_id: str | None = None) -> list[BackendStateRecord]:
        records = self._load_records()
        if backend_id is None:
            return records
        return [record for record in records if record.backend_id == backend_id]

    def _load_active(self) -> dict[str, str]:
        if not self.active_path.exists():
            return {}
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    def _write_active(self, active: dict[str, str]) -> None:
        tmp_path = self.root / "active.json.tmp"
        tmp_path.write_text(json.dumps(active, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.active_path)

    def _registered_state(self, backend_id: str, state_ref: str) -> BackendStateRecord | None:
        for record in self._load_records():
            if record.backend_id == backend_id and record.state_ref == state_ref:
                return record
        return None

    def promote(self, backend_id: str, new_state_ref: str, evolution_run_ref: str) -> None:
        record = self._registered_state(backend_id, new_state_ref)
        if record is None:
            raise ValueError(
                f"Cannot promote unregistered backend state: backend_id={backend_id!r}, "
                f"state_ref={new_state_ref!r}"
            )
        active = self._load_active()
        role = _role_specific_state_key(record)
        if role is not None:
            active[f"{backend_id}:{role}"] = new_state_ref
            active[f"{backend_id}:{role}:evolution_run_ref"] = evolution_run_ref
        else:
            active[backend_id] = new_state_ref
            active[f"{backend_id}:evolution_run_ref"] = evolution_run_ref
        self._write_active(active)

    def resolve_active_state(self, backend_id: str, role: str | None = None) -> str | None:
        active = self._load_active()
        if role and f"{backend_id}:{role}" in active:
            return active[f"{backend_id}:{role}"]
        return active.get(backend_id)


def _role_specific_state_key(record: BackendStateRecord) -> str | None:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    prompt_overlay = metadata.get("prompt_overlay")
    if isinstance(prompt_overlay, dict):
        role = prompt_overlay.get("role")
        if isinstance(role, str) and role:
            return role
    role = metadata.get("role")
    if metadata.get("state_kind") == "prompt_overlay" and isinstance(role, str) and role:
        return role
    return None
