from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from evolab.contracts.lab_state import (
    ArtifactIndexRecord,
    LabStateDigest,
    LabStateIndex,
    RunLedgerRecord,
    SubagentReportRecord,
    TrainingIndexRecord,
)

RecordT = TypeVar("RecordT", bound=BaseModel)


class LabStateRegistryLoadWarning(RuntimeWarning):
    """Warning emitted when a persisted lab-state JSONL line cannot be loaded."""


class FileLabStateRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_run_ledger(self, record: RunLedgerRecord) -> str:
        path = self.root / "run_ledgers" / f"{_safe_ref(record.task_run_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record.task_run_id

    def get_run_ledger(self, task_run_id: str) -> RunLedgerRecord | None:
        path = self.root / "run_ledgers" / f"{_safe_ref(task_run_id)}.json"
        if not path.exists():
            return None
        return RunLedgerRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save_subagent_report(self, record: SubagentReportRecord) -> str:
        self._append("subagent_reports.jsonl", record)
        return record.report_ref

    def list_subagent_reports(self, task_id: str | None = None) -> list[SubagentReportRecord]:
        records = self._load_jsonl("subagent_reports.jsonl", SubagentReportRecord)
        return _filter_task(records, task_id)

    def save_artifact_index_record(self, record: ArtifactIndexRecord) -> str:
        self._append("artifact_index.jsonl", record)
        return record.artifact_ref

    def save_final_artifact_index(self, task_id: str, payload: dict[str, Any]) -> str:
        path = self.root / "final_artifact_indexes" / f"{_safe_ref(task_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_dumps(payload), encoding="utf-8")
        return str(path)

    def list_artifacts(self, task_id: str | None = None) -> list[ArtifactIndexRecord]:
        records = self._load_jsonl("artifact_index.jsonl", ArtifactIndexRecord)
        return _filter_task(records, task_id)

    def save_training_index_record(self, record: TrainingIndexRecord) -> str:
        self._append("training_index.jsonl", record)
        return record.sample_ref

    def list_training_samples(self, task_id: str | None = None) -> list[TrainingIndexRecord]:
        records = self._load_jsonl("training_index.jsonl", TrainingIndexRecord)
        return _filter_task(records, task_id)

    def save_index(self, record: LabStateIndex) -> str:
        self._append("indexes.jsonl", record)
        return record.index_ref

    def list_indexes(self, task_id: str | None = None) -> list[LabStateIndex]:
        records = self._load_jsonl("indexes.jsonl", LabStateIndex)
        if task_id is None:
            return records
        return [record for record in records if record.task.task_id == task_id]

    def latest_index(self, task_id: str | None = None) -> LabStateIndex | None:
        records = self.list_indexes(task_id)
        return records[-1] if records else None

    def save_digest(self, record: LabStateDigest) -> str:
        self._append("digests.jsonl", record)
        return record.digest_ref

    def list_digests(self, task_id: str | None = None) -> list[LabStateDigest]:
        records = self._load_jsonl("digests.jsonl", LabStateDigest)
        return _filter_task(records, task_id)

    def latest_digest(self, task_id: str | None = None) -> LabStateDigest | None:
        records = self.list_digests(task_id)
        return records[-1] if records else None

    def _append(self, relative_path: str, record: BaseModel) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _load_jsonl(self, relative_path: str, model_type: type[RecordT]) -> list[RecordT]:
        path = self.root / relative_path
        if not path.exists():
            return []
        records: list[RecordT] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(model_type.model_validate_json(stripped))
                except Exception as exc:
                    _warn_malformed_jsonl_record(
                        path=path,
                        line_number=line_number,
                        model_type=model_type,
                        error=exc,
                        line=stripped,
                    )
        return records


def _filter_task(records: list[RecordT], task_id: str | None) -> list[RecordT]:
    if task_id is None:
        return records
    return [record for record in records if getattr(record, "task_id", None) == task_id]


def _safe_ref(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"unsafe lab state ref: {value!r}")
    return safe


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def _warn_malformed_jsonl_record(
    *,
    path: Path,
    line_number: int,
    model_type: type[BaseModel],
    error: Exception,
    line: str,
) -> None:
    model_type_name = getattr(model_type, "__name__", str(model_type))
    error_text = _short_text(str(error).replace("\n", " "), limit=240)
    line_prefix = _short_text(line, limit=160)
    warnings.warn(
        "Skipping malformed lab-state JSONL record: "
        f"path={path}; line={line_number}; record_type={model_type_name}; "
        f"error={error_text}; prefix={line_prefix!r}",
        LabStateRegistryLoadWarning,
        stacklevel=3,
    )


def _short_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
