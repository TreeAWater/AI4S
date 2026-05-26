from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from evolab.contracts.records import (
    EvolutionRunRecord,
    LLMCallRecord,
    MetaAgentRunRecord,
    SubagentRunRecord,
    ToolCallTrajectoryRecord,
    TrajectoryEventRecord,
)


class TrajectoryRegistryLoadWarning(RuntimeWarning):
    """Warning emitted when a persisted trajectory JSONL line cannot be loaded."""


class TrajectoryRegistry(ABC):
    @abstractmethod
    def save_meta_agent_run(self, record: MetaAgentRunRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_subagent_run(self, record: SubagentRunRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_llm_call(self, record: LLMCallRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_tool_call_record(self, record: ToolCallTrajectoryRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_event(self, record: TrajectoryEventRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_evolution_run(self, record: EvolutionRunRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_meta_agent_run(self, run_ref: str) -> MetaAgentRunRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_subagent_run(self, run_ref: str) -> SubagentRunRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_llm_call(self, call_ref: str) -> LLMCallRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_tool_call_record(self, record_ref: str) -> ToolCallTrajectoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_event(self, event_ref: str) -> TrajectoryEventRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_evolution_run(self, run_ref: str) -> EvolutionRunRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_meta_agent_runs(self) -> list[MetaAgentRunRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_subagent_runs(self) -> list[SubagentRunRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_llm_calls(self) -> list[LLMCallRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_tool_call_records(self) -> list[ToolCallTrajectoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_events(self) -> list[TrajectoryEventRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_evolution_runs(self) -> list[EvolutionRunRecord]:
        raise NotImplementedError

    @abstractmethod
    def query_subagent_runs(self, filters: dict[str, Any]) -> list[SubagentRunRecord]:
        raise NotImplementedError


class FileTrajectoryRegistry(TrajectoryRegistry):
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, payload: str) -> None:
        with (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def save_meta_agent_run(self, record: MetaAgentRunRecord) -> str:
        self._append("meta_agent", record.model_dump_json())
        return record.run_ref

    def save_subagent_run(self, record: SubagentRunRecord) -> str:
        self._append("subagent", record.model_dump_json())
        return record.run_ref

    def save_llm_call(self, record: LLMCallRecord) -> str:
        self._append("llm_calls", record.model_dump_json())
        return record.call_ref

    def save_tool_call_record(self, record: ToolCallTrajectoryRecord) -> str:
        self._append("tool_calls", record.model_dump_json())
        return record.record_ref

    def save_event(self, record: TrajectoryEventRecord) -> str:
        self._append("events", record.model_dump_json())
        return record.event_ref

    def save_evolution_run(self, record: EvolutionRunRecord) -> str:
        self._append("evolution", record.model_dump_json())
        return record.run_ref

    def _load_records(self, name: str, record_type: Any) -> list[Any]:
        path = self.root / f"{name}.jsonl"
        if not path.exists():
            return []
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(record_type.model_validate_json(stripped))
                except Exception as exc:
                    _warn_malformed_jsonl_record(
                        path=path,
                        line_number=line_number,
                        record_type=record_type,
                        error=exc,
                        line=stripped,
                    )
        return records

    def list_meta_agent_runs(self) -> list[MetaAgentRunRecord]:
        return self._load_records("meta_agent", MetaAgentRunRecord)

    def list_subagent_runs(self) -> list[SubagentRunRecord]:
        return self._load_records("subagent", SubagentRunRecord)

    def list_llm_calls(self) -> list[LLMCallRecord]:
        return self._load_records("llm_calls", LLMCallRecord)

    def list_tool_call_records(self) -> list[ToolCallTrajectoryRecord]:
        return self._load_records("tool_calls", ToolCallTrajectoryRecord)

    def list_events(self) -> list[TrajectoryEventRecord]:
        return self._load_records("events", TrajectoryEventRecord)

    def list_evolution_runs(self) -> list[EvolutionRunRecord]:
        return self._load_records("evolution", EvolutionRunRecord)

    def get_meta_agent_run(self, run_ref: str) -> MetaAgentRunRecord | None:
        return _first_matching(self.list_meta_agent_runs(), "run_ref", run_ref)

    def get_subagent_run(self, run_ref: str) -> SubagentRunRecord | None:
        return _first_matching(self.list_subagent_runs(), "run_ref", run_ref)

    def get_llm_call(self, call_ref: str) -> LLMCallRecord | None:
        return _first_matching(self.list_llm_calls(), "call_ref", call_ref)

    def get_tool_call_record(self, record_ref: str) -> ToolCallTrajectoryRecord | None:
        return _first_matching(self.list_tool_call_records(), "record_ref", record_ref)

    def get_event(self, event_ref: str) -> TrajectoryEventRecord | None:
        return _first_matching(self.list_events(), "event_ref", event_ref)

    def get_evolution_run(self, run_ref: str) -> EvolutionRunRecord | None:
        return _first_matching(self.list_evolution_runs(), "run_ref", run_ref)

    def _validate_subagent_filters(self, filters: dict[str, Any]) -> None:
        unknown_keys = sorted(set(filters) - set(SubagentRunRecord.model_fields))
        if unknown_keys:
            raise ValueError(f"Unknown subagent run filter: {', '.join(unknown_keys)}")

    def _matches_filter(self, record: SubagentRunRecord, key: str, value: Any) -> bool:
        record_value = getattr(record, key)
        if isinstance(record_value, list) and not isinstance(value, list):
            return value in record_value
        return record_value == value

    def query_subagent_runs(self, filters: dict[str, Any]) -> list[SubagentRunRecord]:
        self._validate_subagent_filters(filters)
        results = []
        for record in self.list_subagent_runs():
            if all(self._matches_filter(record, key, value) for key, value in filters.items()):
                results.append(record)
        return results


def _first_matching(records: list[Any], field_name: str, value: str) -> Any | None:
    for record in records:
        if getattr(record, field_name) == value:
            return record
    return None


def _warn_malformed_jsonl_record(
    *,
    path: Path,
    line_number: int,
    record_type: Any,
    error: Exception,
    line: str,
) -> None:
    record_type_name = getattr(record_type, "__name__", str(record_type))
    error_text = _short_text(str(error).replace("\n", " "), limit=240)
    line_prefix = _short_text(line, limit=160)
    warnings.warn(
        "Skipping malformed trajectory JSONL record: "
        f"path={path}; line={line_number}; record_type={record_type_name}; "
        f"error={error_text}; prefix={line_prefix!r}",
        TrajectoryRegistryLoadWarning,
        stacklevel=3,
    )


def _short_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
