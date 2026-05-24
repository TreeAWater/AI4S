from __future__ import annotations

from typing import Any, Protocol, TypeVar
from uuid import uuid4

from evolab.contracts.records import (
    EvolutionRunRecord,
    LLMCallRecord,
    MetaAgentRunRecord,
    SubagentRunRecord,
    ToolCallTrajectoryRecord,
    TrajectoryEventRecord,
)
from evolab.contracts.tools import ToolCallRecord
from evolab.registries.trajectory import TrajectoryRegistry

RecordT = TypeVar("RecordT")


class TrajectoryPrivacyProcessor(Protocol):
    def process(self, record_kind: str, record: RecordT) -> RecordT:
        ...


class DirectTrajectoryPrivacyProcessor:
    def process(self, record_kind: str, record: RecordT) -> RecordT:
        return record


class TrajectoryCollector:
    """Central write path for trajectory data.

    The default processor stores records unchanged. Future DP or redaction logic
    should live behind TrajectoryPrivacyProcessor instead of being scattered
    across runtime call sites.
    """

    def __init__(
        self,
        registry: TrajectoryRegistry | None,
        *,
        privacy_processor: TrajectoryPrivacyProcessor | None = None,
    ) -> None:
        self.registry = registry
        self.privacy_processor = privacy_processor or DirectTrajectoryPrivacyProcessor()

    @property
    def enabled(self) -> bool:
        return self.registry is not None

    def save_llm_call(self, record: LLMCallRecord) -> str | None:
        if self.registry is None:
            return None
        return self.registry.save_llm_call(self._process("llm_call", record))

    def save_meta_agent_run(self, record: MetaAgentRunRecord) -> str | None:
        if self.registry is None:
            return None
        return self.registry.save_meta_agent_run(self._process("meta_agent_run", record))

    def save_subagent_run(self, record: SubagentRunRecord) -> str | None:
        if self.registry is None:
            return None
        return self.registry.save_subagent_run(self._process("subagent_run", record))

    def save_evolution_run(self, record: EvolutionRunRecord) -> str | None:
        if self.registry is None:
            return None
        return self.registry.save_evolution_run(self._process("evolution_run", record))

    def record_event(
        self,
        *,
        event_type: str,
        subject_type: str,
        subject_ref: str | None = None,
        task_id: str | None = None,
        run_ref: str | None = None,
        parent_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if self.registry is None:
            return None
        record = TrajectoryEventRecord(
            event_ref=f"event-{uuid4()}",
            event_type=event_type,
            subject_type=subject_type,
            subject_ref=subject_ref,
            task_id=task_id,
            run_ref=run_ref,
            parent_ref=parent_ref,
            metadata=metadata or {},
        )
        return self.registry.save_event(self._process("event", record))

    def record_tool_call(
        self,
        *,
        run_ref: str,
        task_id: str,
        record: ToolCallRecord,
        role: str | None = None,
        runtime_stage: str | None = None,
        step_index: int | None = None,
        workflow_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if self.registry is None:
            return None
        trajectory_record = ToolCallTrajectoryRecord(
            record_ref=f"tool-call-{uuid4()}",
            run_ref=run_ref,
            task_id=task_id,
            tool_call_id=record.tool_call.call_id,
            tool_name=record.tool_call.name,
            role=role,
            runtime_stage=runtime_stage,
            step_index=step_index,
            workflow_node_id=workflow_node_id,
            record=record,
            artifact_refs=record.result.artifact_refs,
            metadata=metadata or {},
        )
        return self.registry.save_tool_call_record(
            self._process("tool_call", trajectory_record)
        )

    def _process(self, record_kind: str, record: RecordT) -> RecordT:
        return self.privacy_processor.process(record_kind, record)
