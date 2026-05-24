from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.records import LLMCallRecord, MetaAgentRunRecord, SubagentRunRecord
from evolab.contracts.sft import SFTDatasetManifest, SFTDatasetSample
from evolab.registries.trajectory import TrajectoryRegistry


class SFTExportConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    dataset_id: str | None = None
    teacher_backend_ids: list[str] = Field(default_factory=list)
    runtime_stages: list[str] = Field(default_factory=lambda: ["subagent_flat", "workflow_node"])
    actions: list[str] = Field(default_factory=lambda: ["final_answer"])
    source_run_refs: list[str] = Field(default_factory=list)
    source_llm_call_refs: list[str] = Field(default_factory=list)
    include_meta_agent: bool = False
    include_tool_call_samples: bool = False
    val_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    shuffle: bool = False
    seed: int = 0
    max_samples: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SFTExportResult:
    manifest: SFTDatasetManifest
    train_samples: list[SFTDatasetSample]
    val_samples: list[SFTDatasetSample]
    manifest_path: Path
    train_path: Path
    val_path: Path


def export_sft_dataset(
    *,
    trajectory_registry: TrajectoryRegistry,
    output_dir: Path | str,
    config: SFTExportConfig | None = None,
) -> SFTExportResult:
    config = config or SFTExportConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = config.dataset_id or f"sft-{uuid4()}"

    all_calls = trajectory_registry.list_llm_calls()
    meta_by_run = {record.run_ref: record for record in trajectory_registry.list_meta_agent_runs()}
    subagent_by_run = {record.run_ref: record for record in trajectory_registry.list_subagent_runs()}
    calls_by_run = _calls_by_run(all_calls)
    selected_calls = _select_calls(all_calls, config)
    samples = [
        _sample_from_call(
            call,
            calls_by_run=calls_by_run,
            meta_by_run=meta_by_run,
            subagent_by_run=subagent_by_run,
        )
        for call in selected_calls
    ]
    if config.max_samples is not None:
        samples = samples[: config.max_samples]
    if config.shuffle:
        rng = random.Random(config.seed)
        samples = list(samples)
        rng.shuffle(samples)

    train_samples, val_samples = _split_samples(samples, config.val_fraction)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_samples(train_path, train_samples)
    _write_samples(val_path, val_samples)

    manifest = SFTDatasetManifest(
        dataset_id=dataset_id,
        train_path=str(train_path),
        val_path=str(val_path),
        sample_count=len(samples),
        train_count=len(train_samples),
        val_count=len(val_samples),
        source_llm_call_refs=[sample.source_llm_call_ref for sample in samples],
        source_run_refs=sorted({sample.source_run_ref for sample in samples}),
        teacher_backend_ids=sorted({sample.teacher_backend_id for sample in samples}),
        teacher_models=sorted({sample.teacher_model for sample in samples if sample.teacher_model}),
        selection=config.model_dump(mode="json"),
        metadata=config.metadata,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return SFTExportResult(
        manifest=manifest,
        train_samples=train_samples,
        val_samples=val_samples,
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
    )


def _select_calls(calls: list[LLMCallRecord], config: SFTExportConfig) -> list[LLMCallRecord]:
    selected = []
    for call in calls:
        action = _call_action(call)
        runtime_stage = _runtime_stage(call)
        if config.source_llm_call_refs and call.call_ref not in config.source_llm_call_refs:
            continue
        if config.source_run_refs and call.run_ref not in config.source_run_refs:
            continue
        if config.teacher_backend_ids and call.backend_id not in config.teacher_backend_ids:
            continue
        if not config.include_meta_agent and runtime_stage == "meta_agent_dispatch":
            continue
        if config.runtime_stages and runtime_stage not in config.runtime_stages:
            continue
        if action == "tool_call" and not config.include_tool_call_samples:
            continue
        if config.actions and action not in config.actions:
            continue
        if action == "final_answer" and not _has_non_empty_assistant_output(call):
            continue
        selected.append(call)
    return selected


def _sample_from_call(
    call: LLMCallRecord,
    *,
    calls_by_run: dict[str, list[LLMCallRecord]],
    meta_by_run: dict[str, MetaAgentRunRecord],
    subagent_by_run: dict[str, SubagentRunRecord],
) -> SFTDatasetSample:
    owner = subagent_by_run.get(call.run_ref) or meta_by_run.get(call.run_ref)
    messages = _reconstructed_messages(call, calls_by_run.get(call.run_ref, []))
    return SFTDatasetSample(
        sample_id=call.call_ref,
        messages=messages,
        source_llm_call_ref=call.call_ref,
        source_run_ref=call.run_ref,
        task_id=getattr(owner, "task_id", None),
        role=_sample_role(call, owner),
        teacher_backend_id=call.backend_id,
        teacher_model=call.model,
        action=_call_action(call),
        metadata={
            "runtime_stage": _runtime_stage(call),
            "llm_call_metadata": call.metadata,
        },
    )


def _reconstructed_messages(call: LLMCallRecord, run_calls: list[LLMCallRecord]) -> list[Message]:
    prior_tool_outputs = _prior_tool_outputs(call, run_calls)
    messages: list[Message] = []
    for message in call.input_messages:
        if message.role == "tool" and message.tool_call_id in prior_tool_outputs:
            tool_call_message = prior_tool_outputs[message.tool_call_id]
            if not _already_has_tool_call(messages, message.tool_call_id):
                messages.append(tool_call_message)
        messages.append(message.model_copy(deep=True))
    messages.extend(message.model_copy(deep=True) for message in call.output_messages)
    return messages


def _prior_tool_outputs(call: LLMCallRecord, run_calls: list[LLMCallRecord]) -> dict[str, Message]:
    outputs = {}
    for prior in run_calls:
        if prior.call_ref == call.call_ref:
            break
        for output_message in prior.output_messages:
            tool_call = output_message.metadata.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("call_id"), str):
                outputs[tool_call["call_id"]] = output_message.model_copy(deep=True)
    return outputs


def _already_has_tool_call(messages: list[Message], call_id: str) -> bool:
    for message in messages:
        tool_call = message.metadata.get("tool_call")
        if isinstance(tool_call, dict) and tool_call.get("call_id") == call_id:
            return True
    return False


def _split_samples(
    samples: list[SFTDatasetSample],
    val_fraction: float,
) -> tuple[list[SFTDatasetSample], list[SFTDatasetSample]]:
    if not samples or val_fraction <= 0:
        return samples, []
    val_count = int(round(len(samples) * val_fraction))
    if val_count >= len(samples):
        val_count = len(samples) - 1
    if val_count <= 0:
        return samples, []
    return samples[:-val_count], samples[-val_count:]


def _write_samples(path: Path, samples: list[SFTDatasetSample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(mode="json"), sort_keys=True) + "\n")


def _calls_by_run(calls: list[LLMCallRecord]) -> dict[str, list[LLMCallRecord]]:
    grouped: dict[str, list[LLMCallRecord]] = {}
    for call in calls:
        grouped.setdefault(call.run_ref, []).append(call)
    return grouped


def _call_action(call: LLMCallRecord) -> str:
    action = call.metadata.get("action")
    if isinstance(action, str):
        return action
    if not call.output_messages:
        return ""
    output_action = call.output_messages[0].metadata.get("action")
    return output_action if isinstance(output_action, str) else ""


def _runtime_stage(call: LLMCallRecord) -> str:
    stage = call.metadata.get("runtime_stage")
    return stage if isinstance(stage, str) else ""


def _has_non_empty_assistant_output(call: LLMCallRecord) -> bool:
    return any(message.role == "assistant" and bool(message.content.strip()) for message in call.output_messages)


def _sample_role(call: LLMCallRecord, owner: Any) -> str | None:
    role = call.metadata.get("role")
    if isinstance(role, str):
        return role
    owner_role = getattr(owner, "role", None)
    return owner_role if isinstance(owner_role, str) else None
