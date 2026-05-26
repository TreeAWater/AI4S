from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.opsd import OPSDDatasetManifest, OPSDDatasetSample
from evolab.contracts.records import LLMCallRecord, MetaAgentRunRecord, SubagentRunRecord
from evolab.registries.trajectory import TrajectoryRegistry


class OPSDExportConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    dataset_id: str | None = None
    teacher_backend_ids: list[str] = Field(default_factory=list)
    source_run_refs: list[str] = Field(default_factory=list)
    source_llm_call_refs: list[str] = Field(default_factory=list)
    include_tool_choices: bool = True
    include_subagent_choices: bool = False
    val_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    shuffle: bool = False
    seed: int = 0
    max_samples: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class OPSDExportResult:
    manifest: OPSDDatasetManifest
    train_samples: list[OPSDDatasetSample]
    val_samples: list[OPSDDatasetSample]
    manifest_path: Path
    train_path: Path
    val_path: Path


def export_opsd_dataset(
    *,
    trajectory_registry: TrajectoryRegistry,
    output_dir: Path | str,
    config: OPSDExportConfig | None = None,
) -> OPSDExportResult:
    config = config or OPSDExportConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = config.dataset_id or f"opsd-{uuid4()}"

    all_calls = trajectory_registry.list_llm_calls()
    meta_by_run = {record.run_ref: record for record in trajectory_registry.list_meta_agent_runs()}
    subagent_by_run = {record.run_ref: record for record in trajectory_registry.list_subagent_runs()}
    selected_calls = _select_calls(all_calls, config)
    samples = [
        sample
        for call in selected_calls
        if (
            sample := _sample_from_call(
                call,
                config=config,
                meta_by_run=meta_by_run,
                subagent_by_run=subagent_by_run,
            )
        )
        is not None
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

    manifest = OPSDDatasetManifest(
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
        decision_types=sorted({sample.decision_type for sample in samples}),
        selection=config.model_dump(mode="json"),
        metadata=config.metadata,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return OPSDExportResult(
        manifest=manifest,
        train_samples=train_samples,
        val_samples=val_samples,
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
    )


def _select_calls(calls: list[LLMCallRecord], config: OPSDExportConfig) -> list[LLMCallRecord]:
    selected = []
    for call in calls:
        if config.source_llm_call_refs and call.call_ref not in config.source_llm_call_refs:
            continue
        if config.source_run_refs and call.run_ref not in config.source_run_refs:
            continue
        if config.teacher_backend_ids and call.backend_id not in config.teacher_backend_ids:
            continue
        runtime_stage = _runtime_stage(call)
        if runtime_stage == "subagent_flat" and config.include_tool_choices:
            selected.append(call)
        elif runtime_stage == "meta_agent_dispatch" and config.include_subagent_choices:
            selected.append(call)
    return selected


def _sample_from_call(
    call: LLMCallRecord,
    *,
    config: OPSDExportConfig,
    meta_by_run: dict[str, MetaAgentRunRecord],
    subagent_by_run: dict[str, SubagentRunRecord],
) -> OPSDDatasetSample | None:
    runtime_stage = _runtime_stage(call)
    if runtime_stage == "subagent_flat" and config.include_tool_choices:
        return _tool_choice_sample(call, subagent_by_run.get(call.run_ref))
    if runtime_stage == "meta_agent_dispatch" and config.include_subagent_choices:
        return _subagent_choice_sample(call, meta_by_run.get(call.run_ref))
    return None


def _tool_choice_sample(call: LLMCallRecord, owner: SubagentRunRecord | None) -> OPSDDatasetSample | None:
    action = _call_action(call)
    if action != "tool_call":
        return None
    tool_specs = _tool_specs(call)
    if not tool_specs:
        return None
    tool_call = _output_tool_call(call)
    if tool_call is None:
        return None
    tool_name = tool_call.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    return OPSDDatasetSample(
        sample_id=call.call_ref,
        decision_type="tool_choice",
        messages=[message.model_copy(deep=True) for message in call.input_messages],
        candidate_actions=tool_specs,
        chosen_action={
            "action": "tool_call",
            "tool_name": tool_name,
            "arguments": tool_call.get("arguments", {}),
            "call_id": tool_call.get("call_id"),
        },
        source_llm_call_ref=call.call_ref,
        source_run_ref=call.run_ref,
        task_id=getattr(owner, "task_id", None),
        role=_sample_role(call, owner),
        teacher_backend_id=call.backend_id,
        teacher_model=call.model,
        metadata={
            "runtime_stage": _runtime_stage(call),
            "llm_call_metadata": call.metadata,
        },
    )


def _subagent_choice_sample(call: LLMCallRecord, owner: MetaAgentRunRecord | None) -> OPSDDatasetSample | None:
    content = _last_user_content(call)
    if content is None:
        return None
    try:
        state = json.loads(content)
    except json.JSONDecodeError:
        return None
    roles = state.get("available_roles")
    if not isinstance(roles, list) or not roles:
        return None
    decision = _dispatch_decision(call, owner)
    if decision is None or decision.get("action") != "run_subagent":
        return None
    target_role = decision.get("target_role")
    if not isinstance(target_role, str) or not target_role:
        return None
    return OPSDDatasetSample(
        sample_id=call.call_ref,
        decision_type="subagent_choice",
        messages=[message.model_copy(deep=True) for message in call.input_messages],
        candidate_actions=[role for role in roles if isinstance(role, dict)],
        chosen_action={
            "action": "run_subagent",
            "target_role": target_role,
            "instruction": decision.get("instruction"),
            "retrieval_query": decision.get("retrieval_query"),
        },
        source_llm_call_ref=call.call_ref,
        source_run_ref=call.run_ref,
        task_id=getattr(owner, "task_id", None),
        role=_sample_role(call, owner),
        teacher_backend_id=call.backend_id,
        teacher_model=call.model,
        metadata={
            "runtime_stage": _runtime_stage(call),
            "llm_call_metadata": call.metadata,
            "dispatch_state": state,
        },
    )


def _tool_specs(call: LLMCallRecord) -> list[dict[str, Any]]:
    specs = call.metadata.get("tool_specs")
    if not isinstance(specs, list):
        return []
    return [spec for spec in specs if isinstance(spec, dict)]


def _output_tool_call(call: LLMCallRecord) -> dict[str, Any] | None:
    for message in call.output_messages:
        tool_call = message.metadata.get("tool_call")
        if isinstance(tool_call, dict):
            return tool_call
    return None


def _dispatch_decision(call: LLMCallRecord, owner: MetaAgentRunRecord | None) -> dict[str, Any] | None:
    if owner is not None:
        return owner.decision.model_dump(mode="json")
    for message in call.output_messages:
        if message.role != "assistant" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _last_user_content(call: LLMCallRecord) -> str | None:
    for message in reversed(call.input_messages):
        if message.role == "user" and message.content:
            return message.content
    return None


def _split_samples(
    samples: list[OPSDDatasetSample],
    val_fraction: float,
) -> tuple[list[OPSDDatasetSample], list[OPSDDatasetSample]]:
    if not samples or val_fraction <= 0:
        return samples, []
    val_count = int(round(len(samples) * val_fraction))
    if val_count >= len(samples):
        val_count = len(samples) - 1
    if val_count <= 0:
        return samples, []
    return samples[:-val_count], samples[-val_count:]


def _write_samples(path: Path, samples: list[OPSDDatasetSample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(mode="json"), sort_keys=True) + "\n")


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


def _sample_role(call: LLMCallRecord, owner: Any) -> str | None:
    role = call.metadata.get("role")
    if isinstance(role, str):
        return role
    owner_role = getattr(owner, "role", None)
    return owner_role if isinstance(owner_role, str) else None
