from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.dynamic_workflow import (
    DynamicSubAgentSpec,
    DynamicSubagentsConfig,
    DynamicWorkflowSpec,
    DynamicWorkflowTrace,
    DynamicWorkflowValidationReport,
)
from evolab.contracts.generated_tools import TaskEffectiveToolCatalog
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskRequest
from evolab.tools.runtime import ToolRuntime


LLMCallRecorder = Callable[
    [list[Message], LLMGenerationConfig, LLMRuntimeResponse, dict[str, Any]],
    str | None,
]


@dataclass
class DynamicPlanningOutcome:
    spec: DynamicWorkflowSpec | None
    validation_report: DynamicWorkflowValidationReport
    fallback_reason: dict[str, Any] | None = None
    llm_call_refs: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)


@dataclass
class DynamicRuntimeSubAgent:
    spec: DynamicSubAgentSpec
    role: RoleSpec
    skill_bundle: SkillBundle
    tool_names: list[str]
    provenance: dict[str, Any]


class DynamicWorkflowValidationError(ValueError):
    def __init__(self, report: DynamicWorkflowValidationReport):
        super().__init__("dynamic workflow validation failed: " + "; ".join(report.errors))
        self.report = report


class DynamicWorkflowPlanner:
    def __init__(
        self,
        *,
        planner_llm: Any,
        config: DynamicSubagentsConfig,
        tool_runtime: ToolRuntime,
        skill_backend: Any | None = None,
        available_llm_backend_ids: set[str] | None = None,
        llm_call_recorder: LLMCallRecorder | None = None,
        effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
    ) -> None:
        self.planner_llm = planner_llm
        self.config = config
        self.tool_runtime = tool_runtime
        self.skill_backend = skill_backend
        self.available_llm_backend_ids = available_llm_backend_ids or set()
        self.llm_call_recorder = llm_call_recorder
        self.effective_tool_catalog = effective_tool_catalog

    def plan(
        self,
        *,
        request: TaskRequest,
        work_item: dict[str, Any] | None = None,
        role_pool_templates: list[dict[str, Any]] | None = None,
    ) -> DynamicPlanningOutcome:
        messages = self._messages(request=request, work_item=work_item, role_pool_templates=role_pool_templates or [])
        raw_outputs: list[str] = []
        llm_call_refs: list[str] = []
        last_report = DynamicWorkflowValidationReport(
            valid=False,
            errors=["planner was not called"],
            planner_backend_id=self.config.planner_backend.backend_id if self.config.planner_backend else None,
            default_worker_backend_id=self.config.default_worker_backend.backend_id if self.config.default_worker_backend else None,
        )
        for attempt_index in range(self.config.max_planner_retries + 1):
            attempt_messages = messages
            if raw_outputs:
                attempt_messages = [
                    *messages,
                    Message(
                        role="user",
                        content=json.dumps(
                            {
                                "previous_output_was_invalid": True,
                                "validation_errors": last_report.errors,
                                "retry_index": attempt_index,
                                "required_response": "Return a corrected compact DynamicWorkflowSpec JSON object only.",
                                "compression_rules": [
                                    "Do not echo request payloads, work item source file lists, static role definitions, examples, or schemas beyond minimal required fields.",
                                    "metadata must be a small object and must not contain role_pool_templates, work_item, source text, article text, or copied prompt templates.",
                                    "Keep system_prompt, goal, constraints, acceptance_criteria, and planner_rationale_summary concise.",
                                    "Prefer four nodes for extraction tasks: context, extract, validate, write.",
                                ],
                            },
                            sort_keys=True,
                        ),
                    ),
                ]
            generation_config = LLMGenerationConfig(
                model="",
                temperature=0,
                max_output_tokens=_planner_max_output_tokens(self.config),
                metadata={"runtime_stage": "dynamic_workflow_planner", "attempt_index": attempt_index},
            )
            response = self.planner_llm.generate(attempt_messages, [], generation_config)
            if self.llm_call_recorder is not None:
                call_ref = self.llm_call_recorder(
                    attempt_messages,
                    generation_config,
                    response,
                    {"runtime_stage": "dynamic_workflow_planner", "attempt_index": attempt_index},
                )
                if call_ref is not None:
                    llm_call_refs.append(call_ref)
            if response.action.action != "final_answer":
                last_report = last_report.model_copy(
                    update={"errors": [f"planner returned unsupported action {response.action.action!r}"]}
                )
                raw_outputs.append("")
                continue
            raw_output = response.action.content or ""
            raw_outputs.append(raw_output)
            try:
                payload = _parse_planner_json_payload(raw_output)
                payload = _normalize_dynamic_planner_payload(payload)
                spec = DynamicWorkflowSpec.model_validate(payload)
                spec = _bind_dynamic_spec_to_work_item(spec, work_item)
            except Exception as exc:
                last_report = DynamicWorkflowValidationReport(
                    valid=False,
                    errors=[f"invalid planner JSON/spec: {exc}"],
                    planner_backend_id=self.config.planner_backend.backend_id if self.config.planner_backend else None,
                    default_worker_backend_id=self.config.default_worker_backend.backend_id if self.config.default_worker_backend else None,
                    metadata={
                        "attempt_index": attempt_index,
                        "raw_output_prefix": raw_output[:240],
                        "work_item_id": _work_item_id(work_item),
                    },
                )
                continue
            prepared_spec, report = validate_dynamic_workflow_spec(
                spec,
                config=self.config,
                available_llm_backend_ids=self.available_llm_backend_ids,
                tool_runtime=self.tool_runtime,
                skill_backend=self.skill_backend,
                task_id=request.task_id,
                effective_tool_catalog=self.effective_tool_catalog,
            )
            last_report = report
            if report.valid:
                return DynamicPlanningOutcome(
                    spec=prepared_spec,
                    validation_report=report,
                    llm_call_refs=llm_call_refs,
                    raw_outputs=raw_outputs,
                )
        fallback_reason = {
            "reason": "dynamic planning failed validation",
            "errors": last_report.errors,
            "planning_failure_mode": "fail_dynamic_task",
        }
        return DynamicPlanningOutcome(
            spec=None,
            validation_report=last_report,
            fallback_reason=fallback_reason,
            llm_call_refs=llm_call_refs,
            raw_outputs=raw_outputs,
        )

    def _messages(
        self,
        *,
        request: TaskRequest,
        work_item: dict[str, Any] | None,
        role_pool_templates: list[dict[str, Any]],
    ) -> list[Message]:
        effective_allowed_tool_names = _effective_allowed_tool_names(
            config=self.config,
            effective_tool_catalog=self.effective_tool_catalog,
        )
        payload = {
            "task_id": request.task_id,
            "task_goal": request.goal,
            "work_item": work_item or {},
            "scope": self.config.scope,
            "mode": self.config.mode,
            "configured_builtin_allowed_tool_names": list(self.config.allowed_tool_names),
            "effective_allowed_tool_names": effective_allowed_tool_names,
            "effective_tool_catalog": _effective_tool_catalog_json(
                self.effective_tool_catalog,
                tool_runtime=self.tool_runtime,
            ),
            "allowed_tool_names": effective_allowed_tool_names,
            "max_subagents": self.config.max_subagents,
            "max_subagents_per_work_item": self.config.max_subagents_per_work_item,
            "default_worker_backend_id": self.config.default_worker_backend.backend_id
            if self.config.default_worker_backend
            else None,
            "role_pool_templates": _planner_static_subagent_payloads(role_pool_templates),
            "required_response": {
                "format": "JSON only",
                "schema": "DynamicWorkflowSpec",
                "top_level_keys": [
                    "workflow_id",
                    "work_item_id",
                    "task_summary",
                    "article_context_summary",
                    "dynamic_subagents",
                    "workflow_nodes",
                    "workflow_edges",
                    "artifact_contracts",
                    "validation_rules",
                    "planner_rationale_summary",
                    "metadata",
                ],
                "forbidden_top_level_keys": ["name", "version", "description", "tasks", "workflow", "steps"],
                "template": _dynamic_workflow_response_template(
                    task_id=request.task_id,
                    work_item_id=_work_item_id(work_item),
                    default_worker_backend_id=(
                        self.config.default_worker_backend.backend_id if self.config.default_worker_backend else None
                    ),
                    allowed_tool_names=effective_allowed_tool_names,
                ),
                "requirements": [
                    "Define the minimal necessary runtime-only subagents.",
                    "Use exactly the top-level keys shown in required_response.template.",
                    "Do not wrap the response in name/version/tasks/workflow/steps.",
                    "Use allowed tools only.",
                    "Construct a valid DAG.",
                    "Every workflow_node.subagent_id must match a dynamic_subagents[].subagent_id.",
                    "Every workflow_edge may use only source_node_id, target_node_id, relation, reason, and metadata.",
                    "Put edge artifact names inside workflow_edge.metadata.artifact_names if needed.",
                    "artifact_contracts must be a JSON object, not a list.",
                    "For extraction tasks, include a context or file-survey node, an extraction-capable node, a validation or critic node or schema-validation step, and a writer/finalization node.",
                    "Do not generate a context-only or context-to-writer-only workflow for an extraction task.",
                    "When supplementary spreadsheets, workbooks, CSV/TSV, markdown tables, or table-like files are present, include a table/workbook inspection or triage node before extraction.",
                    "For scientific document extraction, prefer explicit intermediate artifacts such as document_inventory.json, candidate_source_files.json, candidate_tables.json, candidate_rows.json, candidate_records.json, validated_records.json, and final_records.jsonl when the allowed tools support them.",
                    "Do not include chain-of-thought or hidden reasoning.",
                    "Use planner_rationale_summary for a concise rationale only.",
                    "Keep metadata compact. Do not copy role_pool_templates, work item source lists, article text, prompt templates, or request payloads into the response.",
                    "Do not generate executable code.",
                    "Prefer two to six subagents unless context clearly requires more.",
                ],
            },
        }
        return [
            Message(
                role="system",
                content=(
                    "You are EvoLab DynamicWorkflowPlanner. Return only a valid JSON "
                    "DynamicWorkflowSpec. Never include chain-of-thought or executable code."
                ),
            ),
            Message(role="user", content=json.dumps(payload, indent=2, sort_keys=True)),
        ]


class DynamicSubAgentFactory:
    def __init__(
        self,
        *,
        config: DynamicSubagentsConfig,
        tool_runtime: ToolRuntime,
        skill_backend: Any,
        available_llm_backend_ids: set[str],
        runtime_policy: RuntimePolicy | None = None,
        agent_memory_backend: BackendBinding | None = None,
        static_roles: list[RoleSpec] | None = None,
        effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
    ) -> None:
        self.config = config
        self.tool_runtime = tool_runtime
        self.skill_backend = skill_backend
        self.available_llm_backend_ids = set(available_llm_backend_ids)
        self.runtime_policy = runtime_policy or RuntimePolicy()
        self.agent_memory_backend = agent_memory_backend
        self.static_roles_by_name = {role.name: role for role in (static_roles or [])}
        self.effective_tool_catalog = effective_tool_catalog

    def create(
        self,
        *,
        spec: DynamicSubAgentSpec,
        task_id: str,
        work_item_id: str | None,
        workflow_id: str,
        planner_backend_id: str | None,
    ) -> DynamicRuntimeSubAgent:
        default_worker = self.config.default_worker_backend.backend_id if self.config.default_worker_backend else None
        static_role = self.static_roles_by_name.get(spec.role_name)
        worker_backend_id = spec.llm_backend_id or (
            static_role.llm_backend.backend_id if static_role is not None else default_worker
        )
        if worker_backend_id not in self.available_llm_backend_ids:
            raise ValueError(f"dynamic subagent {spec.subagent_id!r} references unknown backend {worker_backend_id!r}")
        retrieval_request = _retrieval_request_for_dynamic_spec(spec, task_id=task_id)
        skill_bundle = self.skill_backend.get(retrieval_request)
        tool_names = _dedupe(
            [
                *spec.allowed_tools,
                *(static_role.allowed_tools if static_role is not None else []),
            ]
        )
        if self.config.allow_skill_required_tools:
            tool_names = _dedupe([*tool_names, *skill_bundle.required_tools])
        _validate_tool_names(
            tool_names,
            config=self.config,
            tool_runtime=self.tool_runtime,
            errors=[],
            context=f"dynamic subagent {spec.subagent_id!r}",
            raise_on_error=True,
            effective_tool_catalog=self.effective_tool_catalog,
        )
        tool_bundle = self.tool_runtime.prepare(
            required_tools=tool_names,
            allowed_tools=tool_names,
            policy=self.runtime_policy,
        )
        role_backend = (
            static_role.llm_backend
            if static_role is not None and static_role.llm_backend.backend_id == worker_backend_id
            else BackendBinding(backend_id=worker_backend_id)
        )
        required_skills = _dedupe(
            [
                *spec.required_skills,
                *(static_role.required_skills if static_role is not None else []),
            ]
        )
        role_metadata = {
            **(static_role.metadata if static_role is not None else {}),
            "dynamic_subagent_id": spec.subagent_id,
            "dynamic_workflow_id": workflow_id,
            "dynamic_spec_hash": _spec_hash(spec),
            "consumed_static_agents_md_role": static_role is not None,
        }
        role = RoleSpec(
            name=spec.role_name,
            system_prompt=_dynamic_role_system_prompt(spec, static_role),
            llm_backend=role_backend,
            agent_memory_backend=None,
            allowed_tools=[tool.name for tool in tool_bundle.tool_specs],
            required_skills=required_skills,
            memory_policy={},
            metadata=role_metadata,
        )
        provenance = {
            "planner_backend_id": planner_backend_id,
            "worker_backend_id": worker_backend_id,
            "workflow_id": workflow_id,
            "work_item_id": work_item_id,
            "spec_hash": _spec_hash(spec),
            "consumed_static_agents_md_role": static_role is not None,
            "static_role_prompt_hash": _text_hash(static_role.system_prompt) if static_role is not None else None,
            "generated_at": _utc_now(),
            "retrieval_request": retrieval_request.model_dump(mode="json"),
            "resolved_skill_ids": [skill.skill_id for skill in skill_bundle.skills],
            "prepared_tool_names": [tool.name for tool in tool_bundle.tool_specs],
        }
        return DynamicRuntimeSubAgent(
            spec=spec.model_copy(update={"llm_backend_id": worker_backend_id}),
            role=role,
            skill_bundle=skill_bundle,
            tool_names=[tool.name for tool in tool_bundle.tool_specs],
            provenance=provenance,
        )


def _planner_max_output_tokens(config: DynamicSubagentsConfig) -> int:
    metadata = config.metadata if isinstance(config.metadata, dict) else {}
    for key in ("planner_max_output_tokens", "max_planner_output_tokens"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return max(512, min(parsed, 16384))
    return 4096


def _parse_planner_json_payload(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as original_error:
        text = _strip_json_code_fence(raw_output).strip()
        if text and text != raw_output:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        if start >= 0:
            decoder = json.JSONDecoder()
            try:
                payload, _end = decoder.raw_decode(text[start:])
                return payload
            except json.JSONDecodeError:
                pass
        raise original_error


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if not lines:
        return text
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _planner_static_subagent_payloads(static_subagents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for raw in static_subagents:
        if not isinstance(raw, dict):
            continue
        item: dict[str, Any] = {}
        for key in ("name", "allowed_tools", "required_skills", "memory_policy", "metadata"):
            if key in raw:
                item[key] = raw[key]
        llm_backend = raw.get("llm_backend")
        if isinstance(llm_backend, dict) and isinstance(llm_backend.get("backend_id"), str):
            item["llm_backend_id"] = llm_backend["backend_id"]
        prompt = raw.get("system_prompt")
        if isinstance(prompt, str) and prompt.strip():
            item["system_prompt_summary"] = _compact_text(prompt, limit=240)
        compact.append(item)
    return compact


def _compact_text(text: str, *, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _dynamic_role_system_prompt(spec: DynamicSubAgentSpec, static_role: RoleSpec | None) -> str:
    if static_role is None:
        return spec.system_prompt
    base_prompt = static_role.system_prompt.strip()
    planner_prompt = spec.system_prompt.strip()
    if not planner_prompt or planner_prompt == base_prompt:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "Dynamic workflow assignment context from planner:\n"
        f"{planner_prompt}"
    )


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _work_item_id(work_item: dict[str, Any] | None) -> str | None:
    if not isinstance(work_item, dict):
        return None
    value = work_item.get("work_item_id") or work_item.get("id") or work_item.get("article_id")
    return str(value) if value not in (None, "") else None


def _bind_dynamic_spec_to_work_item(
    spec: DynamicWorkflowSpec,
    work_item: dict[str, Any] | None,
) -> DynamicWorkflowSpec:
    work_item_id = _work_item_id(work_item)
    if work_item_id is None:
        return spec
    metadata = dict(spec.metadata)
    metadata["work_item"] = dict(work_item or {})
    if spec.work_item_id not in (None, work_item_id):
        metadata["planner_work_item_id_overridden"] = spec.work_item_id
    return spec.model_copy(update={"work_item_id": work_item_id, "metadata": metadata})


def _dynamic_workflow_response_template(
    *,
    task_id: str,
    work_item_id: str | None,
    default_worker_backend_id: str | None,
    allowed_tool_names: list[str],
) -> dict[str, Any]:
    preferred_tools = [
        tool
        for tool in ["list_files", "read_text", "inspect_file_metadata", "inspect_table", "write_report"]
        if tool in allowed_tool_names
    ]
    if not preferred_tools:
        preferred_tools = list(allowed_tool_names[:3])
    extraction_tools = [
        tool
        for tool in [
            "build_document_inventory",
            "discover_candidate_source_files",
            "discover_candidate_tables",
            "extract_candidate_rows",
            "build_candidate_records",
            "read_text",
            "inspect_table",
            "read_table_slice",
            "profile_table",
            "write_jsonl",
            "write_report",
        ]
        if tool in allowed_tool_names
    ]
    validation_tools = [
        tool
        for tool in ["validate_candidate_records", "json_schema_validate", "inspect_table", "write_report"]
        if tool in allowed_tool_names
    ]
    writer_tools = [
        tool
        for tool in ["serialize_final_records", "json_schema_validate", "write_jsonl", "write_report"]
        if tool in allowed_tool_names
    ]
    work_item_suffix = work_item_id or task_id
    return {
        "workflow_id": f"dynamic-{work_item_suffix}",
        "work_item_id": work_item_id,
        "task_summary": "One-sentence summary of the assigned task scope.",
        "article_context_summary": "Concise summary of the current work item context.",
        "dynamic_subagents": [
            {
                "subagent_id": "context_agent",
                "role_name": "ContextAgent",
                "goal": "Inspect source context and identify evidence-bearing inputs.",
                "system_prompt": "Use only assigned tools to inspect evidence and write concise, source-grounded outputs.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "output_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "allowed_tools": preferred_tools,
                "required_skills": [],
                "skill_retrieval_request": None,
                "artifact_inputs": [],
                "artifact_outputs": ["context_summary.json"],
                "max_turns": 4,
                "max_tool_calls": 8,
                "llm_backend_id": default_worker_backend_id,
                "constraints": ["Use source-grounded evidence only."],
                "acceptance_criteria": ["Outputs cite inspected source artifacts."],
                "metadata": {},
            },
            {
                "subagent_id": "extraction_agent",
                "role_name": "ExecAgent",
                "goal": "Read concrete candidate sources and produce schema-constrained candidate records with source provenance.",
                "system_prompt": "Use only assigned tools to inspect source files/tables and write source-grounded candidate records.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "output_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "allowed_tools": extraction_tools or preferred_tools,
                "required_skills": [],
                "skill_retrieval_request": None,
                "artifact_inputs": ["context_summary.json"],
                "artifact_outputs": ["candidate_records.json"],
                "max_turns": 6,
                "max_tool_calls": 16,
                "llm_backend_id": default_worker_backend_id,
                "constraints": ["Extract records only from inspected source evidence."],
                "acceptance_criteria": ["Candidate records include article/work-item provenance and source references."],
                "metadata": {},
            },
            {
                "subagent_id": "validation_agent",
                "role_name": "CriticAgent",
                "goal": "Validate candidate records against evidence and schema before final writing.",
                "system_prompt": "Reject unsupported records and write validated record artifacts or an explicit empty-output audit.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "output_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "allowed_tools": validation_tools or writer_tools,
                "required_skills": [],
                "skill_retrieval_request": None,
                "artifact_inputs": ["candidate_records.json"],
                "artifact_outputs": ["validated_records.json"],
                "max_turns": 4,
                "max_tool_calls": 8,
                "llm_backend_id": default_worker_backend_id,
                "constraints": ["Do not use ground truth or unsupported assumptions."],
                "acceptance_criteria": ["Validated records are traceable to source evidence or explicitly rejected."],
                "metadata": {},
            },
            {
                "subagent_id": "writer_agent",
                "role_name": "SchemaWriterAgent",
                "goal": "Write schema-constrained final records from validated evidence.",
                "system_prompt": "Write JSON/JSONL-compatible records only from traceable evidence.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "output_schema": {"type": "object", "properties": {}, "additionalProperties": True},
                "allowed_tools": writer_tools,
                "required_skills": [],
                "skill_retrieval_request": None,
                "artifact_inputs": ["validated_records.json"],
                "artifact_outputs": ["final_records.jsonl"],
                "max_turns": 4,
                "max_tool_calls": 8,
                "llm_backend_id": default_worker_backend_id,
                "constraints": ["Do not invent records."],
                "acceptance_criteria": ["Final records are valid JSONL or an explicit empty result."],
                "metadata": {},
            },
        ],
        "workflow_nodes": [
            {
                "node_id": "context",
                "subagent_id": "context_agent",
                "input_artifacts": [],
                "output_artifacts": ["context_summary.json"],
                "dependencies": [],
                "execution_constraints": {},
            },
            {
                "node_id": "extract",
                "subagent_id": "extraction_agent",
                "input_artifacts": ["context_summary.json"],
                "output_artifacts": ["candidate_records.json"],
                "dependencies": ["context"],
                "execution_constraints": {},
            },
            {
                "node_id": "validate",
                "subagent_id": "validation_agent",
                "input_artifacts": ["candidate_records.json"],
                "output_artifacts": ["validated_records.json"],
                "dependencies": ["extract"],
                "execution_constraints": {},
            },
            {
                "node_id": "write",
                "subagent_id": "writer_agent",
                "input_artifacts": ["validated_records.json"],
                "output_artifacts": ["final_records.jsonl"],
                "dependencies": ["validate"],
                "execution_constraints": {},
            },
        ],
        "workflow_edges": [
            {
                "source_node_id": "context",
                "target_node_id": "extract",
                "metadata": {"artifact_names": ["context_summary.json"]},
            },
            {
                "source_node_id": "extract",
                "target_node_id": "validate",
                "metadata": {"artifact_names": ["candidate_records.json"]},
            },
            {
                "source_node_id": "validate",
                "target_node_id": "write",
                "metadata": {"artifact_names": ["validated_records.json"]},
            },
        ],
        "artifact_contracts": {
            "candidate_records.json": {
                "artifact_name": "candidate_records.json",
                "producer_node_id": "extract",
                "required": True,
            },
            "validated_records.json": {
                "artifact_name": "validated_records.json",
                "producer_node_id": "validate",
                "required": True,
            },
            "final_records.jsonl": {"artifact_name": "final_records.jsonl", "producer_node_id": "write", "required": True}
        },
        "validation_rules": ["Every record must be traceable to source evidence."],
        "planner_rationale_summary": "Concise rationale; no hidden reasoning.",
        "metadata": {},
    }


def _normalize_dynamic_planner_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    metadata = dict(normalized.get("metadata") or {})
    for key in (
        "task_id",
        "task_goal",
        "description",
        "version",
        "scope",
        "work_item",
        "top_level_keys",
        "role_pool_templates",
    ):
        if key in normalized:
            metadata.setdefault(key, normalized.pop(key))
    if metadata:
        normalized["metadata"] = metadata
    normalized.setdefault("planner_rationale_summary", "Dynamic workflow generated from task and work-item context.")

    agents = normalized.get("dynamic_subagents")
    if isinstance(agents, list):
        normalized_agents = []
        for raw_agent in agents:
            if not isinstance(raw_agent, dict):
                normalized_agents.append(raw_agent)
                continue
            agent = dict(raw_agent)
            if "goal" not in agent:
                for alias in ("task", "objective", "purpose"):
                    if isinstance(agent.get(alias), str) and agent[alias].strip():
                        agent["goal"] = agent.pop(alias)
                        break
            if "system_prompt" not in agent:
                for alias in ("prompt", "instructions", "instruction"):
                    if isinstance(agent.get(alias), str) and agent[alias].strip():
                        agent["system_prompt"] = agent.pop(alias)
                        break
            if "role_name" not in agent and isinstance(agent.get("name"), str):
                agent["role_name"] = agent.pop("name")
            if "role_name" not in agent and isinstance(agent.get("subagent_id"), str):
                agent["role_name"] = _role_name_from_subagent_id(agent["subagent_id"])
            if "system_prompt" not in agent and isinstance(agent.get("goal"), str):
                agent["system_prompt"] = (
                    "You are a runtime-only EvoLab dynamic subagent. "
                    f"Complete this goal using only assigned tools and source-grounded evidence: {agent['goal']}"
                )
            if "goal" not in agent and isinstance(agent.get("system_prompt"), str):
                agent["goal"] = agent["system_prompt"]
            if not isinstance(agent.get("output_schema"), dict) or not agent.get("output_schema"):
                agent["output_schema"] = {"type": "object", "additionalProperties": True}
            normalized_agents.append(agent)
        normalized["dynamic_subagents"] = normalized_agents

    nodes = normalized.get("workflow_nodes")
    if (not isinstance(nodes, list) or not nodes) and isinstance(normalized.get("dynamic_subagents"), list):
        synthesized_nodes = _synthesize_workflow_nodes_from_dynamic_subagents(normalized["dynamic_subagents"])
        if synthesized_nodes:
            normalized["workflow_nodes"] = synthesized_nodes
            nodes = synthesized_nodes
    if isinstance(nodes, list):
        normalized_nodes = []
        for raw_node in nodes:
            if not isinstance(raw_node, dict):
                normalized_nodes.append(raw_node)
                continue
            node = dict(raw_node)
            if isinstance(node.get("execution_constraints"), list):
                node["execution_constraints"] = {"constraints": node["execution_constraints"]}
            normalized_nodes.append(node)
        normalized["workflow_nodes"] = normalized_nodes
        if not isinstance(normalized.get("dynamic_subagents"), list) or not normalized.get("dynamic_subagents"):
            synthesized_agents = _synthesize_dynamic_subagents_from_nodes(normalized_nodes)
            if synthesized_agents:
                normalized["dynamic_subagents"] = synthesized_agents

    edges = normalized.get("workflow_edges")
    if isinstance(edges, list):
        normalized_edges = []
        allowed_edge_keys = {"schema_version", "source_node_id", "target_node_id", "relation", "reason", "metadata"}
        for raw_edge in edges:
            if not isinstance(raw_edge, dict):
                normalized_edges.append(raw_edge)
                continue
            edge = dict(raw_edge)
            edge_metadata = dict(edge.get("metadata") or {})
            for key in list(edge):
                if key not in allowed_edge_keys:
                    edge_metadata.setdefault(key, edge.pop(key))
            if edge_metadata:
                edge["metadata"] = edge_metadata
            normalized_edges.append(edge)
        normalized["workflow_edges"] = normalized_edges
    elif isinstance(normalized.get("workflow_nodes"), list):
        synthesized_edges = _synthesize_workflow_edges_from_nodes(normalized["workflow_nodes"])
        if synthesized_edges:
            normalized["workflow_edges"] = synthesized_edges

    contracts = normalized.get("artifact_contracts")
    if isinstance(contracts, list):
        normalized["artifact_contracts"] = {
            str(item.get("artifact_name") or item.get("name") or f"artifact_{index + 1}"): item
            for index, item in enumerate(contracts)
            if isinstance(item, dict)
        }

    return normalized


def _synthesize_workflow_nodes_from_dynamic_subagents(agents: list[Any]) -> list[dict[str, Any]]:
    """Recover a DAG when a planner emits dynamic agents but omits nodes.

    Planner models sometimes provide well-formed dynamic subagents and artifact
    contracts but forget the explicit ``workflow_nodes`` list. The agent order
    and artifact input/output declarations are enough to reconstruct a generic
    dependency chain without changing extraction semantics.
    """

    nodes: list[dict[str, Any]] = []
    used_node_ids: set[str] = set()
    artifact_producers: dict[str, str] = {}
    for index, raw_agent in enumerate(agents):
        if not isinstance(raw_agent, dict):
            continue
        subagent_id = raw_agent.get("subagent_id")
        if not isinstance(subagent_id, str) or not subagent_id.strip():
            continue
        role_name = raw_agent.get("role_name") if isinstance(raw_agent.get("role_name"), str) else ""
        node_id = _unique_dynamic_node_id(
            _node_id_from_dynamic_agent(subagent_id=subagent_id, role_name=role_name, agent=raw_agent),
            used_node_ids,
        )
        inputs = [str(item) for item in raw_agent.get("artifact_inputs", []) if isinstance(item, str)]
        outputs = [str(item) for item in raw_agent.get("artifact_outputs", []) if isinstance(item, str)]
        dependencies = [
            artifact_producers[item]
            for item in inputs
            if item in artifact_producers and artifact_producers[item] != node_id
        ]
        if nodes and _dynamic_agent_needs_prior_context(raw_agent):
            previous_node_id = nodes[-1]["node_id"]
            if previous_node_id not in dependencies:
                dependencies.append(previous_node_id)
        node = {
            "node_id": node_id,
            "subagent_id": subagent_id,
            "input_artifacts": inputs,
            "output_artifacts": outputs,
            "dependencies": _dedupe(dependencies),
            "execution_constraints": {},
            "metadata": {
                "synthesized_from_dynamic_subagent": subagent_id,
                "original_agent_index": index,
            },
        }
        nodes.append(node)
        for artifact in outputs:
            artifact_producers.setdefault(artifact, node_id)
    return nodes


def _synthesize_workflow_edges_from_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            continue
        target = raw_node.get("node_id")
        if not isinstance(target, str) or not target:
            continue
        for dependency in raw_node.get("dependencies", []) or []:
            if not isinstance(dependency, str) or not dependency:
                continue
            edges.append(
                {
                    "source_node_id": dependency,
                    "target_node_id": target,
                    "relation": "depends_on",
                    "reason": "synthesized from dynamic subagent artifact dependencies",
                    "metadata": {"synthesized": True},
                }
            )
    return edges


def _node_id_from_dynamic_agent(*, subagent_id: str, role_name: str, agent: dict[str, Any]) -> str:
    text = " ".join(
        [
            subagent_id,
            role_name,
            str(agent.get("goal") or ""),
            " ".join(str(item) for item in agent.get("artifact_outputs", []) if isinstance(item, str)),
        ]
    ).casefold()
    if any(token in text for token in ("write", "writer", "schema", "final", "serialize", "jsonl")):
        return "write"
    if any(token in text for token in ("validat", "critic", "review", "audit")):
        return "validate"
    if any(token in text for token in ("table", "workbook", "sheet", "column", "triage")):
        return "table"
    if any(token in text for token in ("extract", "exec", "candidate", "record", "sequence")):
        return "extract"
    if any(token in text for token in ("context", "survey", "file", "inventory", "source")):
        return "context"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", subagent_id).strip("_") or "node"


def _unique_dynamic_node_id(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _dynamic_agent_needs_prior_context(agent: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(agent.get("subagent_id") or ""),
            str(agent.get("role_name") or ""),
            str(agent.get("goal") or ""),
            " ".join(str(item) for item in agent.get("artifact_outputs", []) if isinstance(item, str)),
        ]
    ).casefold()
    return any(token in text for token in ("extract", "exec", "validat", "critic", "review", "write", "writer", "final"))


def _synthesize_dynamic_subagents_from_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    """Recover a usable DynamicWorkflowSpec when a planner emits nodes but omits agents.

    Some planner models follow the DAG part of the schema but forget the
    ``dynamic_subagents`` block. When the node list is otherwise explicit, this
    is a recoverable schema-shape issue: each unique ``subagent_id`` can be
    turned into an ephemeral runtime subagent with conservative, generic tool
    access for the node's role. This keeps dynamic mode from falling back to the
    static workflow on an LLM formatting omission.
    """

    agents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            continue
        subagent_id = raw_node.get("subagent_id")
        if not isinstance(subagent_id, str) or not subagent_id or subagent_id in seen:
            continue
        seen.add(subagent_id)
        node_id = raw_node.get("node_id") if isinstance(raw_node.get("node_id"), str) else subagent_id
        artifacts = [
            str(item)
            for item in [
                *(raw_node.get("input_artifacts") if isinstance(raw_node.get("input_artifacts"), list) else []),
                *(raw_node.get("output_artifacts") if isinstance(raw_node.get("output_artifacts"), list) else []),
            ]
        ]
        role_name = _role_name_from_dynamic_node(subagent_id=subagent_id, node_id=node_id, artifacts=artifacts)
        goal = _goal_from_dynamic_node(role_name=role_name, node_id=node_id, artifacts=artifacts)
        agents.append(
            {
                "subagent_id": subagent_id,
                "role_name": role_name,
                "goal": goal,
                "system_prompt": (
                    "You are an ephemeral EvoLab dynamic subagent synthesized from a planner DAG node. "
                    "Complete the assigned node using only allowed tools, source-grounded evidence, "
                    "and the declared artifact contract. Do not invent records or executable code."
                ),
                "input_schema": {"type": "object", "additionalProperties": True},
                "output_schema": {"type": "object", "additionalProperties": True},
                "allowed_tools": _default_tools_for_dynamic_role(role_name),
                "required_skills": [],
                "artifact_inputs": list(raw_node.get("input_artifacts") or []),
                "artifact_outputs": list(raw_node.get("output_artifacts") or []),
                "max_turns": 8,
                "max_tool_calls": 16,
                "constraints": ["Use source-grounded evidence only.", "Respect declared input/output artifacts."],
                "acceptance_criteria": ["Declared output artifacts are produced or an explicit failure reason is written."],
                "metadata": {"synthesized_from_workflow_node": node_id},
            }
        )
    return agents


def _role_name_from_dynamic_node(*, subagent_id: str, node_id: str, artifacts: list[str]) -> str:
    text = " ".join([subagent_id, node_id, *artifacts]).casefold()
    if any(token in text for token in ("write", "writer", "schema", "final", "report", "serialize")):
        return "SchemaWriterAgent"
    if any(token in text for token in ("validate", "validation", "critic", "review", "audit")):
        return "CriticAgent"
    if any(token in text for token in ("extract", "candidate", "row", "record")):
        return "ExecAgent"
    if any(token in text for token in ("context", "survey", "inventory", "source", "table", "workbook", "triage")):
        return "ContextAgent"
    return _role_name_from_subagent_id(subagent_id)


def _goal_from_dynamic_node(*, role_name: str, node_id: str, artifacts: list[str]) -> str:
    outputs = [artifact for artifact in artifacts if artifact]
    output_text = f" and produce {', '.join(outputs)}" if outputs else ""
    if role_name == "ContextAgent":
        return f"Survey source context for workflow node {node_id}{output_text}."
    if role_name == "ExecAgent":
        return f"Extract candidate records for workflow node {node_id}{output_text}."
    if role_name == "CriticAgent":
        return f"Validate candidate evidence for workflow node {node_id}{output_text}."
    if role_name == "SchemaWriterAgent":
        return f"Serialize final workflow artifacts for node {node_id}{output_text}."
    return f"Complete workflow node {node_id}{output_text}."


def _default_tools_for_dynamic_role(role_name: str) -> list[str]:
    common = ["read_text", "inspect_file_metadata", "write_report"]
    if role_name == "ContextAgent":
        return [
            "list_files",
            "read_text",
            "search_text",
            "inspect_file_metadata",
            "extract_sections",
            "inspect_excel_workbook",
            "inspect_table",
            "build_document_inventory",
            "discover_candidate_source_files",
            "discover_candidate_tables",
            "write_report",
        ]
    if role_name == "ExecAgent":
        return [
            "list_files",
            "read_text",
            "search_text",
            "inspect_file_metadata",
            "extract_sections",
            "inspect_excel_workbook",
            "read_excel_sheet",
            "inspect_table",
            "read_table_slice",
            "detect_table_header",
            "normalize_table",
            "profile_table",
            "build_document_inventory",
            "discover_candidate_source_files",
            "discover_candidate_tables",
            "extract_candidate_rows",
            "build_candidate_records",
            "write_jsonl",
            "write_report",
        ]
    if role_name == "CriticAgent":
        return [
            *common,
            "inspect_table",
            "read_table_slice",
            "profile_table",
            "validate_candidate_records",
            "json_schema_validate",
        ]
    if role_name == "SchemaWriterAgent":
        return [
            "read_text",
            "validate_candidate_records",
            "serialize_final_records",
            "json_schema_validate",
            "write_jsonl",
            "write_report",
        ]
    return common


def _role_name_from_subagent_id(subagent_id: str) -> str:
    parts = [part for part in subagent_id.replace("-", "_").split("_") if part]
    if not parts:
        return "DynamicSubAgent"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def validate_dynamic_workflow_spec(
    spec: DynamicWorkflowSpec,
    *,
    config: DynamicSubagentsConfig,
    available_llm_backend_ids: set[str],
    tool_runtime: ToolRuntime,
    skill_backend: Any | None,
    task_id: str,
    effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
) -> tuple[DynamicWorkflowSpec, DynamicWorkflowValidationReport]:
    errors: list[str] = []
    warnings: list[str] = []
    effective_allowed_tool_names = _effective_allowed_tool_names(
        config=config,
        effective_tool_catalog=effective_tool_catalog,
    )
    default_worker_backend_id = config.default_worker_backend.backend_id if config.default_worker_backend else None
    planner_backend_id = config.planner_backend.backend_id if config.planner_backend else None
    if planner_backend_id and planner_backend_id not in available_llm_backend_ids:
        errors.append(f"planner_backend {planner_backend_id!r} is not configured")
    if default_worker_backend_id and default_worker_backend_id not in available_llm_backend_ids:
        errors.append(f"default_worker_backend {default_worker_backend_id!r} is not configured")
    max_subagents = config.max_subagents_per_work_item if spec.work_item_id else config.max_subagents
    if len(spec.dynamic_subagents) > max_subagents:
        errors.append(f"dynamic workflow defines {len(spec.dynamic_subagents)} subagents, exceeds limit {max_subagents}")

    prepared_agents: list[DynamicSubAgentSpec] = []
    worker_backend_ids: list[str] = []
    resolved_skill_ids: list[str] = []
    for agent in spec.dynamic_subagents:
        worker_backend_id = agent.llm_backend_id or default_worker_backend_id
        if not worker_backend_id:
            errors.append(f"dynamic subagent {agent.subagent_id!r} has no worker backend")
        elif worker_backend_id not in available_llm_backend_ids:
            errors.append(f"dynamic subagent {agent.subagent_id!r} references unknown backend {worker_backend_id!r}")
        else:
            worker_backend_ids.append(worker_backend_id)
        if config.require_output_schema and not agent.output_schema:
            errors.append(f"dynamic subagent {agent.subagent_id!r} is missing output_schema")
        _validate_tool_names(
            agent.allowed_tools,
            config=config,
            tool_runtime=tool_runtime,
            errors=errors,
            context=f"dynamic subagent {agent.subagent_id!r}",
            effective_tool_catalog=effective_tool_catalog,
        )
        if skill_backend is not None and (agent.skill_retrieval_request or agent.required_skills):
            try:
                bundle = skill_backend.get(_retrieval_request_for_dynamic_spec(agent, task_id=task_id))
                resolved_skill_ids.extend(skill.skill_id for skill in bundle.skills)
                if config.allow_skill_required_tools:
                    _validate_tool_names(
                        bundle.required_tools,
                        config=config,
                        tool_runtime=tool_runtime,
                        errors=errors,
                        context=f"required skills for dynamic subagent {agent.subagent_id!r}",
                        effective_tool_catalog=effective_tool_catalog,
                    )
            except Exception as exc:
                errors.append(f"skill retrieval failed for dynamic subagent {agent.subagent_id!r}: {exc}")
        prepared_agents.append(agent.model_copy(update={"llm_backend_id": worker_backend_id}))

    prepared_spec = _prepare_dynamic_artifact_dependencies(
        spec.model_copy(update={"dynamic_subagents": prepared_agents}),
        warnings=warnings,
    )
    graph_errors = _dynamic_dag_errors(prepared_spec)
    errors.extend(graph_errors)
    errors.extend(_dynamic_extraction_workflow_errors(prepared_spec, config=config))
    report = DynamicWorkflowValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        planner_backend_id=planner_backend_id,
        default_worker_backend_id=default_worker_backend_id,
        worker_backend_ids=_dedupe(worker_backend_ids),
        allowed_tool_names=effective_allowed_tool_names,
        resolved_skill_ids=_dedupe(resolved_skill_ids),
        metadata={
            "workflow_id": spec.workflow_id,
            "work_item_id": spec.work_item_id,
            "configured_builtin_allowed_tool_names": list(config.allowed_tool_names),
            **(
                {
                    "effective_tool_catalog_task_id": effective_tool_catalog.task_id,
                    "generated_tool_names": list(effective_tool_catalog.generated_tool_names),
                }
                if effective_tool_catalog is not None
                else {}
            ),
        },
    )
    return prepared_spec, report


def _dynamic_extraction_workflow_errors(
    spec: DynamicWorkflowSpec,
    *,
    config: DynamicSubagentsConfig,
) -> list[str]:
    if not _dynamic_workflow_is_extraction_task(spec, config=config):
        return []
    errors: list[str] = []
    if not _has_dynamic_context_or_survey_node(spec):
        errors.append("extraction dynamic workflow requires a context or file-survey node")
    if not _has_dynamic_extraction_node(spec):
        errors.append("extraction dynamic workflow requires an extraction-capable node")
    if not _has_dynamic_validation_step(spec):
        errors.append("extraction dynamic workflow requires a validation, critic, or schema-validation step")
    if not _has_dynamic_writer_step(spec):
        errors.append("extraction dynamic workflow requires a writer or finalization node")
    if _work_item_has_structured_or_supplementary_sources(spec) and not _has_dynamic_table_or_workbook_step(spec):
        errors.append(
            "extraction dynamic workflow for a work item with supplementary or structured files "
            "requires a table/workbook inspection or triage node"
        )
    return errors


def _dynamic_workflow_is_extraction_task(
    spec: DynamicWorkflowSpec,
    *,
    config: DynamicSubagentsConfig,
) -> bool:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    config_metadata = config.metadata if isinstance(config.metadata, dict) else {}
    for source in (metadata, config_metadata):
        value = source.get("extraction_task")
        if value is not None:
            return bool(value)
    text = _dynamic_workflow_search_text(spec, include_work_item=False)
    extraction_terms = ("extract", "extraction", "record", "records", "sequence", "schema-constrained")
    return any(term in text for term in extraction_terms)


def _work_item_has_structured_or_supplementary_sources(spec: DynamicWorkflowSpec) -> bool:
    text = _dynamic_workflow_search_text(spec, include_work_item=True)
    if any(term in text for term in ("supplementary", "spreadsheet", "workbook", ".xlsx", ".xls", ".csv", ".tsv", "table")):
        return True
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    work_item = metadata.get("work_item")
    if isinstance(work_item, dict):
        for key in ("source_files", "exact_source_files", "files"):
            value = work_item.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, str):
                        continue
                    lowered = item.casefold()
                    if any(token in lowered for token in ("supplementary", ".xlsx", ".xls", ".csv", ".tsv")):
                        return True
    return False


def _has_dynamic_context_or_survey_node(spec: DynamicWorkflowSpec) -> bool:
    return _has_dynamic_node_matching(
        spec,
        role_terms=("context", "survey", "file", "inventory", "source", "document"),
        tool_terms=("list_files", "read_text", "inspect_file_metadata", "build_document_inventory"),
    )


def _has_dynamic_extraction_node(spec: DynamicWorkflowSpec) -> bool:
    return _has_dynamic_node_matching(
        spec,
        role_terms=("extract", "exec", "evidence", "candidate", "record", "sequence"),
        tool_terms=("extract_candidate_rows", "build_candidate_records", "write_jsonl"),
    )


def _has_dynamic_validation_step(spec: DynamicWorkflowSpec) -> bool:
    return _has_dynamic_node_matching(
        spec,
        role_terms=("critic", "validat", "review", "schema"),
        tool_terms=("validate_candidate_records", "json_schema_validate"),
    )


def _has_dynamic_writer_step(spec: DynamicWorkflowSpec) -> bool:
    return _has_dynamic_node_matching(
        spec,
        role_terms=("write", "writer", "final", "serialize"),
        tool_terms=("serialize_final_records", "write_jsonl", "write_report"),
        output_terms=("final_records", "biology_component_records", ".jsonl"),
    )


def _has_dynamic_table_or_workbook_step(spec: DynamicWorkflowSpec) -> bool:
    return _has_dynamic_node_matching(
        spec,
        role_terms=("table", "workbook", "sheet", "column", "triage", "recover"),
        tool_terms=(
            "inspect_table",
            "inspect_excel_workbook",
            "read_excel_sheet",
            "read_table_slice",
            "detect_table_header",
            "normalize_table",
            "profile_table",
            "discover_candidate_tables",
            "extract_candidate_rows",
        ),
    )


def _has_dynamic_node_matching(
    spec: DynamicWorkflowSpec,
    *,
    role_terms: tuple[str, ...] = (),
    tool_terms: tuple[str, ...] = (),
    output_terms: tuple[str, ...] = (),
) -> bool:
    for agent in spec.dynamic_subagents:
        agent_text = " ".join(
            [
                agent.subagent_id,
                agent.role_name,
                agent.goal,
                agent.system_prompt,
                " ".join(agent.constraints),
                " ".join(agent.acceptance_criteria),
                " ".join(agent.artifact_outputs),
            ]
        ).casefold()
        if role_terms and any(term in agent_text for term in role_terms):
            return True
        if tool_terms and any(tool in agent.allowed_tools for tool in tool_terms):
            return True
        if output_terms:
            outputs = " ".join(agent.artifact_outputs).casefold()
            if any(term in outputs for term in output_terms):
                return True
    return False


def _dynamic_workflow_search_text(spec: DynamicWorkflowSpec, *, include_work_item: bool) -> str:
    parts = [
        spec.task_summary,
        spec.article_context_summary or "",
        spec.planner_rationale_summary,
        " ".join(spec.validation_rules),
    ]
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    for key in ("task_goal", "description", "scope"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
    if include_work_item:
        work_item = metadata.get("work_item")
        if isinstance(work_item, dict):
            parts.append(json.dumps(work_item, sort_keys=True))
    return "\n".join(parts).casefold()


def _prepare_dynamic_artifact_dependencies(
    spec: DynamicWorkflowSpec,
    *,
    warnings: list[str],
) -> DynamicWorkflowSpec:
    agent_by_id = {agent.subagent_id: agent for agent in spec.dynamic_subagents}
    producer_nodes_by_artifact: dict[str, list[str]] = {}
    for node in spec.workflow_nodes:
        for artifact_name in _effective_node_output_artifacts(node, agent_by_id):
            producer_nodes_by_artifact.setdefault(artifact_name, []).append(node.node_id)

    prepared_nodes = []
    for node in spec.workflow_nodes:
        dependencies = list(node.dependencies)
        added: list[str] = []
        for input_artifact in node.input_artifacts:
            for producer_node_id in producer_nodes_by_artifact.get(input_artifact, []):
                if producer_node_id == node.node_id or producer_node_id in dependencies:
                    continue
                dependencies.append(producer_node_id)
                added.append(producer_node_id)
        if added:
            warnings.append(
                "added artifact dependency for dynamic workflow node "
                f"{node.node_id!r}: {', '.join(_dedupe(added))}"
            )
            node = node.model_copy(update={"dependencies": _dedupe(dependencies)})
        prepared_nodes.append(node)
    return spec.model_copy(update={"workflow_nodes": prepared_nodes})


def _effective_node_output_artifacts(node: Any, agent_by_id: dict[str, DynamicSubAgentSpec]) -> list[str]:
    outputs = [item for item in node.output_artifacts if isinstance(item, str) and item]
    if outputs:
        return _dedupe(outputs)
    agent = agent_by_id.get(node.subagent_id)
    if agent is None:
        return []
    return _dedupe([item for item in agent.artifact_outputs if isinstance(item, str) and item])


def dynamic_workflow_node_order(spec: DynamicWorkflowSpec) -> list[str]:
    node_ids = [node.node_id for node in spec.workflow_nodes]
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: [] for node_id in node_ids}
    for source, target in _dynamic_edges(spec):
        incoming[target] += 1
        outgoing[source].append(target)
    ready = [node_id for node_id in node_ids if incoming[node_id] == 0]
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for target in outgoing[node_id]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(order) != len(node_ids):
        raise ValueError("dynamic workflow DAG contains a cycle")
    return order


def persist_dynamic_workflow_artifacts(
    *,
    lab_root: Path,
    task_id: str,
    spec: DynamicWorkflowSpec | None,
    validation_report: DynamicWorkflowValidationReport,
    trace: DynamicWorkflowTrace | None = None,
    fallback_reason: dict[str, Any] | None = None,
) -> dict[str, str]:
    workflow_id = spec.workflow_id if spec is not None else str(validation_report.metadata.get("workflow_id") or "planning_failed")
    storage_id = _dynamic_workflow_storage_id(spec=spec, validation_report=validation_report, workflow_id=workflow_id)
    root = lab_root / "dynamic_workflows" / _safe_ref(task_id) / storage_id
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if spec is not None:
        paths["dynamic_workflow_spec"] = _write_json(root / "dynamic_workflow_spec.json", spec.model_dump(mode="json"))
        paths["dynamic_subagents"] = _write_json(
            root / "dynamic_subagents.json",
            [agent.model_dump(mode="json") for agent in spec.dynamic_subagents],
        )
    paths["planner_validation_report"] = _write_json(
        root / "planner_validation_report.json",
        validation_report.model_dump(mode="json"),
    )
    if trace is not None:
        paths["dynamic_workflow_trace"] = _write_json(root / "dynamic_workflow_trace.json", trace.model_dump(mode="json"))
        paths["dynamic_subagent_records"] = _write_jsonl(
            root / "dynamic_subagent_records.jsonl",
            trace.node_results,
        )
    if fallback_reason is not None:
        paths["fallback_reason"] = _write_json(root / "fallback_reason.json", fallback_reason)
    return paths


def _dynamic_workflow_storage_id(
    *,
    spec: DynamicWorkflowSpec | None,
    validation_report: DynamicWorkflowValidationReport,
    workflow_id: str,
) -> str:
    safe_workflow_id = _safe_ref(workflow_id)
    work_item_id = spec.work_item_id if spec is not None else validation_report.metadata.get("work_item_id")
    if not isinstance(work_item_id, str) or not work_item_id:
        return safe_workflow_id
    safe_work_item_id = _safe_ref(work_item_id)
    if safe_work_item_id in safe_workflow_id:
        return safe_workflow_id
    return _safe_ref(f"{safe_work_item_id}__{safe_workflow_id}")


def _retrieval_request_for_dynamic_spec(spec: DynamicSubAgentSpec, *, task_id: str) -> RetrievalRequest:
    payload = spec.skill_retrieval_request or {}
    if not isinstance(payload, dict):
        payload = {}
    query = payload.get("query")
    if not isinstance(query, str) or not query:
        query = " ".join([spec.goal, *spec.required_skills]).strip() or spec.role_name
    role = payload.get("role")
    if not isinstance(role, str) or not role:
        role = spec.role_name
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return RetrievalRequest(
        task_id=str(payload.get("task_id") or task_id),
        role=role,
        query=query,
        filters=filters,
        metadata={**metadata, "dynamic_subagent_id": spec.subagent_id},
    )


def _validate_tool_names(
    tool_names: list[str],
    *,
    config: DynamicSubagentsConfig,
    tool_runtime: ToolRuntime,
    errors: list[str],
    context: str,
    raise_on_error: bool = False,
    effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
) -> None:
    allowed = set(_effective_allowed_tool_names(config=config, effective_tool_catalog=effective_tool_catalog))
    local_errors: list[str] = []
    for tool_name in tool_names:
        if tool_name not in allowed:
            local_errors.append(f"{context} requests tool {tool_name!r} outside dynamic_subagents.allowed_tool_names")
        elif tool_runtime._get_effective_spec(tool_name) is None:
            local_errors.append(f"{context} requests unknown tool {tool_name!r}")
    if raise_on_error and local_errors:
        raise ValueError("; ".join(local_errors))
    errors.extend(local_errors)


def _effective_allowed_tool_names(
    *,
    config: DynamicSubagentsConfig,
    effective_tool_catalog: TaskEffectiveToolCatalog | None,
) -> list[str]:
    if effective_tool_catalog is None:
        return list(config.allowed_tool_names)
    return list(effective_tool_catalog.effective_allowed_tool_names)


def _effective_tool_catalog_json(
    effective_tool_catalog: TaskEffectiveToolCatalog | None,
    *,
    tool_runtime: ToolRuntime,
) -> dict[str, Any]:
    if effective_tool_catalog is None:
        return {}
    payload: dict[str, Any] = {}
    for name in effective_tool_catalog.effective_allowed_tool_names:
        spec = effective_tool_catalog.tool_specs_by_name.get(name) or tool_runtime._get_effective_spec(name)
        if spec is not None:
            payload[name] = spec.model_dump(mode="json")
    return payload


def _dynamic_dag_errors(spec: DynamicWorkflowSpec) -> list[str]:
    errors: list[str] = []
    try:
        dynamic_workflow_node_order(spec)
    except Exception as exc:
        errors.append(str(exc))
    return errors


def _dynamic_edges(spec: DynamicWorkflowSpec) -> list[tuple[str, str]]:
    edges = [(edge.source_node_id, edge.target_node_id) for edge in spec.workflow_edges]
    for node in spec.workflow_nodes:
        for dependency in node.dependencies:
            edges.append((dependency, node.node_id))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        deduped.append(edge)
    return deduped


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(_json_compatible(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("".join(json.dumps(_json_compatible(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _spec_hash(spec: DynamicSubAgentSpec) -> str:
    encoded = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"dynamic-subagent-sha256-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _safe_ref(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return safe or "dynamic"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
