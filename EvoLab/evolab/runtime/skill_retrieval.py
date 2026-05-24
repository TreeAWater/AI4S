from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle
from evolab.contracts.tools import ToolBundle
from evolab.tools.runtime import ToolRuntime


class MissingRequiredToolError(RuntimeError):
    def __init__(self, missing_tools: Iterable[str]) -> None:
        self.missing_tools = tuple(_dedupe(missing_tools))
        missing = ", ".join(self.missing_tools)
        super().__init__(f"missing required tools for skill retrieval: {missing}")


@dataclass(frozen=True)
class PreparedSkillRuntimeContext:
    skill_bundle: SkillBundle
    tool_bundle: ToolBundle
    skill_context: dict[str, Any]


def prepare_skill_runtime_context(
    *,
    retrieval_request: RetrievalRequest,
    skill_backend: Any,
    tool_runtime: ToolRuntime | None,
    allowed_tools: Iterable[str],
    policy: RuntimePolicy,
    role_name: str | None = None,
) -> PreparedSkillRuntimeContext:
    skill_bundle = skill_backend.get(retrieval_request)
    skill_bundle = _scope_skill_bundle_for_assignment(
        skill_bundle,
        role_name=role_name or retrieval_request.role,
        query=retrieval_request.query,
        policy_metadata=policy.metadata,
    )
    required_tools = _dedupe(
        [
            *_aggregate_required_tools(skill_bundle),
            *_completion_guard_required_tools(policy.metadata, role_name),
        ]
    )
    if required_tools != skill_bundle.required_tools:
        skill_bundle = skill_bundle.model_copy(update={"required_tools": required_tools})

    if required_tools and tool_runtime is None:
        raise MissingRequiredToolError(required_tools)

    if tool_runtime is None:
        tool_bundle = ToolBundle()
    else:
        tool_bundle = tool_runtime.prepare(
            required_tools=required_tools,
            allowed_tools=allowed_tools,
            policy=policy,
            optional_tools=policy.allowed_human_tools if policy.allow_human_tools else None,
        )
        prepared_tool_names = [spec.name for spec in tool_bundle.tool_specs]
        missing_tools = [name for name in required_tools if name not in prepared_tool_names]
        if missing_tools:
            if not _allows_degraded_tool_preparation(policy.metadata):
                raise MissingRequiredToolError(missing_tools)
            skill_bundle = _bundle_for_available_tools(
                skill_bundle,
                available_tool_names=prepared_tool_names,
                unavailable_tool_names=missing_tools,
            )

    return PreparedSkillRuntimeContext(
        skill_bundle=skill_bundle,
        tool_bundle=tool_bundle,
        skill_context=build_skill_context(skill_bundle),
    )


def build_skill_context(skill_bundle: SkillBundle) -> dict[str, Any]:
    graph_context_summary = skill_bundle.metadata.get("graph_context_summary", {})
    tree_paths = []
    if isinstance(graph_context_summary, dict):
        tree_paths = graph_context_summary.get("retrieval_paths", [])
    context = {
        "backend_id": skill_bundle.backend_id,
        "graph_version_ref": skill_bundle.graph_version_ref,
        "skill_state_ref": skill_bundle.skill_state_ref,
        "selected_skills": [_skill_context_item(skill) for skill in skill_bundle.skills],
        "tree_paths": _json_compatible(tree_paths),
        "graph_context_summary": _json_compatible(graph_context_summary),
        "required_tools": skill_bundle.required_tools,
    }
    for key in (
        "unavailable_required_tools",
        "incompatible_skills",
        "compatible_skill_ids",
        "filtered_out_skill_ids",
        "skill_scope",
        "tool_preparation_scope",
    ):
        if key in skill_bundle.metadata:
            context[key] = _json_compatible(skill_bundle.metadata[key])
    return context


def _skill_context_item(skill: Any) -> dict[str, Any]:
    item = {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "required_tools": skill.required_tools,
        "retrieval": skill.metadata.get("retrieval", {}),
    }
    if skill.artifact_refs:
        item["artifact_refs"] = _json_compatible(skill.artifact_refs)
    if skill.script_refs:
        item["script_refs"] = _json_compatible(skill.script_refs)
    if skill.resource_refs:
        item["resource_refs"] = _json_compatible(skill.resource_refs)
    return item


def _aggregate_required_tools(skill_bundle: SkillBundle) -> list[str]:
    return _dedupe(
        [
            *skill_bundle.required_tools,
            *(tool for skill in skill_bundle.skills for tool in skill.required_tools),
        ]
    )


def _scope_skill_bundle_for_assignment(
    skill_bundle: SkillBundle,
    *,
    role_name: str | None,
    query: str,
    policy_metadata: dict[str, Any],
) -> SkillBundle:
    subagent_policy = policy_metadata.get("subagent_policy")
    if not isinstance(subagent_policy, dict):
        return skill_bundle
    if subagent_policy.get("skill_retrieval_scope") not in {
        "per_internal_dag_node",
        "per_assigned_task",
        "per_subagent_assignment",
    }:
        return skill_bundle
    scope = _skill_scope_for_role(role_name, query)
    if scope is None:
        return skill_bundle
    kept = []
    filtered = []
    for skill in skill_bundle.skills:
        if _skill_matches_scope(skill, scope):
            kept.append(skill)
        else:
            filtered.append(skill)
    if not kept:
        return skill_bundle.model_copy(
            update={
                "metadata": {
                    **skill_bundle.metadata,
                    "skill_scope": scope,
                    "skill_scope_warning": "scope filter matched no skills; original bundle retained",
                }
            }
        )
    kept_ids = {skill.skill_id for skill in kept}
    metadata = {
        **skill_bundle.metadata,
        "skill_scope": scope,
        "filtered_out_skill_ids": [skill.skill_id for skill in filtered],
    }
    trace = metadata.get("retrieval_trace")
    if isinstance(trace, dict):
        metadata["retrieval_trace"] = _filtered_retrieval_trace(trace, kept_ids)
    return skill_bundle.model_copy(
        update={
            "skills": kept,
            "required_tools": _dedupe(tool for skill in kept for tool in skill.required_tools),
            "metadata": metadata,
        }
    )


def _skill_scope_for_role(role_name: str | None, query: str) -> dict[str, Any] | None:
    role = (role_name or "").casefold()
    query_tokens = set(_tokens(query))
    scopes = {
        "surveyagent": {
            "generic_agent_type": "SurveyAgent",
            "include": {
                "survey",
                "intake",
                "inventory",
                "discover",
                "discovery",
                "structure",
                "parsing",
                "localization",
                "reading",
                "artifact",
                "document",
                "paper",
                "table",
                "spreadsheet",
                "source",
                "metadata",
            },
            "exclude": {
                "validation",
                "validate",
                "evaluation",
                "evaluate",
                "ground",
                "truth",
                "dedup",
                "deduplication",
                "conflict",
                "trajectory",
                "mining",
                "feedback",
                "write",
                "reporting",
                "construction",
                "construct",
                "normalization",
                "ontology",
                "alignment",
            },
        },
        "designagent": {
            "generic_agent_type": "DesignAgent",
            "include": {"design", "plan", "schema", "mapping", "strategy", "workflow", "validation", "coverage"},
            "exclude": {"ground", "truth", "evaluation", "trajectory", "mining", "feedback"},
        },
        "execagent": {
            "generic_agent_type": "ExecAgent",
            "include": {
                "execute",
                "extract",
                "extraction",
                "reading",
                "artifact",
                "table",
                "spreadsheet",
                "schema",
                "field",
                "mapping",
                "record",
                "construction",
                "entity",
                "evidence",
                "source",
                "normalization",
                "ontology",
                "alignment",
                "negative",
                "filtering",
            },
            "exclude": {"ground", "truth", "evaluation", "trajectory", "mining", "feedback"},
        },
        "criticagent": {
            "generic_agent_type": "CriticAgent",
            "include": {
                "validation",
                "validate",
                "evaluation",
                "evaluate",
                "dedup",
                "deduplication",
                "conflict",
                "negative",
                "filtering",
                "evidence",
                "audit",
            },
            "exclude": {"trajectory", "mining", "feedback"},
        },
        "writeagent": {
            "generic_agent_type": "WriteAgent",
            "include": {"write", "writing", "report", "reporting", "artifact", "output", "summary", "finalization"},
            "exclude": {
                "ground",
                "truth",
                "evaluation",
                "validate",
                "validation",
                "trajectory",
                "mining",
                "feedback",
                "table",
                "spreadsheet",
                "extraction",
                "extract",
                "ontology",
                "alignment",
                "normalization",
                "construction",
                "mapping",
            },
        },
    }
    scope = scopes.get(role)
    if scope is None:
        return None
    return {
        **scope,
        "query_tokens": sorted(query_tokens),
    }


def _skill_matches_scope(skill: Any, scope: dict[str, Any]) -> bool:
    content_text = " ".join(
        [
            getattr(skill, "skill_id", ""),
            getattr(skill, "name", ""),
            _scope_relevant_content(getattr(skill, "content", "")),
        ]
    ).casefold()
    metadata_text = _scope_relevant_metadata(getattr(skill, "metadata", {})).casefold()
    tokens = set(_tokens(content_text)) | set(_tokens(metadata_text))
    include = set(scope.get("include", set()))
    exclude = set(scope.get("exclude", set()))
    if set(_tokens(metadata_text)) & exclude:
        return False
    return bool(tokens & include)


def _filtered_retrieval_trace(trace: dict[str, Any], kept_ids: set[str]) -> dict[str, Any]:
    filtered = dict(trace)
    for key in ("returned_skill_ids", "directly_matched_skill_ids", "dependency_added_skill_ids", "optional_expanded_skill_ids"):
        value = filtered.get(key)
        if isinstance(value, list):
            filtered[key] = [item for item in value if item in kept_ids]
    steps = filtered.get("relation_expansion_steps")
    if isinstance(steps, list):
        filtered["relation_expansion_steps"] = [
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("source_skill_id") in kept_ids
            and step.get("target_skill_id") in kept_ids
        ]
    return filtered


def _completion_guard_required_tools(policy_metadata: dict[str, Any], role_name: str | None) -> list[str]:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    raw_required = guard_metadata.get("required_tool_calls_before_final", [])
    if not isinstance(raw_required, list):
        return []
    return [name for name in raw_required if isinstance(name, str) and name]


def _completion_guard_metadata(policy_metadata: dict[str, Any], role_name: str | None) -> dict[str, Any]:
    by_role = policy_metadata.get("completion_guards_by_role")
    if isinstance(by_role, dict):
        if isinstance(role_name, str):
            role_guards = by_role.get(role_name)
            if isinstance(role_guards, dict):
                return role_guards
        return {}
    return policy_metadata


def _allows_degraded_tool_preparation(policy_metadata: dict[str, Any]) -> bool:
    subagent_policy = policy_metadata.get("subagent_policy")
    if not isinstance(subagent_policy, dict):
        return False
    return subagent_policy.get("tool_preparation_scope") in {
        "per_internal_dag_node",
        "per_internal_dag_node_or_retrieved_skill",
    }


def _bundle_for_available_tools(
    skill_bundle: SkillBundle,
    *,
    available_tool_names: Iterable[str],
    unavailable_tool_names: Iterable[str],
) -> SkillBundle:
    available = set(available_tool_names)
    unavailable = _dedupe(unavailable_tool_names)
    compatible_skills = []
    incompatible_skills = []
    for skill in skill_bundle.skills:
        missing_for_skill = [name for name in skill.required_tools if name not in available]
        if missing_for_skill:
            incompatible_skills.append(
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "missing_tools": missing_for_skill,
                }
            )
            continue
        compatible_skills.append(skill)
    filtered_required_tools = _dedupe(tool for tool in skill_bundle.required_tools if tool in available)
    metadata = {
        **skill_bundle.metadata,
        "tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill",
        "unavailable_required_tools": unavailable,
        "incompatible_skills": incompatible_skills,
        "compatible_skill_ids": [skill.skill_id for skill in compatible_skills],
    }
    return skill_bundle.model_copy(
        update={
            "skills": compatible_skills,
            "required_tools": filtered_required_tools,
            "metadata": metadata,
        }
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _tokens(value: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9_]+", value.casefold())


def _json_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_json_text(item)}" for key, item in value.items())
    if isinstance(value, list | tuple | set):
        return " ".join(_json_text(item) for item in value)
    return "" if value is None else str(value)


def _scope_relevant_content(content: str) -> str:
    ignored_sections = {
        "validation signals",
        "smoke tests",
        "synthetic tests",
        "system tests",
        "benchmark tests",
        "environment assumptions",
    }
    kept: list[str] = []
    current_ignored = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_ignored = line[:-1].strip().casefold() in ignored_sections
            if current_ignored:
                continue
        if current_ignored:
            continue
        kept.append(line)
    return " ".join(kept)


def _scope_relevant_metadata(metadata: dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        return ""
    parts: list[str] = []
    for key in (
        "domain_tags",
        "task_types",
        "target_category",
        "tags",
        "summary",
        "description",
        "scope",
        "applicability",
        "candidate_metadata",
        "retrieval",
    ):
        value = metadata.get(key)
        if key == "candidate_metadata" and isinstance(value, dict):
            parts.append(
                _json_text(
                    {
                        candidate_key: value.get(candidate_key)
                        for candidate_key in (
                            "domain_tags",
                            "task_types",
                            "target_category",
                            "tags",
                            "summary",
                            "description",
                            "scope",
                            "applicability",
                        )
                        if candidate_key in value
                    }
                )
            )
            continue
        if key == "retrieval" and isinstance(value, dict):
            parts.append(_json_text({"matched_category_path": value.get("matched_category_path")}))
            continue
        parts.append(_json_text(value))
    return " ".join(parts)


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
