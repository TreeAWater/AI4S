from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from evolab.contracts.retrieval import SkillBundle, SkillItem
from evolab.contracts.workflow import WorkflowEdge, WorkflowNode, WorkflowPlan


_MANDATORY_RELATIONS = {"depends_on", "requires", "consumes", "prerequisite"}
_OPTIONAL_RELATIONS = {"related_to", "complements", "validates", "produces", "alternative_to", "specializes"}
_PLANNING_RELATIONS = _MANDATORY_RELATIONS | _OPTIONAL_RELATIONS
_TARGET_BEFORE_SOURCE_RELATIONS = {"depends_on", "requires", "consumes", "prerequisite", "validates"}
_SOURCE_BEFORE_TARGET_RELATIONS = {"produces"}
_SECTION_PATTERN = re.compile(r"^([A-Za-z][A-Za-z ]+):\s*$")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class TopologicalSortResult:
    node_ids: list[str]
    edges: list[WorkflowEdge]
    warnings: list[str]


class SkillWorkflowPlanner:
    def plan(
        self,
        *,
        task_id: str | None,
        task_goal: str,
        role: str | None,
        skill_bundle: SkillBundle,
    ) -> WorkflowPlan:
        nodes = [_node_from_skill(index, skill) for index, skill in enumerate(skill_bundle.skills)]
        node_by_skill_id = {node.skill_id: node for node in nodes}
        edges: list[WorkflowEdge] = []
        planning_warnings: list[str] = []

        _extend_unique_edges(
            edges,
            _relationship_edges(skill_bundle, node_by_skill_id, planning_warnings),
        )
        _extend_unique_edges(edges, _io_inferred_edges(nodes))
        _extend_unique_edges(edges, _phase_order_edges(nodes, edges))

        sort_result = topologically_order_workflow_nodes(nodes, edges)
        planning_warnings.extend(sort_result.warnings)
        ordered_skill_ids = [_node_by_id(nodes)[node_id].skill_id for node_id in sort_result.node_ids]
        retrieval_trace = _dict(skill_bundle.metadata.get("retrieval_trace"))
        graph_context_summary = _dict(skill_bundle.metadata.get("graph_context_summary"))

        plan = WorkflowPlan(
            plan_id=_plan_id(task_id, role, [skill.skill_id for skill in skill_bundle.skills]),
            task_id=task_id,
            task_goal=task_goal,
            role=role,
            nodes=nodes,
            edges=sort_result.edges,
            required_tools=_dedupe([*skill_bundle.required_tools, *(tool for node in nodes for tool in node.required_tools)]),
            expected_artifacts=_dedupe(output for node in nodes for output in node.expected_outputs),
            metadata={
                "source_skill_bundle_backend_id": skill_bundle.backend_id,
                "graph_version_ref": skill_bundle.graph_version_ref,
                "skill_state_ref": skill_bundle.skill_state_ref,
                "graph_context_summary": graph_context_summary,
                "retrieval_trace": _retrieval_trace_summary(retrieval_trace),
                "planning_warnings": planning_warnings,
                "topological_order": ordered_skill_ids,
                "topological_node_order": sort_result.node_ids,
                "direct_skill_ids": _list(retrieval_trace.get("directly_matched_skill_ids")),
                "dependency_completed_skill_ids": _list(retrieval_trace.get("dependency_added_skill_ids")),
                "optional_expanded_skill_ids": _list(retrieval_trace.get("optional_expanded_skill_ids")),
            },
        )
        return plan


def topologically_order_workflow_nodes(
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
) -> TopologicalSortResult:
    node_ids = [node.node_id for node in nodes]
    active_edges = [
        edge
        for edge in edges
        if edge.source_node_id in set(node_ids) and edge.target_node_id in set(node_ids) and edge.source_node_id != edge.target_node_id
    ]
    warnings: list[str] = []

    while True:
        order, has_cycle = _kahn_order(node_ids, active_edges)
        if not has_cycle:
            return TopologicalSortResult(node_ids=order, edges=active_edges, warnings=warnings)

        removable = [
            edge
            for edge in active_edges
            if edge.metadata.get("edge_strength") != "mandatory"
        ]
        if not removable:
            removable = list(active_edges)
        edge_to_remove = sorted(removable, key=_edge_break_priority)[0]
        active_edges.remove(edge_to_remove)
        warnings.append(
            "removed cyclic workflow edge "
            f"{edge_to_remove.source_node_id}->{edge_to_remove.target_node_id} "
            f"({edge_to_remove.relation})"
        )


def topologically_order_plan(plan: WorkflowPlan) -> TopologicalSortResult:
    return topologically_order_workflow_nodes(plan.nodes, plan.edges)


def _kahn_order(node_ids: list[str], edges: list[WorkflowEdge]) -> tuple[list[str], bool]:
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        incoming[edge.target_node_id] += 1
        outgoing[edge.source_node_id].append(edge.target_node_id)
    ready = sorted([node_id for node_id, count in incoming.items() if count == 0], key=position.__getitem__)
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for target_id in sorted(outgoing[node_id], key=position.__getitem__):
            incoming[target_id] -= 1
            if incoming[target_id] == 0:
                ready.append(target_id)
                ready.sort(key=position.__getitem__)
    if len(ordered) == len(node_ids):
        return ordered, False
    remaining = [node_id for node_id in node_ids if node_id not in ordered]
    return [*ordered, *remaining], True


def _node_from_skill(index: int, skill: SkillItem) -> WorkflowNode:
    metadata = _dict(skill.metadata)
    return WorkflowNode(
        node_id=f"node-{index + 1:03d}-{_slug(skill.skill_id)}",
        skill_id=skill.skill_id,
        name=skill.name,
        purpose=_purpose(skill),
        required_inputs=_metadata_or_content_list(skill, "required_inputs", "Required Inputs"),
        expected_outputs=_metadata_or_content_list(skill, "expected_outputs", "Expected Outputs"),
        required_tools=_dedupe(skill.required_tools),
        resource_refs=_resource_refs(skill),
        metadata={
            "skill_metadata": metadata,
            "retrieval": _dict(metadata.get("retrieval")),
            "phase_rank": _phase_rank(skill),
        },
    )


def _relationship_edges(
    skill_bundle: SkillBundle,
    node_by_skill_id: dict[str, WorkflowNode],
    warnings: list[str],
) -> list[WorkflowEdge]:
    edges: list[WorkflowEdge] = []
    trace = _dict(skill_bundle.metadata.get("retrieval_trace"))
    steps = _list(trace.get("relation_expansion_steps"))
    graph_summary = _dict(skill_bundle.metadata.get("graph_context_summary"))
    steps.extend(_list(graph_summary.get("expanded_relationships")))
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        relation = raw_step.get("relation")
        source_skill_id = raw_step.get("source_skill_id")
        target_skill_id = raw_step.get("target_skill_id")
        if not all(isinstance(value, str) for value in (relation, source_skill_id, target_skill_id)):
            continue
        if relation not in _PLANNING_RELATIONS:
            continue
        source_node = node_by_skill_id.get(source_skill_id)
        target_node = node_by_skill_id.get(target_skill_id)
        if source_node is None or target_node is None:
            warnings.append(f"skipped workflow relation with missing skill node: {source_skill_id} {relation} {target_skill_id}")
            continue
        edge_source, edge_target = _oriented_nodes(source_node, target_node, relation)
        edges.append(
            WorkflowEdge(
                source_node_id=edge_source.node_id,
                target_node_id=edge_target.node_id,
                relation=relation,
                reason=str(raw_step.get("reason") or "retrieval_relationship"),
                metadata={
                    "source_skill_id": source_skill_id,
                    "target_skill_id": target_skill_id,
                    "edge_strength": "mandatory" if relation in _MANDATORY_RELATIONS else "optional",
                },
            )
        )
    return edges


def _oriented_nodes(source_node: WorkflowNode, target_node: WorkflowNode, relation: str) -> tuple[WorkflowNode, WorkflowNode]:
    if relation in _TARGET_BEFORE_SOURCE_RELATIONS:
        return target_node, source_node
    if relation in _SOURCE_BEFORE_TARGET_RELATIONS:
        return source_node, target_node
    return _phase_or_lexical_order(source_node, target_node)


def _io_inferred_edges(nodes: list[WorkflowNode]) -> list[WorkflowEdge]:
    edges: list[WorkflowEdge] = []
    for producer in nodes:
        output_tokens = _tokens(" ".join(producer.expected_outputs))
        if not output_tokens:
            continue
        for consumer in nodes:
            if producer.node_id == consumer.node_id:
                continue
            input_tokens = _tokens(" ".join(consumer.required_inputs))
            if not input_tokens or output_tokens.isdisjoint(input_tokens):
                continue
            source, target = _phase_or_lexical_order(producer, consumer)
            if source.node_id != producer.node_id:
                continue
            edges.append(
                WorkflowEdge(
                    source_node_id=producer.node_id,
                    target_node_id=consumer.node_id,
                    relation="inferred_io",
                    reason="expected_outputs overlap required_inputs",
                    metadata={"edge_strength": "inferred"},
                )
            )
    return edges


def _phase_order_edges(nodes: list[WorkflowNode], existing_edges: list[WorkflowEdge]) -> list[WorkflowEdge]:
    if len(nodes) < 2:
        return []
    connected = {(edge.source_node_id, edge.target_node_id) for edge in existing_edges}
    ordered = sorted(nodes, key=lambda node: (int(node.metadata.get("phase_rank", 100)), node.name, node.skill_id))
    edges: list[WorkflowEdge] = []
    for source, target in zip(ordered, ordered[1:]):
        if (source.node_id, target.node_id) in connected:
            continue
        edges.append(
            WorkflowEdge(
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                relation="phase_order",
                reason="deterministic scientific IE phase fallback",
                metadata={"edge_strength": "phase_order"},
            )
        )
    return edges


def _phase_or_lexical_order(left: WorkflowNode, right: WorkflowNode) -> tuple[WorkflowNode, WorkflowNode]:
    left_key = (int(left.metadata.get("phase_rank", 100)), left.name, left.skill_id)
    right_key = (int(right.metadata.get("phase_rank", 100)), right.name, right.skill_id)
    return (left, right) if left_key <= right_key else (right, left)


def _phase_rank(skill: SkillItem) -> int:
    text = f"{skill.skill_id} {skill.name}".casefold()
    phase_markers = [
        ("document_intake", 10),
        ("paper_structure_parsing", 20),
        ("section_localization", 30),
        ("supplementary_artifact_discovery", 40),
        ("multi_format_artifact_reading", 50),
        ("table_structure_understanding", 60),
        ("schema_interpretation", 70),
        ("field_mapping", 80),
        ("structured_record_construction", 90),
        ("ontology_alignment", 100),
        ("entity_normalization", 110),
        ("entity_validation", 120),
        ("negative_pattern_filtering", 130),
        ("result_validation", 140),
        ("deduplication", 150),
        ("conflict_resolution", 150),
        ("ground_truth", 160),
        ("evaluation", 160),
        ("reporting", 170),
        ("trajectory_pattern_mining", 180),
    ]
    for marker, rank in phase_markers:
        if marker in text:
            return rank
    return 1000


def _metadata_or_content_list(skill: SkillItem, metadata_key: str, section_title: str) -> list[str]:
    metadata = _dict(skill.metadata)
    value = metadata.get(metadata_key)
    if value is None:
        candidate_metadata = _dict(metadata.get("candidate_metadata"))
        value = candidate_metadata.get(metadata_key)
    parsed = _string_list(value)
    if parsed:
        return parsed
    return _content_section_list(skill.content, section_title)


def _content_section_list(content: str, section_title: str) -> list[str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = _SECTION_PATTERN.match(line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current is None or not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        elif re.match(r"^\d+\.\s+", line):
            line = re.sub(r"^\d+\.\s+", "", line).strip()
        sections.setdefault(current, []).append(line)
    return sections.get(section_title, [])


def _resource_refs(skill: SkillItem) -> list[Any]:
    refs: list[Any] = [*_list(skill.resource_refs)]
    metadata = _dict(skill.metadata)
    refs.extend(_list(metadata.get("resources")))
    refs.extend(_list(_dict(metadata.get("candidate_metadata")).get("resources")))
    return _dedupe_json(refs)


def _purpose(skill: SkillItem) -> str:
    for line in skill.content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.endswith(":"):
            return stripped
    return skill.name


def _retrieval_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "graph_version",
        "selected_category_paths",
        "returned_skill_ids",
        "directly_matched_skill_ids",
        "relation_expansion_steps",
        "dependency_added_skill_ids",
        "optional_expanded_skill_ids",
        "skipped_relationships",
        "conflict_warnings",
        "coverage_report",
    ]
    return {key: trace[key] for key in keys if key in trace}


def _extend_unique_edges(edges: list[WorkflowEdge], additions: list[WorkflowEdge]) -> None:
    seen = {(edge.source_node_id, edge.target_node_id, edge.relation) for edge in edges}
    for edge in additions:
        key = (edge.source_node_id, edge.target_node_id, edge.relation)
        if key in seen:
            continue
        edges.append(edge)
        seen.add(key)


def _node_by_id(nodes: list[WorkflowNode]) -> dict[str, WorkflowNode]:
    return {node.node_id: node for node in nodes}


def _edge_break_priority(edge: WorkflowEdge) -> tuple[int, str, str, str]:
    strength = str(edge.metadata.get("edge_strength") or "")
    strength_rank = {"phase_order": 0, "inferred": 1, "optional": 2, "mandatory": 3}.get(strength, 1)
    return (strength_rank, edge.relation, edge.source_node_id, edge.target_node_id)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(value) if len(token) > 2}


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").casefold()
    return slug or "skill"


def _plan_id(task_id: str | None, role: str | None, skill_ids: list[str]) -> str:
    seed = "|".join([task_id or "", role or "", *skill_ids])
    return f"workflow-{uuid5(NAMESPACE_URL, seed)}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _dedupe(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_json(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped

