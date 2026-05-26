from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evolab.backends.skills.candidates import CandidateSkill
from evolab.backends.skills.graph_indexer import (
    CategoryIndex,
    GraphTreeIndexer,
    _get_ancestors,
    _get_subtree_category_ids,
)
from evolab.backends.skills.graph_schema import SkillCategoryNode, SkillGraph
from evolab.contracts.retrieval import RetrievalRequest, SkillRef


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_MANDATORY_COMPLETION_RELATIONS = {"depends_on", "requires", "consumes", "prerequisite"}
_HELPFUL_EXPANSION_RELATIONS = {
    "related_to",
    "complements",
    "validates",
    "produces",
    "alternative_to",
    "specializes",
    "refines",
}
_WARNING_ONLY_RELATIONS = {"conflicts_with", "replaces", "deprecated_by"}
_OPTIONAL_EXPANSION_MIN_SCORE = 0.1
_TOKEN_ALIASES = {
    "evaluation": {"evaluate"},
    "evaluate": {"evaluation"},
    "extraction": {"extract"},
    "extract": {"extraction"},
    "retrieval": {"retrieve"},
    "retrieve": {"retrieval"},
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class QueryInfo:
    query: str
    tokens: set[str]
    top_k: int | None
    metadata: dict[str, Any]
    context: str | None = None
    domain_tags: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    target_category: str | None = None
    requested_capabilities: tuple[str, ...] = ()
    requested_tasks: tuple[str, ...] = ()


@dataclass
class RetrievalCandidate:
    skill: CandidateSkill
    score: float
    source: str
    reasons: list[str] = field(default_factory=list)
    relation_path: list[str] = field(default_factory=list)
    matched_category_id: str | None = None
    matched_category_path: str | None = None
    retrieved_by: str | None = None


@dataclass(frozen=True)
class RetrievalPath:
    category_ids: tuple[str, ...]
    score: float

    @property
    def endpoint_category_id(self) -> str:
        return self.category_ids[-1]


@dataclass(frozen=True)
class GraphSkillSearchResult:
    ranked_candidates: list[RetrievalCandidate]
    skills: list[SkillRef]
    required_tools: list[str]
    metadata: dict[str, Any]
    query: QueryInfo


@dataclass(frozen=True)
class RelationshipExpansionResult:
    candidates_by_id: dict[str, RetrievalCandidate]
    expanded_relationships: list[dict[str, str]]
    relation_expansion_steps: list[dict[str, str]]
    dependency_added_skill_ids: list[str]
    optional_expanded_skill_ids: list[str]
    skipped_relationships: list[dict[str, str]]
    conflict_warnings: list[str]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_as_text(item)}" for key, item in value.items())
    return str(value)


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _metadata_value(request: RetrievalRequest, *keys: str) -> Any:
    for container in (request.metadata, request.filters):
        for key in keys:
            if key in container:
                return container[key]
    return None


def _parse_top_k(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _parse_query_info(request: RetrievalRequest) -> QueryInfo:
    context = _metadata_value(request, "context")
    context_text = _as_text(context) or None
    query_text = " ".join(part for part in (request.query, context_text) if part)
    metadata = {**request.filters, **request.metadata}
    target_category = _metadata_value(request, "target_category", "category_id")
    return QueryInfo(
        query=request.query,
        tokens=_tokens(query_text),
        top_k=_parse_top_k(_metadata_value(request, "top_k")),
        metadata=metadata,
        context=context_text,
        domain_tags=_as_string_tuple(_metadata_value(request, "domain_tags", "domains")),
        task_types=_as_string_tuple(_metadata_value(request, "task_types", "tasks")),
        required_tools=_as_string_tuple(_metadata_value(request, "required_tools", "tools")),
        target_category=target_category if isinstance(target_category, str) else None,
        requested_capabilities=_as_string_tuple(
            _metadata_value(
                request,
                "scientific_process_capability",
                "process_capability",
                "root_capability",
                "capability",
            )
        ),
        requested_tasks=_as_string_tuple(_metadata_value(request, "scientific_task", "task_category", "task")),
    )


def _category_tokens(category: Any) -> set[str]:
    return _tokens(
        " ".join(
            [
                category.category_id,
                category.name,
                category.description or "",
                _as_text(category.metadata.get("domain_tags")),
                _as_text(category.metadata.get("task_types")),
            ]
        )
    )


def _category_identity_tokens(category: Any) -> set[str]:
    return _tokens(" ".join([category.category_id, category.name]))


def _matches_category_name_or_id(value: str, category: Any) -> bool:
    normalized = value.casefold()
    return normalized in {category.category_id.casefold(), category.name.casefold()}


def _dedupe_categories(categories: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped = []
    for category in categories:
        if category.category_id in seen:
            continue
        seen.add(category.category_id)
        deduped.append(category)
    return deduped


def _match_root_capabilities(query: QueryInfo, index: CategoryIndex) -> list[Any]:
    requested_matches = [
        category
        for category in index.process_capabilities
        if any(_matches_category_name_or_id(value, category) for value in query.requested_capabilities)
    ]
    if requested_matches:
        return _dedupe_categories(requested_matches)

    matched: list[Any] = []
    for category in index.process_capabilities:
        if query.tokens and not query.tokens.isdisjoint(_category_tokens(category)):
            matched.append(category)
    return _dedupe_categories(matched)


def _match_scientific_tasks(query: QueryInfo, index: CategoryIndex, roots: list[Any]) -> list[Any]:
    candidates: list[SkillCategoryNode] = []
    if roots:
        for root in roots:
            candidates.extend(
                index.by_id[category_id]
                for category_id in index.get_descendants(root.category_id)
                if category_id in index.task_category_ids
            )
    else:
        candidates = list(index.scientific_tasks)

    exact_matches = [category for category in candidates if _task_category_has_exact_match(query, category)]
    if exact_matches:
        return _sort_task_matches(query, index, exact_matches)

    scored_token_matches = [(category, _task_category_score(query, index, category)) for category in candidates]
    max_score = max((score for _, score in scored_token_matches), default=0.0)
    token_matches = [category for category, score in scored_token_matches if score > 0 and score >= max_score * 0.5]
    if token_matches:
        return _sort_task_matches(query, index, token_matches)

    if roots:
        return _dedupe_categories(candidates)
    return []


def _task_category_metadata_values(category: SkillCategoryNode, key: str) -> tuple[str, ...]:
    return _as_string_tuple(category.metadata.get(key))


def _task_category_has_exact_match(query: QueryInfo, category: SkillCategoryNode) -> bool:
    return any(_matches_category_name_or_id(value, category) for value in query.requested_tasks) or (
        query.target_category is not None and _matches_category_name_or_id(query.target_category, category)
    )


def _task_category_score(query: QueryInfo, index: CategoryIndex, category: SkillCategoryNode) -> float:
    score = 0.0
    lexical_overlap = query.tokens & _category_tokens(category)
    if lexical_overlap:
        score += len(lexical_overlap) * 2.0
    if _task_category_has_exact_match(query, category):
        score += 100.0
    for tag in _task_category_metadata_values(category, "domain_tags"):
        if tag in query.domain_tags or (query.tokens and not query.tokens.isdisjoint(_tokens(tag))):
            score += 4.0
    for task_type in _task_category_metadata_values(category, "task_types"):
        if task_type in query.task_types or (query.tokens and not query.tokens.isdisjoint(_tokens(task_type))):
            score += 4.0
    if score > 0:
        score += index.depth_by_id.get(category.category_id, 0) * 0.25
    return score


def _sort_task_matches(query: QueryInfo, index: CategoryIndex, categories: list[SkillCategoryNode]) -> list[SkillCategoryNode]:
    return sorted(
        _dedupe_categories(categories),
        key=lambda category: (
            -_task_category_score(query, index, category),
            -index.depth_by_id.get(category.category_id, 0),
            category.name,
        ),
    )


def _requested_category_values(query: QueryInfo) -> tuple[str, ...]:
    values = [*query.requested_tasks]
    if query.target_category is not None:
        values.append(query.target_category)
    return tuple(values)


def _category_contains_requested_value(query: QueryInfo, index: CategoryIndex, category: SkillCategoryNode) -> bool:
    requested_values = _requested_category_values(query)
    if not requested_values:
        return False
    for category_id in _get_subtree_category_ids(index, category.category_id):
        descendant = index.by_id.get(category_id)
        if descendant is None:
            continue
        if any(_matches_category_name_or_id(value, descendant) for value in requested_values):
            return True
    return False


def _category_incremental_tokens(index: CategoryIndex, category: SkillCategoryNode, ancestor_ids: tuple[str, ...]) -> set[str]:
    tokens = _category_tokens(category)
    for ancestor_id in ancestor_ids:
        ancestor = index.by_id.get(ancestor_id)
        if ancestor is not None:
            tokens -= _category_tokens(ancestor)
    return tokens


def _category_incremental_identity_tokens(
    index: CategoryIndex,
    category: SkillCategoryNode,
    ancestor_ids: tuple[str, ...],
) -> set[str]:
    tokens = _category_identity_tokens(category)
    for ancestor_id in ancestor_ids:
        ancestor = index.by_id.get(ancestor_id)
        if ancestor is not None:
            tokens -= _category_identity_tokens(ancestor)
    return tokens


def _expanded_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_TOKEN_ALIASES.get(token, set()))
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
    return expanded


def _direct_child_category_score(
    query: QueryInfo,
    index: CategoryIndex,
    category: SkillCategoryNode,
    ancestor_ids: tuple[str, ...],
) -> float:
    score = 0.0
    if _task_category_has_exact_match(query, category):
        score += 100.0
    elif _category_contains_requested_value(query, index, category):
        score += 50.0

    query_tokens = _expanded_tokens(query.tokens)
    identity_overlap = query_tokens & _category_incremental_identity_tokens(index, category, ancestor_ids)
    incremental_overlap = query_tokens & _category_incremental_tokens(index, category, ancestor_ids)
    if identity_overlap:
        score += len(identity_overlap) * 3.0
    elif len(incremental_overlap) >= 2:
        score += len(incremental_overlap) * 1.0

    for tag in _task_category_metadata_values(category, "domain_tags"):
        if tag in query.domain_tags:
            score += 4.0

    for task_type in _task_category_metadata_values(category, "task_types"):
        if task_type in query.task_types or (query.tokens and not query.tokens.isdisjoint(_tokens(task_type))):
            score += 4.0

    if score > 0:
        score += index.depth_by_id.get(category.category_id, 0) * 0.25
    return score


def _match_direct_child_categories(
    query: QueryInfo,
    index: CategoryIndex,
    parent_category_id: str,
    path_ids: tuple[str, ...],
) -> list[tuple[SkillCategoryNode, float]]:
    scored_children = [
        (child, _direct_child_category_score(query, index, child, path_ids))
        for child in index.children_by_parent.get(parent_category_id, [])
        if child.category_id in index.task_category_ids
    ]
    return [(child, score) for child, score in scored_children if score > 0]


def _build_retrieval_paths(query: QueryInfo, index: CategoryIndex, matched_roots: list[Any]) -> list[RetrievalPath]:
    explicit_target_path = _explicit_target_category_path(query, index)
    if explicit_target_path is not None:
        return [explicit_target_path]

    roots = matched_roots or index.process_capabilities
    paths: list[RetrievalPath] = []
    explicit_root_match = bool(matched_roots)

    def descend(path_ids: tuple[str, ...], path_score: float, seen: set[str]) -> list[RetrievalPath]:
        current_id = path_ids[-1]
        if current_id in seen:
            return []
        child_matches = _match_direct_child_categories(query, index, current_id, path_ids)
        if not child_matches:
            return [RetrievalPath(category_ids=path_ids, score=round(path_score, 4))]

        child_paths: list[RetrievalPath] = []
        for child, child_score in child_matches:
            child_paths.extend(descend((*path_ids, child.category_id), path_score + child_score, seen | {current_id}))
        return child_paths

    for root in roots:
        root_paths = descend((root.category_id,), 0.0, set())
        if explicit_root_match:
            paths.extend(root_paths)
        else:
            paths.extend(path for path in root_paths if len(path.category_ids) > 1)

    deduped: dict[tuple[str, ...], RetrievalPath] = {}
    for path in paths:
        current = deduped.get(path.category_ids)
        if current is None or path.score > current.score:
            deduped[path.category_ids] = path
    return list(deduped.values())


def _explicit_target_category_path(query: QueryInfo, index: CategoryIndex) -> RetrievalPath | None:
    if query.target_category is None:
        return None
    for category in index.by_id.values():
        if not _matches_category_name_or_id(query.target_category, category):
            continue
        return RetrievalPath(
            category_ids=tuple(_category_path_ids(index, category.category_id)),
            score=100.0 + index.depth_by_id.get(category.category_id, 0) * 0.25,
        )
    return None


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(value) if token.lower() not in _STOP_WORDS}


def _raw_strings(values: list[str]) -> list[str]:
    return [value for value in values if isinstance(value, str)]


def _candidate_search_text(skill: CandidateSkill) -> str:
    return " ".join(
        [
            skill.name,
            skill.description,
            *skill.domain_tags,
            *skill.task_types,
            skill.target_category or "",
            skill.scope,
            *skill.applicability,
            *skill.procedure,
            *skill.required_tools,
            *skill.scripts,
            *skill.resources,
            *skill.examples,
        ]
    )


def _render_section(title: str, value: str | list[str]) -> str | None:
    if isinstance(value, str):
        if not value:
            return None
        return f"{title}:\n{value}"
    values = _raw_strings(value)
    if not values:
        return None
    if title == "Procedure":
        body = "\n".join(f"{index}. {item}" for index, item in enumerate(values, start=1))
    else:
        body = "\n".join(f"- {item}" for item in values)
    return f"{title}:\n{body}"


def _render_candidate_content(skill: CandidateSkill) -> str:
    sections = [
        _render_section("Description", skill.description),
        _render_section("Scope", skill.scope),
        _render_section("Applicability", skill.applicability),
        _render_section("Limitations", skill.limitations),
        _render_section("Required Inputs", skill.required_inputs),
        _render_section("Expected Outputs", skill.expected_outputs),
        _render_section("Dependencies", skill.dependencies),
        _render_section("Environment Assumptions", skill.environment_assumptions),
        _render_section("Procedure", skill.procedure),
        _render_section("Examples", skill.examples),
    ]
    return "\n\n".join(section for section in sections if section)


def _candidate_metadata(skill: CandidateSkill) -> dict[str, Any]:
    return {
        "source_type": skill.source_type,
        "source_uri": skill.source_uri,
        "provenance": skill.provenance,
        "domain_tags": skill.domain_tags,
        "task_types": skill.task_types,
        "target_category": skill.target_category,
        "scripts": skill.scripts,
        "resources": skill.resources,
        "smoke_tests": skill.smoke_tests,
        "synthetic_tests": skill.synthetic_tests,
        "system_tests": skill.system_tests,
        "benchmark_tests": skill.benchmark_tests,
        "validation_signals": skill.validation_signals,
        "confidence": skill.confidence,
        "candidate_metadata": skill.metadata,
    }


def _to_skill_ref(skill: CandidateSkill) -> SkillRef:
    return SkillRef(
        skill_id=skill.skill_id,
        name=skill.name,
        content=_render_candidate_content(skill),
        required_tools=sorted(set(skill.required_tools)),
        metadata=_candidate_metadata(skill),
    )


def _category_label(category_id: str, index: CategoryIndex) -> str:
    category = index.by_id.get(category_id)
    return category.name if category else category_id


def _category_path_ids(index: CategoryIndex, category_id: str) -> list[str]:
    return [*_get_ancestors(index, category_id), category_id]


def _category_path(index: CategoryIndex, category_id: str | None) -> str | None:
    if category_id is None or category_id not in index.by_id:
        return None
    return " > ".join(_category_label(path_id, index) for path_id in _category_path_ids(index, category_id))


def _score_candidate(
    skill: CandidateSkill,
    query: QueryInfo,
    index: CategoryIndex,
    matched_task_ids: set[str],
    source: str,
    relation_path: list[str],
) -> RetrievalCandidate:
    score = 0.0
    reasons: list[str] = []

    lexical_overlap = sorted(query.tokens & _tokens(_candidate_search_text(skill)))
    if lexical_overlap:
        score += min(len(lexical_overlap), 8) * 0.5
        reasons.append("lexical overlap")

    for tag in sorted(set(skill.domain_tags)):
        if tag in query.domain_tags or (query.tokens and not query.tokens.isdisjoint(_tokens(tag))):
            score += 2.0
            reasons.append(f"domain tag match: {tag}")

    for task_type in sorted(set(skill.task_types)):
        if task_type in query.task_types or (query.tokens and not query.tokens.isdisjoint(_tokens(task_type))):
            score += 2.0
            reasons.append(f"task type match: {task_type}")

    for tool in sorted(set(skill.required_tools)):
        if tool in query.required_tools or (query.tokens and not query.tokens.isdisjoint(_tokens(tool))):
            score += 1.5
            reasons.append(f"required tool match: {tool}")

    if skill.target_category and query.target_category == skill.target_category:
        score += 3.0
        reasons.append(f"target category match: {skill.target_category}")

    if skill.target_category and skill.target_category in matched_task_ids:
        score += 3.0
        reasons.append(f"matched scientific task {_category_label(skill.target_category, index)}")

    if source == "belongs_to_category" and relation_path and relation_path[-1] in matched_task_ids:
        score += 3.0
        reasons.append(f"matched scientific task {_category_label(relation_path[-1], index)}")

    if not reasons:
        reasons.append("relationship expansion")

    return RetrievalCandidate(
        skill=skill,
        score=round(score, 4),
        source=source,
        reasons=reasons,
        relation_path=relation_path,
        matched_category_id=relation_path[-1] if relation_path else None,
        matched_category_path=_category_path(index, relation_path[-1]) if relation_path else None,
    )


def _merge_retrieval_candidate(candidates_by_id: dict[str, RetrievalCandidate], candidate: RetrievalCandidate) -> None:
    current = candidates_by_id.get(candidate.skill.skill_id)
    if current is None:
        candidates_by_id[candidate.skill.skill_id] = candidate
        return
    if candidate.score > current.score:
        current.score = candidate.score
        current.source = candidate.source
        current.relation_path = candidate.relation_path
        current.matched_category_id = candidate.matched_category_id
        current.matched_category_path = candidate.matched_category_path
        current.retrieved_by = candidate.retrieved_by
    for reason in candidate.reasons:
        if reason not in current.reasons:
            current.reasons.append(reason)


def _path_retrieval_category_ids(index: CategoryIndex, retrieval_paths: list[RetrievalPath]) -> set[str]:
    return {category_id for path in retrieval_paths for category_id in path.category_ids if category_id in index.by_id}


def _retrieval_path_for_category(retrieval_paths: list[RetrievalPath], category_id: str) -> tuple[RetrievalPath, str] | None:
    endpoint_paths = [path for path in retrieval_paths if path.endpoint_category_id == category_id]
    if endpoint_paths:
        return max(endpoint_paths, key=lambda path: (path.score, len(path.category_ids))), "direct"
    ancestor_paths = [path for path in retrieval_paths if category_id in path.category_ids]
    if ancestor_paths:
        return max(ancestor_paths, key=lambda path: (path.score, len(path.category_ids))), "path_ancestor"
    return None


def _retrieve_path_seed_candidates(
    graph: SkillGraph,
    query: QueryInfo,
    index: CategoryIndex,
    retrieval_paths: list[RetrievalPath],
) -> tuple[dict[str, RetrievalCandidate], set[str]]:
    path_category_ids = _path_retrieval_category_ids(index, retrieval_paths)
    endpoint_category_ids = {path.endpoint_category_id for path in retrieval_paths}
    candidates_by_id: dict[str, RetrievalCandidate] = {}

    if not retrieval_paths:
        for skill in graph.skills:
            if not isinstance(skill, CandidateSkill):
                continue
            candidate = _score_candidate(skill, query, index, set(), "lexical", [])
            if candidate.score > 0:
                _merge_retrieval_candidate(candidates_by_id, candidate)
        return candidates_by_id, set()

    active_belongs_to_edges = [
        edge
        for edge in graph.edges
        if edge.relation == "belongs_to_category" and not edge.deprecated and edge.target_id in path_category_ids
    ]
    belongs_to_by_skill: dict[str, list[str]] = {}
    for edge in active_belongs_to_edges:
        belongs_to_by_skill.setdefault(edge.source_id, []).append(edge.target_id)

    for skill in graph.skills:
        if not isinstance(skill, CandidateSkill):
            continue
        category_matches: list[tuple[str, str]] = []
        if skill.target_category and skill.target_category in path_category_ids:
            category_path = _retrieval_path_for_category(retrieval_paths, skill.target_category)
            if category_path is not None:
                category_matches.append((skill.target_category, "target_category"))
        for category_id in belongs_to_by_skill.get(skill.skill_id, []):
            if _retrieval_path_for_category(retrieval_paths, category_id) is not None:
                category_matches.append((category_id, "belongs_to_category"))

        for category_id, source in category_matches:
            path_match = _retrieval_path_for_category(retrieval_paths, category_id)
            if path_match is None:
                continue
            _, retrieved_by = path_match
            candidate = _score_candidate(skill, query, index, endpoint_category_ids, source, [category_id])
            candidate.matched_category_id = category_id
            candidate.matched_category_path = _category_path(index, category_id)
            candidate.retrieved_by = retrieved_by
            path = candidate.matched_category_path
            if path:
                candidate.reasons.append(f"category path: {path}")
            _merge_retrieval_candidate(candidates_by_id, candidate)

    return candidates_by_id, path_category_ids


def _expand_relationships(
    graph: SkillGraph,
    seeds_by_id: dict[str, RetrievalCandidate],
) -> RelationshipExpansionResult:
    skill_by_id = {skill.skill_id: skill for skill in graph.skills if isinstance(skill, CandidateSkill)}
    candidates_by_id = dict(seeds_by_id)
    expanded: list[dict[str, str]] = []
    relation_expansion_steps: list[dict[str, str]] = []
    dependency_added_skill_ids: list[str] = []
    optional_expanded_skill_ids: list[str] = []
    skipped_relationships: list[dict[str, str]] = []
    conflict_warnings: list[str] = []

    seed_ids = set(seeds_by_id)
    sorted_seed_candidates = sorted(
        seeds_by_id.values(),
        key=lambda candidate: (-candidate.score, -candidate.skill.confidence, candidate.skill.name),
    )
    sorted_seed_ids = [candidate.skill.skill_id for candidate in sorted_seed_candidates]

    for edge in graph.edges:
        if edge.deprecated or edge.relation == "belongs_to_category":
            continue
        source_is_seed = edge.source_id in seed_ids
        target_is_seed = edge.target_id in seed_ids
        if not source_is_seed and not target_is_seed:
            continue
        if edge.relation not in _WARNING_ONLY_RELATIONS:
            continue

        seed_id = edge.source_id if source_is_seed else edge.target_id
        neighbor_id = edge.target_id if source_is_seed else edge.source_id
        warning = f"{seed_id} {edge.relation} {neighbor_id}"
        if warning not in conflict_warnings:
            conflict_warnings.append(warning)
        skipped = {
            "source_skill_id": seed_id,
            "target_skill_id": neighbor_id,
            "relation": edge.relation,
            "reason": "warning_only",
        }
        if skipped not in skipped_relationships:
            skipped_relationships.append(skipped)

    for seed_id in sorted_seed_ids:
        seed = seeds_by_id[seed_id]
        for edge in graph.edges:
            if edge.deprecated or edge.relation not in _MANDATORY_COMPLETION_RELATIONS:
                continue
            if edge.source_id != seed_id:
                continue
            dependency_id = edge.target_id
            if dependency_id not in skill_by_id:
                skipped_relationships.append(
                    {
                        "source_skill_id": seed_id,
                        "target_skill_id": dependency_id,
                        "relation": edge.relation,
                        "reason": "missing_target_skill",
                    }
                )
                continue
            if dependency_id not in candidates_by_id:
                dependency = skill_by_id[dependency_id]
                weight = edge.weight if edge.weight is not None else 1.0
                relation_candidate = RetrievalCandidate(
                    skill=dependency,
                    score=round(seed.score * weight, 4),
                    source=f"relationship:{edge.relation}",
                    reasons=[f"dependency_added via {edge.relation} from {seed_id}"],
                    relation_path=[seed_id, edge.relation, dependency_id],
                )
                _merge_retrieval_candidate(candidates_by_id, relation_candidate)
            if dependency_id not in dependency_added_skill_ids:
                dependency_added_skill_ids.append(dependency_id)
            step = {
                "source_skill_id": seed_id,
                "target_skill_id": dependency_id,
                "relation": edge.relation,
                "reason": "dependency_added",
            }
            if step not in relation_expansion_steps:
                relation_expansion_steps.append(step)
            expanded_step = {
                "source_skill_id": seed_id,
                "relation": edge.relation,
                "target_skill_id": dependency_id,
            }
            if expanded_step not in expanded:
                expanded.append(expanded_step)

    for seed_id in sorted_seed_ids:
        seed = seeds_by_id[seed_id]
        for edge in graph.edges:
            if edge.deprecated or edge.relation not in _HELPFUL_EXPANSION_RELATIONS:
                continue
            source_is_seed = edge.source_id == seed_id
            target_is_seed = edge.target_id == seed_id
            if not source_is_seed and not target_is_seed:
                continue
            neighbor_id = edge.target_id if source_is_seed else edge.source_id
            if neighbor_id not in skill_by_id:
                skipped_relationships.append(
                    {
                        "source_skill_id": seed_id,
                        "target_skill_id": neighbor_id,
                        "relation": edge.relation,
                        "reason": "missing_target_skill",
                    }
                )
                continue
            if neighbor_id in candidates_by_id:
                continue

            weight = edge.weight if edge.weight is not None else 0.5
            relation_score = round(seed.score * weight, 4)
            if relation_score < _OPTIONAL_EXPANSION_MIN_SCORE:
                skipped_relationships.append(
                    {
                        "source_skill_id": seed_id,
                        "target_skill_id": neighbor_id,
                        "relation": edge.relation,
                        "reason": "below_score_threshold",
                    }
                )
                continue

            neighbor = skill_by_id[neighbor_id]
            relation_candidate = RetrievalCandidate(
                skill=neighbor,
                score=relation_score,
                source=f"relationship:{edge.relation}",
                reasons=[f"one-hop {edge.relation} from {seed_id}"],
                relation_path=[seed_id, edge.relation, neighbor_id],
            )
            _merge_retrieval_candidate(candidates_by_id, relation_candidate)
            if neighbor_id not in optional_expanded_skill_ids:
                optional_expanded_skill_ids.append(neighbor_id)
            step = {
                "source_skill_id": seed_id,
                "target_skill_id": neighbor_id,
                "relation": edge.relation,
                "reason": "optional_relationship_added",
            }
            if step not in relation_expansion_steps:
                relation_expansion_steps.append(step)
            expanded_step = {
                "source_skill_id": seed_id,
                "relation": edge.relation,
                "target_skill_id": neighbor_id,
            }
            if expanded_step not in expanded:
                expanded.append(expanded_step)

    return RelationshipExpansionResult(
        candidates_by_id=candidates_by_id,
        expanded_relationships=expanded,
        relation_expansion_steps=relation_expansion_steps,
        dependency_added_skill_ids=dependency_added_skill_ids,
        optional_expanded_skill_ids=optional_expanded_skill_ids,
        skipped_relationships=skipped_relationships,
        conflict_warnings=conflict_warnings,
    )


def _rank_relationship_candidates(
    candidates_by_id: dict[str, RetrievalCandidate],
    seeds_by_id: dict[str, RetrievalCandidate],
    dependency_added_skill_ids: list[str],
    optional_expanded_skill_ids: list[str],
    relation_expansion_steps: list[dict[str, str]],
) -> list[RetrievalCandidate]:
    direct_rank = {
        candidate.skill.skill_id: index
        for index, candidate in enumerate(
            sorted(
                seeds_by_id.values(),
                key=lambda candidate: (-candidate.score, -candidate.skill.confidence, candidate.skill.name),
            )
        )
    }
    dependency_rank = {skill_id: index for index, skill_id in enumerate(dependency_added_skill_ids)}
    optional_rank = {skill_id: index for index, skill_id in enumerate(optional_expanded_skill_ids)}
    dependency_edges: dict[str, set[str]] = {}

    for step in relation_expansion_steps:
        if step["reason"] != "dependency_added":
            continue
        source_id = step["source_skill_id"]
        target_id = step["target_skill_id"]
        if source_id in candidates_by_id and target_id in candidates_by_id:
            dependency_edges.setdefault(target_id, set()).add(source_id)

    def priority(skill_id: str) -> tuple[int, int, float, str]:
        candidate = candidates_by_id[skill_id]
        if skill_id in dependency_rank and skill_id not in direct_rank:
            return (0, dependency_rank[skill_id], -candidate.score, candidate.skill.name)
        if skill_id in direct_rank:
            return (1, direct_rank[skill_id], -candidate.score, candidate.skill.name)
        if skill_id in optional_rank:
            return (2, optional_rank[skill_id], -candidate.score, candidate.skill.name)
        return (3, 0, -candidate.score, candidate.skill.name)

    incoming_counts = {skill_id: 0 for skill_id in candidates_by_id}
    for prerequisite_id, dependent_ids in dependency_edges.items():
        incoming_counts.setdefault(prerequisite_id, 0)
        for dependent_id in dependent_ids:
            incoming_counts[dependent_id] = incoming_counts.get(dependent_id, 0) + 1

    ready = sorted([skill_id for skill_id, count in incoming_counts.items() if count == 0], key=priority)
    ordered_ids: list[str] = []
    while ready:
        skill_id = ready.pop(0)
        if skill_id in ordered_ids:
            continue
        ordered_ids.append(skill_id)
        for dependent_id in sorted(dependency_edges.get(skill_id, set()), key=priority):
            incoming_counts[dependent_id] -= 1
            if incoming_counts[dependent_id] == 0:
                ready.append(dependent_id)
                ready.sort(key=priority)

    if len(ordered_ids) < len(candidates_by_id):
        remaining = [skill_id for skill_id in candidates_by_id if skill_id not in ordered_ids]
        ordered_ids.extend(sorted(remaining, key=priority))

    return [candidates_by_id[skill_id] for skill_id in ordered_ids]


def _coverage_report(candidates: list[RetrievalCandidate]) -> dict[str, Any]:
    checks = {
        "objective": any(candidate.skill.description or candidate.skill.scope for candidate in candidates),
        "input_output_contract": any(
            candidate.skill.required_inputs and candidate.skill.expected_outputs for candidate in candidates
        ),
        "required_tools": any(candidate.skill.required_tools for candidate in candidates),
        "domain_tags": any(candidate.skill.domain_tags for candidate in candidates),
        "procedure": any(candidate.skill.procedure for candidate in candidates),
        "failure_modes": any(candidate.skill.limitations for candidate in candidates),
    }
    covered = [name for name, is_covered in checks.items() if is_covered]
    missing = [name for name, is_covered in checks.items() if not is_covered]
    return {"sufficient": not missing, "covered": covered, "missing": missing}


def _to_skill_ref_with_retrieval(candidate: RetrievalCandidate) -> SkillRef:
    skill_ref = _to_skill_ref(candidate.skill)
    skill_ref.metadata["retrieval"] = {
        "score": candidate.score,
        "source": candidate.source,
        "reasons": candidate.reasons,
        "relation_path": candidate.relation_path,
        "matched_category_id": candidate.matched_category_id,
        "matched_category_path": candidate.matched_category_path,
        "retrieved_by": candidate.retrieved_by,
    }
    return skill_ref


def _category_trace(category: Any) -> dict[str, Any]:
    trace = {"category_id": category.category_id, "name": category.name}
    if category.parent_category_id is not None:
        trace["parent_category_id"] = category.parent_category_id
    return trace


def _retrieval_path_trace(index: CategoryIndex, path: RetrievalPath) -> dict[str, Any]:
    endpoint_category_id = path.endpoint_category_id
    return {
        "category_ids": list(path.category_ids),
        "endpoint_category_id": endpoint_category_id,
        "category_path": _category_path(index, endpoint_category_id),
    }


class GraphSkillSearcher:
    def __init__(self, graph: SkillGraph, indexer: GraphTreeIndexer):
        self.graph = graph
        self.indexer = indexer
        self.index = indexer.index

    def search(self, request: RetrievalRequest) -> GraphSkillSearchResult:
        query = _parse_query_info(request)
        matched_roots = _match_root_capabilities(query, self.index)
        retrieval_paths = _build_retrieval_paths(query, self.index, matched_roots)
        matched_task_ids: list[str] = []
        for path in retrieval_paths:
            endpoint_category_id = path.endpoint_category_id
            if endpoint_category_id in self.index.task_category_ids and endpoint_category_id not in matched_task_ids:
                matched_task_ids.append(endpoint_category_id)
        matched_tasks = [self.index.by_id[category_id] for category_id in matched_task_ids]
        seed_candidates, retrieved_tree_category_ids = _retrieve_path_seed_candidates(
            self.graph,
            query,
            self.index,
            retrieval_paths,
        )
        expansion = _expand_relationships(self.graph, seed_candidates)
        ranked_candidates = _rank_relationship_candidates(
            expansion.candidates_by_id,
            seed_candidates,
            expansion.dependency_added_skill_ids,
            expansion.optional_expanded_skill_ids,
            expansion.relation_expansion_steps,
        )
        if query.top_k is not None:
            ranked_candidates = ranked_candidates[: query.top_k]
        returned_skill_ids = [candidate.skill.skill_id for candidate in ranked_candidates]
        returned_skill_id_set = set(returned_skill_ids)
        relation_expansion_steps = [
            step for step in expansion.relation_expansion_steps if step["target_skill_id"] in returned_skill_id_set
        ]
        dependency_added_skill_ids = [
            skill_id for skill_id in expansion.dependency_added_skill_ids if skill_id in returned_skill_id_set
        ]
        optional_expanded_skill_ids = [
            skill_id for skill_id in expansion.optional_expanded_skill_ids if skill_id in returned_skill_id_set
        ]
        skipped_relationships = list(expansion.skipped_relationships)
        for step in expansion.relation_expansion_steps:
            if step["target_skill_id"] not in returned_skill_id_set:
                skipped_relationships.append({**step, "reason": "top_k_pruned"})
        expanded_relationships = [
            {
                "source_skill_id": step["source_skill_id"],
                "relation": step["relation"],
                "target_skill_id": step["target_skill_id"],
            }
            for step in relation_expansion_steps
        ]

        skills = [_to_skill_ref_with_retrieval(candidate) for candidate in ranked_candidates]
        required_tools = sorted({tool for skill in skills for tool in skill.required_tools})
        coverage = _coverage_report(ranked_candidates)
        deepest_matched_depth = max(
            (self.index.depth_by_id.get(category.category_id, 0) for category in matched_tasks),
            default=None,
        )
        deepest_matched_categories = [
            category
            for category in matched_tasks
            if deepest_matched_depth is not None
            and self.index.depth_by_id.get(category.category_id, 0) == deepest_matched_depth
        ]
        graph_context_summary = {
            "graph_version": self.graph.version,
            "counts": {
                "skills": len(self.graph.skills),
                "categories": len(self.graph.categories),
                "edges": len(self.graph.edges),
                "process_capabilities": len(self.index.process_capabilities),
                "scientific_tasks": len(self.index.scientific_tasks),
                "returned_specific_abilities": len(ranked_candidates),
            },
            "matched_root_capabilities": [_category_trace(category) for category in matched_roots],
            "matched_scientific_tasks": [_category_trace(category) for category in matched_tasks],
            "matched_category_paths": [
                path for path in (_category_path(self.index, category.category_id) for category in matched_tasks) if path
            ],
            "retrieval_paths": [_retrieval_path_trace(self.index, path) for path in retrieval_paths],
            "deepest_matched_categories": [
                {"category_id": category.category_id, "name": category.name} for category in deepest_matched_categories
            ],
            "retrieved_from_subtree_category_ids": sorted(
                retrieved_tree_category_ids,
                key=lambda category_id: (
                    self.index.depth_by_id.get(category_id, 0),
                    self.index.by_id[category_id].name if category_id in self.index.by_id else category_id,
                ),
            ),
            "retrieved_from_tree_category_ids": sorted(
                retrieved_tree_category_ids,
                key=lambda category_id: (
                    self.index.depth_by_id.get(category_id, 0),
                    self.index.by_id[category_id].name if category_id in self.index.by_id else category_id,
                ),
            ),
            "returned_specific_abilities": [
                {"skill_id": candidate.skill.skill_id, "name": candidate.skill.name} for candidate in ranked_candidates
            ],
            "expanded_relationships": expanded_relationships,
            "selected_category_paths": [
                path for path in (_category_path(self.index, category.category_id) for category in matched_tasks) if path
            ],
            "directly_matched_skill_ids": list(seed_candidates),
            "dependency_completed_skill_ids": dependency_added_skill_ids,
            "optional_relationship_expanded_skill_ids": optional_expanded_skill_ids,
            "coverage_report": coverage,
            "warnings": expansion.conflict_warnings,
            "conflict_warnings": expansion.conflict_warnings,
        }
        if "graph_context_summary" in self.graph.metadata:
            graph_context_summary["source_summary"] = self.graph.metadata["graph_context_summary"]

        retrieval_trace = {
            "graph_version": self.graph.version,
            "selected_category_paths": graph_context_summary["matched_category_paths"],
            "retrieval_paths": graph_context_summary["retrieval_paths"],
            "returned_skill_ids": returned_skill_ids,
            "directly_matched_skill_ids": list(seed_candidates),
            "relation_expansion_steps": relation_expansion_steps,
            "dependency_added_skill_ids": dependency_added_skill_ids,
            "optional_expanded_skill_ids": optional_expanded_skill_ids,
            "skipped_relationships": skipped_relationships,
            "conflict_warnings": expansion.conflict_warnings,
            "coverage_report": coverage,
        }
        metadata = {
            "graph_context_summary": graph_context_summary,
            "retrieval_trace": retrieval_trace,
            "matched_skill_ids": [candidate.skill.skill_id for candidate in ranked_candidates],
        }
        return GraphSkillSearchResult(
            ranked_candidates=ranked_candidates,
            skills=skills,
            required_tools=required_tools,
            metadata=metadata,
            query=query,
        )
