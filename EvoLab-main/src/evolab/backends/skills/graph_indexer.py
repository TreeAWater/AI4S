from __future__ import annotations

from dataclasses import dataclass

from evolab.backends.skills.graph_schema import SkillCategoryNode, SkillGraph


@dataclass(frozen=True)
class CategoryIndex:
    by_id: dict[str, SkillCategoryNode]
    children_by_parent: dict[str, list[SkillCategoryNode]]
    parent_by_child: dict[str, str]
    root_capability_ids: set[str]
    task_category_ids: set[str]
    depth_by_id: dict[str, int]
    root_by_category_id: dict[str, str]
    process_capabilities: list[SkillCategoryNode]
    scientific_tasks: list[SkillCategoryNode]

    def get_ancestors(self, category_id: str) -> list[str]:
        return _get_ancestors(self, category_id)

    def get_descendants(self, category_id: str, max_depth: int | None = None) -> list[str]:
        return _get_descendants(self, category_id, max_depth=max_depth)

    def get_subtree_category_ids(self, category_id: str) -> set[str]:
        return _get_subtree_category_ids(self, category_id)

    def get_root_for_category(self, category_id: str) -> str | None:
        return _get_root_for_category(self, category_id)


class GraphTreeIndexer:
    def __init__(self, graph: SkillGraph):
        self.graph = graph
        self.index = _build_category_index(graph)

    def get_ancestors(self, category_id: str) -> list[str]:
        return self.index.get_ancestors(category_id)

    def get_descendants(self, category_id: str, max_depth: int | None = None) -> list[str]:
        return self.index.get_descendants(category_id, max_depth=max_depth)

    def get_subtree_category_ids(self, category_id: str) -> set[str]:
        return self.index.get_subtree_category_ids(category_id)

    def get_root_for_category(self, category_id: str) -> str | None:
        return self.index.get_root_for_category(category_id)


def _build_category_index(graph: SkillGraph) -> CategoryIndex:
    by_id = {category.category_id: category for category in graph.categories}
    children_by_parent: dict[str, list[SkillCategoryNode]] = {}
    parent_by_child: dict[str, str] = {}
    process_capabilities: list[SkillCategoryNode] = []
    scientific_tasks: list[SkillCategoryNode] = []
    root_capability_ids: set[str] = set()
    task_category_ids: set[str] = set()
    for category in graph.categories:
        if category.parent_category_id:
            children_by_parent.setdefault(category.parent_category_id, []).append(category)
            parent_by_child[category.category_id] = category.parent_category_id
        if category.layer == "scientific_process_capability":
            process_capabilities.append(category)
            root_capability_ids.add(category.category_id)
        elif category.layer == "scientific_task":
            scientific_tasks.append(category)
            task_category_ids.add(category.category_id)

    depth_by_id: dict[str, int] = {}
    root_by_category_id: dict[str, str] = {}

    def visit(category_id: str, root_id: str, depth: int, path: set[str]) -> None:
        if category_id in path:
            return
        current_depth = depth_by_id.get(category_id)
        if current_depth is None or depth < current_depth:
            depth_by_id[category_id] = depth
            root_by_category_id[category_id] = root_id
        for child in children_by_parent.get(category_id, []):
            visit(child.category_id, root_id, depth + 1, path | {category_id})

    for root in process_capabilities:
        visit(root.category_id, root.category_id, 0, set())

    def infer_unrooted_depth(category_id: str, path: set[str]) -> int:
        if category_id in depth_by_id:
            return depth_by_id[category_id]
        if category_id in path:
            depth_by_id[category_id] = 0
            return 0
        parent_id = parent_by_child.get(category_id)
        if parent_id is None or parent_id not in by_id:
            depth_by_id[category_id] = 0
            return 0
        depth_by_id[category_id] = infer_unrooted_depth(parent_id, path | {category_id}) + 1
        if parent_id in root_by_category_id:
            root_by_category_id[category_id] = root_by_category_id[parent_id]
        return depth_by_id[category_id]

    for category_id in by_id:
        infer_unrooted_depth(category_id, set())

    return CategoryIndex(
        by_id=by_id,
        children_by_parent=children_by_parent,
        parent_by_child=parent_by_child,
        root_capability_ids=root_capability_ids,
        task_category_ids=task_category_ids,
        depth_by_id=depth_by_id,
        root_by_category_id=root_by_category_id,
        process_capabilities=process_capabilities,
        scientific_tasks=scientific_tasks,
    )


def _get_ancestors(index: CategoryIndex, category_id: str) -> list[str]:
    ancestors: list[str] = []
    current_id = category_id
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        parent_id = index.parent_by_child.get(current_id)
        if parent_id is None:
            break
        ancestors.append(parent_id)
        current_id = parent_id
    return list(reversed(ancestors))


def _get_descendants(index: CategoryIndex, category_id: str, max_depth: int | None = None) -> list[str]:
    descendants: list[str] = []

    def visit(parent_id: str, relative_depth: int) -> None:
        if max_depth is not None and relative_depth > max_depth:
            return
        for child in index.children_by_parent.get(parent_id, []):
            descendants.append(child.category_id)
            visit(child.category_id, relative_depth + 1)

    visit(category_id, 1)
    return descendants


def _get_subtree_category_ids(index: CategoryIndex, category_id: str) -> set[str]:
    return {category_id, *_get_descendants(index, category_id)}


def _get_root_for_category(index: CategoryIndex, category_id: str) -> str | None:
    return index.root_by_category_id.get(category_id)
