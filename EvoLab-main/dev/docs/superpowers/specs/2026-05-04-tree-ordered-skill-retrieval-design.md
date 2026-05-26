# Tree-Ordered Skill Retrieval Design

Date: 2026-05-04

## Goal

Change `GraphSkillBackend` retrieval from subtree expansion to strict tree-ordered traversal. Retrieval should move from a scientific process capability to direct child task categories one step at a time, while still returning multiple skills when a task request spans multiple task branches.

## Architecture

`GraphSkillBackend.get()` will build one or more retrieval paths through the category tree. Each path starts at a matched root capability when available, otherwise at every root capability so task-level queries can still find their branch. At each node, only direct children are scored; children that match the request become independent branches, and traversal continues recursively until no direct child matches.

Candidate skills are retrieved only from category ids on the selected paths. A skill attached to a path endpoint is a direct match. A skill attached to an ancestor on the path is retained as a generic path skill and marked separately. Descendant categories that were not reached by stepwise traversal are not searched.

## Data Flow

1. Parse `RetrievalRequest` into `QueryInfo`.
2. Build `CategoryIndex`.
3. Match root capabilities from explicit metadata or query tokens.
4. Build strict retrieval paths by scoring only direct children at each step.
5. Retrieve direct category-bound skills from selected path categories.
6. Expand supported skill-to-skill relationships from those seeds.
7. Rank, apply `top_k`, and return existing `SkillBundle` shape with richer retrieval metadata.

## Metadata

The graph context summary will include `retrieval_paths`, each containing category ids and a rendered category path. `retrieved_from_tree_category_ids` will list category ids used for direct skill lookup. The previous subtree summary key may remain for compatibility, but it will no longer imply descendant expansion.

Skill retrieval metadata will keep existing fields and use `retrieved_by="direct"` for endpoint matches and `retrieved_by="path_ancestor"` for generic skills attached to an ancestor category.

## Testing

Tests will cover:

- A high-level task category no longer retrieves skills attached only to descendant categories.
- A deep request reaches a descendant only by matching each direct child step.
- A multi-branch request returns skills from multiple strict paths.
- Existing ranking, relationship expansion, coverage, and lexical fallback behavior still works.
