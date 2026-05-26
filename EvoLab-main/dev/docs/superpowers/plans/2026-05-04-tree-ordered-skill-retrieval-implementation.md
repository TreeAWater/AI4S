# Tree-Ordered Skill Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GraphSkillBackend` retrieve candidate skills by strict stepwise tree traversal while allowing multiple matching task branches per request.

**Architecture:** Add a small internal retrieval-path model and traversal helpers to `evolab/backends/skills/graph.py`. Replace subtree candidate retrieval in `GraphSkillBackend.get()` with path-based candidate retrieval while preserving `SkillBundle` and `SkillRef` public contracts.

**Tech Stack:** Python 3.11, Pydantic v2 models already in the repository, pytest.

---

## File Structure

- `evolab/backends/skills/graph.py`: add retrieval path helpers and switch seed retrieval to path categories.
- `tests/test_graph_skill_backend.py`: add red tests for strict traversal and multi-branch retrieval, then update subtree expectations.

## Task 1: Red Tests For Strict Tree Traversal

- [x] Add a test proving a high-level task match does not retrieve descendant-only skills.
- [x] Add a test proving a request can return multiple skills from multiple strict paths.
- [x] Run the targeted tests and confirm they fail before implementation.

## Task 2: Implement Path-Based Retrieval

- [x] Add `RetrievalPath` and direct-child scoring helpers.
- [x] Build strict retrieval paths in `GraphSkillBackend.get()`.
- [x] Retrieve seed candidates only from selected path categories.
- [x] Update graph context metadata with path summaries.

## Task 3: Verify And Clean Up

- [x] Run targeted graph skill backend tests.
- [x] Run the full pytest suite.
- [x] Inspect the diff for unrelated changes.
