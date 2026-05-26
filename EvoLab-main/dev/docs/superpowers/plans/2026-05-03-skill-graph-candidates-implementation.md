# Skill Graph CandidateSkill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GraphSkillBackend` use docx-aligned `CandidateSkill` nodes internally while keeping `SkillBundle` / `SkillRef` unchanged for runtime and prompt builder consumers.

**Architecture:** Keep `SkillBackend` in `evolab/backends/skills/base.py`, put candidate node models in `candidates.py`, graph-level models in `graph_schema.py`, and filesystem retrieval/update behavior in `graph.py`. `GraphSkillBackend.get(...)` validates graph JSON into backend-internal models, renders matched candidates into `SkillRef`, and returns the existing public `SkillBundle`.

**Tech Stack:** Python 3.11-style type hints, Pydantic v2 through `StrictBaseModel`, pytest, UTF-8 JSON and JSONL filesystem storage.

---

## File Structure

- `evolab/backends/skills/base.py`: `SkillBackend` ABC.
- `evolab/backends/skills/candidates.py`: `CandidateSkill` and `SkillSourceType`.
- `evolab/backends/skills/graph_schema.py`: `SkillGraph`, `SkillGraphEdge`, `SkillCategoryNode`, `MissingSkillReport`, `SkillUpdateSummary`.
- `evolab/backends/skills/graph.py`: filesystem graph backend implementation.
- `evolab/backends/skills/__init__.py`: package root exports.
- `tests/test_graph_skill_backend.py`: backend package, schema, retrieval, rendering, invalid-node, and update-log tests.

## Task 1: Skill Backend Package Layout And ABC

**Files:**
- Create: `evolab/backends/skills/__init__.py`
- Create: `evolab/backends/skills/base.py`
- Delete: `evolab/backends/skills.py`
- Modify: `tests/test_graph_skill_backend.py`

- [ ] **Step 1: Write failing package layout and ABC tests**

Add these imports and tests near the top of `tests/test_graph_skill_backend.py`:

```python
from evolab.backends.skills import GraphSkillBackend, SkillBackend
from evolab.backends.skills.base import SkillBackend as BaseModuleSkillBackend
from evolab.backends.skills.graph import GraphSkillBackend as GraphModuleSkillBackend
from evolab.contracts.retrieval import SkillBundle


def test_skill_backend_package_exports_base_and_graph_modules():
    assert SkillBackend is BaseModuleSkillBackend
    assert GraphSkillBackend is GraphModuleSkillBackend


def test_graph_skill_backend_inherits_skill_backend_base(tmp_path):
    assert issubclass(GraphSkillBackend, SkillBackend)
    assert isinstance(GraphSkillBackend(tmp_path / "skills.json"), SkillBackend)


def test_skill_backend_base_requires_contract_methods():
    class MissingGet(SkillBackend):
        backend_id = "missing_get"

        def look_at(self, event):
            return event

    class MissingLookAt(SkillBackend):
        backend_id = "missing_look_at"

        def get(self, request):
            return SkillBundle(skills=[], required_tools=[], backend_id=self.backend_id)

    with pytest.raises(TypeError, match="abstract"):
        MissingGet()
    with pytest.raises(TypeError, match="abstract"):
        MissingLookAt()
```

- [ ] **Step 2: Run package layout tests to verify failure**

Run: `pytest tests/test_graph_skill_backend.py::test_skill_backend_package_exports_base_and_graph_modules tests/test_graph_skill_backend.py::test_graph_skill_backend_inherits_skill_backend_base tests/test_graph_skill_backend.py::test_skill_backend_base_requires_contract_methods -v`

Expected on a clean flat-module tree: FAIL with `ModuleNotFoundError: No module named 'evolab.backends.skills.base'; 'evolab.backends.skills' is not a package` or a failure showing `SkillBackend` is still a protocol-like non-ABC.

- [ ] **Step 3: Create package root and base ABC**

Create `evolab/backends/skills/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evolab.contracts.retrieval import RetrievalRequest, SkillBundle


class SkillBackend(ABC):
    backend_id: str

    @abstractmethod
    def get(self, request: RetrievalRequest) -> SkillBundle:
        raise NotImplementedError

    @abstractmethod
    def look_at(self, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
```

Create `evolab/backends/skills/__init__.py`:

```python
from evolab.backends.skills.base import SkillBackend
from evolab.backends.skills.graph import GraphSkillBackend

__all__ = ["GraphSkillBackend", "SkillBackend"]
```

Move the existing `GraphSkillBackend` implementation from `evolab/backends/skills.py` to `evolab/backends/skills/graph.py`, import `SkillBackend` from `evolab.backends.skills.base`, and define `class GraphSkillBackend(SkillBackend):`.

Delete `evolab/backends/skills.py` after the package exists. Python cannot safely use both a module and a package for the same import target.

- [ ] **Step 4: Run package layout tests to verify pass**

Run: `pytest tests/test_graph_skill_backend.py::test_skill_backend_package_exports_base_and_graph_modules tests/test_graph_skill_backend.py::test_graph_skill_backend_inherits_skill_backend_base tests/test_graph_skill_backend.py::test_skill_backend_base_requires_contract_methods -v`

Expected: PASS.

- [ ] **Step 5: Commit package layout**

```bash
git add evolab/backends/skills tests/test_graph_skill_backend.py
git add -u evolab/backends/skills.py
git commit -m "refactor: split skill backend package"
```

## Task 2: CandidateSkill And Graph Schema Models

**Files:**
- Create: `evolab/backends/skills/candidates.py`
- Create: `evolab/backends/skills/graph_schema.py`
- Modify: `evolab/backends/skills/__init__.py`
- Modify: `tests/test_graph_skill_backend.py`

- [ ] **Step 1: Write failing CandidateSkill and SkillGraph tests**

Add these imports to `tests/test_graph_skill_backend.py`:

```python
from pydantic import ValidationError

from evolab.backends.skills import CandidateSkill, SkillGraph
```

Add this helper:

```python
def _candidate_skill(**overrides):
    data = {
        "skill_id": "skill-1",
        "name": "Pytest failure triage",
        "description": "Use pytest output to isolate failing behavior.",
        "source_type": "human",
        "source_uri": "human://seed/pytest-triage",
        "provenance": {"author": "seed"},
        "domain_tags": ["testing", "debugging"],
        "task_types": ["regression"],
        "target_category": "testing",
        "scope": "Python test failure diagnosis",
        "applicability": ["pytest output is available"],
        "limitations": ["Does not fix production bugs automatically"],
        "required_inputs": ["pytest failure output"],
        "expected_outputs": ["root cause hypothesis", "focused test command"],
        "dependencies": ["pytest"],
        "environment_assumptions": ["repository has pytest configured"],
        "procedure": [
            "Run the failing pytest node with verbose output.",
            "Read the assertion failure and traceback.",
            "Inspect the smallest code path that explains the failure.",
        ],
        "required_tools": ["pytest", "rg"],
        "scripts": ["pytest {test_node} -v"],
        "resources": ["docs/testing.md"],
        "examples": ["pytest tests/test_example.py::test_failure -v"],
        "smoke_tests": ["pytest --version"],
        "synthetic_tests": ["pytest tests/test_example.py -q"],
        "system_tests": ["pytest -q"],
        "benchmark_tests": ["pytest tests/test_regressions.py -q"],
        "validation_signals": ["human_seeded"],
        "confidence": 0.8,
        "metadata": {"priority": "high"},
    }
    data.update(overrides)
    return data
```

Add these tests:

```python
def test_candidate_skill_accepts_docx_aligned_definition():
    candidate = CandidateSkill(**_candidate_skill())

    assert candidate.skill_id == "skill-1"
    assert candidate.source_type == "human"
    assert candidate.domain_tags == ["testing", "debugging"]
    assert candidate.procedure[0] == "Run the failing pytest node with verbose output."
    assert candidate.confidence == 0.8


def test_candidate_skill_rejects_invalid_source_type_and_confidence():
    with pytest.raises(ValidationError):
        CandidateSkill(**_candidate_skill(source_type="invalid"))

    with pytest.raises(ValidationError):
        CandidateSkill(**_candidate_skill(confidence=1.5))


def test_skill_graph_accepts_candidate_skill_nodes():
    graph = SkillGraph(
        version="graph-v2",
        skills=[_candidate_skill()],
        categories=[
            {
                "category_id": "testing",
                "name": "Testing",
                "description": "Testing workflows",
            }
        ],
        edges=[
            {
                "source_id": "skill-1",
                "target_id": "testing",
                "relation": "belongs_to_category",
                "weight": 0.9,
            }
        ],
        metadata={"owner": "skills"},
    )

    assert graph.schema_version == "v1"
    assert graph.skills[0].skill_id == "skill-1"
    assert graph.categories[0].category_id == "testing"
    assert graph.edges[0].relation == "belongs_to_category"
```

- [ ] **Step 2: Run schema tests to verify failure**

Run: `pytest tests/test_graph_skill_backend.py::test_candidate_skill_accepts_docx_aligned_definition tests/test_graph_skill_backend.py::test_candidate_skill_rejects_invalid_source_type_and_confidence tests/test_graph_skill_backend.py::test_skill_graph_accepts_candidate_skill_nodes -v`

Expected: FAIL with `ImportError` for missing `CandidateSkill` or `SkillGraph`.

- [ ] **Step 3: Implement CandidateSkill**

Create `evolab/backends/skills/candidates.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


SkillSourceType = Literal[
    "github",
    "paper",
    "pubmed",
    "package",
    "notebook",
    "database",
    "web",
    "human",
]


class CandidateSkill(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    skill_id: str
    name: str
    description: str

    source_type: SkillSourceType
    source_uri: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    domain_tags: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    target_category: str | None = None

    scope: str
    applicability: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    environment_assumptions: list[str] = Field(default_factory=list)

    procedure: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    smoke_tests: list[str] = Field(default_factory=list)
    synthetic_tests: list[str] = Field(default_factory=list)
    system_tests: list[str] = Field(default_factory=list)
    benchmark_tests: list[str] = Field(default_factory=list)
    validation_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Implement graph schema models**

Create `evolab/backends/skills/graph_schema.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from evolab.backends.skills.candidates import CandidateSkill
from evolab.contracts.common import StrictBaseModel


class SkillCategoryNode(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    category_id: str
    name: str
    description: str | None = None
    parent_category_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillGraphEdge(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_id: str
    target_id: str
    relation: str
    weight: float | None = Field(default=None, ge=0, le=1)
    deprecated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissingSkillReport(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    missing_capability: str
    reason: str
    can_be_solved_by_existing_tools: bool
    risk_level: Literal["low", "medium", "high"]
    on_demand_synthesis_allowed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillUpdateSummary(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_run_id: str | None = None
    candidate_skill_id: str | None = None
    update_type: str
    affected_skill_ids: list[str] = Field(default_factory=list)
    affected_edges: list[dict[str, Any]] = Field(default_factory=list)
    decision_rationale: str | None = None
    validation_signals: list[str] = Field(default_factory=list)
    graph_version_before: str | None = None
    graph_version_after: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: dict[str, Any] = Field(default_factory=dict)


class SkillGraph(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    version: str = "v1"
    skills: list[CandidateSkill] = Field(default_factory=list)
    categories: list[SkillCategoryNode] = Field(default_factory=list)
    edges: list[SkillGraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Export schema classes**

Update `evolab/backends/skills/__init__.py`:

```python
from evolab.backends.skills.base import SkillBackend
from evolab.backends.skills.candidates import CandidateSkill, SkillSourceType
from evolab.backends.skills.graph import GraphSkillBackend
from evolab.backends.skills.graph_schema import (
    MissingSkillReport,
    SkillCategoryNode,
    SkillGraph,
    SkillGraphEdge,
    SkillUpdateSummary,
)

__all__ = [
    "CandidateSkill",
    "GraphSkillBackend",
    "MissingSkillReport",
    "SkillBackend",
    "SkillCategoryNode",
    "SkillGraph",
    "SkillGraphEdge",
    "SkillSourceType",
    "SkillUpdateSummary",
]
```

- [ ] **Step 6: Run schema tests to verify pass**

Run: `pytest tests/test_graph_skill_backend.py::test_candidate_skill_accepts_docx_aligned_definition tests/test_graph_skill_backend.py::test_candidate_skill_rejects_invalid_source_type_and_confidence tests/test_graph_skill_backend.py::test_skill_graph_accepts_candidate_skill_nodes -v`

Expected: PASS.

- [ ] **Step 7: Commit schema models**

```bash
git add evolab/backends/skills/__init__.py evolab/backends/skills/candidates.py evolab/backends/skills/graph_schema.py tests/test_graph_skill_backend.py
git commit -m "feat: add candidate skill graph schema"
```

## Task 3: GraphSkillBackend Candidate Retrieval And Rendering

**Files:**
- Modify: `evolab/backends/skills/graph.py`
- Modify: `tests/test_graph_skill_backend.py`

- [ ] **Step 1: Replace old skill graph fixtures with CandidateSkill fixtures**

Update tests that write graph JSON so each skill uses `_candidate_skill(...)`. For example:

```python
graph_path.write_text(
    json.dumps(
        {
            "schema_version": "v1",
            "version": "graph-v2",
            "skills": [
                _candidate_skill(
                    skill_id="skill-1",
                    name="Pytest failure triage",
                    description="Use pytest output to isolate failing behavior.",
                    domain_tags=["testing", "debugging"],
                    task_types=["regression"],
                    required_tools=["pytest"],
                    procedure=["Run pytest with verbose output.", "Inspect the failing assertion."],
                ),
                _candidate_skill(
                    skill_id="skill-2",
                    name="Release notes",
                    description="Write a concise changelog.",
                    domain_tags=["docs"],
                    task_types=["documentation"],
                    required_tools=["git"],
                    procedure=["Collect merged changes.", "Group user-visible changes."],
                ),
            ],
            "categories": [
                {
                    "category_id": "testing",
                    "name": "Testing",
                    "description": "Testing workflows",
                }
            ],
            "edges": [
                {
                    "source_id": "skill-1",
                    "target_id": "testing",
                    "relation": "belongs_to_category",
                    "weight": 0.9,
                }
            ],
            "metadata": {"graph_context_summary": "Seed graph for tests"},
        }
    ),
    encoding="utf-8",
)
```

- [ ] **Step 2: Add failing tests for CandidateSkill rendering and metadata**

Add:

```python
def test_get_renders_candidate_skill_operational_contract(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v2",
                "skills": [
                    _candidate_skill(
                        skill_id="skill-1",
                        procedure=["Run pytest with verbose output.", "Inspect the failing assertion."],
                        scripts=["pytest {test_node} -v"],
                        resources=["docs/testing.md"],
                        examples=["pytest tests/test_example.py::test_failure -v"],
                    )
                ],
                "edges": [],
                "metadata": {"graph_context_summary": "Testing graph"},
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("pytest regression"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-1"]
    assert bundle.graph_version_ref == "graph-v2"
    assert bundle.metadata["graph_context_summary"] == "Testing graph"
    assert bundle.metadata["matched_skill_ids"] == ["skill-1"]
    assert "Description:\nUse pytest output to isolate failing behavior." in bundle.skills[0].content
    assert "Procedure:\n1. Run pytest with verbose output.\n2. Inspect the failing assertion." in bundle.skills[0].content
    assert "Required Inputs:\n- pytest failure output" in bundle.skills[0].content
    assert bundle.skills[0].metadata["scripts"] == ["pytest {test_node} -v"]
    assert bundle.skills[0].metadata["resources"] == ["docs/testing.md"]
    assert bundle.skills[0].metadata["confidence"] == 0.8
    assert bundle.required_tools == ["pytest", "rg"]
```

Add:

```python
def test_unmatched_query_returns_missing_skill_report(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "v1",
                "skills": [_candidate_skill()],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("database migration"))

    assert bundle.skills == []
    assert bundle.required_tools == []
    assert bundle.metadata["matched_skill_ids"] == []
    assert bundle.metadata["missing_skill_report"] == {
        "schema_version": "v1",
        "missing_capability": "database migration",
        "reason": "No CandidateSkill matched the retrieval query.",
        "can_be_solved_by_existing_tools": False,
        "risk_level": "medium",
        "on_demand_synthesis_allowed": False,
        "metadata": {},
    }
```

- [ ] **Step 3: Run retrieval tests to verify failure**

Run: `pytest tests/test_graph_skill_backend.py::test_get_renders_candidate_skill_operational_contract tests/test_graph_skill_backend.py::test_unmatched_query_returns_missing_skill_report -v`

Expected: FAIL because current `GraphSkillBackend` still expects the old dict shape and does not render CandidateSkill operational content.

- [ ] **Step 4: Implement graph loading and partial skill validation**

In `evolab/backends/skills/graph.py`, replace raw graph loading with these helpers:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evolab.backends.skills.base import SkillBackend
from evolab.backends.skills.candidates import CandidateSkill
from evolab.backends.skills.graph_schema import MissingSkillReport, SkillGraph, SkillUpdateSummary
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillRef


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
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
_EMPTY_GRAPH = {
    "schema_version": "v1",
    "version": "v1",
    "skills": [],
    "categories": [],
    "edges": [],
    "metadata": {},
}


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(value) if token.lower() not in _STOP_WORDS}


def _raw_strings(values: list[str]) -> list[str]:
    return [value for value in values if isinstance(value, str)]
```

Add these methods to `GraphSkillBackend`:

```python
    def _load_raw_graph(self) -> dict[str, Any]:
        with self.graph_path.open(encoding="utf-8") as graph_file:
            graph = json.load(graph_file)
        if not isinstance(graph, dict):
            raise ValueError("skill graph must be a JSON object")
        for field in ("skills", "categories", "edges"):
            if field in graph and not isinstance(graph[field], list):
                raise ValueError(f"skill graph field '{field}' must be a list")
        return graph

    def _load_graph(self) -> tuple[SkillGraph, list[dict[str, Any]]]:
        raw_graph = self._load_raw_graph()
        valid_skills: list[CandidateSkill] = []
        skipped_skills: list[dict[str, Any]] = []
        for index, raw_skill in enumerate(raw_graph.get("skills", [])):
            if not isinstance(raw_skill, dict):
                skipped_skills.append({"index": index, "reason": "skill entry must be an object"})
                continue
            try:
                valid_skills.append(CandidateSkill.model_validate(raw_skill))
            except ValidationError as exc:
                skipped_skills.append({"index": index, "reason": str(exc)})

        graph_payload = {
            "schema_version": raw_graph.get("schema_version", "v1"),
            "version": raw_graph.get("version", "v1"),
            "skills": [skill.model_dump(mode="json") for skill in valid_skills],
            "categories": raw_graph.get("categories", []),
            "edges": raw_graph.get("edges", []),
            "metadata": raw_graph.get("metadata", {}),
        }
        try:
            return SkillGraph.model_validate(graph_payload), skipped_skills
        except ValidationError as exc:
            raise ValueError(f"invalid skill graph: {exc}") from exc
```

- [ ] **Step 5: Implement candidate matching, rendering, and bundle metadata**

Add these helpers to `evolab/backends/skills/graph.py`:

```python
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
```

Replace `get(...)` with:

```python
    def get(self, request: RetrievalRequest) -> SkillBundle:
        graph, skipped_skills = self._load_graph()
        query_tokens = _tokens(request.query)
        matched_candidates: list[CandidateSkill] = []

        for skill in graph.skills:
            if query_tokens and query_tokens.isdisjoint(_tokens(_candidate_search_text(skill))):
                continue
            matched_candidates.append(skill)

        skills = [_to_skill_ref(skill) for skill in matched_candidates]
        required_tools = sorted({tool for skill in skills for tool in skill.required_tools})

        metadata: dict[str, Any] = {
            "graph_context_summary": graph.metadata.get("graph_context_summary"),
            "matched_skill_ids": [skill.skill_id for skill in matched_candidates],
        }
        if skipped_skills:
            metadata["skipped_skills"] = skipped_skills
        if query_tokens and not matched_candidates:
            metadata["missing_skill_report"] = MissingSkillReport(
                missing_capability=request.query,
                reason="No CandidateSkill matched the retrieval query.",
                can_be_solved_by_existing_tools=False,
                risk_level="medium",
                on_demand_synthesis_allowed=False,
            ).model_dump(mode="json")

        return SkillBundle(
            skills=skills,
            required_tools=required_tools,
            backend_id=self.backend_id,
            graph_version_ref=graph.version,
            metadata=metadata,
        )
```

- [ ] **Step 6: Run retrieval tests to verify pass**

Run: `pytest tests/test_graph_skill_backend.py::test_get_renders_candidate_skill_operational_contract tests/test_graph_skill_backend.py::test_unmatched_query_returns_missing_skill_report -v`

Expected: PASS.

- [ ] **Step 7: Run all graph backend tests**

Run: `pytest tests/test_graph_skill_backend.py -v`

Expected: PASS.

- [ ] **Step 8: Commit retrieval implementation**

```bash
git add evolab/backends/skills/graph.py tests/test_graph_skill_backend.py
git commit -m "feat: retrieve candidate skill graph nodes"
```

## Task 4: New Empty Graph Schema, Invalid Nodes, And Update Summary Logs

**Files:**
- Modify: `evolab/backends/skills/graph.py`
- Modify: `tests/test_graph_skill_backend.py`

- [ ] **Step 1: Write failing tests for empty graph initialization and invalid-node behavior**

Update the constructor test assertion:

```python
def test_constructor_initializes_missing_graph(tmp_path):
    graph_path = tmp_path / "nested" / "skills.json"

    backend = GraphSkillBackend(graph_path)

    assert backend.graph_path == graph_path
    assert json.loads(graph_path.read_text(encoding="utf-8")) == {
        "schema_version": "v1",
        "version": "v1",
        "skills": [],
        "categories": [],
        "edges": [],
        "metadata": {},
    }
```

Replace invalid-entry expectations:

```python
def test_invalid_skill_entries_are_skipped_and_reported(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "v1",
                "skills": [
                    {"skill_id": "bad-1", "name": "Missing CandidateSkill fields"},
                    "not-a-skill",
                    _candidate_skill(
                        skill_id="skill-1",
                        name="Pytest helper",
                        description="Use pytest for focused tests.",
                        domain_tags=["testing"],
                    ),
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("pytest"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-1"]
    assert len(bundle.metadata["skipped_skills"]) == 2
    assert bundle.metadata["skipped_skills"][0]["index"] == 0
    assert "Field required" in bundle.metadata["skipped_skills"][0]["reason"]
    assert bundle.metadata["skipped_skills"][1] == {"index": 1, "reason": "skill entry must be an object"}
```

- [ ] **Step 2: Write failing test for update summary JSONL**

Replace the existing look-at test with:

```python
def test_look_at_writes_update_summary_jsonl_with_versions(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v2",
                "skills": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)
    event = {
        "source_run_id": "run-1",
        "candidate_skill_id": "skill-1",
        "update_type": "skipped",
        "affected_skill_ids": ["skill-1"],
        "affected_edges": [{"source_id": "skill-1", "target_id": "testing"}],
        "decision_rationale": "Observation recorded without graph mutation.",
        "validation_signals": ["human_feedback"],
        "provenance": {"observer": "test"},
    }

    result = backend.look_at(event)

    update_log = tmp_path / "skills.updates.jsonl"
    logged = json.loads(update_log.read_text(encoding="utf-8").splitlines()[0])
    assert result == {
        "status": "recorded",
        "update_log": str(update_log),
        "graph_version_before": "graph-v2",
        "graph_version_after": "graph-v2",
    }
    assert logged["source_run_id"] == "run-1"
    assert logged["candidate_skill_id"] == "skill-1"
    assert logged["update_type"] == "skipped"
    assert logged["affected_skill_ids"] == ["skill-1"]
    assert logged["graph_version_before"] == "graph-v2"
    assert logged["graph_version_after"] == "graph-v2"
    assert logged["provenance"] == {"observer": "test"}
    assert "timestamp" in logged
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_graph_skill_backend.py::test_constructor_initializes_missing_graph tests/test_graph_skill_backend.py::test_invalid_skill_entries_are_skipped_and_reported tests/test_graph_skill_backend.py::test_look_at_writes_update_summary_jsonl_with_versions -v`

Expected: FAIL because the constructor and look-at behavior still use the older graph and event shapes.

- [ ] **Step 4: Update constructor to write new empty graph**

In `GraphSkillBackend.__init__`, keep:

```python
    def __init__(self, graph_path: Path | str):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.exists():
            self.graph_path.write_text(json.dumps(_EMPTY_GRAPH), encoding="utf-8")
```

The `_EMPTY_GRAPH` constant must be:

```python
_EMPTY_GRAPH = {
    "schema_version": "v1",
    "version": "v1",
    "skills": [],
    "categories": [],
    "edges": [],
    "metadata": {},
}
```

- [ ] **Step 5: Implement update summary logging**

Replace `look_at(...)` in `evolab/backends/skills/graph.py`:

```python
    def look_at(self, event: dict[str, Any]) -> dict[str, Any]:
        raw_graph = self._load_raw_graph()
        graph_version = raw_graph.get("version")
        graph_version_ref = graph_version if isinstance(graph_version, str) else None
        update_log = self.graph_path.with_suffix(".updates.jsonl")
        summary = SkillUpdateSummary(
            source_run_id=event.get("source_run_id") or event.get("run_ref"),
            candidate_skill_id=event.get("candidate_skill_id") or event.get("skill_id"),
            update_type=event.get("update_type", "recorded"),
            affected_skill_ids=event.get("affected_skill_ids", []),
            affected_edges=event.get("affected_edges", []),
            decision_rationale=event.get("decision_rationale"),
            validation_signals=event.get("validation_signals", []),
            graph_version_before=graph_version_ref,
            graph_version_after=graph_version_ref,
            provenance=event.get("provenance", {}),
        )
        with update_log.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(summary.model_dump(mode="json"), sort_keys=True) + "\n")
        return {
            "status": "recorded",
            "update_log": str(update_log),
            "graph_version_before": graph_version_ref,
            "graph_version_after": graph_version_ref,
        }
```

- [ ] **Step 6: Run update and invalid-node tests to verify pass**

Run: `pytest tests/test_graph_skill_backend.py::test_constructor_initializes_missing_graph tests/test_graph_skill_backend.py::test_invalid_skill_entries_are_skipped_and_reported tests/test_graph_skill_backend.py::test_look_at_writes_update_summary_jsonl_with_versions -v`

Expected: PASS.

- [ ] **Step 7: Commit update logging**

```bash
git add evolab/backends/skills/graph.py tests/test_graph_skill_backend.py
git commit -m "feat: log skill graph update summaries"
```

## Task 5: Final Verification And Cleanups

**Files:**
- Modify: `tests/test_graph_skill_backend.py`
- Modify: `evolab/backends/skills/__init__.py`
- Modify: `evolab/backends/skills/candidates.py`
- Modify: `evolab/backends/skills/graph_schema.py`
- Modify: `evolab/backends/skills/graph.py`

- [ ] **Step 1: Run targeted skill backend suite**

Run: `pytest tests/test_graph_skill_backend.py -v`

Expected: all graph skill backend tests pass. The suite should include package exports, ABC contract, CandidateSkill schema validation, SkillGraph validation, CandidateSkill retrieval, missing-skill report, required tool aggregation, invalid-node reporting, update summary logging, and blank extension point errors.

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Remove Python cache directories produced by tests**

Run:

```bash
find evolab tests -type d -name __pycache__ -prune -exec rm -rf {} +
```

Expected: command exits 0 and `git status --short` does not list `__pycache__` paths.

- [ ] **Step 4: Check changed files**

Run: `git status --short`

Expected: only planned skill backend code and test files are modified or untracked:

```text
 M tests/test_graph_skill_backend.py
?? evolab/backends/skills/candidates.py
?? evolab/backends/skills/graph_schema.py
```

The user-supplied developer documentation may remain untracked as:

```text
?? "docs/superpowers/specs/Self-Evolving Agents for Scientific Research_ Developer Documentation.docx"
```

Do not stage that file unless the user explicitly asks to add the source document to git.

If package-layout changes are part of the same final candidate, `git status --short` may also show:

```text
 D evolab/backends/skills.py
?? evolab/backends/skills/
```

- [ ] **Step 5: Commit final candidate changes**

If Tasks 1 through 4 were committed separately, no extra commit is needed after Step 2 passes. If the work was done as one batch, commit it:

```bash
git add evolab/backends/skills tests/test_graph_skill_backend.py
git add -u evolab/backends/skills.py
git commit -m "feat: implement candidate skill graph backend"
```

Expected: commit succeeds.
