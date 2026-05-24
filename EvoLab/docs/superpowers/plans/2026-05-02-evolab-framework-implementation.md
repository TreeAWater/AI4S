# EvoLab Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EvoLab framework skeleton with real OpenAI-compatible LLM API integration, Mem0 SDK memory integration, filesystem graph skills, filesystem queues/registries, TaskWorker rollout flow, EvolveWorker promotion flow, and explicit blank extension points for unfinished backend-private algorithms.

**Architecture:** Use Pydantic v2 for external contracts and `Protocol` for service boundaries. TaskWorker initializes configured LLM, memory, skill, and tool runtimes at startup, then runs synchronous rollout loops. EvolveWorker consumes independent training jobs, validates promotion, and registers / promotes backend state only through `FileBackendStateRegistry`.

**Final skeleton reconciliation:** TaskWorker and EvolveWorker are independent workers; TaskWorker does not import EvolveWorker, does not schedule evolve jobs, and does not maintain `evolve_pending`. `TaskRuntime` has an injectable dispatch loop, but its default `run(...)` raises `NotImplementedError` instead of returning fake completion. `ToolRuntime` is synchronous in-process support, not a separate tool worker. `ApiLLMBackend` is the real OpenAI Responses adapter with strict schema validation and tool-call parsing; `LocalTrainableLLMBackend`, blank trainer algorithms, and GraphSkillBackend mining / rewiring are explicit `NotImplementedError` extension points. Tests may use test-local fakes; product code has no mock backend or trainer.

**Tech Stack:** Python 3.11+, Pydantic v2, OpenAI Python SDK, Mem0 Python SDK (`mem0ai`), pytest, filesystem JSON/JSONL storage.

**References:**
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Mem0 Python SDK quickstart: https://docs.mem0.ai/open-source/python-quickstart
- Mem0 Platform quickstart: https://docs.mem0.ai/platform/quickstart

---

## File Structure

Create:

- `pyproject.toml`: package metadata, runtime dependencies, test config.
- `evolab/__init__.py`: package marker.
- `evolab/contracts/common.py`: shared `Message`, `ArtifactRef`, refs, policy, budget.
- `evolab/contracts/task.py`: `TaskRequest`, `TaskJob`, task provenance and proposer relation models.
- `evolab/contracts/dispatch.py`: `DispatchDecision` and action enum.
- `evolab/contracts/retrieval.py`: retrieval, memory, skill bundle models.
- `evolab/contracts/tools.py`: tool specs, calls, results, traces.
- `evolab/contracts/llm.py`: runtime request/response, subagent actions, generation config.
- `evolab/contracts/evolution.py`: evolution request/result, proposer inputs, metrics.
- `evolab/contracts/records.py`: LLM, meta-agent, subagent, evolution run records.
- `evolab/contracts/state.py`: backend state records and transitions.
- `evolab/config/task_config.py`: task, role, backend binding configs.
- `evolab/backends/llm.py`: `LLMBackend` / `LLMRuntime` protocols, OpenAI Responses `ApiLLMBackend`, blank local backend.
- `evolab/backends/memory.py`: `MemoryBackend` protocol and `Mem0MemoryBackend`.
- `evolab/backends/skills.py`: `SkillBackend` protocol and filesystem `GraphSkillBackend` using canonical `skill_id` and UTF-8 JSON/JSONL.
- `evolab/backends/trainers.py`: trainer protocol and blank trainer base.
- `evolab/tools/runtime.py`: in-process `ToolRegistry` and `ToolRuntime`.
- `evolab/lab/layout.py`: filesystem Lab path helper.
- `evolab/lab/queue.py`: filesystem work queue.
- `evolab/registries/task.py`: filesystem task registry.
- `evolab/registries/trajectory.py`: filesystem trajectory registry.
- `evolab/registries/backend_state.py`: filesystem backend state registry.
- `evolab/runtime/prompt_builder.py`: canonical prompt builder.
- `evolab/runtime/promotion.py`: promotion validation.
- `evolab/runtime/task_runtime.py`: thin TaskRuntime rollout orchestration.
- `evolab/runtime/task_worker.py`: TaskWorker startup and queue consumption.
- `evolab/runtime/evolve_worker.py`: EvolveWorker queue consumption and trainer dispatch.
- `tests/`: matching unit tests.

Do not create product mock backends or product mock trainers. Tests may define test-local fakes inside test files.

---

### Task 1: Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `evolab/__init__.py`
- Create: `tests/test_imports.py`

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_imports.py
def test_package_imports():
    import evolab

    assert evolab.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_imports.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'evolab'`.

- [ ] **Step 3: Add package metadata and dependencies**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "evolab"
version = "0.1.0"
description = "Self-evolving agents for scientific research"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7",
  "openai>=1.109.0",
  "mem0ai>=0.1.0",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```python
# evolab/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_imports.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml evolab/__init__.py tests/test_imports.py
git commit -m "feat: add project foundation"
```

---

### Task 2: Core Contracts

**Files:**
- Create: `evolab/contracts/common.py`
- Create: `evolab/contracts/task.py`
- Create: `evolab/contracts/dispatch.py`
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write contract validation tests**

```python
# tests/test_contracts.py
import pytest
from pydantic import ValidationError

from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest


def test_dispatch_decision_accepts_run_subagent():
    decision = DispatchDecision(
        action=DispatchAction.RUN_SUBAGENT,
        target_role="solver",
        instruction="Solve the problem.",
        retrieval_query="relevant prior work",
    )
    assert decision.action == DispatchAction.RUN_SUBAGENT


def test_proposer_task_requires_relation():
    with pytest.raises(ValidationError):
        TaskRequest(
            task_id="task-proposed-1",
            origin=TaskOrigin.PROPOSER,
            purpose=TaskPurpose.TRAINING_ROLLOUT,
            goal="Variant task",
        )


def test_human_task_does_not_require_relation():
    request = TaskRequest(
        task_id="task-human-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Find a biomarker.",
    )
    assert request.proposed_task_relation is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contracts.py -q`

Expected: FAIL with import errors for missing contract modules.

- [ ] **Step 3: Implement shared common contracts**

```python
# evolab/contracts/common.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Message(StrictBaseModel):
    schema_version: str = "v1"
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(StrictBaseModel):
    schema_version: str = "v1"
    uri: str
    type: Literal["text", "code", "dataset", "model_adapter", "image", "paper", "log", "other"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class GitRef(StrictBaseModel):
    schema_version: str = "v1"
    uri: str
    commit: str | None = None
    branch: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputSpec(StrictBaseModel):
    schema_version: str = "v1"
    name: str
    description: str
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimePolicy(StrictBaseModel):
    schema_version: str = "v1"
    max_tool_steps: int = 20
    allow_human_tools: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionBudget(StrictBaseModel):
    schema_version: str = "v1"
    max_wall_clock_s: int | None = None
    max_train_samples: int | None = None
    max_cost_usd: float | None = None
    deadline_at: datetime | None = None
```

- [ ] **Step 4: Implement task and dispatch contracts**

```python
# evolab/contracts/task.py
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from evolab.contracts.common import StrictBaseModel


class TaskOrigin(StrEnum):
    HUMAN = "human"
    PROPOSER = "proposer"
    BENCHMARK = "benchmark"
    SCHEDULER = "scheduler"


class TaskPurpose(StrEnum):
    SCIENCE = "science"
    TRAINING_ROLLOUT = "training_rollout"
    EVALUATION = "evaluation"
    REGRESSION = "regression"


class ProposedTaskRelationType(StrEnum):
    SUBPROBLEM = "subproblem"
    ANALOGY = "analogy"
    DIFFICULTY_VARIANT = "difficulty_variant"
    SKILL_PROBE = "skill_probe"
    FAILURE_REPAIR = "failure_repair"
    COUNTEREXAMPLE = "counterexample"
    DATA_VARIANT = "data_variant"
    ABLATION = "ablation"
    REGRESSION = "regression"
    CURRICULUM_STEP = "curriculum_step"


class ProposerInputRef(StrictBaseModel):
    schema_version: str = "v1"
    ref_type: str
    ref_id: str
    role: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposedTaskRelation(StrictBaseModel):
    schema_version: str = "v1"
    human_anchor_task_refs: list[str] = Field(default_factory=list)
    human_anchor_trajectory_refs: list[str] = Field(default_factory=list)
    proposer_input_refs: list[ProposerInputRef] = Field(default_factory=list)
    relation_type: ProposedTaskRelationType
    relation_rationale: str
    target_capabilities: list[str] = Field(default_factory=list)
    expected_transfer: str
    eval_target_task_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_human_anchor(self) -> "ProposedTaskRelation":
        if not self.human_anchor_task_refs and not self.human_anchor_trajectory_refs:
            raise ValueError("proposed task relation requires a human anchor task or trajectory")
        return self


class TaskRequest(StrictBaseModel):
    schema_version: str = "v1"
    task_id: str
    origin: TaskOrigin
    purpose: TaskPurpose
    goal: str
    task_config_ref: str | None = None
    task_payload_uri: str | None = None
    producer_ref: str | None = None
    parent_task_id: str | None = None
    round_id: str | None = None
    proposed_task_relation: ProposedTaskRelation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_proposer_relation(self) -> "TaskRequest":
        if self.origin == TaskOrigin.PROPOSER and self.proposed_task_relation is None:
            raise ValueError("origin=proposer requires proposed_task_relation")
        return self


class TaskJob(StrictBaseModel):
    schema_version: str = "v1"
    job_id: str
    request_payload_uri: str
    enqueued_at: str
```

```python
# evolab/contracts/dispatch.py
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from evolab.contracts.common import OutputSpec, StrictBaseModel


class DispatchAction(StrEnum):
    RUN_SUBAGENT = "run_subagent"
    FINISH_TASK = "finish_task"
    ASK_HUMAN = "ask_human"
    ABORT = "abort"


class DispatchDecision(StrictBaseModel):
    schema_version: str = "v1"
    action: DispatchAction
    target_role: str | None = None
    instruction: str | None = None
    retrieval_query: str | None = None
    expected_outputs: list[OutputSpec] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_fields(self) -> "DispatchDecision":
        if self.action == DispatchAction.RUN_SUBAGENT:
            if not self.target_role:
                raise ValueError("run_subagent requires target_role")
            if not self.instruction:
                raise ValueError("run_subagent requires instruction")
        return self
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evolab/contracts/common.py evolab/contracts/task.py evolab/contracts/dispatch.py tests/test_contracts.py
git commit -m "feat: add core contracts"
```

---

### Task 3: Runtime Contracts

**Files:**
- Create: `evolab/contracts/retrieval.py`
- Create: `evolab/contracts/tools.py`
- Create: `evolab/contracts/llm.py`
- Create: `evolab/contracts/evolution.py`
- Create: `evolab/contracts/records.py`
- Create: `evolab/contracts/state.py`
- Test: `tests/test_runtime_contracts.py`

- [ ] **Step 1: Write serialization tests**

```python
# tests/test_runtime_contracts.py
from evolab.contracts.common import Message
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.retrieval import RetrievalRequest
from evolab.contracts.tools import ToolCall, ToolResult


def test_retrieval_request_round_trips():
    request = RetrievalRequest(task_id="task-1", role="solver", query="prior failures")
    loaded = RetrievalRequest.model_validate_json(request.model_dump_json())
    assert loaded.query == "prior failures"


def test_tool_call_and_result_round_trip():
    call = ToolCall(call_id="call-1", name="read_file", arguments={"path": "x"})
    result = ToolResult(call_id="call-1", status="ok", content="done")
    assert ToolCall.model_validate(call.model_dump()).name == "read_file"
    assert ToolResult.model_validate(result.model_dump()).status == "ok"


def test_evolution_request_requires_mode():
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="local",
        artifact_root_uri="lab/evolution/llm/run-1",
        trigger_trajectory_ref="traj-1",
    )
    assert request.mode == LLMEvolutionMode.BASICS
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_runtime_contracts.py -q`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement retrieval contracts**

```python
# evolab/contracts/retrieval.py
from __future__ import annotations

from typing import Any

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class RetrievalRequest(StrictBaseModel):
    schema_version: str = "v1"
    task_id: str
    role: str
    query: str
    task_origin: str | None = None
    task_purpose: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(StrictBaseModel):
    schema_version: str = "v1"
    memory_id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryBundle(StrictBaseModel):
    schema_version: str = "v1"
    items: list[MemoryItem] = Field(default_factory=list)
    backend_id: str
    state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillRef(StrictBaseModel):
    schema_version: str = "v1"
    skill_id: str
    name: str
    content: str
    required_tools: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillBundle(StrictBaseModel):
    schema_version: str = "v1"
    skills: list[SkillRef] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    backend_id: str
    graph_version_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Implement tool and LLM contracts**

```python
# evolab/contracts/tools.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class ToolSpec(StrictBaseModel):
    schema_version: str = "v1"
    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class ToolBundle(StrictBaseModel):
    schema_version: str = "v1"
    tool_specs: list[ToolSpec] = Field(default_factory=list)


class ToolCall(StrictBaseModel):
    schema_version: str = "v1"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictBaseModel):
    schema_version: str = "v1"
    call_id: str
    status: Literal["ok", "error"]
    content: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(StrictBaseModel):
    schema_version: str = "v1"
    tool_call: ToolCall
    result: ToolResult


class ToolTrace(StrictBaseModel):
    schema_version: str = "v1"
    run_ref: str
    calls: list[ToolCallRecord] = Field(default_factory=list)
```

```python
# evolab/contracts/llm.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.tools import ToolCall


class LLMGenerationConfig(StrictBaseModel):
    schema_version: str = "v1"
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_json_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRuntimeRequest(StrictBaseModel):
    schema_version: str = "v1"
    messages: list[Message]
    tool_specs: list[dict[str, Any]] = Field(default_factory=list)
    generation_config: LLMGenerationConfig


class SubAgentAction(StrictBaseModel):
    schema_version: str = "v1"
    action: Literal["tool_call", "final_answer", "ask_human", "abort"]
    content: str | None = None
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRuntimeResponse(StrictBaseModel):
    schema_version: str = "v1"
    action: SubAgentAction
    raw_response: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Implement evolution, records, and state contracts**

```python
# evolab/contracts/evolution.py
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, EvolutionBudget, StrictBaseModel
from evolab.contracts.task import ProposerInputRef


class LLMEvolutionMode(StrEnum):
    BASICS = "basics"
    CONSOLIDATION = "consolidation"


class LabSignals(StrictBaseModel):
    schema_version: str = "v1"
    solve_rate: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceSnapshot(StrictBaseModel):
    schema_version: str = "v1"
    snapshot_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMEvolutionRequest(StrictBaseModel):
    schema_version: str = "v1"
    mode: LLMEvolutionMode
    backend_id: str
    previous_state_ref: str | None = None
    artifact_root_uri: str
    budget: EvolutionBudget = Field(default_factory=EvolutionBudget)
    trigger_trajectory_ref: str | None = None
    proposer_input_refs: list[ProposerInputRef] = Field(default_factory=list)
    lab_signals: LabSignals | None = None
    instance_snapshots: list[InstanceSnapshot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardEvolutionMetrics(StrictBaseModel):
    schema_version: str = "v1"
    n_train_samples: int | None = None
    eval_score_before: float | None = None
    eval_score_after: float | None = None
    eval_metric_name: str | None = None
    promotion_threshold: float | None = None
    promotion_margin: float | None = None


class LLMEvolutionResult(StrictBaseModel):
    schema_version: str = "v1"
    status: Literal["promoted_candidate", "not_recommended", "skipped", "failed"]
    new_state_ref: str | None = None
    recommend_for_promotion: bool = False
    lora_role: Literal["solver", "skill_distilled", "composed"] | None = None
    standard_metrics: StandardEvolutionMetrics = Field(default_factory=StandardEvolutionMetrics)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

```python
# evolab/contracts/state.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class BackendStateRecord(StrictBaseModel):
    schema_version: str = "v1"
    state_ref: str
    backend_id: str
    backend_type: Literal["llm", "memory", "skill"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_from_task_id: str | None = None
    created_from_run_ref: str | None = None
    parent_state_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    active: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
```

```python
# evolab/contracts/records.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, Message, StrictBaseModel
from evolab.contracts.dispatch import DispatchDecision
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionResult
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.tools import ToolCallRecord


class LLMCallRecord(StrictBaseModel):
    schema_version: str = "v1"
    call_ref: str
    run_ref: str
    backend_id: str
    model: str
    input_messages: list[Message]
    output_messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetaAgentRunRecord(StrictBaseModel):
    schema_version: str = "v1"
    run_ref: str
    task_id: str
    decision: DispatchDecision
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubagentRunRecord(StrictBaseModel):
    schema_version: str = "v1"
    run_ref: str
    task_id: str
    task_origin: str
    task_purpose: str
    producer_ref: str | None = None
    round_id: str | None = None
    human_anchor_task_refs: list[str] = Field(default_factory=list)
    human_anchor_trajectory_refs: list[str] = Field(default_factory=list)
    proposed_relation_type: str | None = None
    expected_transfer: str | None = None
    stage_index: int
    role: str
    instruction: str
    retrieval_request: RetrievalRequest
    memory_bundle: MemoryBundle
    skill_bundle: SkillBundle
    prompt_messages: list[Message]
    llm_call_refs: list[str] = Field(default_factory=list)
    llm_backend_id: str
    llm_backend_config_ref: str | None = None
    llm_backend_state_ref: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    output_messages: list[Message] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class EvolutionRunRecord(StrictBaseModel):
    schema_version: str = "v1"
    run_ref: str
    mode: LLMEvolutionMode
    backend_id: str
    result_status: Literal["promoted_candidate", "not_recommended", "skipped", "failed"]
    result: LLMEvolutionResult
    training_trajectory_refs: list[str] = Field(default_factory=list)
    consumed_instance_snapshot_refs: list[str] = Field(default_factory=list)
    parent_evolution_run_refs: list[str] = Field(default_factory=list)
    lora_role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_runtime_contracts.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evolab/contracts tests/test_runtime_contracts.py
git commit -m "feat: add runtime contracts"
```

---

### Task 4: Filesystem Lab And Queue

**Files:**
- Create: `evolab/lab/layout.py`
- Create: `evolab/lab/queue.py`
- Test: `tests/test_lab_queue.py`

- [ ] **Step 1: Write filesystem queue tests**

```python
# tests/test_lab_queue.py
from pathlib import Path

from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue


def test_lab_layout_creates_core_dirs(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    assert layout.tasks_queue_dir.exists()
    assert layout.evolve_queue_dir.exists()
    assert layout.trajectory_dir.exists()


def test_file_queue_enqueue_claim_done(tmp_path: Path):
    queue = FileWorkQueue(tmp_path / "queue")
    queue.ensure()
    queue.enqueue("job-1", {"job_id": "job-1", "value": 1})
    claimed = queue.claim("worker-1")
    assert claimed is not None
    assert claimed.payload["job_id"] == "job-1"
    queue.mark_done(claimed)
    assert (tmp_path / "queue" / "done").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_lab_queue.py -q`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement Lab layout**

```python
# evolab/lab/layout.py
from __future__ import annotations

from pathlib import Path


class LabLayout:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def tasks_queue_dir(self) -> Path:
        return self.root / "queues" / "tasks"

    @property
    def evolve_queue_dir(self) -> Path:
        return self.root / "queues" / "evolve"

    @property
    def trajectory_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def registries_dir(self) -> Path:
        return self.root / "registries"

    def task_dir(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id

    def evolution_run_dir(self, run_ref: str) -> Path:
        return self.root / "evolution" / "llm" / run_ref

    def ensure(self) -> None:
        dirs = [
            self.root / "configs",
            self.tasks_queue_dir,
            self.evolve_queue_dir,
            self.trajectory_dir / "meta_agent",
            self.trajectory_dir / "subagent",
            self.trajectory_dir / "llm_calls",
            self.trajectory_dir / "evolution",
            self.registries_dir / "trajectory",
            self.registries_dir / "backend_state",
            self.registries_dir / "task",
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Implement filesystem queue**

```python
# evolab/lab/queue.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    path: Path
    payload: dict[str, Any]


class FileWorkQueue:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def ensure(self) -> None:
        for name in ["queued", "claimed", "done", "failed", "skipped"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def enqueue(self, job_id: str, payload: dict[str, Any]) -> Path:
        self.ensure()
        path = self.root / "queued" / f"{job_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def claim(self, worker_id: str) -> ClaimedJob | None:
        self.ensure()
        for path in sorted((self.root / "queued").glob("*.json")):
            claimed_path = self.root / "claimed" / path.name
            try:
                os.replace(path, claimed_path)
            except FileNotFoundError:
                continue
            payload = json.loads(claimed_path.read_text(encoding="utf-8"))
            payload["claimed_by"] = worker_id
            payload["claimed_at"] = datetime.now(timezone.utc).isoformat()
            claimed_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return ClaimedJob(job_id=payload["job_id"], path=claimed_path, payload=payload)
        return None

    def mark_done(self, job: ClaimedJob) -> None:
        os.replace(job.path, self.root / "done" / job.path.name)

    def mark_failed(self, job: ClaimedJob, error: str) -> None:
        payload = dict(job.payload)
        payload["error"] = error
        job.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(job.path, self.root / "failed" / job.path.name)

    def mark_skipped(self, job: ClaimedJob, reason: str) -> None:
        payload = dict(job.payload)
        payload["skip_reason"] = reason
        job.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(job.path, self.root / "skipped" / job.path.name)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_lab_queue.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evolab/lab tests/test_lab_queue.py
git commit -m "feat: add filesystem lab queue"
```

---

### Task 5: Filesystem Registries

**Files:**
- Create: `evolab/registries/trajectory.py`
- Create: `evolab/registries/backend_state.py`
- Create: `evolab/registries/task.py`
- Test: `tests/test_registries.py`

- [ ] **Step 1: Write registry tests**

```python
# tests/test_registries.py
from pathlib import Path

from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.task import FileTaskRegistry


def test_backend_state_promote_and_resolve(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path)
    record = BackendStateRecord(state_ref="state-1", backend_id="llm-api", backend_type="llm")
    registry.register_candidate(record)
    registry.promote("llm-api", "state-1", "evo-1")
    assert registry.resolve_active_state("llm-api") == "state-1"


def test_task_registry_query_by_origin(tmp_path: Path):
    registry = FileTaskRegistry(tmp_path)
    request = TaskRequest(
        task_id="task-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Find target.",
    )
    registry.save_task_request(request)
    assert [item.task_id for item in registry.query_by_origin(TaskOrigin.HUMAN)] == ["task-1"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_registries.py -q`

Expected: FAIL with missing registry modules.

- [ ] **Step 3: Implement backend state registry**

```python
# evolab/registries/backend_state.py
from __future__ import annotations

import json
from pathlib import Path

from evolab.contracts.state import BackendStateRecord


class FileBackendStateRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "states.jsonl"
        self.active_path = self.root / "active.json"

    def register_candidate(self, record: BackendStateRecord) -> None:
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _load_active(self) -> dict[str, str]:
        if not self.active_path.exists():
            return {}
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    def promote(self, backend_id: str, new_state_ref: str, evolution_run_ref: str) -> None:
        active = self._load_active()
        active[backend_id] = new_state_ref
        active[f"{backend_id}:evolution_run_ref"] = evolution_run_ref
        self.active_path.write_text(json.dumps(active, indent=2, sort_keys=True), encoding="utf-8")

    def resolve_active_state(self, backend_id: str, role: str | None = None) -> str | None:
        active = self._load_active()
        if role and f"{backend_id}:{role}" in active:
            return active[f"{backend_id}:{role}"]
        return active.get(backend_id)
```

- [ ] **Step 4: Implement task registry**

```python
# evolab/registries/task.py
from __future__ import annotations

from pathlib import Path

from evolab.contracts.task import TaskOrigin, TaskRequest


class FileTaskRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_task_request(self, request: TaskRequest) -> Path:
        path = self.root / f"{request.task_id}.json"
        path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, task_id: str) -> TaskRequest:
        return TaskRequest.model_validate_json((self.root / f"{task_id}.json").read_text(encoding="utf-8"))

    def query_by_origin(self, origin: TaskOrigin) -> list[TaskRequest]:
        results = []
        for path in sorted(self.root.glob("*.json")):
            request = TaskRequest.model_validate_json(path.read_text(encoding="utf-8"))
            if request.origin == origin:
                results.append(request)
        return results

    def query_by_human_anchor(self, task_ref: str) -> list[TaskRequest]:
        results = []
        for path in sorted(self.root.glob("*.json")):
            request = TaskRequest.model_validate_json(path.read_text(encoding="utf-8"))
            relation = request.proposed_task_relation
            if relation and task_ref in relation.human_anchor_task_refs:
                results.append(request)
        return results
```

- [ ] **Step 5: Implement trajectory registry**

```python
# evolab/registries/trajectory.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from evolab.contracts.records import EvolutionRunRecord, LLMCallRecord, MetaAgentRunRecord, SubagentRunRecord


class FileTrajectoryRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, payload: str) -> None:
        with (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def save_meta_agent_run(self, record: MetaAgentRunRecord) -> str:
        self._append("meta_agent", record.model_dump_json())
        return record.run_ref

    def save_subagent_run(self, record: SubagentRunRecord) -> str:
        self._append("subagent", record.model_dump_json())
        return record.run_ref

    def save_llm_call(self, record: LLMCallRecord) -> str:
        self._append("llm_calls", record.model_dump_json())
        return record.call_ref

    def save_evolution_run(self, record: EvolutionRunRecord) -> str:
        self._append("evolution", record.model_dump_json())
        return record.run_ref

    def query_subagent_runs(self, filters: dict[str, Any]) -> list[SubagentRunRecord]:
        path = self.root / "subagent.jsonl"
        if not path.exists():
            return []
        results = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = SubagentRunRecord.model_validate_json(line)
            if all(getattr(record, key) == value for key, value in filters.items()):
                results.append(record)
        return results
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_registries.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evolab/registries tests/test_registries.py
git commit -m "feat: add filesystem registries"
```

---

### Task 6: Real LLM API Backend

**Files:**
- Create: `evolab/backends/llm.py`
- Test: `tests/test_api_llm_backend.py`

- [ ] **Step 1: Write API backend tests with test-local client fake**

```python
# tests/test_api_llm_backend.py
import os

import pytest

from evolab.backends.llm import ApiLLMBackend, ApiLLMBackendConfig
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig


class _FakeResponses:
    def create(self, **kwargs):
        class Response:
            output_text = "Final answer"

            def model_dump(self):
                return {"output_text": self.output_text, "kwargs": kwargs}

        return Response()


class _FakeClient:
    responses = _FakeResponses()


def test_api_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"))


def test_api_backend_generates_final_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient())
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )
    assert response.action.action == "final_answer"
    assert response.action.content == "Final answer"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_api_llm_backend.py -q`

Expected: FAIL with missing backend module.

- [ ] **Step 3: Implement real API backend wrapper**

```python
# evolab/backends/llm.py
from __future__ import annotations

import os
from typing import Any, Protocol

from pydantic import BaseModel

from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction


class LLMRuntime(Protocol):
    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        ...


class LLMBackend(Protocol):
    backend_id: str

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        ...


class ApiLLMBackendConfig(BaseModel):
    provider: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class OpenAIResponsesRuntime:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        input_messages = [{"role": message.role, "content": message.content} for message in messages]
        kwargs: dict[str, Any] = {
            "model": generation_config.model or self.model,
            "input": input_messages,
        }
        if generation_config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = generation_config.max_output_tokens
        if generation_config.temperature is not None:
            kwargs["temperature"] = generation_config.temperature
        if tool_specs:
            kwargs["tools"] = tool_specs
        if generation_config.response_json_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": generation_config.response_json_schema,
                    "strict": True,
                }
            }
        response = self.client.responses.create(**kwargs)
        content = getattr(response, "output_text", None) or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=content),
            raw_response=raw,
        )


class ApiLLMBackend:
    def __init__(self, config: ApiLLMBackendConfig, client: Any | None = None, backend_id: str = "api_llm"):
        self.config = config
        self.backend_id = backend_id
        api_key = os.environ.get(config.api_key_env)
        if not api_key and client is None:
            raise ValueError(f"missing API key in environment variable {config.api_key_env}")
        if client is None:
            from openai import OpenAI

            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            client = OpenAI(**client_kwargs)
        self.client = client

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        if state_ref is not None:
            raise ValueError("ApiLLMBackend does not support trainable state_ref")
        return OpenAIResponsesRuntime(self.client, self.config.model)


class LocalTrainableLLMBackend:
    backend_id = "local_trainable"

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        raise NotImplementedError("local trainable LLM runtime is not implemented in V1")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api_llm_backend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/llm.py tests/test_api_llm_backend.py
git commit -m "feat: add real api llm backend"
```

---

### Task 7: Mem0 Memory Backend

**Files:**
- Create: `evolab/backends/memory.py`
- Test: `tests/test_mem0_memory_backend.py`

- [ ] **Step 1: Write Mem0 adapter tests with test-local client fake**

```python
# tests/test_mem0_memory_backend.py
from evolab.backends.memory import Mem0MemoryBackend, Mem0MemoryConfig
from evolab.contracts.common import Message
from evolab.contracts.retrieval import RetrievalRequest


class _FakeMem0Client:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []

    def add(self, messages, user_id=None, **kwargs):
        self.add_calls.append({"messages": messages, "user_id": user_id, "kwargs": kwargs})
        return {"results": [{"id": "m1", "memory": "stored"}]}

    def search(self, query, **kwargs):
        self.search_calls.append({"query": query, "kwargs": kwargs})
        return {"results": [{"id": "m1", "memory": "remembered", "score": 0.9}]}


def test_mem0_search_maps_request_to_client():
    client = _FakeMem0Client()
    backend = Mem0MemoryBackend(Mem0MemoryConfig(user_id_template="{task_id}:{role}"), client=client)
    bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="q"))
    assert bundle.items[0].content == "remembered"
    assert client.search_calls[0]["kwargs"]["user_id"] == "task-1:solver"


def test_mem0_add_maps_messages_to_client():
    client = _FakeMem0Client()
    backend = Mem0MemoryBackend(Mem0MemoryConfig(user_id_template="{task_id}:{role}"), client=client)
    backend.add(task_id="task-1", role="solver", messages=[Message(role="user", content="hello")])
    assert client.add_calls[0]["messages"] == [{"role": "user", "content": "hello"}]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_mem0_memory_backend.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement Mem0 backend**

```python
# evolab/backends/memory.py
from __future__ import annotations

import os
from typing import Any, Protocol

from pydantic import BaseModel

from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, MemoryItem, RetrievalRequest


class MemoryBackend(Protocol):
    backend_id: str

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        ...


class Mem0MemoryConfig(BaseModel):
    api_key_env: str = "MEM0_API_KEY"
    user_id_template: str = "{task_id}:{role}"
    use_platform_client: bool = False


class Mem0MemoryBackend:
    backend_id = "mem0"

    def __init__(self, config: Mem0MemoryConfig, client: Any | None = None):
        self.config = config
        if client is None:
            if config.use_platform_client:
                api_key = os.environ.get(config.api_key_env)
                if not api_key:
                    raise ValueError(f"missing Mem0 API key in {config.api_key_env}")
                from mem0 import MemoryClient

                client = MemoryClient(api_key=api_key)
            else:
                from mem0 import Memory

                client = Memory()
        self.client = client

    def _user_id(self, task_id: str, role: str) -> str:
        return self.config.user_id_template.format(task_id=task_id, role=role)

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        user_id = self._user_id(request.task_id, request.role)
        result = self.client.search(request.query, user_id=user_id)
        raw_items = result.get("results", result if isinstance(result, list) else [])
        items = [
            MemoryItem(
                memory_id=str(item.get("id", "")),
                content=str(item.get("memory", item.get("content", ""))),
                score=item.get("score"),
                metadata={k: v for k, v in item.items() if k not in {"id", "memory", "content", "score"}},
            )
            for item in raw_items
        ]
        return MemoryBundle(items=items, backend_id=self.backend_id, metadata={"user_id": user_id})

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, Any]:
        user_id = self._user_id(task_id, role)
        mem0_messages = [{"role": message.role, "content": message.content} for message in messages]
        return self.client.add(mem0_messages, user_id=user_id)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mem0_memory_backend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/memory.py tests/test_mem0_memory_backend.py
git commit -m "feat: add mem0 memory backend"
```

---

### Task 8: Filesystem GraphSkillBackend

**Files:**
- Create: `evolab/backends/skills.py`
- Test: `tests/test_graph_skill_backend.py`

- [ ] **Step 1: Write skill backend tests**

```python
# tests/test_graph_skill_backend.py
import json
from pathlib import Path

from evolab.backends.skills import GraphSkillBackend
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle


def test_graph_skill_backend_gets_matching_skill(tmp_path: Path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "skills": [
                    {
                        "skill_id": "s1",
                        "name": "Literature Search",
                        "content": "Search papers.",
                        "required_tools": ["search"],
                        "tags": ["literature"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)
    bundle = backend.get(RetrievalRequest(task_id="task-1", role="solver", query="literature"))
    assert isinstance(bundle, SkillBundle)
    assert bundle.skills[0].skill_id == "s1"


def test_graph_skill_backend_look_at_records_update_summary(tmp_path: Path):
    backend = GraphSkillBackend(tmp_path / "skills.json")
    backend.look_at({"run_ref": "run-1", "summary": "observed"})
    assert (tmp_path / "skills.updates.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_graph_skill_backend.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement filesystem skill graph**

```python
# evolab/backends/skills.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillRef


class SkillBackend(Protocol):
    backend_id: str

    def get(self, request: RetrievalRequest) -> SkillBundle:
        ...

    def look_at(self, event: dict[str, Any]) -> dict[str, Any]:
        ...


class GraphSkillBackend:
    backend_id = "graph_skill"

    def __init__(self, graph_path: Path | str):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.exists():
            self.graph_path.write_text(json.dumps({"version": "v1", "skills": []}, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        return json.loads(self.graph_path.read_text(encoding="utf-8"))

    def get(self, request: RetrievalRequest) -> SkillBundle:
        graph = self._load()
        query = request.query.lower()
        matched = []
        for item in graph.get("skills", []):
            haystack = " ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("content", "")),
                    " ".join(item.get("tags", [])),
                ]
            ).lower()
            if not query or any(token in haystack for token in query.split()):
                matched.append(
                    SkillRef(
                        skill_id=item["skill_id"],
                        name=item["name"],
                        content=item.get("content", ""),
                        required_tools=item.get("required_tools", []),
                        metadata={k: v for k, v in item.items() if k not in {"skill_id", "name", "content", "required_tools"}},
                    )
                )
        required_tools = sorted({tool for skill in matched for tool in skill.required_tools})
        return SkillBundle(skills=matched, required_tools=required_tools, backend_id=self.backend_id, graph_version_ref=graph.get("version"))

    def look_at(self, event: dict[str, Any]) -> dict[str, Any]:
        update_path = self.graph_path.with_suffix(".updates.jsonl")
        with update_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return {"status": "recorded", "update_log": str(update_path)}

    def mine_resources(self) -> None:
        raise NotImplementedError("resource mining is a blank extension point in V1")

    def rewire_edges(self) -> None:
        raise NotImplementedError("skill graph rewiring is a blank extension point in V1")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_graph_skill_backend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/backends/skills.py tests/test_graph_skill_backend.py
git commit -m "feat: add filesystem skill backend"
```

---

### Task 9: ToolRuntime

**Files:**
- Create: `evolab/tools/runtime.py`
- Test: `tests/test_tool_runtime.py`

- [ ] **Step 1: Write ToolRuntime tests**

```python
# tests/test_tool_runtime.py
from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def test_tool_runtime_filters_allowed_tools():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="search", description="Search", parameters_schema={}), lambda args: "ok")
    runtime = ToolRuntime(registry)
    bundle = runtime.prepare(required_tools=["search"], allowed_tools=["search"], policy=RuntimePolicy())
    assert bundle.tool_specs[0].name == "search"


def test_tool_runtime_execute_returns_result():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="search", description="Search", parameters_schema={}), lambda args: "found")
    runtime = ToolRuntime(registry)
    result = runtime.execute_tool_name("call-1", "search", {"q": "x"})
    assert result.status == "ok"
    assert result.content == "found"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_tool_runtime.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement in-process ToolRuntime**

```python
# evolab/tools/runtime.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolBundle, ToolCall, ToolResult, ToolSpec


ToolHandler = Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def get_handler(self, name: str) -> ToolHandler:
        if name not in self._handlers:
            raise ValueError(f"tool is not registered: {name}")
        return self._handlers[name]


class ToolRuntime:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def prepare(self, required_tools: list[str], allowed_tools: list[str], policy: RuntimePolicy) -> ToolBundle:
        specs = []
        for name in required_tools:
            if name not in allowed_tools:
                continue
            spec = self.registry.get_spec(name)
            if spec is not None:
                specs.append(spec)
        return ToolBundle(tool_specs=specs)

    def execute(self, call: ToolCall) -> ToolResult:
        return self.execute_tool_name(call.call_id, call.name, call.arguments)

    def execute_tool_name(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            handler = self.registry.get_handler(name)
            content = handler(arguments)
            return ToolResult(call_id=call_id, status="ok", content=content)
        except Exception as exc:
            return ToolResult(call_id=call_id, status="error", content=str(exc))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tool_runtime.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evolab/tools/runtime.py tests/test_tool_runtime.py
git commit -m "feat: add in-process tool runtime"
```

---

### Task 10: Promotion Gate And EvolveWorker

**Files:**
- Create: `evolab/backends/trainers.py`
- Create: `evolab/runtime/promotion.py`
- Create: `evolab/runtime/evolve_worker.py`
- Test: `tests/test_promotion.py`

- [ ] **Step 1: Write promotion tests**

```python
# tests/test_promotion.py
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest, LLMEvolutionResult, StandardEvolutionMetrics
from evolab.runtime.promotion import validate_promotion


def test_promotion_requires_state_ref():
    request = LLMEvolutionRequest(mode=LLMEvolutionMode.BASICS, backend_id="local", artifact_root_uri="/tmp/evo")
    result = LLMEvolutionResult(status="promoted_candidate", recommend_for_promotion=True, lora_role="solver")
    errors = validate_promotion(result, request)
    assert "new_state_ref is empty" in errors


def test_promotion_accepts_valid_result():
    request = LLMEvolutionRequest(mode=LLMEvolutionMode.BASICS, backend_id="local", previous_state_ref="old", artifact_root_uri="/tmp/evo")
    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-new",
        lora_role="solver",
        standard_metrics=StandardEvolutionMetrics(eval_score_after=0.8),
        artifact_refs=[ArtifactRef(uri="/tmp/evo/adapter.bin", type="model_adapter")],
    )
    assert validate_promotion(result, request) == []
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_promotion.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement trainer protocol and promotion validation**

```python
# evolab/backends/trainers.py
from __future__ import annotations

from typing import Protocol

from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult


class LLMTrainer(Protocol):
    trainer_id: str

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        ...


class BlankTrainer:
    trainer_id = "blank"

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        raise NotImplementedError("training algorithm is not implemented")
```

```python
# evolab/runtime/promotion.py
from __future__ import annotations

from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest, LLMEvolutionResult


def expected_roles_for(mode: LLMEvolutionMode) -> set[str]:
    if mode == LLMEvolutionMode.BASICS:
        return {"solver"}
    if mode == LLMEvolutionMode.CONSOLIDATION:
        return {"skill_distilled"}
    return set()


def validate_promotion(result: LLMEvolutionResult, request: LLMEvolutionRequest) -> list[str]:
    errors: list[str] = []
    if not result.recommend_for_promotion:
        return errors
    if not result.new_state_ref:
        errors.append("new_state_ref is empty")
    if not any(ref.uri.startswith(request.artifact_root_uri) for ref in result.artifact_refs):
        errors.append("no artifact under artifact_root_uri")
    if request.previous_state_ref is not None and result.standard_metrics.eval_score_after is None:
        errors.append("eval_score_after missing for non-cold-start evolve")
    if result.lora_role not in expected_roles_for(request.mode):
        errors.append(f"lora_role {result.lora_role} mismatch with mode {request.mode}")
    return errors
```

- [ ] **Step 4: Implement EvolveWorker skeleton**

```python
# evolab/runtime/evolve_worker.py
from __future__ import annotations

from pathlib import Path

from evolab.backends.trainers import LLMTrainer
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.state import BackendStateRecord
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.runtime.promotion import validate_promotion


class EvolveWorker:
    def __init__(
        self,
        queue: FileWorkQueue,
        trainers: dict[str, LLMTrainer],
        backend_state_registry: FileBackendStateRegistry,
        worker_id: str,
    ):
        self.queue = queue
        self.trainers = trainers
        self.backend_state_registry = backend_state_registry
        self.worker_id = worker_id

    def run_once(self) -> bool:
        job = self.queue.claim(self.worker_id)
        if job is None:
            return False
        payload_path = Path(job.payload["request_payload_uri"])
        request = LLMEvolutionRequest.model_validate_json(payload_path.read_text(encoding="utf-8"))
        trainer = self.trainers.get(request.backend_id)
        if trainer is None:
            self.queue.mark_skipped(job, f"no trainer for backend_id {request.backend_id!r}")
            return True
        try:
            Path(request.artifact_root_uri).mkdir(parents=True, exist_ok=True)
            result = trainer.train(request)
            if result.status == "failed":
                self.queue.mark_failed(job, result.metadata.get("error", "trainer returned failed result"))
                return True
            if result.status == "skipped":
                self.queue.mark_skipped(job, result.metadata.get("reason", "trainer returned skipped result"))
                return True
            errors = validate_promotion(result, request)
            if errors:
                self.queue.mark_failed(job, "; ".join(errors))
                return True
            if result.recommend_for_promotion and result.new_state_ref:
                self.backend_state_registry.register_candidate(
                    BackendStateRecord(
                        state_ref=result.new_state_ref,
                        backend_id=request.backend_id,
                        backend_type="llm",
                        created_from_run_ref=job.job_id,
                        parent_state_refs=[request.previous_state_ref] if request.previous_state_ref else [],
                        artifact_refs=result.artifact_refs,
                    )
                )
                self.backend_state_registry.promote(request.backend_id, result.new_state_ref, job.job_id)
            self.queue.mark_done(job)
        except NotImplementedError as exc:
            self.queue.mark_skipped(job, str(exc))
        except Exception as exc:
            self.queue.mark_failed(job, str(exc))
        return True
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_promotion.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evolab/backends/trainers.py evolab/runtime/promotion.py evolab/runtime/evolve_worker.py tests/test_promotion.py
git commit -m "feat: add evolution promotion gate"
```

---

### Task 11: TaskRuntime And TaskWorker Skeleton

**Files:**
- Create: `evolab/config/task_config.py`
- Create: `evolab/runtime/prompt_builder.py`
- Create: `evolab/runtime/task_runtime.py`
- Create: `evolab/runtime/task_worker.py`
- Test: `tests/test_task_worker.py`

- [ ] **Step 1: Write TaskWorker startup test**

```python
# tests/test_task_worker.py
from pathlib import Path

from evolab.lab.layout import LabLayout
from evolab.runtime.task_worker import TaskWorker


def test_task_worker_initializes_lab(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    worker = TaskWorker(layout=layout, worker_id="worker-1")
    worker.startup()
    assert layout.tasks_queue_dir.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_task_worker.py -q`

Expected: FAIL with missing task worker module.

- [ ] **Step 3: Implement minimal config and prompt builder**

```python
# evolab/config/task_config.py
from __future__ import annotations

from pydantic import Field

from evolab.contracts.common import RuntimePolicy, StrictBaseModel


class BackendBinding(StrictBaseModel):
    backend_id: str
    config_ref: str | None = None
    state_ref: str | None = None


class RoleSpec(StrictBaseModel):
    name: str
    system_prompt: str
    llm_backend: BackendBinding
    allowed_tools: list[str] = Field(default_factory=list)


class TaskConfig(StrictBaseModel):
    task_id: str
    goal: str
    roles: dict[str, RoleSpec] = Field(default_factory=dict)
    max_dispatch_steps: int = 20
    runtime_policy: RuntimePolicy = Field(default_factory=RuntimePolicy)
```

```python
# evolab/runtime/prompt_builder.py
from __future__ import annotations

from evolab.config.task_config import RoleSpec
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, SkillBundle


class PromptBuilder:
    def build(self, role: RoleSpec, instruction: str, memory: MemoryBundle, skills: SkillBundle) -> list[Message]:
        memory_text = "\n".join(item.content for item in memory.items)
        skill_text = "\n".join(skill.content for skill in skills.skills)
        return [
            Message(role="system", content=role.system_prompt),
            Message(role="user", content=f"Instruction:\n{instruction}\n\nMemory:\n{memory_text}\n\nSkills:\n{skill_text}"),
        ]
```

- [ ] **Step 4: Implement TaskRuntime and TaskWorker skeleton**

```python
# evolab/runtime/task_runtime.py
from __future__ import annotations

from evolab.contracts.task import TaskRequest


class TaskRuntime:
    def __init__(self, dispatch_loop=None):
        self.dispatch_loop = dispatch_loop

    def run(self, request: TaskRequest) -> dict[str, str]:
        if self.dispatch_loop is not None:
            return self.dispatch_loop(request)
        raise NotImplementedError("task dispatch loop is not implemented")
```

```python
# evolab/runtime/task_worker.py
from __future__ import annotations

from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime
from evolab.tools.runtime import ToolRegistry, ToolRuntime


class TaskWorker:
    def __init__(self, layout: LabLayout, worker_id: str):
        self.layout = layout
        self.worker_id = worker_id
        self.task_runtime = TaskRuntime()

    def startup(self) -> None:
        self.layout.ensure()
        self.task_queue = FileWorkQueue(self.layout.tasks_queue_dir)
        self.task_queue.ensure()
        self.backend_state_registry = FileBackendStateRegistry(self.layout.registries_dir / "backend_state")
        self.task_registry = FileTaskRegistry(self.layout.registries_dir / "task")
        self.trajectory_registry = FileTrajectoryRegistry(self.layout.registries_dir / "trajectory")
        self.tool_registry = ToolRegistry()
        self.tool_runtime = ToolRuntime(self.tool_registry)
        self.prompt_builder = PromptBuilder()

    def run_once(self) -> dict[str, str] | None:
        job = self.task_queue.claim(self.worker_id)
        if job is None:
            return None
        from evolab.contracts.task import TaskRequest

        request = TaskRequest.model_validate_json(open(job.payload["request_payload_uri"], encoding="utf-8").read())
        result = self.task_runtime.run(request)
        self.task_queue.mark_done(job)
        return result
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_task_worker.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evolab/config evolab/runtime tests/test_task_worker.py
git commit -m "feat: add task worker skeleton"
```

---

### Task 12: Full Test Pass And Documentation Check

**Files:**
- Modify: `docs/spec.md`
- Modify: `docs/superpowers/specs/2026-05-02-evolab-framework-design.md`
- Modify: `docs/superpowers/plans/2026-05-02-evolab-framework-implementation.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`

Expected: PASS for all tests added by this plan.

- [ ] **Step 2: Search for product mock implementations**

Run: `rg -n "class Mock|MockBackend|MockTrainer|fake training|mock backend|mock trainer" evolab tests docs`

Expected: no matches under `evolab/`; matches under `tests/` are acceptable only for names that start with `_Fake`.

- [ ] **Step 3: Verify blank extension points are explicit**

Run: `rg -n "NotImplementedError|blank extension point|not implemented in V1" evolab docs`

Expected: matches for local trainable LLM, graph skill mining/rewiring, and blank trainer paths.

- [ ] **Step 4: Commit final verification changes**

```bash
git add docs evolab tests pyproject.toml
git commit -m "test: verify framework skeleton"
```
