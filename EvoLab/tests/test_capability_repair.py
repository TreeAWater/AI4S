from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.repair import SkillOverlayPatch, ToolOverlayPatch
from evolab.contracts.retrieval import (
    MemoryBundle,
    MemoryItem,
    RetrievalRequest,
    SkillBundle,
    SkillRef,
    SkillUpdateResult,
)
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolResult, ToolSpec
from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.capability_repair import (
    CapabilityRepairRuntime,
    FailureDetector,
    RepairValidator,
    TaskLocalSkillOverlay,
    TaskLocalToolOverlay,
)
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime
from evolab.runtime.task_worker import TaskWorker
from evolab.tools.runtime import ToolRegistry, ToolRuntime


TABLE_BLOCK = {
    "caption": "Table S1 Promoter activity",
    "start_line": 309,
    "end_line": 471,
}
TABLE_HEADERS = ["names", "promoter_sequences", "fi_od600"]
TABLE_ROWS = [
    ["Names", "Promoter sequences", "FI/OD600"],
    ["J23119", "TTGACAGCAATCTCAATCCTAGGTATAAT", "51747"],
    ["S1", "AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATAATGCGAGCA", "449926"],
]


class StaticMemoryRuntime:
    def __init__(self) -> None:
        self.search_requests: list[RetrievalRequest] = []
        self.add_calls: list[tuple[str, str, list[Message]]] = []

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        self.search_requests.append(request)
        return MemoryBundle(
            backend_id="memory-local",
            state_ref="memory-state-before",
            items=[MemoryItem(memory_id="memory-1", content="Prior note", score=0.5)],
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        self.add_calls.append((task_id, role, messages))
        return {"status": "updated", "state_ref": f"{role}-state-after"}


class StaticSkillRuntime:
    def __init__(self) -> None:
        self.get_requests: list[RetrievalRequest] = []
        self.look_at_events: list[dict] = []

    def get(self, request: RetrievalRequest) -> SkillBundle:
        self.get_requests.append(request)
        return _skill_bundle()

    def look_at(self, event: dict) -> SkillUpdateResult:
        self.look_at_events.append(event)
        return SkillUpdateResult(
            status="recorded",
            update_summary={"repair_trajectory_seen": "repair_trajectory" in event["metadata"]},
            graph_version_ref="skill-graph-v1",
            skill_state_ref="skill-state-v2",
        )


class ScriptedLLMRuntime:
    def __init__(self, responses: list[LLMRuntimeResponse]):
        self.responses = responses
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return self.responses.pop(0)


class InterruptingCapabilityRepairRuntime(CapabilityRepairRuntime):
    def validate_and_retry(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("interrupt after repair planning")


def _skill_bundle() -> SkillBundle:
    return SkillBundle(
        backend_id="skill-local",
        graph_version_ref="skill-graph-v1",
        skill_state_ref="skill-state-v1",
        required_tools=["read_table_slice"],
        skills=[
            SkillRef(
                skill_id="skill-table-extraction",
                name="Scientific table extraction",
                content="Extract structured records from scientific tables.",
                required_tools=["read_table_slice"],
                metadata={"retrieval": {"category_path": "Science > Tables"}},
            )
        ],
        metadata={"graph_context_summary": {"graph_version": "skill-graph-v1"}},
    )


def _request(task_id: str = "task-1") -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Extract promoter records from the recoverable supplementary table.",
    )


def _role() -> RoleSpec:
    return RoleSpec(
        name="solver",
        system_prompt="You solve extraction tasks.",
        llm_backend=BackendBinding(backend_id="llm-local"),
        allowed_tools=["read_table_slice"],
    )


def _tool_spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} tool", parameters_schema={"type": "object"})


def _register_table_like_tools(registry: ToolRegistry) -> None:
    registry.register(
        _tool_spec("inspect_table"),
        lambda arguments: ToolResult(
            call_id="inspect-local",
            status="ok",
            content="inspected table with 3 rows",
            metadata={
                "row_count": len(TABLE_ROWS),
                "headers": TABLE_HEADERS,
                "plain_text_table_block": TABLE_BLOCK,
                "warnings": [],
            },
        ),
    )

    def _read_table_slice(arguments: dict) -> ToolResult:
        start_row = int(arguments.get("start_row", 0))
        end_row = int(arguments.get("end_row", len(TABLE_ROWS)))
        if 0 <= start_row <= len(TABLE_ROWS) and 0 <= end_row <= len(TABLE_ROWS) and start_row < end_row:
            rows = TABLE_ROWS[start_row:end_row]
        else:
            rows = []
        return ToolResult(
            call_id="base-local",
            status="ok",
            content=f"read {len(rows)} table rows",
            metadata={
                "path": arguments.get("path", "/tmp/table.md"),
                "start_row": start_row,
                "end_row": end_row,
                "headers": TABLE_HEADERS,
                "rows": rows,
                "warnings": [],
                "plain_text_table_block": TABLE_BLOCK,
            },
        )

    registry.register(_tool_spec("read_table_slice"), _read_table_slice)


def test_failure_detector_detects_invalid_table_slice_bounds():
    detector = FailureDetector()

    signal = detector.detect_tool_failure(
        task_id="task-1",
        subagent_id="subagent-1",
        step_id="step-1",
        tool_call=ToolCall(
            call_id="call-1",
            name="read_table_slice",
            arguments={"path": "/tmp/table.md", "start_row": 9, "end_row": 3},
        ),
        tool_result=ToolResult(
            call_id="call-1",
            status="ok",
            content="read 0 table rows",
            metadata={"rows": []},
        ),
        active_skill_bundle=_skill_bundle(),
        task_context={},
    )

    assert signal is not None
    assert signal.failure_type == "invalid_tool_arguments"
    assert signal.suspected_cause == "start_row_greater_than_end_row"


def test_failure_detector_detects_source_line_vs_row_index_mismatch():
    detector = FailureDetector()

    signal = detector.detect_tool_failure(
        task_id="task-1",
        subagent_id="subagent-1",
        step_id="step-1",
        tool_call=ToolCall(
            call_id="call-1",
            name="read_table_slice",
            arguments={
                "path": "/tmp/table.md",
                "table_caption": "Table S1",
                "start_row": 309,
                "end_row": 471,
            },
        ),
        tool_result=ToolResult(
            call_id="call-1",
            status="ok",
            content="read 0 table rows",
            metadata={
                "rows": [],
                "row_count": 3,
                "plain_text_table_block": TABLE_BLOCK,
                "warnings": [],
            },
        ),
        active_skill_bundle=_skill_bundle(),
        task_context={},
    )

    assert signal is not None
    assert signal.failure_type == "coordinate_system_mismatch"
    assert "source_line" in signal.suspected_cause


def test_tool_overlay_safe_wrapper_normalizes_source_line_coordinates():
    registry = ToolRegistry()
    _register_table_like_tools(registry)
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["read_table_slice"],
        allowed_tools=["read_table_slice"],
        policy=RuntimePolicy(),
    )
    overlay = TaskLocalToolOverlay(
        patches=[
            ToolOverlayPatch(
                patch_id="patch-read-table-slice-safe",
                name="read_table_slice_safe",
                base_tool_name="read_table_slice",
                strategy="safe_read_table_slice_wrapper",
            )
        ]
    )

    runtime.apply_runtime_tool_overlay(overlay)
    result = runtime.execute(
        ToolCall(
            call_id="call-1",
            name="read_table_slice_safe",
            arguments={
                "path": "/tmp/table.md",
                "table_caption": "Table S1",
                "start_row": 309,
                "end_row": 471,
            },
        )
    )

    assert result.status == "ok"
    assert result.metadata["rows"][1][0] == "J23119"
    assert "normalized source line coordinates" in " ".join(result.metadata["warnings"])


def test_skill_overlay_adds_runtime_tool_use_policy_without_mutating_global_bundle():
    bundle = _skill_bundle()
    overlay = TaskLocalSkillOverlay(
        patches=[
            SkillOverlayPatch(
                patch_id="patch-table-coordinates",
                target_skill_name="Scientific table extraction",
                principles=[
                    "distinguish source-file line coordinates from table-relative row coordinates"
                ],
                failure_modes=["source_line_as_row_index"],
                recovery_strategies=[
                    "normalize source line ranges to table-relative row bounds before retry"
                ],
            )
        ]
    )

    merged = overlay.apply(bundle)

    assert "source-file line coordinates" in merged.skills[0].content
    assert merged.metadata["runtime_skill_overlay"]["patches"][0]["patch_id"] == "patch-table-coordinates"
    assert "source-file line coordinates" not in bundle.skills[0].content


def test_repair_validator_validates_safe_table_slice_patch():
    registry = ToolRegistry()
    _register_table_like_tools(registry)
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["read_table_slice"],
        allowed_tools=["read_table_slice"],
        policy=RuntimePolicy(),
    )
    validator = RepairValidator()
    plan = CapabilityRepairRuntime().planner.plan(
        CapabilityRepairRuntime().detector.detect_tool_failure(
            task_id="task-1",
            subagent_id="subagent-1",
            step_id="step-1",
            tool_call=ToolCall(
                call_id="call-1",
                name="read_table_slice",
                arguments={
                    "path": "/tmp/table.md",
                    "table_caption": "Table S1",
                    "start_row": 309,
                    "end_row": 471,
                },
            ),
            tool_result=ToolResult(
                call_id="call-1",
                status="ok",
                content="read 0 table rows",
                metadata={"rows": [], "plain_text_table_block": TABLE_BLOCK, "row_count": 3},
            ),
            active_skill_bundle=_skill_bundle(),
            task_context={},
        )
    )

    validation = validator.validate_repair_plan(
        plan=plan,
        tool_runtime=runtime,
        failed_tool_call=ToolCall(
            call_id="call-1",
            name="read_table_slice",
            arguments={
                "path": "/tmp/table.md",
                "table_caption": "Table S1",
                "start_row": 309,
                "end_row": 471,
            },
        ),
    )

    assert validation.valid is True
    assert validation.before_summary["row_count"] == 0
    assert validation.after_summary["row_count"] > 0
    assert validation.normal_behavior_ok is True


def test_runtime_repair_retries_failed_step_and_persists_repair_trajectory(tmp_path: Path):
    request = _request()
    memory = StaticMemoryRuntime()
    skill = StaticSkillRuntime()
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="read_table_slice",
                        arguments={
                            "path": "/tmp/table.md",
                            "table_caption": "Table S1",
                            "start_row": 309,
                            "end_row": 471,
                        },
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    registry = ToolRegistry()
    _register_table_like_tools(registry)
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={"solver": _role()},
        runtime_policy=RuntimePolicy(
            enable_runtime_capability_repair=True,
            max_repair_attempts_per_step=1,
            max_repair_attempts_per_task=2,
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=trajectory_registry,
        backend_state_registry=FileBackendStateRegistry(tmp_path / "backend-state"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert len(saved.tool_calls) == 2
    assert saved.tool_calls[0].result.metadata["rows"] == []
    assert saved.tool_calls[1].tool_call.name == "read_table_slice_safe"
    assert saved.tool_calls[1].result.metadata["rows"][1][0] == "J23119"
    assert saved.metadata["repair_trajectory"][0]["failure_signal"]["failure_type"] == "coordinate_system_mismatch"
    assert saved.metadata["repair_trajectory"][0]["validation_result"]["valid"] is True
    event_types = [event.event_type for event in trajectory_registry.list_events()]
    assert "repair_detected" in event_types
    assert "repair_retried" in event_types


def test_skill_backend_observation_receives_repair_trajectory_without_global_mutation(tmp_path: Path):
    request = _request()
    memory = StaticMemoryRuntime()
    skill = StaticSkillRuntime()
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="read_table_slice",
                        arguments={
                            "path": "/tmp/table.md",
                            "table_caption": "Table S1",
                            "start_row": 309,
                            "end_row": 471,
                        },
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    registry = ToolRegistry()
    _register_table_like_tools(registry)
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id=request.task_id,
            goal=request.goal,
            roles={"solver": _role()},
            runtime_policy=RuntimePolicy(enable_runtime_capability_repair=True),
        ),
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    assert skill.look_at_events
    assert skill.look_at_events[0]["metadata"]["repair_trajectory"]
    assert backend_state_registry.resolve_active_state("skill-local") is None


def test_interrupted_subagent_persists_partial_repair_postmortem_record(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    request_path = layout.root / "requests" / f"{request.task_id}.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    FileWorkQueue(layout.tasks_queue_dir).enqueue("job-1", {"request_payload_uri": str(request_path)})

    memory = StaticMemoryRuntime()
    skill = StaticSkillRuntime()
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="read_table_slice",
                        arguments={
                            "path": "/tmp/table.md",
                            "table_caption": "Table S1",
                            "start_row": 309,
                            "end_row": 471,
                        },
                    ),
                )
            )
        ]
    )
    registry = ToolRegistry()
    _register_table_like_tools(registry)
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_config=TaskConfig(
            task_id=request.task_id,
            goal=request.goal,
            roles={"solver": _role()},
            runtime_policy=RuntimePolicy(enable_runtime_capability_repair=True),
        ),
        task_runtime=TaskRuntime(
            task_config=TaskConfig(
                task_id=request.task_id,
                goal=request.goal,
                roles={"solver": _role()},
                runtime_policy=RuntimePolicy(enable_runtime_capability_repair=True),
            ),
            prompt_builder=PromptBuilder(),
            tool_runtime=ToolRuntime(registry),
            trajectory_registry=FileTrajectoryRegistry(layout.registries_dir / "trajectory"),
            backend_state_registry=FileBackendStateRegistry(layout.registries_dir / "backend_state"),
            llm_runtimes={"llm-local": llm},
            memory_runtimes={"memory-local": memory},
            skill_runtimes={"skill-local": skill},
            capability_repair_runtime=InterruptingCapabilityRepairRuntime(),
        ),
    )
    worker.startup()

    result = worker.run_once()

    assert result is None
    events = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_events()
    event_types = [event.event_type for event in events]
    assert "repair_planned" in event_types
    assert "subagent_postmortem" in event_types
    postmortem = next(event for event in events if event.event_type == "subagent_postmortem")
    assert postmortem.metadata["repair_trajectory"]
