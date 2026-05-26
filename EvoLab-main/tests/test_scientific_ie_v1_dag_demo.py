from pathlib import Path
import shutil

from evolab.backends.skills import GraphSkillBackend
from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime as _TaskRuntime
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.scientific_ie import register_scientific_ie_tools


class EmptyMemory:
    def search(self, request: RetrievalRequest) -> MemoryBundle:
        return MemoryBundle(backend_id="memory-local")

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        return {"status": "updated", "state_ref": "memory-after"}


class DemoLLM:
    def __init__(self):
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []
        self.step = 0
        self.wrote_records = False

    def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
        self.calls.append((messages, tool_specs, generation_config))
        self.step += 1
        tool_names = {spec["name"] for spec in tool_specs}
        if not self.wrote_records and "write_jsonl" in tool_names:
            self.wrote_records = True
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="write-records",
                        name="write_jsonl",
                        arguments={"records": [{"component_name": "mock component"}]},
                    ),
                )
            )
        if self.step == 3 and "ask_human" in tool_names:
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="ask-human",
                        name="ask_human",
                        arguments={"question": "Proceed conservatively?", "context": "demo"},
                    ),
                )
            )
        return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=f"node {self.step} done"))


class DynamicPlanner:
    def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
        payload = {
            "workflow_id": "wf-scientific-ie-demo",
            "task_summary": "Extract biological component records.",
            "article_context_summary": "unit test",
            "dynamic_subagents": [
                {
                    "subagent_id": "solver",
                    "role_name": "solver",
                    "goal": "Extract biological component records.",
                    "system_prompt": "Extract scientific records.",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "allowed_tools": SCIENTIFIC_IE_ALLOWED_TOOLS,
                    "llm_backend_id": "llm-local",
                }
            ],
            "workflow_nodes": [{"node_id": "node-solver", "subagent_id": "solver"}],
            "workflow_edges": [],
            "artifact_contracts": {},
            "validation_rules": [],
            "planner_rationale_summary": "Run the scientific IE solver.",
            "metadata": {"extraction_task": False},
        }
        import json

        return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(payload)))


def TaskRuntime(**kwargs) -> _TaskRuntime:
    task_config = kwargs.get("task_config")
    if isinstance(task_config, TaskConfig) and task_config.dynamic_subagents is None:
        root = getattr(kwargs.get("trajectory_registry"), "root", None)
        agents_ref = (
            str(root.parent / "agents.md")
            if isinstance(root, Path)
            else str(Path("/tmp/evolab-test-agents") / "scientific-ie-v1-agents.md")
        )
        kwargs["task_config"] = task_config.model_copy(
            update={
                "agents_ref": task_config.agents_ref or agents_ref,
                "dynamic_subagents": DynamicSubagentsConfig(
                    enabled=True,
                    mode="dynamic",
                    planner_backend={"backend_id": "planner-local"},
                    default_worker_backend={"backend_id": "llm-local"},
                    allowed_worker_backend_ids=["llm-local"],
                    allowed_tool_names=SCIENTIFIC_IE_ALLOWED_TOOLS,
                    max_planner_retries=0,
                    require_output_schema=False,
                )
            }
        )
        kwargs["llm_runtimes"] = {"planner-local": DynamicPlanner(), **(kwargs.get("llm_runtimes") or {})}
    return _TaskRuntime(**kwargs)


SCIENTIFIC_IE_ALLOWED_TOOLS = [
    "list_files",
    "read_text",
    "inspect_file_metadata",
    "extract_sections",
    "search_text",
    "read_table_slice",
    "inspect_table",
    "inspect_excel_workbook",
    "read_excel_sheet",
    "detect_table_header",
    "normalize_table",
    "profile_table",
    "json_schema_validate",
    "write_jsonl",
    "write_report",
    "ask_human",
    "request_human_review",
    "notify_human",
]


def test_scientific_ie_v1_dag_demo_uses_real_seed_graph_packages_tools_artifacts_and_look_at(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    graph_copy = tmp_path / "scientific_ie_seed_graph_v1.json"
    shutil.copy2(repo_root / "dev" / "configs" / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json", graph_copy)
    skill_backend = GraphSkillBackend(graph_copy, repo_root=repo_root / "dev")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    registry = ToolRegistry()
    register_scientific_ie_tools(registry, artifact_root=tmp_path / "tool-artifacts")
    request = TaskRequest(
        task_id="bio-ie-demo",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Extract biological component records from a scientific article with supplementary tables and sequences.",
        metadata={"domain_package": "dev/domain_packages/biology_component_extraction_v1"},
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        runtime_policy=RuntimePolicy(enable_workflow_planning=True, allow_human_tools=True, max_workflow_nodes=20),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Extract scientific records.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=SCIENTIFIC_IE_ALLOWED_TOOLS,
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        backend_state_registry=backend_state_registry,
        tool_artifact_root_factory=lambda request, run_ref: tmp_path / "artifacts" / run_ref,
        llm_runtimes={"llm-local": DemoLLM()},
        memory_runtimes={"memory-local": EmptyMemory()},
        skill_runtimes={"skill-local": skill_backend},
    )

    result = runtime.run(request)
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    skill_ids = [skill.skill_id for skill in saved.skill_bundle.skills]
    assert "skill.structured_record_construction.v1" in skill_ids
    assert "skill.extraction_schema_interpretation.v1" in skill_ids
    forbidden = ["promoter", "rbs", "terminator", "grna", "microbe_trait", "chemical_reaction", "material_property"]
    assert not any(term in skill_id.casefold() for term in forbidden for skill_id in skill_ids)
    assert result["execution_mode"] == "dynamic"
    assert saved.metadata["dispatch_metadata"]["dynamic_workflow_id"] == "wf-scientific-ie-demo"
    assert saved.metadata["tool_trace"]["calls"]
    assert saved.artifact_refs
    assert any(call["tool_call"]["name"] == "write_jsonl" for call in saved.metadata["tool_trace"]["calls"])
    assert saved.metadata["skill_observation_request"]["run_ref"] == result["run_ref"]
    assert (graph_copy.with_suffix(".updates.jsonl")).exists()
    assert saved.skill_bundle.skill_state_ref == "scientific-ie-v1"
    state_record = backend_state_registry.get_state("scientific-ie-v1")
    assert state_record is not None
    assert state_record.backend_id == "graph_skill"
    assert state_record.backend_type == "skill"
    assert state_record.created_from_task_id == request.task_id
    assert state_record.created_from_run_ref == result["run_ref"]
