from pathlib import Path

import pytest

from evolab.backends.embeddings import FakeEmbeddingBackend
from evolab.backends.memory import MethodMemoryBackend
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime


class ScriptedLLM:
    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))


class EmptySkill:
    def get(self, request: RetrievalRequest) -> SkillBundle:
        return SkillBundle(backend_id="skill-local")

    def look_at(self, event: dict) -> dict[str, str]:
        return {"status": "recorded"}


class StaticMemory:
    def __init__(self, update_result: dict):
        self.update_result = update_result

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        return MemoryBundle(
            backend_id="memory-local",
            state_ref=f"before:{request.role}",
            metadata={
                "memory_scope": request.filters.get("memory_scope", "agent"),
                "memory_scope_id": request.filters.get("memory_scope_id", f"agent:{request.role}"),
            },
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict:
        return self.update_result


class FailingEmbeddingRuntime:
    def embed(self, texts: list[str], *, purpose: str):
        raise RuntimeError("mem0 embedding unavailable")

def _native_mem0_backend(
    tmp_path: Path,
    *,
    backend_id: str = "mem0-local",
    embedding_runtime=None,
) -> MethodMemoryBackend:
    backend = MethodMemoryBackend(
        backend_id=backend_id,
        method=Mem0MemoryMethod(
            store_path=tmp_path / f"{backend_id}.sqlite",
            llm_backend_id="llm-local",
            embedding_backend_id="embed-local",
        ),
    )
    backend.bind_runtimes(
        llm_runtimes={"llm-local": ScriptedLLM()},
        embedding_runtimes={"embed-local": embedding_runtime or FakeEmbeddingBackend().instantiate(None)},
    )
    return backend


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="task-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Check memory failure handling.",
    )


def _runtime(
    tmp_path: Path,
    *,
    memory,
    backend_state_registry: FileBackendStateRegistry | None = None,
) -> TaskRuntime:
    return TaskRuntime(
        task_config=TaskConfig(
            task_id="task-1",
            goal="Check memory failure handling.",
            roles={
                "solver": RoleSpec(
                    name="solver",
                    system_prompt="Solve.",
                    llm_backend=BackendBinding(backend_id="llm-local"),
                )
            },
        ),
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": ScriptedLLM()},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": EmptySkill()},
    )


def test_mem0_empty_search_result_is_valid_bundle_with_state_ref(tmp_path: Path):
    backend = _native_mem0_backend(tmp_path)

    bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="nothing"))

    assert bundle.items == []
    assert bundle.backend_id == "mem0-local"
    assert bundle.state_ref == "method://mem0/10:mem0-local/5:agent/12:agent:solver/v0"
    assert bundle.metadata["memory_scope"] == "agent"
    assert bundle.metadata["memory_scope_id"] == "agent:solver"
    assert bundle.metadata["memory_method"] == "mem0"


def test_mem0_search_failure_raises_clear_error(tmp_path: Path):
    backend = _native_mem0_backend(tmp_path, embedding_runtime=FailingEmbeddingRuntime())

    with pytest.raises(RuntimeError, match="mem0 embedding unavailable"):
        backend.search(RetrievalRequest(task_id="task-1", role="solver", query="history"))


def test_mem0_add_failure_is_recordable_failed_update_without_new_active_state(tmp_path: Path):
    backend = _native_mem0_backend(tmp_path)

    result = backend.add("task-1", "solver", [Message(role="assistant", content="remember")])

    assert result.status == "failed"
    assert result.state_ref == "method://mem0/10:mem0-local/5:agent/12:agent:solver/v0"
    assert result.previous_state_ref == "method://mem0/10:mem0-local/5:agent/12:agent:solver/v0"
    assert result.metadata["error_type"] == "extraction_parse"


def test_runtime_records_missing_memory_state_ref_without_registering_state(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path / "backend-state")
    runtime = _runtime(tmp_path, memory=StaticMemory({"status": "updated"}), backend_state_registry=registry)

    result = runtime.run(_request())

    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.metadata["agent_memory_update_result"] == {"status": "updated"}
    assert saved.metadata["task_memory_update_result"] == {"status": "updated"}
    assert registry.list_states() == []


def test_runtime_records_degraded_memory_update_without_registering_state(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path / "backend-state")
    runtime = _runtime(
        tmp_path,
        memory=StaticMemory(
            {
                "status": "degraded",
                "state_ref": "memory-after",
                "metadata": {"reason": "partial external update"},
            }
        ),
        backend_state_registry=registry,
    )

    result = runtime.run(_request())

    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.metadata["agent_memory_update_result"]["status"] == "degraded"
    assert saved.metadata["agent_memory_update_result"]["metadata"]["reason"] == "partial external update"
    assert registry.list_states() == []


def test_runtime_preserves_mem0_add_failure_metadata_without_registering_state(tmp_path: Path):
    registry = FileBackendStateRegistry(tmp_path / "backend-state")
    backend = _native_mem0_backend(tmp_path)
    runtime = _runtime(tmp_path, memory=backend, backend_state_registry=registry)

    result = runtime.run(_request())

    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.metadata["agent_memory_update_result"]["status"] == "failed"
    assert saved.metadata["agent_memory_update_result"]["metadata"]["error_type"] == "extraction_parse"
    assert saved.metadata["task_memory_update_result"]["status"] == "failed"
    assert registry.list_states() == []
