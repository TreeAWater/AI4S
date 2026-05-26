import json
from pathlib import Path

from evolab.backends.embeddings.fake import FakeEmbeddingRuntime
from evolab.backends.memory import MethodMemoryBackend
from evolab.backends.memory.methods.everos import EverOSMemoryMethod
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import RetrievalRequest


class ScriptedLLMRuntime:
    def __init__(self, contents: list[dict]):
        self.contents = list(contents)
        self.calls = []

    def generate(self, messages, tool_specs, generation_config):
        self.calls.append(
            {
                "messages": messages,
                "tool_specs": tool_specs,
                "generation_config": generation_config,
            }
        )
        if not self.contents:
            raise AssertionError("unexpected LLM call")
        return LLMRuntimeResponse(
            action=SubAgentAction(
                action="final_answer",
                content=json.dumps(self.contents.pop(0)),
            )
        )


def _memcell_payload(
    *,
    summary: str = "Sigma70 promoter extraction succeeded",
    episode: str = "The solver extracted Sigma70 promoter evidence and rejected primer-only rows.",
    fact: str = "Sigma70 promoter rows require direct regulatory DNA evidence.",
) -> dict:
    return {
        "memcells": [
            {
                "summary": summary,
                "episode": episode,
                "salience": 0.9,
                "atomic_facts": [{"text": fact, "entities": ["Sigma70"]}],
                "foresights": [
                    {
                        "text": "Future biology component extraction should audit primer-only rows.",
                        "evidence": "The solver rejected primer-only rows.",
                        "validity": "future extraction tasks",
                    }
                ],
                "agent_case": {
                    "task_intent": "Extract promoter component records",
                    "approach": "Survey evidence, reject primers, write accepted JSONL records.",
                    "key_insight": "Promoter evidence must be direct, not inferred from sequencing primer labels.",
                    "quality_score": 0.86,
                },
                "agent_skills": [
                    {
                        "name": "Primer rejection audit",
                        "description": "Distinguish promoter evidence from primer metadata.",
                        "content": "Check record context and reject primer-only sequence mentions.",
                        "confidence": 0.8,
                    }
                ],
            }
        ]
    }


def _scene_payload(summary: str = "Promoter extraction evidence auditing") -> dict:
    return {
        "title": "Biology component extraction",
        "summary": summary,
        "tags": ["biology", "promoter", "Sigma70"],
    }


def _backend(tmp_path: Path, llm: ScriptedLLMRuntime, *, recollection_mode: str = "scene"):
    method = EverOSMemoryMethod(
        store_path=tmp_path / "everos.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
        scene_similarity_threshold=0.0,
        recollection_mode=recollection_mode,
    )
    backend = MethodMemoryBackend(
        backend_id="everos-memory",
        method=method,
        default_search_threshold=0.0,
        default_search_top_k=5,
    )
    backend.bind_runtimes(
        llm_runtimes={"llm": llm},
        embedding_runtimes={"embed": FakeEmbeddingRuntime("embed", dimensions=8)},
    )
    return backend, method


def test_everos_memory_method_extracts_memcells_consolidates_scenes_and_searches(tmp_path: Path):
    llm = ScriptedLLMRuntime([_memcell_payload(), _scene_payload()])
    backend, method = _backend(tmp_path, llm)

    result = backend.add(
        "task-1",
        "solver",
        [Message(role="assistant", content="Solver extracted Sigma70 promoter evidence.")],
    )

    assert result.status == "updated"
    assert result.metadata["memcell_count"] == 1
    assert result.metadata["scene_count"] == 1
    assert result.metadata["record_count"] == 5
    assert result.state_ref == "method://everos/13:everos-memory/5:agent/12:agent:solver/v1"
    scenes = method.store.list_memscenes("agent", "agent:solver")
    assert len(scenes) == 1
    assert scenes[0]["title"] == "Biology component extraction"

    bundle = backend.search(
        RetrievalRequest(
            task_id="task-2",
            role="solver",
            query="How should Sigma70 promoter extraction handle primers?",
        )
    )

    assert bundle.metadata["memory_method"] == "everos"
    assert bundle.metadata["recollection_mode"] == "scene"
    assert bundle.items
    assert bundle.items[0].memory_id.startswith("everos:memscene:")
    assert "MemScene: Biology component extraction" in bundle.items[0].content
    assert "Sigma70 promoter rows require direct regulatory DNA evidence" in bundle.items[0].content


def test_everos_memory_method_updates_existing_scene_when_similarity_threshold_matches(tmp_path: Path):
    llm = ScriptedLLMRuntime(
        [
            _memcell_payload(),
            _scene_payload("Initial promoter extraction scene."),
            _memcell_payload(
                summary="LuxR promoter extraction reused the audit pattern",
                episode="The reviewer applied the primer rejection audit to LuxR promoter evidence.",
                fact="LuxR promoter records must separate regulatory sequence evidence from primer labels.",
            ),
            _scene_payload("Merged promoter extraction scene across Sigma70 and LuxR."),
        ]
    )
    backend, method = _backend(tmp_path, llm)

    backend.add("task-1", "solver", [Message(role="assistant", content="First extraction.")])
    second = backend.add("task-2", "solver", [Message(role="assistant", content="Second extraction.")])

    scenes = method.store.list_memscenes("agent", "agent:solver")
    assert second.state_ref == "method://everos/13:everos-memory/5:agent/12:agent:solver/v2"
    assert len(scenes) == 1
    assert scenes[0]["member_count"] == 2
    assert scenes[0]["summary"] == "Merged promoter extraction scene across Sigma70 and LuxR."


def test_everos_agentic_recollection_uses_llm_to_select_reconstructed_context(tmp_path: Path):
    llm = ScriptedLLMRuntime(
        [
            _memcell_payload(),
            _scene_payload(),
            {
                "selected_scenes": [
                    {
                        "scene_id": "",
                        "selected_memory_ids": [],
                        "reconstructed_context": "Use direct promoter evidence and reject primer-only rows.",
                        "rationale": "The scene contains the relevant extraction audit pattern.",
                    }
                ]
            },
        ]
    )
    backend, method = _backend(tmp_path, llm, recollection_mode="agentic")
    backend.add("task-1", "solver", [Message(role="assistant", content="Store extraction memory.")])
    scene_id = method.store.list_memscenes("agent", "agent:solver")[0]["scene_id"]
    llm.contents[-1]["selected_scenes"][0]["scene_id"] = scene_id

    bundle = backend.search(
        RetrievalRequest(
            task_id="task-2",
            role="solver",
            query="What evidence rule should I remember for promoter extraction?",
        )
    )

    assert len(llm.calls) == 3
    assert bundle.metadata["recollection_mode"] == "agentic"
    assert bundle.metadata["recollection_status"] == "selected"
    assert bundle.items[0].content == "Use direct promoter evidence and reject primer-only rows."
    assert bundle.items[0].metadata["recollection_rationale"].startswith("The scene contains")
