import json
from pathlib import Path
from types import SimpleNamespace

from evolab.backends.trainers import SFTTrainer, SFTTrainerConfig
from evolab.backends.trainers import sft as sft_module
from evolab.backends.trainers.sft import _tokenize_sft_sample
from evolab.contracts.common import Message
from evolab.contracts.evolution import EvolutionBudget, LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.local_trainable import LocalTrainableStateManifest
from evolab.contracts.records import LLMCallRecord, SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.evolution_executor import EvolutionExecutor
from evolab.runtime.sft_exporter import SFTExportConfig, export_sft_dataset


def _subagent_run(run_ref: str = "subagent-1") -> SubagentRunRecord:
    return SubagentRunRecord(
        run_ref=run_ref,
        task_id="task-1",
        task_origin=TaskOrigin.HUMAN,
        task_purpose=TaskPurpose.SCIENCE,
        stage_index=0,
        role="solver",
        instruction="Solve it.",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="prior work"),
        memory_bundle=MemoryBundle(backend_id="memory-local"),
        skill_bundle=SkillBundle(backend_id="skill-local"),
        prompt_messages=[Message(role="user", content="Solve it.")],
        llm_call_refs=["call-tool", "call-final"],
        llm_backend_id="teacher-api",
        output_messages=[Message(role="assistant", content="Final answer.")],
    )


def _seed_tool_use_calls(registry: FileTrajectoryRegistry) -> None:
    registry.save_subagent_run(_subagent_run())
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-tool",
            run_ref="subagent-1",
            backend_id="teacher-api",
            model="gpt-teacher",
            input_messages=[Message(role="user", content="Use lookup.")],
            output_messages=[
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "action": "tool_call",
                        "tool_call": {
                            "schema_version": "v1",
                            "call_id": "tool-1",
                            "name": "lookup",
                            "arguments": {"query": "catalyst"},
                        },
                    },
                )
            ],
            metadata={"runtime_stage": "subagent_flat", "action": "tool_call", "role": "solver"},
        )
    )
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-final",
            run_ref="subagent-1",
            backend_id="teacher-api",
            model="gpt-teacher",
            input_messages=[
                Message(role="user", content="Use lookup."),
                Message(role="tool", content="lookup result", tool_call_id="tool-1", name="lookup"),
            ],
            output_messages=[Message(role="assistant", content="Final answer.", metadata={"action": "final_answer"})],
            metadata={"runtime_stage": "subagent_flat", "action": "final_answer", "role": "solver"},
        )
    )


def _seed_second_final_call(registry: FileTrajectoryRegistry) -> None:
    registry.save_subagent_run(_subagent_run("subagent-2"))
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-final-2",
            run_ref="subagent-2",
            backend_id="teacher-api",
            model="gpt-teacher",
            input_messages=[Message(role="user", content="Solve another.")],
            output_messages=[Message(role="assistant", content="Second answer.", metadata={"action": "final_answer"})],
            metadata={"runtime_stage": "subagent_flat", "action": "final_answer", "role": "solver"},
        )
    )


def test_sft_exporter_writes_manifest_and_reconstructs_tool_call_context(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)

    result = export_sft_dataset(
        trajectory_registry=registry,
        output_dir=tmp_path / "sft",
        config=SFTExportConfig(teacher_backend_ids=["teacher-api"], source_run_refs=["subagent-1"]),
    )

    assert result.manifest.sample_count == 1
    assert result.manifest.train_count == 1
    rows = [json.loads(line) for line in result.train_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_llm_call_ref"] == "call-final"
    messages = rows[0]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["metadata"]["tool_call"]["name"] == "lookup"
    assert messages[2]["tool_call_id"] == "tool-1"
    assert messages[-1]["content"] == "Final answer."


def test_sft_exporter_can_include_tool_call_samples(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)

    result = export_sft_dataset(
        trajectory_registry=registry,
        output_dir=tmp_path / "sft",
        config=SFTExportConfig(
            actions=["tool_call", "final_answer"],
            include_tool_call_samples=True,
            source_run_refs=["subagent-1"],
        ),
    )

    assert result.manifest.sample_count == 2
    assert [sample.action for sample in result.train_samples] == ["tool_call", "final_answer"]


def test_sft_trainer_exports_dataset_and_dry_run_adapter(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)
    trainer = SFTTrainer(
        trajectory_registry=registry,
        config=SFTTrainerConfig(promote_dry_run=True),
    )
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="teacher-api",
        previous_state_ref="state-before",
        artifact_root_uri=str(tmp_path / "artifacts"),
        trigger_trajectory_ref="subagent-1",
    )

    result = trainer.train(request)

    assert result.status == "promoted_candidate"
    assert result.new_state_ref is not None
    assert result.new_state_ref.startswith("local-trainable://teacher-api/state/")
    assert result.standard_metrics.n_train_samples == 1
    artifact_roles = {artifact.metadata.get("role") for artifact in result.artifact_refs}
    assert {
        "sft_train",
        "sft_val",
        "sft_manifest",
        "sft_dry_run_adapter",
        "local_trainable_state",
    }.issubset(artifact_roles)
    manifest_uri = result.metadata["manifest_uri"]
    assert json.loads(Path(manifest_uri).read_text(encoding="utf-8"))["train_count"] == 1
    state_artifact = next(
        artifact for artifact in result.artifact_refs if artifact.metadata.get("role") == "local_trainable_state"
    )
    state_manifest = LocalTrainableStateManifest.model_validate_json(
        Path(state_artifact.uri).read_text(encoding="utf-8")
    )
    assert state_manifest.state_ref == result.new_state_ref
    assert state_manifest.parent_state_ref == "state-before"
    assert state_manifest.created_by_trainer == "sft"
    assert state_manifest.dataset_manifest_uri == manifest_uri


def test_sft_trainer_result_passes_promotion_guard_in_dry_run(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)
    trainer = SFTTrainer(
        trajectory_registry=registry,
        config=SFTTrainerConfig(promote_dry_run=True),
    )
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="teacher-api",
        previous_state_ref="state-before",
        artifact_root_uri=str(tmp_path / "artifacts"),
        trigger_trajectory_ref="subagent-1",
    )
    executor = EvolutionExecutor(FileBackendStateRegistry(tmp_path / "states"), worker_id="worker-1")

    outcome = executor.run(request=request, trainer=trainer, run_ref="evo-1")

    assert outcome.promoted is True
    assert outcome.promotion_errors == []


def test_sft_trainer_honors_request_max_train_samples_budget(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)
    _seed_second_final_call(registry)
    trainer = SFTTrainer(
        trajectory_registry=registry,
        config=SFTTrainerConfig(
            promote_dry_run=True,
            export=SFTExportConfig(source_run_refs=["subagent-1", "subagent-2"]),
        ),
    )
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="teacher-api",
        artifact_root_uri=str(tmp_path / "artifacts"),
        budget=EvolutionBudget(max_train_samples=1),
    )

    result = trainer.train(request)

    assert result.standard_metrics.n_train_samples == 1
    manifest_uri = result.metadata["manifest_uri"]
    assert json.loads(Path(manifest_uri).read_text(encoding="utf-8"))["train_count"] == 1


def test_sft_trainer_allows_zero_train_sample_budget(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_use_calls(registry)
    trainer = SFTTrainer(trajectory_registry=registry, config=SFTTrainerConfig())
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="teacher-api",
        artifact_root_uri=str(tmp_path / "artifacts"),
        trigger_trajectory_ref="subagent-1",
        budget=EvolutionBudget(max_train_samples=0),
    )

    result = trainer.train(request)

    assert result.status == "skipped"
    assert result.metadata["train_count"] == 0


def test_sft_trainer_uses_processing_class_for_new_transformers_trainer_signature():
    class NewTrainer:
        def __init__(self, *, processing_class=None):
            self.processing_class = processing_class

    tokenizer = object()
    helper = getattr(sft_module, "_trainer_tokenizer_kwargs", None)

    assert helper is not None
    assert helper(NewTrainer, tokenizer) == {"processing_class": tokenizer}


def test_sft_tokenization_masks_rendered_prompt_prefix_once():
    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in text])

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is False
            return "".join(f"<{message['role']}>{message['content']}" for message in messages)

    sample = {
        "messages": [
            {"role": "user", "content": "question", "metadata": {}},
            {"role": "assistant", "content": "answer", "metadata": {}},
        ]
    }

    tokenized = _tokenize_sft_sample(sample, FakeTokenizer(), max_length=128)

    first_label_index = next(index for index, label in enumerate(tokenized["labels"]) if label != -100)
    rendered_prompt = "<user>question"
    rendered_full = "<user>question<assistant>answer"
    assert tokenized["input_ids"][:first_label_index] == [ord(char) for char in rendered_prompt]
    assert tokenized["input_ids"][first_label_index:-1] == [
        ord(char) for char in rendered_full[len(rendered_prompt) :]
    ]
