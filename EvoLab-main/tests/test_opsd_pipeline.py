import json
from pathlib import Path
from types import SimpleNamespace

from evolab.backends.trainers import OPSDTrainer, OPSDTrainerConfig
from evolab.backends.trainers import opsd as opsd_module
from evolab.cli import _build_evolution_backends, run_clean_demo, run_export_opsd, run_train_opsd
from evolab.contracts.common import Message
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.evolution import EvolutionBudget, LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.local_trainable import LocalTrainableStateManifest
from evolab.contracts.records import LLMCallRecord, MetaAgentRunRecord, SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.lab.layout import LabLayout
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.opsd_exporter import OPSDExportConfig, export_opsd_dataset


def _layout(lab_root: Path) -> LabLayout:
    return LabLayout(lab_root)


def _trajectory_registry(lab_root: Path) -> FileTrajectoryRegistry:
    return FileTrajectoryRegistry(_layout(lab_root).registries_dir / "trajectory")


def _backend_state_registry(lab_root: Path) -> FileBackendStateRegistry:
    return FileBackendStateRegistry(_layout(lab_root).registries_dir / "backend_state")


def _seed_tool_choice_call(registry: FileTrajectoryRegistry) -> None:
    registry.save_subagent_run(
        SubagentRunRecord(
            run_ref="subagent-1",
            task_id="task-1",
            task_origin=TaskOrigin.HUMAN,
            task_purpose=TaskPurpose.SCIENCE,
            stage_index=0,
            role="solver",
            instruction="Use the available tools.",
            retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="tool choice"),
            memory_bundle=MemoryBundle(backend_id="memory-local"),
            skill_bundle=SkillBundle(backend_id="skill-local"),
            prompt_messages=[Message(role="user", content="Use the available tools.")],
            llm_call_refs=["call-tool"],
            llm_backend_id="teacher-api",
        )
    )
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-tool",
            run_ref="subagent-1",
            backend_id="teacher-api",
            model="gpt-teacher",
            input_messages=[Message(role="user", content="Read the article.")],
            output_messages=[
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "action": "tool_call",
                        "tool_call": {
                            "schema_version": "v1",
                            "call_id": "tool-1",
                            "name": "read_text",
                            "arguments": {"path": "inputs/article.md"},
                        },
                    },
                )
            ],
            metadata={
                "runtime_stage": "subagent_flat",
                "action": "tool_call",
                "role": "solver",
                "tool_specs": [
                    {"name": "read_text", "description": "Read a file."},
                    {"name": "write_report", "description": "Write a report."},
                ],
            },
        )
    )


def _seed_subagent_choice_call(registry: FileTrajectoryRegistry) -> None:
    registry.save_meta_agent_run(
        MetaAgentRunRecord(
            run_ref="meta-1",
            task_id="task-1",
            decision=DispatchDecision(
                action=DispatchAction.RUN_SUBAGENT,
                target_role="solver",
                instruction="Solve first.",
                retrieval_query="solve",
            ),
            metadata={"llm_call_refs": ["call-meta"]},
        )
    )
    registry.save_llm_call(
        LLMCallRecord(
            call_ref="call-meta",
            run_ref="meta-1",
            backend_id="teacher-api",
            model="gpt-teacher",
            input_messages=[
                Message(role="system", content="Dispatch roles."),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "available_roles": [
                                {"name": "solver", "system_prompt": "Solve."},
                                {"name": "reviewer", "system_prompt": "Review."},
                            ],
                            "completed_runs": [],
                            "goal": "Do the task.",
                        },
                        sort_keys=True,
                    ),
                ),
            ],
            output_messages=[
                Message(
                    role="assistant",
                    content=(
                        '{"action":"run_subagent","target_role":"solver",'
                        '"instruction":"Solve first.","retrieval_query":"solve"}'
                    ),
                    metadata={"action": "final_answer"},
                )
            ],
            metadata={
                "runtime_stage": "meta_agent_dispatch",
                "action": "final_answer",
                "role": "demo-meta",
            },
        )
    )


def test_opsd_exporter_writes_tool_choice_samples(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_choice_call(registry)

    result = export_opsd_dataset(
        trajectory_registry=registry,
        output_dir=tmp_path / "opsd",
        config=OPSDExportConfig(teacher_backend_ids=["teacher-api"]),
    )

    assert result.manifest.sample_count == 1
    row = json.loads(result.train_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["decision_type"] == "tool_choice"
    assert row["chosen_action"]["tool_name"] == "read_text"
    assert [option["name"] for option in row["candidate_actions"]] == ["read_text", "write_report"]
    assert row["source_llm_call_ref"] == "call-tool"


def test_opsd_exporter_writes_subagent_choice_samples(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_subagent_choice_call(registry)

    result = export_opsd_dataset(
        trajectory_registry=registry,
        output_dir=tmp_path / "opsd",
        config=OPSDExportConfig(include_subagent_choices=True, include_tool_choices=False),
    )

    assert result.manifest.sample_count == 1
    row = json.loads(result.train_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["decision_type"] == "subagent_choice"
    assert row["chosen_action"]["target_role"] == "solver"
    assert [option["name"] for option in row["candidate_actions"]] == ["solver", "reviewer"]
    assert row["metadata"]["runtime_stage"] == "meta_agent_dispatch"


def test_opsd_trainer_exports_dataset_and_dry_run_adapter(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_choice_call(registry)
    trainer = OPSDTrainer(
        trajectory_registry=registry,
        config=OPSDTrainerConfig(promote_dry_run=True),
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
        "opsd_train",
        "opsd_val",
        "opsd_manifest",
        "opsd_dry_run_adapter",
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
    assert state_manifest.created_by_trainer == "opsd"
    assert state_manifest.dataset_manifest_uri == manifest_uri


def test_opsd_trainer_honors_request_max_train_samples_budget(tmp_path: Path):
    registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    _seed_tool_choice_call(registry)
    _seed_subagent_choice_call(registry)
    trainer = OPSDTrainer(
        trajectory_registry=registry,
        config=OPSDTrainerConfig(
            promote_dry_run=True,
            export=OPSDExportConfig(include_subagent_choices=True),
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


def test_opsd_trainer_config_accepts_transformers_backend():
    config = OPSDTrainerConfig(
        training_backend="transformers",
        base_model_ref="sshleifer/tiny-gpt2",
    )

    assert config.training_backend == "transformers"
    assert config.base_model_ref == "sshleifer/tiny-gpt2"


def test_opsd_tokenization_masks_prompt_and_trains_chosen_action_only():
    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in text])

    sample = {
        "decision_type": "tool_choice",
        "messages": [{"role": "user", "content": "Pick a tool.", "metadata": {}}],
        "candidate_actions": [
            {"name": "read_text", "description": "Read a file."},
            {"name": "write_report", "description": "Write a report."},
        ],
        "chosen_action": {"action": "tool_call", "tool_name": "read_text", "arguments": {"path": "a.md"}},
    }
    helper = getattr(opsd_module, "_tokenize_opsd_sample", None)

    assert helper is not None
    tokenized = helper(sample, FakeTokenizer(), max_length=512)

    first_label_index = next(index for index, label in enumerate(tokenized["labels"]) if label != -100)
    completion_text = "".join(chr(token) for token in tokenized["input_ids"][first_label_index:-1])
    assert "CHOSEN_ACTION:" in completion_text
    assert '"tool_name": "read_text"' in completion_text


def test_clean_run_config_builds_opsd_backend(tmp_path: Path):
    backends = _build_evolution_backends(
        {
            "evolution": {
                "backends": {
                    "teacher-api": {
                        "type": "opsd",
                        "promote_dry_run": True,
                        "export": {"include_subagent_choices": True},
                    }
                }
            }
        },
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
    )

    trainer = backends["teacher-api"]
    assert isinstance(trainer, OPSDTrainer)
    assert trainer.config.promote_dry_run is True
    assert trainer.config.export.include_subagent_choices is True


def test_run_export_opsd_writes_dataset_from_lab_root(tmp_path: Path):
    lab_root = tmp_path / "lab"
    registry = _trajectory_registry(lab_root)
    _seed_tool_choice_call(registry)

    result = run_export_opsd(
        lab_root=lab_root,
        teacher_backend_ids=["teacher-api"],
    )

    manifest = result["manifest"]
    assert manifest.sample_count == 1
    assert Path(result["train_path"]).is_file()
    assert Path(result["manifest_path"]).parent == lab_root / "artifacts" / "opsd"


def test_run_train_opsd_uses_clean_run_tool_choice_trajectory(tmp_path: Path):
    lab_root = tmp_path / "demo-lab"
    run_clean_demo(Path("dev/configs/demo_v0.yaml"), lab_root)

    result = run_train_opsd(
        lab_root=lab_root,
        backend_id="fake-llm",
        artifact_root=tmp_path / "opsd-train",
        teacher_backend_ids=["fake-llm"],
        promote_dry_run=True,
    )

    assert result["result"].status == "promoted_candidate"
    assert result["promoted"] is True
    assert result["promotion_errors"] == []
    assert result["result"].standard_metrics.n_train_samples == 1
    train_rows = [
        json.loads(line)
        for line in (Path(result["artifact_root"]) / "dataset" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["decision_type"] for row in train_rows] == ["tool_choice"]
    assert train_rows[0]["chosen_action"]["tool_name"] == "read_file"
    state_registry = _backend_state_registry(lab_root)
    assert state_registry.resolve_active_state("fake-llm") == result["result"].new_state_ref


def test_run_train_opsd_records_evolution_run(tmp_path: Path):
    lab_root = tmp_path / "lab"
    registry = _trajectory_registry(lab_root)
    _seed_tool_choice_call(registry)

    result = run_train_opsd(
        lab_root=lab_root,
        backend_id="teacher-api",
        source_run_refs=["subagent-1"],
        promote_dry_run=True,
    )

    assert result["result"].status == "promoted_candidate"
    saved_runs = registry.list_evolution_runs()
    assert len(saved_runs) == 1
    assert saved_runs[0].metadata["worker_id"] == "train-opsd"
    assert saved_runs[0].training_trajectory_refs == ["subagent-1"]
