from __future__ import annotations

from pathlib import Path

import pytest

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.memory import NullMemoryBackend
from evolab.backends.skills import FakeSkillBackend
from evolab.cli import _compile_experiment_config, run_clean_demo
from evolab.config.task_config import BackendBinding, MetaAgentSpec, RoleSpec, TaskConfig
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.registries.trajectory import FileTrajectoryRegistry


def test_seed_agents_without_dynamic_subagents_compile_to_dynamic_default(tmp_path: Path):
    config = _static_short_config(tmp_path)

    compiled = _compile_experiment_config(config, config_path=tmp_path / "static.yaml")

    task_config = TaskConfig.model_validate(compiled["task_config"])
    assert task_config.dynamic_subagents is not None
    assert task_config.dynamic_subagents.enabled is True
    assert list(task_config.roles) == ["SurveyAgent"]


def test_dynamic_disabled_short_config_is_rejected(tmp_path: Path):
    config = _static_short_config(tmp_path)
    config["dynamic_subagents"] = {
        "enabled": False,
        "planner_backend": {"backend_id": "fake-llm"},
        "default_worker_backend": {"backend_id": "fake-llm"},
    }

    with pytest.raises(ValueError, match="dynamic_subagents must be enabled"):
        _compile_experiment_config(config, config_path=tmp_path / "disabled.yaml")


def test_task_runtime_rejects_default_static_role_order():
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id="task-1",
            goal="Run static roles.",
            meta_agent=MetaAgentSpec(
                system_prompt="This old meta dispatch path must not run.",
                llm_backend=BackendBinding(backend_id="fake-llm"),
            ),
            roles={
                "Solver": RoleSpec(
                    name="Solver",
                    system_prompt="Solve.",
                    llm_backend=BackendBinding(backend_id="fake-llm"),
                )
            },
        ),
        llm_runtimes={"fake-llm": FakeLLMRuntime(default_content="done")},
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={"skill": FakeSkillBackend(skills=[])},
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="dynamic_subagents.enabled=true"):
        runtime.run(
            TaskRequest(
                task_id="task-1",
                origin=TaskOrigin.HUMAN,
                purpose=TaskPurpose.SCIENCE,
                goal="Run static roles.",
            )
        )


def test_task_runtime_rejects_hybrid_dynamic_mode():
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id="task-1",
            goal="Run hybrid mode.",
            roles={
                "Solver": RoleSpec(
                    name="Solver",
                    system_prompt="Solve.",
                    llm_backend=BackendBinding(backend_id="fake-llm"),
                )
            },
            dynamic_subagents=DynamicSubagentsConfig(
                enabled=True,
                mode="hybrid",
                planner_backend={"backend_id": "fake-llm"},
                default_worker_backend={"backend_id": "fake-llm"},
            ),
        ),
        llm_runtimes={"fake-llm": FakeLLMRuntime(default_content="done")},
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={"skill": FakeSkillBackend(skills=[])},
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="mode=dynamic"):
        runtime.run(
            TaskRequest(
                task_id="task-1",
                origin=TaskOrigin.HUMAN,
                purpose=TaskPurpose.SCIENCE,
                goal="Run hybrid mode.",
            )
        )


def test_task_runtime_rejects_dynamic_mode_without_agents_ref():
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id="task-1",
            goal="Run dynamic mode without role pool.",
            roles={
                "Solver": RoleSpec(
                    name="Solver",
                    system_prompt="Solve.",
                    llm_backend=BackendBinding(backend_id="fake-llm"),
                )
            },
            dynamic_subagents=DynamicSubagentsConfig(
                enabled=True,
                mode="dynamic",
                planner_backend={"backend_id": "fake-llm"},
                default_worker_backend={"backend_id": "fake-llm"},
            ),
        ),
        llm_runtimes={"fake-llm": FakeLLMRuntime(default_content="done")},
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={"skill": FakeSkillBackend(skills=[])},
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="agents_ref"):
        runtime.run(
            TaskRequest(
                task_id="task-1",
                origin=TaskOrigin.HUMAN,
                purpose=TaskPurpose.SCIENCE,
                goal="Run dynamic mode without role pool.",
            )
        )


def test_static_clean_run_disabled_dynamic_config_is_rejected(tmp_path: Path):
    config_path = tmp_path / "static-clean.yaml"
    config_path.write_text(
        """
lab_root: unused
files:
  inputs/source.txt: "alpha"
task: Read inputs/source.txt, summarize, then end.
meta_agent:
  system_prompt: Return route JSON only.
dynamic_subagents:
  enabled: false
seed_agents:
  SurveyAgent:
    system_prompt: Survey.
backends:
  llm:
    fake-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"route":"SurveyAgent","instruction":"Read inputs/source.txt."}'
        - action:
            action: tool_call
            tool_call:
              call_id: read-source
              name: read_text
              arguments:
                path: inputs/source.txt
        - action:
            action: final_answer
            content: "alpha seen"
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"done","metadata":{"final_answer":"static done"}}'
  memory:
    mem0-agent-memory:
      type: "null"
    mem0-task-memory:
      type: "null"
  skill:
    fake-skill:
      type: fake
      skills:
        - skill_id: read-source
          name: Read source
          content: Read source.
          required_tools: ["read_text"]
""",
        encoding="utf-8",
    )
    lab_root = tmp_path / "lab"

    with pytest.raises(ValueError, match="dynamic_subagents must be enabled"):
        run_clean_demo(config_path, lab_root)


def test_dynamic_config_validates_backend_ids_in_short_config(tmp_path: Path):
    config = _static_short_config(tmp_path)
    config["dynamic_subagents"] = {
        "enabled": True,
        "planner_backend": {"backend_id": "missing"},
        "default_worker_backend": {"backend_id": "fake-llm"},
    }

    try:
        _compile_experiment_config(config, config_path=tmp_path / "bad.yaml")
    except ValueError as exc:
        assert "unknown LLM backend" in str(exc)
    else:
        raise AssertionError("expected dynamic backend validation to fail")


def _static_short_config(tmp_path: Path):
    return {
        "lab_root": str(tmp_path / "lab"),
        "task": "Read a file.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "seed_agents": {"SurveyAgent": {"system_prompt": "Survey."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }
