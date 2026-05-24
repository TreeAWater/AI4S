from __future__ import annotations

from pathlib import Path

from evolab.cli import _compile_experiment_config, run_clean_demo
from evolab.config.task_config import TaskConfig
from evolab.registries.trajectory import FileTrajectoryRegistry


def test_static_yaml_without_dynamic_subagents_compiles_unchanged(tmp_path: Path):
    config = _static_short_config(tmp_path)

    compiled = _compile_experiment_config(config, config_path=tmp_path / "static.yaml")

    task_config = TaskConfig.model_validate(compiled["task_config"])
    assert task_config.dynamic_subagents is None
    assert list(task_config.roles) == ["SurveyAgent"]


def test_dynamic_disabled_compiles_but_static_execution_is_unchanged(tmp_path: Path):
    config = _static_short_config(tmp_path)
    config["dynamic_subagents"] = {
        "enabled": False,
        "planner_backend": {"backend_id": "fake-llm"},
        "default_worker_backend": {"backend_id": "fake-llm"},
    }
    compiled = _compile_experiment_config(config, config_path=tmp_path / "disabled.yaml")
    task_config = TaskConfig.model_validate(compiled["task_config"])

    assert task_config.dynamic_subagents is not None
    assert task_config.dynamic_subagents.enabled is False


def test_static_clean_run_still_uses_existing_meta_subagent_flow(tmp_path: Path):
    config_path = tmp_path / "static-clean.yaml"
    config_path.write_text(
        """
lab_root: unused
files:
  inputs/source.txt: "alpha"
task: Read inputs/source.txt, summarize, then end.
meta_agent:
  system_prompt: Return route JSON only.
subagents:
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

    result = run_clean_demo(config_path, lab_root)

    assert result["final_answer"] == "static done"
    assert "execution_mode" not in result
    trajectory = FileTrajectoryRegistry(lab_root / "registries" / "trajectory")
    assert [run.role for run in trajectory.list_subagent_runs()] == ["SurveyAgent"]


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
        "subagents": {"SurveyAgent": {"system_prompt": "Survey."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }
