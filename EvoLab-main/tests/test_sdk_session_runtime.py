import json
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec
from evolab.backends.llm import ApiLLMBackend
from evolab.session_runtime import _build_llm_backends


def _config(lab: Path) -> SessionConfig:
    return SessionConfig(
        lab_dir=lab,
        task=TaskSpec(
            goal="Write a report.",
            resources="Use input.txt.",
            expected_outputs="report.md",
            success_criteria="report.md exists.",
        ),
        llm={"default": {"type": "fake", "responses": []}},
        memory={"task": {"type": "null"}},
        skills={"default": {"type": "fake", "skills": []}},
    )


def test_initialize_creates_agents_tools_skills_memory_dirs(tmp_path: Path):
    lab = tmp_path / "lab"

    EvoLabSession(_config(lab)).initialize()

    state = lab / ".evolab"
    assert (state / "AGENTS.md").is_file()
    assert (state / "tools").is_dir()
    assert (state / "skills").is_dir()
    assert (state / "memory").is_dir()
    assert (state / "configs").is_dir()
    assert not (lab / "AGENTS.md").exists()
    assert not (lab / "memory").exists()
    assert not (lab / "tools").exists()
    assert not (lab / "skills").exists()


def test_initialize_reuses_existing_lab_without_deleting_user_files(tmp_path: Path):
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "input.txt").write_text("keep me", encoding="utf-8")

    EvoLabSession(_config(lab)).initialize()
    EvoLabSession(_config(lab)).initialize()

    assert (lab / "input.txt").read_text(encoding="utf-8") == "keep me"


def test_sdk_run_writes_report_and_returns_none(tmp_path: Path):
    lab = tmp_path / "lab"
    workflow = {
        "workflow_id": "wf-report",
        "task_summary": "Write report.",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write report.md",
                "system_prompt": "Write the final report.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "One writer is enough.",
    }
    session = EvoLabSession(
        SessionConfig(
            lab_dir=lab,
            task=TaskSpec(
                goal="Write a report.",
                resources="No external files.",
                expected_outputs="report.md",
                success_criteria="report.md exists.",
            ),
            llm={
                "default": {
                    "type": "fake",
                    "responses": [
                        {
                            "action": {
                                "action": "final_answer",
                                "content": (
                                    '{"route":"END","instruction":"No generated tool needed.",'
                                    '"metadata":{"no_generated_tool_reason":"Built-ins are enough."}}'
                                ),
                            }
                        },
                        {
                            "action": {
                                "action": "final_answer",
                                "content": (
                                    '{"route":"END","instruction":"Keep role pool.",'
                                    '"metadata":{"no_role_pool_update_reason":"Seed is enough."}}'
                                ),
                            }
                        },
                        {"action": {"action": "final_answer", "content": json.dumps(workflow)}},
                        {
                            "action": {
                                "action": "tool_call",
                                "tool_calls": [
                                    {
                                        "call_id": "write-1",
                                        "name": "write_report",
                                        "arguments": {"path": "report.md", "content": "Done."},
                                    }
                                ],
                            }
                        },
                        {"action": {"action": "final_answer", "content": "Report written."}},
                    ],
                }
            },
            memory={"task": {"type": "null"}},
            skills={"default": {"type": "fake", "skills": []}},
            tools={"builtin": True},
        )
    )

    assert session.run() is None
    assert (lab / "report.md").read_text(encoding="utf-8") == "Done."
    assert not (lab / "queues").exists()
    assert (lab / ".evolab" / "queues" / "tasks").is_dir()


def test_sdk_llm_backend_uses_configured_env_file(tmp_path: Path):
    env_file = tmp_path / "sdk.env"
    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    config = _config(tmp_path / "lab").model_copy(
        update={
            "env_file": env_file,
            "llm": {
                "api": {
                    "type": "api",
                    "api": "openai-responses",
                    "model": "gpt-4.1-mini",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
        }
    )

    backend = _build_llm_backends(config)["api"]

    assert isinstance(backend, ApiLLMBackend)
    assert backend.config.model == "gpt-4.1-mini"
