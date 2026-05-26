import json

from evolab.config.agents import parse_agents_markdown, render_agents_markdown
from evolab.config.task_config import BackendBinding, RoleSpec


def test_agents_markdown_round_trip_with_aliases():
    text = """
# Agents

```json
{
  "schema_version": "v1",
  "agents": [
    {
      "name": "ExecAgent",
      "prompt": "Execute the task.",
      "llm_backend": "local-qwen",
      "agent_memory_backend": "agent-memory",
      "toolset": ["read_text", "write_jsonl"],
      "skillset": ["skill.extract.v1"],
      "memory_policy": {"scope": "agent"},
      "metadata": {"owner": "meta"}
    }
  ]
}
```
"""

    roles = parse_agents_markdown(text)

    assert list(roles) == ["ExecAgent"]
    role = roles["ExecAgent"]
    assert role.system_prompt == "Execute the task."
    assert role.llm_backend.backend_id == "local-qwen"
    assert role.agent_memory_backend is not None
    assert role.agent_memory_backend.backend_id == "agent-memory"
    assert role.allowed_tools == ["read_text", "write_jsonl"]
    assert role.required_skills == ["skill.extract.v1"]
    assert role.memory_policy == {"scope": "agent"}

    rendered = render_agents_markdown(roles)
    payload = json.loads(rendered.split("```json", 1)[1].split("```", 1)[0])
    assert payload["agents"][0]["name"] == "ExecAgent"


def test_agents_markdown_can_render_roles_mapping():
    rendered = render_agents_markdown(
        {
            "Reviewer": RoleSpec(
                name="Reviewer",
                system_prompt="Review results.",
                llm_backend=BackendBinding(backend_id="review-llm"),
            )
        }
    )

    roles = parse_agents_markdown(rendered)

    assert roles["Reviewer"].system_prompt == "Review results."
    assert roles["Reviewer"].llm_backend.backend_id == "review-llm"


def test_agents_markdown_accepts_bare_role_mapping_with_schema_version():
    roles = parse_agents_markdown(
        json.dumps(
            {
                "schema_version": "v1",
                "ExecAgent": {
                    "system_prompt": "Execute.",
                    "llm_backend": {"backend_id": "exec-llm"},
                },
            }
        )
    )

    assert roles["ExecAgent"].system_prompt == "Execute."
