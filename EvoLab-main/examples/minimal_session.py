from __future__ import annotations

import json
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def main() -> None:
    lab_dir = Path("/tmp/evolab-minimal-session")
    workflow = {
        "workflow_id": "wf-minimal-report",
        "task_summary": "Write a minimal report.",
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
        "planner_rationale_summary": "A single writer can complete the task.",
    }
    session = EvoLabSession(
        SessionConfig(
            lab_dir=lab_dir,
            task=TaskSpec(
                goal="Write a short report.",
                resources="No external files.",
                expected_outputs="report.md",
                success_criteria="report.md exists and contains the final answer.",
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
                                    '"metadata":{"no_generated_tool_reason":"Built-in tools are sufficient."}}'
                                ),
                            }
                        },
                        {
                            "action": {
                                "action": "final_answer",
                                "content": (
                                    '{"route":"END","instruction":"Keep the current role pool.",'
                                    '"metadata":{"no_role_pool_update_reason":"GeneralistAgent is sufficient."}}'
                                ),
                            }
                        },
                        {"action": {"action": "final_answer", "content": json.dumps(workflow)}},
                        {
                            "action": {
                                "action": "tool_call",
                                "tool_calls": [
                                    {
                                        "call_id": "write-report",
                                        "name": "write_report",
                                        "arguments": {
                                            "path": "report.md",
                                            "content": "EvoLab minimal session completed.",
                                        },
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
            runtime={"task_id": "minimal-session"},
        )
    )
    session.run()
    print(f"Wrote {lab_dir / 'report.md'}")


if __name__ == "__main__":
    main()
