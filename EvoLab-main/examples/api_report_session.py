from __future__ import annotations

from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def main() -> None:
    lab_dir = Path("~/evolab-labs/report-session").expanduser()
    session = EvoLabSession(
        SessionConfig(
            env_file=Path(".env"),
            lab_dir=lab_dir,
            task=TaskSpec(
                goal="Write a concise research note from the files in papers/.",
                resources="The Lab folder contains papers/ with source documents and notes/ with user annotations.",
                expected_outputs="report.md at the Lab root.",
                success_criteria="The report is grounded in the supplied files and clearly separates evidence from uncertainty.",
                optional_context="Prefer a short executive summary followed by evidence bullets.",
            ),
            llm={
                "default": {
                    "type": "api",
                    "api": "openai-responses",
                    "model": "gpt-4.1-mini",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            memory={"task": {"type": "null"}},
            skills={"default": {"type": "fake", "skills": []}},
            tools={"builtin": True},
        )
    )
    session.run()


if __name__ == "__main__":
    main()
