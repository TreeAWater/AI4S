from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def test_task_spec_renders_canonical_prompt():
    task = TaskSpec(
        goal="Extract facts.",
        resources="Use files in data/.",
        expected_outputs="Write report.md.",
        success_criteria="Every claim cites a source.",
        optional_context="Prefer concise output.",
    )

    assert task.to_prompt() == (
        "1. goal: Extract facts.\n"
        "2. resources: Use files in data/.\n"
        "3. expected_outputs: Write report.md.\n"
        "4. success_criteria: Every claim cites a source.\n"
        "5. optional_context: Prefer concise output."
    )


def test_task_spec_omits_optional_context_when_not_provided():
    task = TaskSpec(
        goal="Extract facts.",
        resources="Use files in data/.",
        expected_outputs="Write report.md.",
        success_criteria="Every claim cites a source.",
    )

    assert task.to_prompt() == (
        "1. goal: Extract facts.\n"
        "2. resources: Use files in data/.\n"
        "3. expected_outputs: Write report.md.\n"
        "4. success_criteria: Every claim cites a source."
    )


def test_session_run_returns_none(tmp_path: Path):
    session = EvoLabSession(
        SessionConfig(
            lab_dir=tmp_path / "lab",
            task=TaskSpec(
                goal="Write a report.",
                resources="No external files.",
                expected_outputs="report.md",
                success_criteria="report.md exists.",
            ),
            llm={"default": {"type": "fake", "responses": []}},
            memory={"task": {"type": "null"}},
            skills={"default": {"type": "fake", "skills": []}},
        )
    )

    assert session.run() is None
