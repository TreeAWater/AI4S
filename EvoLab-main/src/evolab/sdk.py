from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


class TaskSpec(StrictBaseModel):
    goal: str
    resources: str
    expected_outputs: str
    success_criteria: str
    optional_context: str | None = None

    def to_prompt(self) -> str:
        lines = [
            f"1. goal: {self.goal}",
            f"2. resources: {self.resources}",
            f"3. expected_outputs: {self.expected_outputs}",
            f"4. success_criteria: {self.success_criteria}",
        ]
        if self.optional_context is not None:
            lines.append(f"5. optional_context: {self.optional_context}")
        return "\n".join(lines)


class SessionConfig(StrictBaseModel):
    lab_dir: Path | str
    task: TaskSpec | str
    env_file: Path | str | None = None
    llm: dict[str, Any]
    embeddings: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    seed_roles: dict[str, Any] | None = None
    meta_agent: dict[str, Any] | None = None


class EvoLabSession:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    @property
    def lab_dir(self) -> Path:
        return Path(self.config.lab_dir)

    @property
    def state_dir(self) -> Path:
        return self.lab_dir / ".evolab"

    def initialize(self) -> None:
        from evolab.session_runtime import initialize_lab

        initialize_lab(self.config)

    def run(self) -> None:
        from evolab.session_runtime import run_session

        run_session(self.config)
        return None
