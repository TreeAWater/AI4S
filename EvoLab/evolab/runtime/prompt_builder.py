from __future__ import annotations

import json
from typing import Any

from evolab.config.task_config import RoleSpec
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, SkillBundle


class PromptBuilder:
    @staticmethod
    def build(
        role: RoleSpec,
        instruction: str,
        memory: MemoryBundle | str | None,
        skills: SkillBundle | str | None,
        skill_context: dict[str, Any] | None = None,
    ) -> list[Message]:
        sections = [
            f"Instruction:\n{instruction}",
            f"Memory:\n{_memory_text(memory)}",
            f"Skills:\n{_skill_text(skills)}",
        ]
        if skill_context is not None:
            sections.append(f"Skill Context:\n{_skill_context_text(skill_context)}")
        return [
            Message(role="system", content=role.system_prompt),
            Message(
                role="user",
                content="\n\n".join(sections),
            ),
        ]


def _memory_text(memory: MemoryBundle | str | None) -> str:
    if memory is None:
        return ""
    if isinstance(memory, str):
        return memory
    return "\n".join(item.content for item in memory.items)


def _skill_text(skills: SkillBundle | str | None) -> str:
    if skills is None:
        return ""
    if isinstance(skills, str):
        return skills
    return "\n\n".join(_format_skill(skill) for skill in skills.skills)


def _format_skill(skill: Any) -> str:
    name = getattr(skill, "name", "")
    content = getattr(skill, "content", "")
    if name:
        return f"{name}:\n{content}"
    return str(content)


def _skill_context_text(skill_context: dict[str, Any]) -> str:
    return json.dumps(skill_context, sort_keys=True)
