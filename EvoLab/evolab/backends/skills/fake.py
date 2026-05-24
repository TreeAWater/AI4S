from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from evolab.backends.skills.base import SkillBackend
from evolab.contracts.retrieval import (
    RetrievalRequest,
    SkillBundle,
    SkillItem,
    SkillObservationRequest,
    SkillUpdateResult,
)


class FakeSkillBackend(SkillBackend):
    backend_id = "fake_skill"

    def __init__(
        self,
        *,
        skills: list[SkillItem] | None = None,
        graph_version_ref: str | None = "fake-skill-graph-v1",
        skill_state_ref: str | None = "fake-skill-state-v1",
        next_skill_state_ref: str | None = None,
        backend_id: str | None = None,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.skills = list(skills or [])
        self.graph_version_ref = graph_version_ref
        self.skill_state_ref = skill_state_ref
        self.next_skill_state_ref = next_skill_state_ref or skill_state_ref
        self.get_requests: list[RetrievalRequest] = []
        self.look_at_events: list[dict[str, Any]] = []
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> "FakeSkillBackend":
        self.instantiated_state_refs.append(state_ref)
        if state_ref is not None:
            self.skill_state_ref = state_ref
        return self

    def get(self, request: RetrievalRequest) -> SkillBundle:
        self.get_requests.append(request)
        return SkillBundle(
            backend_id=self.backend_id,
            graph_version_ref=self.graph_version_ref,
            skill_state_ref=self.skill_state_ref,
            skills=list(self.skills),
            required_tools=_dedupe(tool for skill in self.skills for tool in skill.required_tools),
        )

    def look_at(self, event: dict[str, Any] | SkillObservationRequest) -> SkillUpdateResult:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        self.look_at_events.append(payload)
        return SkillUpdateResult(
            status="recorded",
            update_summary={"observed_runs": len(self.look_at_events)},
            graph_version_ref=self.graph_version_ref,
            skill_state_ref=self.next_skill_state_ref,
        )


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
