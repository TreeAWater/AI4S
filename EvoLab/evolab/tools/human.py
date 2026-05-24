from __future__ import annotations

import json
from typing import Any, Protocol

from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.runtime import ToolRegistry


_MOCK_TIMESTAMP = "1970-01-01T00:00:00Z"


class HumanToolAdapter(Protocol):
    def ask_human(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...

    def request_human_review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...

    def notify_human(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class MockHumanToolAdapter:
    def ask_human(self, arguments: dict[str, Any]) -> dict[str, Any]:
        options = arguments.get("options")
        selected_option = options[0] if isinstance(options, list) and options else None
        return {
            "response": "MOCK_HUMAN_RESPONSE: proceed with conservative extraction.",
            "selected_option": selected_option,
            "approved": True,
            "corrections": [],
            "reviewer_id": "mock-human",
            "timestamp": _MOCK_TIMESTAMP,
        }

    def request_human_review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        instructions = str(arguments.get("instructions", ""))
        status = "needs_revision" if "conflict" in instructions.casefold() else "approved"
        return {
            "review_status": status,
            "comments": "MOCK_HUMAN_REVIEW: deterministic review result.",
            "corrections": [],
            "reviewer_id": "mock-human",
            "timestamp": _MOCK_TIMESTAMP,
        }

    def notify_human(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "delivered": True,
            "timestamp": _MOCK_TIMESTAMP,
        }


def register_human_tools(registry: ToolRegistry, adapter: HumanToolAdapter | None = None) -> None:
    adapter = adapter or MockHumanToolAdapter()
    _register_if_missing(registry, _ask_human_spec(), _handler("ask_human", adapter.ask_human))
    _register_if_missing(
        registry,
        _request_human_review_spec(),
        _handler("request_human_review", adapter.request_human_review),
    )
    _register_if_missing(registry, _notify_human_spec(), _handler("notify_human", adapter.notify_human))


def _register_if_missing(registry: ToolRegistry, spec: ToolSpec, handler: Any) -> None:
    if registry.get_spec(spec.name) is None:
        registry.register(spec, handler)


def _handler(tool_name: str, func: Any) -> Any:
    def handle(arguments: dict[str, Any]) -> ToolResult:
        payload = func(arguments)
        return ToolResult(
            call_id=f"{tool_name}-handler",
            status="ok",
            content=json.dumps(payload, sort_keys=True),
            metadata=payload,
        )

    return handle


def _human_metadata() -> dict[str, Any]:
    return {"requires_human": True}


def _ask_human_spec() -> ToolSpec:
    return ToolSpec(
        name="ask_human",
        description="Ask a human collaborator a concrete question during task execution.",
        parameters_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "expected_response_type": {
                    "type": "string",
                    "enum": ["free_text", "choice", "approval", "correction", "annotation"],
                },
                "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
                "blocking": {"type": "boolean"},
                "artifact_refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "context"],
        },
        metadata=_human_metadata(),
    )


def _request_human_review_spec() -> ToolSpec:
    return ToolSpec(
        name="request_human_review",
        description="Ask a human to review an artifact, extraction result, workflow decision, or validation conflict.",
        parameters_schema={
            "type": "object",
            "properties": {
                "artifact_ref": {"type": "string"},
                "review_type": {
                    "type": "string",
                    "enum": ["schema_validation", "evidence_check", "conflict_resolution", "approval", "general"],
                },
                "instructions": {"type": "string"},
                "blocking": {"type": "boolean"},
                "context": {"type": "string"},
            },
            "required": ["artifact_ref", "review_type", "instructions", "blocking"],
        },
        metadata=_human_metadata(),
    )


def _notify_human_spec() -> ToolSpec:
    return ToolSpec(
        name="notify_human",
        description="Send a non-blocking notification to a human collaborator.",
        parameters_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "context": {"type": "string"},
                "artifact_refs": {"type": "array", "items": {"type": "string"}},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["message"],
        },
        metadata=_human_metadata(),
    )
