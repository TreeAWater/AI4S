from __future__ import annotations

from collections import deque
from typing import Any

from evolab.backends.llm.base import LLMBackend, LLMRuntime
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeRequest, LLMRuntimeResponse, SubAgentAction


class FakeLLMRuntime(LLMRuntime):
    def __init__(
        self,
        *,
        default_content: str = "fake response",
        responses: list[LLMRuntimeResponse] | None = None,
    ):
        self.default_content = default_content
        self.responses = deque(responses or [])
        self.requests: list[LLMRuntimeRequest] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.requests.append(
            LLMRuntimeRequest(
                messages=messages,
                tool_specs=tool_specs,
                generation_config=generation_config,
            )
        )
        if self.responses:
            return self.responses.popleft()
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=self.default_content),
            raw_response={"backend": "fake"},
        )


class FakeLLMBackend(LLMBackend):
    backend_id = "fake_llm"

    def __init__(
        self,
        *,
        default_content: str = "fake response",
        responses: list[LLMRuntimeResponse] | None = None,
        backend_id: str | None = None,
    ):
        if backend_id is not None:
            self.backend_id = backend_id
        self.default_content = default_content
        self.responses = list(responses or [])
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> FakeLLMRuntime:
        self.instantiated_state_refs.append(state_ref)
        return FakeLLMRuntime(default_content=self.default_content, responses=list(self.responses))
