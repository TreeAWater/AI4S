from evolab.backends.llm import FakeLLMBackend, FakeLLMRuntime, LLMBackend
from evolab.backends.llm.fake import FakeLLMBackend as PackageFakeLLMBackend
from evolab.backends.llm.fake import FakeLLMRuntime as PackageFakeLLMRuntime
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction


def test_fake_llm_backend_instantiates_fake_runtime_and_records_state_refs():
    backend = FakeLLMBackend(default_content="default")

    runtime = backend.instantiate("state-1")

    assert isinstance(backend, LLMBackend)
    assert isinstance(runtime, FakeLLMRuntime)
    assert backend.instantiated_state_refs == ["state-1"]
    assert FakeLLMBackend is PackageFakeLLMBackend
    assert FakeLLMRuntime is PackageFakeLLMRuntime


def test_fake_llm_runtime_records_requests_and_returns_queued_then_default_responses():
    queued = LLMRuntimeResponse(
        action=SubAgentAction(action="final_answer", content="queued"),
        raw_response={"source": "queued"},
    )
    runtime = FakeLLMRuntime(default_content="default", responses=[queued])
    config = LLMGenerationConfig(model="fake-model")

    first = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"type": "function", "name": "lookup"}],
        generation_config=config,
    )
    second = runtime.generate(
        messages=[Message(role="user", content="again")],
        tool_specs=[],
        generation_config=config,
    )

    assert first == queued
    assert second.action.action == "final_answer"
    assert second.action.content == "default"
    assert len(runtime.requests) == 2
    assert runtime.requests[0].messages == [Message(role="user", content="hello")]
    assert runtime.requests[0].tool_specs == [{"type": "function", "name": "lookup"}]
    assert runtime.requests[0].generation_config == config
