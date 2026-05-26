import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evolab.backends.llm import (
    ApiLLMBackend,
    ApiLLMBackendConfig,
    LLMBackend,
    LocalTrainableLLMBackend,
    serialize_message_for_responses,
)
from evolab.backends.llm.api import ApiLLMBackend as PackageApiLLMBackend
from evolab.backends.llm.api import OpenAIChatCompletionsRuntime
from evolab.backends.llm.base import LLMBackend as BaseLLMBackend
from evolab.backends.llm.local import LocalTrainableLLMBackend as PackageLocalTrainableLLMBackend
from evolab.backends.trainers import LLMTrainer
from evolab.contracts.common import Message
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.llm import LLMGenerationConfig


class _FakeResponses:
    def __init__(self, response=None):
        self.calls = []
        self.response = response
        self.responses = list(response) if isinstance(response, list) else None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses is not None:
            return self.responses.pop(0)
        if self.response is not None:
            return self.response

        class Response:
            output_text = "Final answer"

            def model_dump(self):
                return {"output_text": self.output_text, "kwargs": kwargs}

        return Response()


class _FakeClient:
    def __init__(self, response=None):
        self.responses = _FakeResponses(response=response)


class _FakeChatCompletions:
    def __init__(self, response=None):
        self.calls = []
        self.response = response
        self.responses = list(response) if isinstance(response, list) else None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses is not None:
            return self.responses.pop(0)
        if self.response is not None:
            return self.response

        class Message:
            content = "Final chat answer"
            tool_calls = None

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

            def model_dump(self):
                return {"choices": [{"message": {"content": "Final chat answer"}}], "kwargs": kwargs}

        return Response()


class _FakeChat:
    def __init__(self, response=None):
        self.completions = _FakeChatCompletions(response=response)


class _FakeChatClient:
    def __init__(self, response=None):
        self.chat = _FakeChat(response=response)


class _RetryableAPIError(Exception):
    status_code = 502


def test_concrete_llm_backends_inherit_from_llm_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    api_backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"),
        client=_FakeClient(),
    )

    assert issubclass(ApiLLMBackend, LLMBackend)
    assert issubclass(LocalTrainableLLMBackend, LLMBackend)
    assert isinstance(api_backend, LLMBackend)
    assert isinstance(LocalTrainableLLMBackend(), LLMBackend)
    assert inspect.isabstract(LLMBackend)
    assert LLMBackend.__abstractmethods__ == {"instantiate"}
    assert LLMBackend is BaseLLMBackend
    assert ApiLLMBackend is PackageApiLLMBackend
    assert LocalTrainableLLMBackend is PackageLocalTrainableLLMBackend


def test_api_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"))


def test_api_backend_generates_final_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient())
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )
    assert response.action.action == "final_answer"
    assert response.action.content == "Final answer"


def test_api_backend_can_use_chat_completions_runtime_for_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeChatClient()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api="openai-chat-completions",
            model="deepseek/deepseek-v4-flash",
            base_url="https://openrouter.ai/api/v1",
        ),
        client=client,
        backend_id="openrouter-deepseek-v4-flash",
    )
    runtime = backend.instantiate(state_ref=None)

    response = runtime.generate(
        messages=[Message(role="system", content="System."), Message(role="user", content="hello")],
        tool_specs=[
            {
                "name": "lookup",
                "description": "Lookup records.",
                "parameters_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert isinstance(runtime, OpenAIChatCompletionsRuntime)
    assert response.action.action == "final_answer"
    assert response.action.content == "Final chat answer"
    assert client.chat.completions.calls == [
        {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup records.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                }
            ],
        }
    ]


def test_api_backend_applies_default_max_output_tokens_to_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeChatClient()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api="openai-chat-completions",
            model="qwen/qwen3-30b-a3b-instruct-2507",
            max_output_tokens=4096,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert client.chat.completions.calls[0]["max_tokens"] == 4096


def test_generation_config_max_output_tokens_overrides_api_backend_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeChatClient()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api="openai-chat-completions",
            model="qwen/qwen3-30b-a3b-instruct-2507",
            max_output_tokens=4096,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="", max_output_tokens=1024),
    )

    assert client.chat.completions.calls[0]["max_tokens"] == 1024


def test_chat_completions_runtime_parses_tool_call_and_replays_tool_history(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Function:
        name = "lookup"
        arguments = '{"query": "hello"}'

    class ToolCall:
        id = "call-1"
        function = Function()

    class ToolMessage:
        content = None
        tool_calls = [ToolCall()]

    class FinalMessage:
        content = "Final after tool"
        tool_calls = None

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, message):
            self.choices = [Choice(message)]

        def model_dump(self):
            return {"choices": [{"message": {"content": getattr(self.choices[0].message, "content", None)}}]}

    client = _FakeChatClient([Response(ToolMessage()), Response(FinalMessage())])
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", api="openai-chat-completions", model="deepseek/deepseek-v4-flash"),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    tool_response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"name": "lookup", "parameters_schema": {"type": "object"}}],
        generation_config=LLMGenerationConfig(model=""),
    )
    final_response = runtime.generate(
        messages=[Message(role="tool", content="lookup result", tool_call_id="call-1")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert tool_response.action.action == "tool_call"
    assert tool_response.action.tool_call is not None
    assert tool_response.action.tool_call.name == "lookup"
    assert tool_response.action.tool_call.arguments == {"query": "hello"}
    assert final_response.action.action == "final_answer"
    assert final_response.action.content == "Final after tool"
    assert client.chat.completions.calls[1]["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query": "hello"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "lookup result"},
    ]


def test_chat_completions_runtime_parses_multiple_tool_calls_and_replays_one_assistant_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class LookupFunction:
        name = "lookup"
        arguments = '{"query": "hello"}'

    class InspectFunction:
        name = "inspect"
        arguments = '{"path": "input.md"}'

    class LookupToolCall:
        id = "call-1"
        function = LookupFunction()

    class InspectToolCall:
        id = "call-2"
        function = InspectFunction()

    class ToolMessage:
        content = None
        tool_calls = [LookupToolCall(), InspectToolCall()]

    class FinalMessage:
        content = "Final after tools"
        tool_calls = None

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, message):
            self.choices = [Choice(message)]

        def model_dump(self):
            return {"choices": [{"message": {"content": getattr(self.choices[0].message, "content", None)}}]}

    client = _FakeChatClient([Response(ToolMessage()), Response(FinalMessage())])
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", api="openai-chat-completions", model="deepseek/deepseek-v4-flash"),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    tool_response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"name": "lookup", "parameters_schema": {"type": "object"}}],
        generation_config=LLMGenerationConfig(model=""),
    )
    final_response = runtime.generate(
        messages=[
            Message(role="tool", content="lookup result", tool_call_id="call-1"),
            Message(role="tool", content="inspect result", tool_call_id="call-2"),
        ],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert tool_response.action.action == "tool_call"
    assert [call.name for call in tool_response.action.tool_calls] == ["lookup", "inspect"]
    assert [call.call_id for call in tool_response.action.tool_calls] == ["call-1", "call-2"]
    assert tool_response.action.tool_call == tool_response.action.tool_calls[0]
    assert final_response.action.action == "final_answer"
    assert client.chat.completions.calls[1]["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query": "hello"}'},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": '{"path": "input.md"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "lookup result"},
        {"role": "tool", "tool_call_id": "call-2", "content": "inspect result"},
    ]


def test_chat_completions_runtime_accepts_python_dict_style_tool_arguments(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Function:
        name = "read_text"
        arguments = "{'path': 'input.md', 'max_chars': 1200}"

    class ToolCall:
        id = "call-1"
        function = Function()

    class ToolMessage:
        content = None
        tool_calls = [ToolCall()]

    class Choice:
        message = ToolMessage()

    class Response:
        choices = [Choice()]

        def model_dump(self):
            return {"choices": [{"message": {"tool_calls": [{"id": "call-1"}]}}]}

    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", api="openai-chat-completions", model="deepseek/deepseek-v4-flash"),
        client=_FakeChatClient(Response()),
    )
    runtime = backend.instantiate(state_ref=None)

    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"name": "read_text", "parameters_schema": {"type": "object"}}],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert response.action.tool_call is not None
    assert response.action.tool_call.arguments == {"path": "input.md", "max_chars": 1200}


def test_chat_completions_runtime_preserves_non_object_tool_arguments_as_raw(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Function:
        name = "write_report"
        arguments = '"plain report content"'

    class ToolCall:
        id = "call-1"
        function = Function()

    class ToolMessage:
        content = None
        tool_calls = [ToolCall()]

    class Choice:
        message = ToolMessage()

    class Response:
        choices = [Choice()]

        def model_dump(self):
            return {"choices": [{"message": {"tool_calls": [{"id": "call-1"}]}}]}

    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", api="openai-chat-completions", model="deepseek/deepseek-v4-flash"),
        client=_FakeChatClient(Response()),
    )
    runtime = backend.instantiate(state_ref=None)

    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"name": "write_report", "parameters_schema": {"type": "object"}}],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert response.action.tool_call is not None
    assert response.action.tool_call.arguments == {"_raw_arguments": "plain report content"}


def test_chat_completions_runtime_passes_timeout_to_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeChatClient()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api="openai-chat-completions",
            model="deepseek/deepseek-v4-flash",
            timeout_seconds=45.0,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert client.chat.completions.calls[0]["timeout"] == 45.0


def test_chat_completions_runtime_passes_extra_body_to_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeChatClient()
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api="openai-chat-completions",
            model="qwen-local",
            extra_body=extra_body,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model=""),
    )

    assert client.chat.completions.calls[0]["extra_body"] == extra_body


def test_api_backend_retries_retryable_response_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _RetryableAPIError("Error code: 502 - origin_bad_gateway")

            class Response:
                output_text = "Recovered final answer"

                def model_dump(self):
                    return {"output_text": self.output_text}

            return Response()

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            model="gpt-4.1-mini",
            max_retries=1,
            retry_initial_delay_seconds=0.0,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.action == "final_answer"
    assert response.action.content == "Recovered final answer"
    assert len(client.responses.calls) == 2


def test_api_backend_retries_json_decode_errors_from_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise json.JSONDecodeError("Expecting value", "", 0)

            class Response:
                output_text = "Recovered after parse error"

                def model_dump(self):
                    return {"output_text": self.output_text}

            return Response()

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            model="gpt-4.1-mini",
            max_retries=1,
            retry_initial_delay_seconds=0.0,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.content == "Recovered after parse error"
    assert len(client.responses.calls) == 2


def test_api_backend_does_not_retry_non_retryable_response_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class NonRetryableAPIError(Exception):
        status_code = 403

    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise NonRetryableAPIError("Error code: 403 - insufficient balance")

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    backend = ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            model="gpt-4.1-mini",
            max_retries=3,
            retry_initial_delay_seconds=0.0,
        ),
        client=client,
    )
    runtime = backend.instantiate(state_ref=None)

    with pytest.raises(NonRetryableAPIError):
        runtime.generate(
            messages=[Message(role="user", content="hello")],
            tool_specs=[],
            generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
        )

    assert len(client.responses.calls) == 1


def test_api_backend_reads_nested_message_output_text_when_output_text_is_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        output_text = ""
        output = [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Nested final answer",
                    }
                ],
            }
        ]

        def model_dump(self):
            return {"output": self.output}

    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient(Response()))
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.action == "final_answer"
    assert response.action.content == "Nested final answer"


def test_api_backend_returns_diagnostic_final_answer_for_empty_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        output_text = ""
        output = [{"type": "reasoning", "summary": []}]

        def model_dump(self):
            return {"id": "response-empty", "output": self.output}

    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient(Response()))
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.action == "final_answer"
    assert response.action.content == '{"error": "empty_model_response"}'


def test_api_backend_forwards_generation_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=tools,
        generation_config=LLMGenerationConfig(
            model="override-model",
            temperature=0.2,
            max_output_tokens=128,
            response_json_schema=schema,
        ),
    )

    assert client.responses.calls == [
        {
            "model": "override-model",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 128,
            "temperature": 0.2,
            "tools": tools,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
    ]


def test_api_backend_allows_provider_parallel_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        generation_config=LLMGenerationConfig(model="override-model"),
    )

    assert "parallel_tool_calls" not in client.responses.calls[0]


def test_api_backend_converts_evolab_tool_specs_to_openai_response_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[
            {
                "schema_version": "v1",
                "name": "write_report",
                "description": "Write a report.",
                "parameters_schema": {"type": "object", "properties": {"content": {"type": "string"}}},
                "metadata": {"local": True},
            }
        ],
        generation_config=LLMGenerationConfig(model="override-model"),
    )

    assert client.responses.calls[0]["tools"] == [
        {
            "type": "function",
            "name": "write_report",
            "description": "Write a report.",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
        }
    ]


def test_api_backend_adds_empty_properties_to_object_tool_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[
            {
                "name": "inspect_table",
                "description": "Inspect a table.",
                "parameters_schema": {"type": "object"},
            }
        ],
        generation_config=LLMGenerationConfig(model="override-model"),
    )

    assert client.responses.calls[0]["tools"][0]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_api_backend_preserves_response_continuation_items_and_previous_response_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)
    prior_items = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "hello"}',
        },
    ]

    runtime.generate(
        messages=[Message(role="tool", content="lookup result", tool_call_id="call-1")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(
            model="gpt-4.1-mini",
            previous_response_id="resp_123",
            response_input_items=prior_items,
        ),
    )

    assert client.responses.calls == [
        {
            "model": "gpt-4.1-mini",
            "input": [
                *prior_items,
                {"type": "function_call_output", "call_id": "call-1", "output": "lookup result"},
            ],
            "previous_response_id": "resp_123",
        }
    ]


def test_api_runtime_replays_previous_function_call_item_for_tool_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class ToolCallResponse:
        output = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"query": "hello"}',
            }
        ]

        def model_dump(self):
            return {"output": self.output}

    client = _FakeClient(ToolCallResponse())
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)
    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-5.1-codex"),
    )
    runtime.generate(
        messages=[Message(role="tool", content="lookup result", tool_call_id="call-1")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-5.1-codex"),
    )

    assert client.responses.calls[1]["input"] == [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "hello"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "lookup result"},
    ]


def test_api_runtime_replays_matching_function_calls_for_full_tool_history(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        def __init__(self, call_id: str | None = None):
            self.output = []
            self.output_text = "Final answer"
            if call_id is not None:
                self.output = [
                    {
                        "type": "function_call",
                        "id": None,
                        "call_id": call_id,
                        "name": "lookup",
                        "arguments": '{"query": "hello"}',
                    }
                ]
                self.output_text = ""

        def model_dump(self):
            return {"output": self.output, "output_text": self.output_text}

    client = _FakeClient([Response("call-1"), Response("call-2"), Response()])
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)
    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-5.1-codex"),
    )
    runtime.generate(
        messages=[
            Message(role="user", content="hello"),
            Message(role="tool", content="lookup result 1", tool_call_id="call-1"),
        ],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-5.1-codex"),
    )
    runtime.generate(
        messages=[
            Message(role="user", content="hello"),
            Message(role="tool", content="lookup result 1", tool_call_id="call-1"),
            Message(role="tool", content="lookup result 2", tool_call_id="call-2"),
        ],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-5.1-codex"),
    )

    assert client.responses.calls[2]["input"] == [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "id": None,
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "hello"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "lookup result 1"},
        {
            "type": "function_call",
            "id": None,
            "call_id": "call-2",
            "name": "lookup",
            "arguments": '{"query": "hello"}',
        },
        {"type": "function_call_output", "call_id": "call-2", "output": "lookup result 2"},
    ]


def test_api_backend_allows_explicit_empty_object_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient()
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="default-model"), client=client)
    runtime = backend.instantiate(state_ref=None)
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="override-model", response_json_schema=schema),
    )

    assert client.responses.calls[0]["text"]["format"]["schema"] == schema


def test_api_backend_rejects_invalid_provider():
    with pytest.raises(ValidationError):
        ApiLLMBackendConfig(provider="anthropic", model="claude")


def test_api_backend_evolve_returns_skipped_result(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient())
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="aigocode-gpt",
        artifact_root_uri=str(tmp_path / "artifacts"),
    )

    result = backend.evolve(request)

    assert result.status == "skipped"
    assert result.recommend_for_promotion is False
    assert result.metadata["backend_id"] == "api_llm"
    assert "does not train" in result.metadata["reason"]


def test_local_trainable_llm_backend_is_rollout_only():
    backend = LocalTrainableLLMBackend(backend_id="aigocode-gpt")

    assert isinstance(backend, LLMBackend)
    assert not isinstance(backend, LLMTrainer)
    assert not hasattr(backend, "train")
    assert not hasattr(backend, "evolve")


def test_local_trainable_base_runtime_generates_default_answer():
    backend = LocalTrainableLLMBackend(
        backend_id="aigocode-gpt",
        default_content="base local answer",
    )

    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="local"),
    )

    assert response.action.action == "final_answer"
    assert response.action.content == "base local answer"
    assert response.raw_response["backend_id"] == "aigocode-gpt"
    assert response.raw_response["state_ref"] is None


def test_api_backend_config_records_hosting_mode():
    local_config = ApiLLMBackendConfig(
        provider="openai",
        api="openai-chat-completions",
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
        hosting="local",
    )
    remote_config = ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini", hosting="remote")

    assert local_config.hosting == "local"
    assert remote_config.hosting == "remote"


def test_local_trainable_state_routes_to_openai_compatible_serving_runtime(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    state_ref = "local-trainable://aigocode-gpt/state/local-served"
    manifest_path = tmp_path / "local_trainable_state.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend_id": "aigocode-gpt",
                "state_ref": state_ref,
                "created_by_trainer": "sft",
                "serving": {
                    "api": "openai-chat-completions",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "evolab-local",
                    "hosting": "local",
                    "api_key_env": "LOCAL_LLM_API_KEY",
                },
            }
        ),
        encoding="utf-8",
    )

    backend = LocalTrainableLLMBackend(backend_id="aigocode-gpt")
    runtime = backend.instantiate(str(manifest_path))

    assert isinstance(runtime, OpenAIChatCompletionsRuntime)
    assert runtime.model == "evolab-local"
    assert runtime.client.base_url == "http://127.0.0.1:8000/v1/"
    assert runtime.client.api_key == "dummy-local-key"


def test_api_backend_parses_function_call_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        output = [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"query": "hello"}',
                "call_id": "call-1",
            }
        ]

        def model_dump(self):
            return {"output": self.output}

    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"),
        client=_FakeClient(Response()),
    )
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.action == "tool_call"
    assert response.action.tool_call is not None
    assert response.action.tool_call.call_id == "call-1"
    assert response.action.tool_call.name == "lookup"
    assert response.action.tool_call.arguments == {"query": "hello"}


def test_api_backend_parses_multiple_function_call_responses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        output = [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"query": "hello"}',
                "call_id": "call-1",
            },
            {
                "type": "function_call",
                "name": "inspect",
                "arguments": '{"path": "input.md"}',
                "call_id": "call-2",
            },
        ]

        def model_dump(self):
            return {"output": self.output}

    backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"),
        client=_FakeClient(Response()),
    )
    runtime = backend.instantiate(state_ref=None)
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="gpt-4.1-mini"),
    )

    assert response.action.action == "tool_call"
    assert [call.call_id for call in response.action.tool_calls] == ["call-1", "call-2"]
    assert [call.name for call in response.action.tool_calls] == ["lookup", "inspect"]


def test_tool_message_serializes_to_function_call_output():
    item = serialize_message_for_responses(Message(role="tool", content="result", tool_call_id="call-1"))

    assert item == {"type": "function_call_output", "call_id": "call-1", "output": "result"}
    assert "role" not in item


def test_api_backend_rejects_invalid_strict_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient())
    runtime = backend.instantiate(state_ref=None)

    with pytest.raises(ValueError):
        runtime.generate(
            messages=[Message(role="user", content="hello")],
            tool_specs=[],
            generation_config=LLMGenerationConfig(
                model="gpt-4.1-mini",
                response_json_schema={"type": "array"},
            ),
        )


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        {
            "type": "object",
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            ]
        },
        {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "answer": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    ]
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"answer": {"$ref": "#/$defs/answer"}},
            "required": ["answer"],
            "additionalProperties": False,
            "$defs": {
                "answer": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ],
)
def test_api_backend_rejects_non_strict_output_schemas(monkeypatch, schema):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ApiLLMBackend(ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"), client=_FakeClient())
    runtime = backend.instantiate(state_ref=None)

    with pytest.raises(ValueError):
        runtime.generate(
            messages=[Message(role="user", content="hello")],
            tool_specs=[],
            generation_config=LLMGenerationConfig(
                model="gpt-4.1-mini",
                response_json_schema=schema,
            ),
        )
