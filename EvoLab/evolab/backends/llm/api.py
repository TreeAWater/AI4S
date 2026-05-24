from __future__ import annotations

import ast
import json
import os
import time
from typing import Any, Literal

from pydantic import Field

from evolab.backends.llm.base import LLMBackend, LLMRuntime
from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.tools import ToolCall


class ApiLLMBackendConfig(StrictBaseModel):
    provider: Literal["openai"]
    api: Literal["openai-responses", "openai-chat-completions"] = "openai-responses"
    hosting: Literal["remote", "local"] = "remote"
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_initial_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=30.0, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


def serialize_message_for_responses(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        if not message.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        }
    return {"role": message.role, "content": message.content}


def _schema_allows_object(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "object":
        return True
    if isinstance(schema_type, list):
        return "object" in schema_type
    return False


def _validate_strict_schema_node(schema: Any, path: str, is_root: bool = False) -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"response_json_schema node at {path} must be an object")

    if is_root:
        if "anyOf" in schema:
            raise ValueError("response_json_schema must not use top-level anyOf")
        if schema.get("type") != "object":
            raise ValueError("response_json_schema must be a JSON schema object with type='object'")

    if _schema_allows_object(schema):
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"object schema at {path} requires additionalProperties=False")

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"object schema properties at {path} must be an object")

        required = schema.get("required")
        required_keys = set(required) if isinstance(required, list) else set()
        if (
            not isinstance(required, list)
            or len(required) != len(required_keys)
            or required_keys != set(properties)
        ):
            raise ValueError(f"object schema at {path} requires all properties to be listed in required")
        for name, property_schema in properties.items():
            _validate_strict_schema_node(property_schema, f"{path}.properties.{name}")

    items = schema.get("items")
    if items is not None:
        _validate_strict_schema_node(items, f"{path}.items")

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list):
            raise ValueError(f"anyOf at {path} must be a list")
        for index, branch_schema in enumerate(any_of):
            _validate_strict_schema_node(branch_schema, f"{path}.anyOf[{index}]")

    for definitions_key in ("$defs", "definitions"):
        definitions = schema.get(definitions_key)
        if definitions is not None:
            if not isinstance(definitions, dict):
                raise ValueError(f"{definitions_key} at {path} must be an object")
            for name, definition_schema in definitions.items():
                _validate_strict_schema_node(definition_schema, f"{path}.{definitions_key}.{name}")


def _strict_json_schema_format(schema: dict[str, Any]) -> dict[str, Any]:
    _validate_strict_schema_node(schema, "root", is_root=True)
    return {
        "format": {
            "type": "json_schema",
            "name": "structured_output",
            "schema": schema,
            "strict": True,
        }
    }


def _item_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                loaded = parser(arguments)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict):
                return loaded
            return {"_raw_arguments": loaded}
        return {"_raw_arguments": arguments}
    return {"_raw_arguments": arguments}


def _tool_call_from_response_item(item: Any) -> ToolCall | None:
    if _item_value(item, "type") != "function_call":
        return None

    name = _item_value(item, "name")
    call_id = _item_value(item, "call_id") or _item_value(item, "id")
    if not name or not call_id:
        raise ValueError("function_call response items require name and call_id")

    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=_parse_tool_arguments(_item_value(item, "arguments")),
    )


def _response_tool_calls(response: Any) -> list[ToolCall]:
    output = _item_value(response, "output") or []
    tool_calls: list[ToolCall] = []
    for item in output:
        tool_call = _tool_call_from_response_item(item)
        if tool_call is not None:
            tool_calls.append(tool_call)
    return tool_calls


def _text_from_response_output(response: Any) -> str:
    output = _item_value(response, "output") or []
    text_parts: list[str] = []
    for item in output:
        content = _item_value(item, "content") or []
        for content_item in content:
            if _item_value(content_item, "type") == "output_text":
                text = _item_value(content_item, "text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts)


def _raw_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {}


def _response_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        if spec.get("type") == "function":
            tools.append(spec)
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool specs require a non-empty name")
        description = spec.get("description")
        parameters = _response_tool_parameters(
            spec.get("parameters") or spec.get("parameters_schema") or {"type": "object"}
        )
        tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }
        if isinstance(description, str) and description:
            tool["description"] = description
        tools.append(tool)
    return tools


def _response_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    normalized = dict(parameters)
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized["properties"] = {}
    return normalized


def serialize_message_for_chat(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        if not message.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": message.content}


def _chat_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in _response_tools(tool_specs):
        function_spec = {
            "name": spec["name"],
            "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
        }
        if isinstance(spec.get("description"), str) and spec["description"]:
            function_spec["description"] = spec["description"]
        tools.append({"type": "function", "function": function_spec})
    return tools


def _chat_input_messages(
    messages: list[Message],
    assistant_tool_call_messages: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    input_messages: list[dict[str, Any]] = []
    emitted_assistant_message_keys: set[tuple[str, ...]] = set()
    for message in messages:
        if message.role == "tool" and message.tool_call_id:
            tool_call_id = message.tool_call_id
            assistant_tool_call = assistant_tool_call_messages.get(tool_call_id)
            if assistant_tool_call is not None:
                assistant_key = _chat_assistant_tool_call_message_key(assistant_tool_call)
                if assistant_key not in emitted_assistant_message_keys:
                    input_messages.append(dict(assistant_tool_call))
                    emitted_assistant_message_keys.add(assistant_key)
        input_messages.append(serialize_message_for_chat(message))
    return input_messages


def _chat_assistant_tool_call_message_key(message: dict[str, Any]) -> tuple[str, ...]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return ()
    return tuple(str(item.get("id") or "") for item in tool_calls if isinstance(item, dict))


def _chat_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    _validate_strict_schema_node(schema, "root", is_root=True)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema,
            "strict": True,
        },
    }


def _first_chat_choice_message(response: Any) -> Any | None:
    choices = _item_value(response, "choices") or []
    if not choices:
        return None
    return _item_value(choices[0], "message")


def _chat_tool_calls(message: Any) -> list[Any]:
    raw_tool_calls = _item_value(message, "tool_calls")
    if not raw_tool_calls:
        return []
    return list(raw_tool_calls)


def _tool_call_from_chat_tool_call(item: Any) -> ToolCall:
    function = _item_value(item, "function")
    name = _item_value(function, "name")
    arguments = _item_value(function, "arguments")
    call_id = _item_value(item, "id")
    if not name or not call_id:
        raise ValueError("chat tool calls require id and function name")
    return ToolCall(call_id=call_id, name=name, arguments=_parse_tool_arguments(arguments))


def _tool_calls_from_chat_tool_calls(items: list[Any]) -> list[ToolCall]:
    return [_tool_call_from_chat_tool_call(item) for item in items]


def _chat_assistant_tool_call_message(items: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _chat_assistant_tool_call_item(item)
            for item in items
        ],
    }


def _chat_assistant_tool_call_item(item: Any) -> dict[str, Any]:
    function = _item_value(item, "function")
    return {
        "id": _item_value(item, "id"),
        "type": _item_value(item, "type") or "function",
        "function": {
            "name": _item_value(function, "name"),
            "arguments": _item_value(function, "arguments") or "",
        },
    }


class OpenAIResponsesRuntime:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        max_retries: int = 2,
        retry_initial_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
        timeout_seconds: float | None = None,
        default_max_output_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_initial_delay_seconds = retry_initial_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.default_max_output_tokens = default_max_output_tokens
        self.extra_body = dict(extra_body or {})
        self._function_call_items: dict[str, dict[str, Any]] = {}

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        input_messages = _response_input_messages(
            messages,
            generation_config.response_input_items,
            self._function_call_items,
        )
        kwargs: dict[str, Any] = {
            "model": generation_config.model or self.model,
            "input": input_messages,
        }
        if generation_config.previous_response_id is not None:
            kwargs["previous_response_id"] = generation_config.previous_response_id
        max_output_tokens = generation_config.max_output_tokens or self.default_max_output_tokens
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if generation_config.temperature is not None:
            kwargs["temperature"] = generation_config.temperature
        if tool_specs:
            kwargs["tools"] = _response_tools(tool_specs)
        if generation_config.response_json_schema:
            kwargs["text"] = _strict_json_schema_format(generation_config.response_json_schema)
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        response = self._create_response_with_retries(kwargs)
        raw = _raw_response(response)
        self._remember_function_call_items(raw)
        tool_calls = _response_tool_calls(response)
        if tool_calls:
            return LLMRuntimeResponse(
                action=SubAgentAction(action="tool_call", tool_calls=tool_calls),
                raw_response=raw,
            )
        content = (
            _item_value(response, "output_text")
            or _text_from_response_output(response)
            or '{"error": "empty_model_response"}'
        )
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=content),
            raw_response=raw,
        )

    def _create_response_with_retries(self, kwargs: dict[str, Any]) -> Any:
        attempt = 0
        delay = self.retry_initial_delay_seconds
        while True:
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable_api_error(exc):
                    raise
                if delay > 0:
                    time.sleep(min(delay, self.retry_max_delay_seconds))
                delay = min(delay * 2 if delay > 0 else 0, self.retry_max_delay_seconds)
                attempt += 1

    def _remember_function_call_items(self, raw: dict[str, Any]) -> None:
        for item in _function_call_items_from_raw_response(raw):
            call_id = item.get("call_id") or item.get("id")
            if isinstance(call_id, str) and call_id:
                self._function_call_items[call_id] = item


class OpenAIChatCompletionsRuntime:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        max_retries: int = 2,
        retry_initial_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
        timeout_seconds: float | None = None,
        default_max_output_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_initial_delay_seconds = retry_initial_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.default_max_output_tokens = default_max_output_tokens
        self.extra_body = dict(extra_body or {})
        self._assistant_tool_call_messages: dict[str, dict[str, Any]] = {}

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        kwargs: dict[str, Any] = {
            "model": generation_config.model or self.model,
            "messages": _chat_input_messages(messages, self._assistant_tool_call_messages),
        }
        max_output_tokens = generation_config.max_output_tokens or self.default_max_output_tokens
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if generation_config.temperature is not None:
            kwargs["temperature"] = generation_config.temperature
        if tool_specs:
            kwargs["tools"] = _chat_tools(tool_specs)
        if generation_config.response_json_schema:
            kwargs["response_format"] = _chat_response_format(generation_config.response_json_schema)
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        response = self._create_chat_completion_with_retries(kwargs)
        raw = _raw_response(response)
        message = _first_chat_choice_message(response)
        raw_tool_calls = _chat_tool_calls(message)
        if raw_tool_calls:
            assistant_tool_call_message = _chat_assistant_tool_call_message(raw_tool_calls)
            tool_calls = _tool_calls_from_chat_tool_calls(raw_tool_calls)
            for tool_call in tool_calls:
                self._assistant_tool_call_messages[tool_call.call_id] = assistant_tool_call_message
            return LLMRuntimeResponse(
                action=SubAgentAction(action="tool_call", tool_calls=tool_calls),
                raw_response=raw,
            )
        content = _item_value(message, "content") if message is not None else None
        if not isinstance(content, str) or not content:
            content = '{"error": "empty_model_response"}'
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=content),
            raw_response=raw,
        )

    def _create_chat_completion_with_retries(self, kwargs: dict[str, Any]) -> Any:
        attempt = 0
        delay = self.retry_initial_delay_seconds
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable_api_error(exc):
                    raise
                if delay > 0:
                    time.sleep(min(delay, self.retry_max_delay_seconds))
                delay = min(delay * 2 if delay > 0 else 0, self.retry_max_delay_seconds)
                attempt += 1


def _response_input_messages(
    messages: list[Message],
    response_input_items: list[dict[str, Any]],
    function_call_items: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    input_messages = [dict(item) for item in response_input_items]
    seen_function_call_ids = {
        item.get("call_id") or item.get("id")
        for item in input_messages
        if item.get("type") == "function_call"
    }
    for message in messages:
        item = serialize_message_for_responses(message)
        if item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id not in seen_function_call_ids:
                function_call_item = function_call_items.get(call_id)
                if function_call_item is not None:
                    input_messages.append(dict(function_call_item))
                    seen_function_call_ids.add(call_id)
        input_messages.append(item)
    return input_messages


def _function_call_items_from_raw_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    output = raw.get("output")
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    text = str(exc).casefold()
    if "retryable': true" in text or '"retryable": true' in text:
        return True
    retryable_markers = (
        "origin_bad_gateway",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "rate limit",
        "too many requests",
        "request timeout",
        "temporarily unavailable",
    )
    return any(marker in text for marker in retryable_markers)


class ApiLLMBackend(LLMBackend):
    def __init__(
        self,
        config: ApiLLMBackendConfig,
        client: Any | None = None,
        backend_id: str = "api_llm",
        api_key: str | None = None,
    ):
        self.config = config
        self.backend_id = backend_id
        resolved_api_key = api_key or os.environ.get(config.api_key_env)
        if not resolved_api_key and config.hosting == "local":
            resolved_api_key = "dummy-local-key"
        if not resolved_api_key and client is None:
            raise ValueError(f"missing API key in environment variable {config.api_key_env}")
        if client is None:
            from openai import OpenAI

            client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            if config.timeout_seconds is not None:
                client_kwargs["timeout"] = config.timeout_seconds
            client = OpenAI(**client_kwargs)
        self.client = client

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        if state_ref is not None:
            raise ValueError("ApiLLMBackend does not support trainable state_ref")
        if self.config.api == "openai-chat-completions":
            return OpenAIChatCompletionsRuntime(
                self.client,
                self.config.model,
                max_retries=self.config.max_retries,
                retry_initial_delay_seconds=self.config.retry_initial_delay_seconds,
                retry_max_delay_seconds=self.config.retry_max_delay_seconds,
                timeout_seconds=self.config.timeout_seconds,
                default_max_output_tokens=self.config.max_output_tokens,
                extra_body=self.config.extra_body,
            )
        return OpenAIResponsesRuntime(
            self.client,
            self.config.model,
            max_retries=self.config.max_retries,
            retry_initial_delay_seconds=self.config.retry_initial_delay_seconds,
            retry_max_delay_seconds=self.config.retry_max_delay_seconds,
            timeout_seconds=self.config.timeout_seconds,
            default_max_output_tokens=self.config.max_output_tokens,
            extra_body=self.config.extra_body,
        )

    def evolve(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        return LLMEvolutionResult(
            status="skipped",
            metadata={
                "backend_id": self.backend_id,
                "request_backend_id": request.backend_id,
                "reason": "ApiLLMBackend does not train or mutate remote API model state in V1",
            },
        )
