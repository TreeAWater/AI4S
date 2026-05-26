from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from evolab.backends.llm.api import ApiLLMBackend, ApiLLMBackendConfig
from evolab.backends.llm.base import LLMBackend, LLMRuntime
from evolab.contracts.common import Message
from evolab.contracts.local_trainable import LocalTrainableStateManifest
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction


class LocalTrainableRuntime:
    def __init__(
        self,
        *,
        backend_id: str,
        state_ref: str | None,
        default_content: str,
        state_manifest: LocalTrainableStateManifest | None = None,
    ) -> None:
        self.backend_id = backend_id
        self.state_ref = state_ref
        self.default_content = default_content
        self.state_manifest = state_manifest

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        raw_response = {
            "backend_id": self.backend_id,
            "state_ref": self.state_ref,
            "model": generation_config.model,
            "n_messages": len(messages),
            "n_tool_specs": len(tool_specs),
        }
        if self.state_manifest is not None:
            raw_response["state_manifest"] = self.state_manifest.model_dump(mode="json")
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=self.default_content),
            raw_response=raw_response,
        )


class LocalTrainableLLMBackend(LLMBackend):
    backend_id = "local_trainable"

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        default_content: str = "local trainable mock response",
        state_registry: object | None = None,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.default_content = default_content
        self.state_registry = state_registry
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        self.instantiated_state_refs.append(state_ref)
        manifest = self._load_state_manifest(state_ref) if state_ref else None
        if manifest is not None and manifest.serving is not None:
            return _runtime_from_serving_manifest(manifest.serving)
        default_content = manifest.default_content if manifest is not None else self.default_content
        return LocalTrainableRuntime(
            backend_id=self.backend_id,
            state_ref=state_ref,
            default_content=default_content,
            state_manifest=manifest,
        )

    def _load_state_manifest(self, state_ref: str) -> LocalTrainableStateManifest:
        path = _local_path_from_uri(state_ref)
        if path is not None and path.is_file():
            return LocalTrainableStateManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if self.state_registry is not None:
            get_state = getattr(self.state_registry, "get_state", None)
            if callable(get_state):
                record = get_state(state_ref)
                if record is not None:
                    for artifact in getattr(record, "artifact_refs", []):
                        if artifact.metadata.get("role") != "local_trainable_state":
                            continue
                        artifact_path = _local_path_from_uri(artifact.uri)
                        if artifact_path is not None and artifact_path.is_file():
                            return LocalTrainableStateManifest.model_validate_json(
                                artifact_path.read_text(encoding="utf-8")
                            )
        raise ValueError(f"LocalTrainableLLMBackend cannot resolve state_ref: {state_ref!r}")


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None


def _runtime_from_serving_manifest(serving: dict[str, Any]) -> LLMRuntime:
    api = serving.get("api", "openai-chat-completions")
    model = serving.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("local trainable serving manifest requires a non-empty model")
    base_url = serving.get("base_url") or serving.get("baseUrl")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("local trainable serving base_url must be a string")
    hosting = serving.get("hosting", "local")
    api_key_env = serving.get("api_key_env", "LOCAL_LLM_API_KEY")
    return ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api=api,
            hosting=hosting,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=_optional_float(serving.get("timeout_seconds")),
            max_retries=_int_value(serving.get("max_retries"), 2),
            retry_initial_delay_seconds=_float_value(serving.get("retry_initial_delay_seconds"), 1.0),
            retry_max_delay_seconds=_float_value(serving.get("retry_max_delay_seconds"), 30.0),
        ),
        backend_id="local_trainable_serving",
    ).instantiate(state_ref=None)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float_value(value, 0.0)


def _float_value(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("serving numeric option must be a number")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError("serving numeric option must be a number")


def _int_value(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("serving integer option must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("serving integer option must be an integer")


def write_local_trainable_state_manifest(
    path: Path,
    *,
    backend_id: str,
    state_ref: str,
    parent_state_ref: str | None,
    created_by_trainer: str,
    adapter_uri: str | None,
    dataset_manifest_uri: str | None,
    default_content: str,
    metadata: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = LocalTrainableStateManifest(
        backend_id=backend_id,
        state_ref=state_ref,
        parent_state_ref=parent_state_ref,
        created_by_trainer=created_by_trainer,
        adapter_uri=adapter_uri,
        dataset_manifest_uri=dataset_manifest_uri,
        default_content=default_content,
        metadata=metadata,
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
