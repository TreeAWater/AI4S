from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolBundle, ToolCall, ToolResult, ToolSpec

ToolHandler = Callable[[dict[str, Any]], str | ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def tool_names(self) -> list[str]:
        return list(self._specs)

    def get_handler(self, name: str) -> ToolHandler:
        try:
            return self._handlers[name]
        except KeyError as exc:
            raise ValueError(f"tool {name!r} is not registered") from exc


class ToolRuntime:
    def __init__(self, registry: ToolRegistry) -> None:
        if registry is None:
            raise ValueError("ToolRuntime requires a ToolRegistry")
        self._registry = registry
        self._runtime_specs: dict[str, ToolSpec] = {}
        self._runtime_handlers: dict[str, ToolHandler] = {}
        self._generated_task_id: str | None = None
        self._generated_specs: dict[str, ToolSpec] = {}
        self._generated_handlers: dict[str, ToolHandler] = {}
        self._generated_provenance: dict[str, dict[str, Any]] = {}
        self._prepared_tool_names: set[str] | None = None
        self._policy: RuntimePolicy | None = None
        self._human_request_count = 0

    def prepare(
        self,
        required_tools: Iterable[str],
        allowed_tools: Iterable[str],
        policy: RuntimePolicy,
        optional_tools: Iterable[str] | None = None,
    ) -> ToolBundle:
        self._runtime_specs = {}
        self._runtime_handlers = {}
        allowed = set(allowed_tools)
        seen: set[str] = set()
        tool_specs: list[ToolSpec] = []
        self._policy = policy
        self._human_request_count = 0
        if policy.max_tool_steps == 0:
            self._prepared_tool_names = set()
            return ToolBundle()
        for name in [*required_tools, *(optional_tools or [])]:
            if name in seen or name not in allowed:
                continue
            seen.add(name)
            spec = self._get_effective_spec(name)
            if spec is not None:
                if spec.metadata.get("requires_human") is True and not policy.allow_human_tools:
                    continue
                tool_specs.append(spec)
        self._prepared_tool_names = {spec.name for spec in tool_specs}
        return ToolBundle(tool_specs=tool_specs)

    def execute(self, call: ToolCall) -> ToolResult:
        return self.execute_tool_name(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )

    def apply_runtime_tool_overlay(self, overlay: Any) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for patch in getattr(overlay, "patches", []):
            strategy = getattr(patch, "strategy", "")
            if strategy != "safe_read_table_slice_wrapper":
                raise ValueError(f"unsupported runtime tool overlay strategy: {strategy!r}")
            spec, handler = self._build_safe_read_table_slice_wrapper(patch)
            self.register_runtime_tool(spec, handler)
            specs.append(spec)
        return specs

    def register_runtime_tool(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._runtime_specs:
            raise ValueError(f"runtime tool {spec.name!r} is already registered")
        self._runtime_specs[spec.name] = spec
        self._runtime_handlers[spec.name] = handler
        if self._prepared_tool_names is not None:
            self._prepared_tool_names.add(spec.name)

    def mark_tool_prepared(self, name: str) -> None:
        if self._prepared_tool_names is not None:
            self._prepared_tool_names.add(name)

    def prepared_tool_specs(self) -> list[ToolSpec]:
        if self._prepared_tool_names is None:
            return []
        specs: list[ToolSpec] = []
        for name in sorted(self._prepared_tool_names):
            spec = self._get_effective_spec(name)
            if spec is not None:
                specs.append(spec)
        return specs

    def generated_tool_specs(self) -> list[ToolSpec]:
        return list(self._generated_specs.values())

    def activate_generated_tool_scope(self, task_id: str) -> None:
        if not task_id:
            raise ValueError("generated tool scope requires task_id")
        if self._generated_task_id == task_id:
            return
        if self._generated_specs:
            raise ValueError("cannot switch generated tool scope while generated tools are active")
        self._generated_task_id = task_id

    def reset_task_generated_tools(self, task_id: str | None = None) -> None:
        if task_id is not None and self._generated_task_id not in {None, task_id}:
            raise ValueError("cannot reset generated tools for a different task scope")
        generated_names = set(self._generated_specs)
        self._generated_task_id = None
        self._generated_specs = {}
        self._generated_handlers = {}
        self._generated_provenance = {}
        if self._prepared_tool_names is not None:
            self._prepared_tool_names = {
                name
                for name in self._prepared_tool_names
                if name not in generated_names or self._registry.get_spec(name) is not None
            }

    def register_task_generated_tool(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        provenance: dict[str, Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        if self._generated_task_id is None:
            raise ValueError("activate generated tool scope before registering generated tools")
        if self._registry.get_spec(spec.name) is not None and not replace_existing:
            raise ValueError(f"generated tool {spec.name!r} would replace a built-in tool")
        if spec.name in self._generated_specs and not replace_existing:
            raise ValueError(f"generated tool {spec.name!r} is already registered")
        self._generated_specs[spec.name] = spec
        self._generated_handlers[spec.name] = handler
        self._generated_provenance[spec.name] = dict(provenance or {})

    def generated_tool_names(self) -> list[str]:
        return list(self._generated_specs)

    def generated_tool_provenance(self, name: str) -> dict[str, Any]:
        return dict(self._generated_provenance.get(name) or {})

    def execute_tool_name(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self._execute_tool_name(
            call_id=call_id,
            name=name,
            arguments=arguments,
            include_runtime_tools=True,
            enforce_prepared=True,
        )

    def execute_registered_tool_name(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self._execute_tool_name(
            call_id=call_id,
            name=name,
            arguments=arguments,
            include_runtime_tools=False,
            enforce_prepared=False,
        )

    def _execute_tool_name(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        include_runtime_tools: bool,
        enforce_prepared: bool,
    ) -> ToolResult:
        try:
            is_generated_call = include_runtime_tools and name in self._generated_specs
            if enforce_prepared and self._prepared_tool_names is not None and name not in self._prepared_tool_names:
                return ToolResult(
                    call_id=call_id,
                    status="error",
                    content=f"tool {name!r} was not prepared for this run",
                    metadata={"error_type": "unprepared_tool"},
                )
            spec = self._get_effective_spec(name) if include_runtime_tools else self._registry.get_spec(name)
            if spec is not None and spec.metadata.get("requires_human") is True:
                policy = self._policy or RuntimePolicy()
                if not policy.allow_human_tools:
                    return ToolResult(
                        call_id=call_id,
                        status="error",
                        content=f"human tool {name!r} is disabled by runtime policy",
                        metadata={"error_type": "human_tools_disabled"},
                    )
                if self._human_request_count >= policy.max_human_requests_per_run:
                    return ToolResult(
                        call_id=call_id,
                        status="error",
                        content="maximum human tool requests exceeded",
                        metadata={"error_type": "max_human_requests_exceeded"},
                    )
                self._human_request_count += 1
            handler = self._get_effective_handler(name) if include_runtime_tools else self._registry.get_handler(name)
            output = handler(arguments)
            if isinstance(output, ToolResult):
                result = output.model_copy(update={"call_id": call_id})
            else:
                if not isinstance(output, str):
                    raise TypeError(
                        f"tool {name!r} returned non-string output: "
                        f"{type(output).__name__}"
                    )
                result = ToolResult(call_id=call_id, status="ok", content=output)
            return self._with_generated_tool_metadata(result, name=name, enabled=is_generated_call)
        except Exception as exc:
            return self._with_generated_tool_metadata(
                ToolResult(call_id=call_id, status="error", content=str(exc)),
                name=name,
                enabled=include_runtime_tools and name in self._generated_specs,
            )

    def _with_generated_tool_metadata(self, result: ToolResult, *, name: str, enabled: bool) -> ToolResult:
        if not enabled:
            return result
        metadata = dict(result.metadata)
        metadata["generated_tool"] = {
            "task_id": self._generated_task_id,
            **self._generated_provenance.get(name, {}),
        }
        return result.model_copy(update={"metadata": metadata})

    def _get_effective_spec(self, name: str) -> ToolSpec | None:
        return self._runtime_specs.get(name) or self._generated_specs.get(name) or self._registry.get_spec(name)

    def _get_effective_handler(self, name: str) -> ToolHandler:
        if name in self._runtime_handlers:
            return self._runtime_handlers[name]
        if name in self._generated_handlers:
            return self._generated_handlers[name]
        return self._registry.get_handler(name)

    def _build_safe_read_table_slice_wrapper(self, patch: Any) -> tuple[ToolSpec, ToolHandler]:
        base_tool_name = str(getattr(patch, "base_tool_name"))
        tool_name = str(getattr(patch, "name"))
        spec = ToolSpec(
            name=tool_name,
            description=getattr(patch, "description", None) or "Safe wrapper around read_table_slice",
            parameters_schema={"type": "object"},
            metadata={
                "runtime_overlay": True,
                "base_tool_name": base_tool_name,
                "strategy": getattr(patch, "strategy", ""),
            },
        )

        def handler(arguments: dict[str, Any]) -> ToolResult:
            normalized_arguments = dict(arguments)
            warnings: list[str] = []
            diagnostics: dict[str, Any] = {"base_tool_name": base_tool_name}

            inspect_arguments = {
                key: value
                for key, value in arguments.items()
                if key in {"path", "table_caption", "sheet_name", "delimiter"}
            }
            inspection = self.execute_registered_tool_name(
                call_id=f"{tool_name}-inspect",
                name="inspect_table",
                arguments=inspect_arguments,
            )
            row_count = _metadata_row_count(inspection.metadata)
            block = inspection.metadata.get("plain_text_table_block")

            start_row = _maybe_int(normalized_arguments.get("start_row"), default=0)
            end_value = normalized_arguments.get("end_row")
            end_row = _maybe_int(end_value, default=row_count if row_count is not None else start_row)
            if start_row is not None and end_row is not None and start_row > end_row:
                normalized_arguments["start_row"] = end_row
                normalized_arguments["end_row"] = start_row
                warnings.append("normalized source slice bounds with start_row > end_row")
                start_row, end_row = end_row, start_row

            if (
                isinstance(block, dict)
                and isinstance(block.get("start_line"), int)
                and isinstance(block.get("end_line"), int)
                and row_count is not None
                and start_row is not None
                and end_row is not None
                and (start_row >= row_count or end_row > row_count)
            ):
                source_start = block["start_line"]
                source_end = block["end_line"]
                overlaps_source_lines = not (end_row <= source_start or start_row > source_end)
                if overlaps_source_lines:
                    relative_start = max(0, start_row - source_start)
                    relative_end = min(row_count, max(relative_start + 1, end_row - source_start))
                    normalized_arguments["start_row"] = relative_start
                    normalized_arguments["end_row"] = relative_end
                    warnings.append("normalized source line coordinates to table-relative row indexes")
                    diagnostics["source_line_normalization"] = {
                        "source_start_row": start_row,
                        "source_end_row": end_row,
                        "normalized_start_row": relative_start,
                        "normalized_end_row": relative_end,
                    }

            result = self.execute_registered_tool_name(
                call_id=f"{tool_name}-base",
                name=base_tool_name,
                arguments=normalized_arguments,
            )
            rows = result.metadata.get("rows", [])
            if (
                result.status == "ok"
                and isinstance(rows, list)
                and not rows
                and row_count is not None
                and row_count > 0
                and arguments.get("table_caption")
            ):
                fallback_arguments = dict(normalized_arguments)
                fallback_arguments["start_row"] = 0
                fallback_arguments["end_row"] = row_count
                fallback = self.execute_registered_tool_name(
                    call_id=f"{tool_name}-fallback",
                    name=base_tool_name,
                    arguments=fallback_arguments,
                )
                fallback_rows = fallback.metadata.get("rows", [])
                if fallback.status == "ok" and isinstance(fallback_rows, list) and fallback_rows:
                    result = fallback
                    warnings.append("retried full caption read after empty slice")
                    diagnostics["caption_fallback"] = True

            metadata = dict(result.metadata)
            metadata["warnings"] = [*metadata.get("warnings", []), *warnings]
            metadata["repair_diagnostics"] = diagnostics
            return result.model_copy(update={"metadata": metadata})

        return spec, handler


def _maybe_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata_row_count(metadata: dict[str, Any]) -> int | None:
    if isinstance(metadata.get("row_count"), int):
        return int(metadata["row_count"])
    rows = metadata.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return None
