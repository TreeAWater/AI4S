from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.paths import resolve_path_arguments
from evolab.tools.runtime import ToolRegistry


def register_schema_tools(registry: ToolRegistry, *, base_dir: str | Path | None = None) -> None:
    _register_if_missing(
        registry,
        _json_schema_validate_spec(),
        lambda arguments: json_schema_validate(
            resolve_path_arguments(arguments, base_dir=base_dir, names=("schema_path", "instance_path"))
        ),
    )


def json_schema_validate(arguments: dict[str, Any]) -> ToolResult:
    schema = _schema_from_arguments(arguments)
    instance = _instance_from_arguments(arguments)
    many = bool(arguments.get("many", False))
    instances = instance if many and isinstance(instance, list) else [instance]
    errors: list[dict[str, Any]] = []
    try:
        from jsonschema import Draft202012Validator

        validator = Draft202012Validator(schema)
        for index, item in enumerate(instances):
            for error in sorted(validator.iter_errors(item), key=lambda item: list(item.path)):
                errors.append({"index": index, "path": list(error.path), "message": error.message})
    except ModuleNotFoundError:
        for index, item in enumerate(instances):
            errors.extend({"index": index, **error} for error in _minimal_validate(schema, item, path=[]))
    payload = {"valid": not errors, "errors": errors, "checked_count": len(instances)}
    return ToolResult(
        call_id="json_schema_validate",
        status="ok",
        content="schema validation passed" if not errors else f"schema validation found {len(errors)} errors",
        metadata=payload,
    )


def _minimal_validate(schema: dict[str, Any], instance: Any, *, path: list[Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_type(instance, expected_type):
        errors.append({"path": path, "message": f"expected type {expected_type}"})
        return errors
    if expected_type == "object" or isinstance(instance, dict):
        required = schema.get("required", [])
        if isinstance(required, list) and isinstance(instance, dict):
            for field in required:
                if field not in instance:
                    errors.append({"path": [*path, field], "message": "required field missing"})
        properties = schema.get("properties", {})
        if isinstance(properties, dict) and isinstance(instance, dict):
            for field, subschema in properties.items():
                if field in instance and isinstance(subschema, dict):
                    errors.extend(_minimal_validate(subschema, instance[field], path=[*path, field]))
    if expected_type == "array" and isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                errors.extend(_minimal_validate(item_schema, item, path=[*path, index]))
    return errors


def _matches_type(instance: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(instance, item) for item in expected_type)
    mapping = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    expected = mapping.get(str(expected_type))
    if expected is None:
        return True
    if expected_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected_type == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    return isinstance(instance, expected)


def _schema_from_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    schema = arguments.get("schema")
    if isinstance(schema, dict):
        return schema
    schema_path = arguments.get("schema_path")
    if isinstance(schema_path, str) and schema_path:
        payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    raise ValueError("schema or schema_path must provide a JSON schema object")


def _instance_from_arguments(arguments: dict[str, Any]) -> Any:
    if "instance" in arguments:
        return arguments["instance"]
    instance_path = arguments.get("instance_path")
    if isinstance(instance_path, str) and instance_path:
        path = Path(instance_path)
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError("instance or instance_path must be provided")


def _json_schema_validate_spec() -> ToolSpec:
    return ToolSpec(
        name="json_schema_validate",
        description=(
            "Validate JSON-compatible data against a JSON schema. Provide either "
            "schema or schema_path, and either instance or instance_path."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "schema": {"type": "object", "description": "JSON schema object."},
                "schema_path": {
                    "type": "string",
                    "description": "Path to a JSON schema file. Use this for file-backed extraction tasks.",
                },
                "instance": {
                    "description": "JSON-compatible instance, object, array, scalar, or list of instances to validate.",
                },
                "instance_path": {"type": "string", "description": "Path to a JSON or JSONL instance file."},
                "many": {"type": "boolean", "description": "Treat instance as a list of instances."},
            },
            "required": [],
        },
    )


def _register_if_missing(registry: ToolRegistry, spec: ToolSpec, handler: Any) -> None:
    if registry.get_spec(spec.name) is None:
        registry.register(spec, handler)
