from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from typing import Any

from pydantic import ValidationError

from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.generated_tools import (
    GeneratedToolCapabilityGrant,
    GeneratedToolPackage,
    GeneratedToolRegistration,
    GeneratedToolValidationResult,
    TaskEffectiveToolCatalog,
)
from evolab.contracts.llm import LLMGenerationConfig
from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.runtime import ToolRuntime


_RUNNER_SOURCE = r'''
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def _emit(payload):
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _preload_import_modules(import_modules, module_dir):
    module_dir = Path(module_dir).resolve(strict=False)
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    changed_cwd = False
    import os as _os

    try:
        if original_cwd.resolve(strict=False) == module_dir:
            _os.chdir(str(module_dir.parent))
            changed_cwd = True
        sys.path = _sys_path_without_module_dir(original_sys_path, module_dir, original_cwd)
        importlib.invalidate_caches()
        for item in import_modules:
            required = True
            module_name = item
            if isinstance(item, dict):
                module_name = item.get("name")
                required = bool(item.get("required", True))
            if not isinstance(module_name, str) or not module_name.strip():
                continue
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if not required and exc.name == module_name:
                    continue
                raise RuntimeError(f"generated tool import preload failed for {module_name!r}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"generated tool import preload failed for {module_name!r}: {exc}") from exc
    finally:
        sys.path = original_sys_path
        if changed_cwd:
            _os.chdir(str(original_cwd))
        importlib.invalidate_caches()


def _sys_path_without_module_dir(sys_path, module_dir, original_cwd):
    sanitized = []
    for entry in sys_path:
        if _sys_path_entry_points_to(entry, module_dir, original_cwd):
            continue
        sanitized.append(entry)
    return sanitized


def _sys_path_entry_points_to(entry, module_dir, original_cwd):
    if not isinstance(entry, str):
        return False
    candidate = original_cwd if entry == "" else Path(entry)
    if not candidate.is_absolute():
        candidate = original_cwd / candidate
    return candidate.resolve(strict=False) == module_dir


def _install_file_access_guard(request):
    sys.dont_write_bytecode = True
    module_dir = Path(request["module_path"]).parent
    allowed_read_roots = _resolved_roots([module_dir, *(request.get("allowed_read_roots") or [])])
    allowed_write_root = request.get("allowed_write_root")
    allowed_write_roots = _resolved_roots([allowed_write_root] if allowed_write_root else [])

    def audit(event, args):
        if event != "open" or not args:
            return
        path = args[0]
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else None
        if isinstance(path, int):
            raise PermissionError("generated tool file descriptor access is not allowed")
        if path is None:
            return
        resolved = Path(path).resolve(strict=False)
        if _access_writes(mode, flags) and not _path_is_under(resolved, allowed_write_roots):
            raise PermissionError(f"generated tool file write denied outside allowed_write_root: {resolved}")
        if _access_reads(mode, flags) and not _path_is_under(resolved, allowed_read_roots):
            raise PermissionError(f"generated tool file read denied outside allowed_read_roots: {resolved}")

    sys.addaudithook(audit)


def _resolved_roots(roots):
    return [Path(root).resolve(strict=False) for root in roots if isinstance(root, str) and root]


def _path_is_under(path, roots):
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _mode_writes(mode):
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _mode_reads(mode):
    return "r" in mode or "+" in mode or not _mode_writes(mode)


def _access_writes(mode, flags):
    mode_text = str(mode or "")
    if mode_text and mode_text != "None":
        return _mode_writes(mode_text)
    if isinstance(flags, int):
        import os
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(flags & write_flags)
    return False


def _access_reads(mode, flags):
    mode_text = str(mode or "")
    if mode_text and mode_text != "None":
        return _mode_reads(mode_text)
    if isinstance(flags, int):
        import os
        return (flags & os.O_WRONLY) == 0 or bool(flags & os.O_RDWR)
    return True


def _load_module(module_path):
    spec = importlib.util.spec_from_file_location("generated_tool_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    request = json.loads(sys.stdin.read() or "{}")
    try:
        sys.dont_write_bytecode = True
        module_dir = Path(request["module_path"]).parent
        _preload_import_modules(request.get("import_modules") or [], module_dir)
        _install_file_access_guard(request)
        module = _load_module(request["module_path"])
        action = request.get("action")
        if action == "spec":
            tool_spec = getattr(module, "TOOL_SPEC", None)
            if not isinstance(tool_spec, dict):
                raise RuntimeError("generated tool module must define TOOL_SPEC as a dict")
            run = getattr(module, "run", None)
            if not callable(run):
                raise RuntimeError("generated tool module must define callable run(arguments, context)")
            _emit({"ok": True, "tool_spec": tool_spec})
            return
        if action != "run":
            raise RuntimeError(f"unknown generated tool runner action: {action!r}")
        run = getattr(module, "run", None)
        if not callable(run):
            raise RuntimeError("generated tool module must define callable run(arguments, context)")
        output = run(request.get("arguments") or {}, request.get("context") or {})
        if isinstance(output, dict):
            status = output.get("status", "ok")
            content = output.get("content", "")
            metadata = output.get("metadata", {})
        else:
            status = "ok"
            content = output
            metadata = {}
        if status not in {"ok", "error"}:
            raise RuntimeError(f"generated tool returned invalid status: {status!r}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True, sort_keys=True)
        if not isinstance(metadata, dict):
            metadata = {}
        _emit({"ok": True, "result": {"status": status, "content": content, "metadata": metadata}})
    except Exception as exc:
        _emit({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
'''


class GeneratedToolBuilder:
    def __init__(self, llm_runtime: Any) -> None:
        self.llm_runtime = llm_runtime

    def build(
        self,
        *,
        task_id: str,
        task_goal: str,
        run_ref: str,
        built_in_tool_specs: list[ToolSpec],
        generated_tool_specs: list[ToolSpec],
        role_pool_templates: list[Any],
        artifact_root: str | Path,
        capability_grant: GeneratedToolCapabilityGrant,
        failure_signal: Any | None = None,
        requested_tool_name: str | None = None,
    ) -> GeneratedToolPackage:
        prompt_payload = {
            "instruction": (
                "Return only a JSON object matching GeneratedToolPackage. Include complete Python source files. "
                "The primary module must define TOOL_SPEC as a dict and run(arguments, context) as a callable."
            ),
            "task_id": task_id,
            "task_goal": task_goal,
            "run_ref": run_ref,
            "built_in_tool_specs": [_model_dump_jsonable(spec) for spec in built_in_tool_specs],
            "generated_tool_specs": [_model_dump_jsonable(spec) for spec in generated_tool_specs],
            "role_pool_templates": [_model_dump_jsonable(template) for template in role_pool_templates],
            "artifact_root": str(artifact_root),
            "capability_grant": capability_grant.model_dump(mode="json"),
            "failure_signal": _model_dump_jsonable(failure_signal) if failure_signal is not None else None,
            "requested_tool_name": requested_tool_name,
            "output_contract": {
                "schema": "GeneratedToolPackage",
                "required": ["tool_name", "reason", "manifest", "files", "primary_module"],
                "file_requirement": (
                    "files must be non-empty and contain complete Python source for primary_module, "
                    "including TOOL_SPEC and run(arguments, context)"
                ),
            },
        }
        response = self.llm_runtime.generate(
            [Message(role="user", content=json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True))],
            [],
            LLMGenerationConfig(
                model="",
                temperature=0,
                max_output_tokens=4096,
                metadata={"runtime_stage": "generated_tool_builder"},
            ),
        )
        action = getattr(response, "action", None)
        if getattr(action, "action", None) != "final_answer":
            raise RuntimeError("generated tool builder expected final_answer response")
        content = getattr(action, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("generated tool builder returned no final_answer content")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"generated tool builder returned invalid JSON: {exc}") from exc
        try:
            return GeneratedToolPackage.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(f"generated tool builder returned invalid GeneratedToolPackage: {exc}") from exc


class GeneratedToolRuntime:
    def __init__(
        self,
        artifact_root: str | Path,
        *,
        tool_runtime: ToolRuntime,
        policy: RuntimePolicy,
        trajectory_collector: Any | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.tool_runtime = tool_runtime
        self.policy = policy
        self.trajectory_collector = trajectory_collector

    def register_package(
        self,
        *,
        package: GeneratedToolPackage,
        task_id: str,
        run_ref: str,
        context: dict[str, Any] | None = None,
    ) -> GeneratedToolRegistration:
        registered_name = _unique_registered_tool_name(
            _registered_tool_name(task_id, run_ref, package.tool_name),
            _known_tool_names(self.tool_runtime),
        )
        package_dir = self.artifact_root / "generated_tools" / registered_name
        grant = package.capability_grant if package.capability_grant is not None else _default_grant(package_dir)
        effective_grant = _effective_grant(grant, policy=self.policy, package_dir=package_dir)
        validation_errors = self._preflight(
            package,
            grant=grant,
            effective_grant=effective_grant,
            package_dir=package_dir,
        )
        package_dir.mkdir(parents=True, exist_ok=True)
        for file in package.files:
            destination = package_dir / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content, encoding="utf-8")

        primary_path = package_dir / package.primary_module
        runner_path = package_dir / "_generated_tool_runner.py"
        runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")
        import_modules = _package_import_modules(package)
        runtime_context = {
            **(context or {}),
            "task_id": task_id,
            "capability_grant": effective_grant.model_dump(mode="json"),
        }
        tool_spec = _fallback_spec(package, registered_name)
        validation = GeneratedToolValidationResult(
            valid=not validation_errors,
            status="passed" if not validation_errors else "failed",
            errors=validation_errors,
        )
        if not validation_errors:
            validation, tool_spec = self._validate_load_and_smoke(
                package=package,
                registered_name=registered_name,
                module_path=primary_path,
                runner_path=runner_path,
                context=runtime_context,
                allowed_env_keys=effective_grant.allowed_env_keys,
                allowed_read_roots=effective_grant.allowed_read_roots,
                allowed_write_root=effective_grant.allowed_write_root,
                import_modules=import_modules,
            )

        registration = GeneratedToolRegistration(
            requested_tool_name=package.tool_name,
            registered_tool_name=registered_name,
            task_id=task_id,
            run_ref=run_ref,
            tool_spec=tool_spec,
            module_path=str(primary_path),
            code_hash=package.package_hash,
            validation=validation,
            capability_grant=effective_grant,
            metadata={"package_dir": str(package_dir), "reason": package.reason},
        )
        if validation.valid:
            try:
                self.tool_runtime.register_task_generated_tool(
                    tool_spec,
                    _subprocess_handler(
                        module_path=primary_path,
                        runner_path=runner_path,
                        timeout_s=self.policy.generated_tool_execution_timeout_s,
                        max_output_bytes=self.policy.generated_tool_max_output_bytes,
                        context=runtime_context,
                        allowed_env_keys=effective_grant.allowed_env_keys,
                        allowed_read_roots=effective_grant.allowed_read_roots,
                        allowed_write_root=effective_grant.allowed_write_root,
                        import_modules=import_modules,
                    ),
                    provenance={
                        "requested_tool_name": package.tool_name,
                        "registered_tool_name": registered_name,
                        "code_hash": package.package_hash,
                        "module_path": str(primary_path),
                        "validation": validation.model_dump(mode="json"),
                    },
                )
            except ValueError as exc:
                registration.validation = GeneratedToolValidationResult(
                    valid=False,
                    status="failed",
                    errors=[str(exc)],
                    smoke_test_results=validation.smoke_test_results,
                )
        return registration

    def build_effective_tool_catalog(
        self,
        *,
        task_id: str,
        builtin_allowed_tool_names: list[str],
    ) -> TaskEffectiveToolCatalog:
        return build_effective_tool_catalog(
            task_id=task_id,
            builtin_allowed_tool_names=builtin_allowed_tool_names,
            tool_runtime=self.tool_runtime,
        )

    def _preflight(
        self,
        package: GeneratedToolPackage,
        *,
        grant: GeneratedToolCapabilityGrant,
        effective_grant: GeneratedToolCapabilityGrant,
        package_dir: Path,
    ) -> list[str]:
        errors: list[str] = []
        if not self.policy.allow_runtime_tool_creation:
            errors.append("runtime tool creation is disabled by policy")
        if len(self.tool_runtime.generated_tool_names()) >= self.policy.max_generated_tools_per_task:
            errors.append("maximum generated tools per task exceeded")
        if len(package.files) > self.policy.max_generated_tool_files:
            errors.append("maximum generated tool files exceeded")
        if package.source_bytes > self.policy.max_generated_tool_source_bytes:
            errors.append("maximum generated tool source bytes exceeded")
        parameters_schema = package.manifest.get("parameters_schema")
        schema_error = _parameters_schema_error(
            parameters_schema,
            source="generated tool manifest parameters_schema",
        )
        if schema_error is not None:
            errors.append(schema_error)
        metadata = package.manifest.get("metadata")
        if isinstance(metadata, dict):
            errors.extend(_private_reasoning_metadata_errors(metadata, source="generated tool manifest metadata"))
        errors.extend(_capability_grant_errors(grant, policy=self.policy))
        for file in package.files:
            if file.path.endswith(".py"):
                errors.extend(
                    _static_python_errors(
                        file.content,
                        effective_grant=effective_grant,
                        policy_allowed_imports=set(self.policy.generated_tool_allowed_imports),
                        package_dir=package_dir,
                        grant=effective_grant,
                    )
                )
        return errors

    def _validate_load_and_smoke(
        self,
        *,
        package: GeneratedToolPackage,
        registered_name: str,
        module_path: Path,
        runner_path: Path,
        context: dict[str, Any],
        allowed_env_keys: list[str],
        allowed_read_roots: list[str],
        allowed_write_root: str | None,
        import_modules: list[dict[str, Any]],
    ) -> tuple[GeneratedToolValidationResult, ToolSpec]:
        errors: list[str] = []
        smoke_results: list[dict[str, Any]] = []
        tool_spec = _fallback_spec(package, registered_name)

        spec_run = _run_generated_subprocess(
            runner_path=runner_path,
            module_path=module_path,
            action="spec",
            timeout_s=self.policy.generated_tool_validation_timeout_s,
            max_output_bytes=self.policy.generated_tool_max_output_bytes,
            allowed_env_keys=allowed_env_keys,
            allowed_read_roots=allowed_read_roots,
            allowed_write_root=allowed_write_root,
            import_modules=import_modules,
        )
        if not spec_run.get("ok"):
            errors.append(f"generated tool spec load failed: {spec_run.get('error')}")
        else:
            loaded_spec = spec_run.get("tool_spec")
            if not isinstance(loaded_spec, dict):
                errors.append("generated tool TOOL_SPEC must be a dict")
            else:
                try:
                    tool_spec = _tool_spec_from_loaded(loaded_spec, registered_name, package)
                except (TypeError, ValidationError, ValueError) as exc:
                    errors.append(f"generated tool TOOL_SPEC is invalid: {exc}")

        for smoke_test in package.smoke_tests:
            if errors:
                break
            run_payload = _run_generated_subprocess(
                runner_path=runner_path,
                module_path=module_path,
                action="run",
                arguments=smoke_test.arguments,
                context={**context, **smoke_test.context},
                timeout_s=self.policy.generated_tool_validation_timeout_s,
                max_output_bytes=self.policy.generated_tool_max_output_bytes,
                allowed_env_keys=allowed_env_keys,
                allowed_read_roots=allowed_read_roots,
                allowed_write_root=allowed_write_root,
                import_modules=import_modules,
            )
            smoke_results.append({"name": smoke_test.name, **run_payload})
            if not run_payload.get("ok"):
                errors.append(f"smoke test {smoke_test.name!r} failed: {run_payload.get('error')}")
                continue
            result = run_payload.get("result")
            if not isinstance(result, dict):
                errors.append(f"smoke test {smoke_test.name!r} did not return a result")
                continue
            status = result.get("status")
            if smoke_test.expect_status is not None and status != smoke_test.expect_status:
                errors.append(
                    f"smoke test {smoke_test.name!r} returned status {status!r}, "
                    f"expected {smoke_test.expect_status!r}"
                )
            elif smoke_test.expect_status is None and status != "ok":
                errors.append(f"smoke test {smoke_test.name!r} returned status {status!r}")

        return (
            GeneratedToolValidationResult(
                valid=not errors,
                status="passed" if not errors else "failed",
                errors=errors,
                smoke_test_results=smoke_results,
            ),
            tool_spec,
        )


def _model_dump_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _model_dump_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump_jsonable(item) for item in value]
    return value


def _package_import_modules(package: GeneratedToolPackage) -> list[dict[str, Any]]:
    local_module_roots = _package_local_module_roots(package)
    modules: dict[str, bool] = {}
    for file in package.files:
        if not file.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(file.content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            for preload in _import_modules_from_node(node):
                module_name = preload["name"]
                root = module_name.split(".")[0]
                if root in local_module_roots:
                    continue
                modules[module_name] = modules.get(module_name, False) or preload["required"]
    return [{"name": name, "required": required} for name, required in modules.items()]


def _package_local_module_roots(package: GeneratedToolPackage) -> set[str]:
    roots: set[str] = set()
    for file in package.files:
        if not file.path.endswith(".py"):
            continue
        path = Path(file.path)
        if path.name == "__init__.py" and path.parts:
            roots.add(path.parts[0])
        else:
            roots.add(path.parts[0].removesuffix(".py"))
    return {root for root in roots if root}


def _import_modules_from_node(node: ast.AST) -> list[dict[str, Any]]:
    if isinstance(node, ast.Import):
        return [{"name": alias.name, "required": True} for alias in node.names if alias.name]
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        modules = [{"name": node.module, "required": True}]
        modules.extend(
            {"name": f"{node.module}.{alias.name}", "required": False}
            for alias in node.names
            if alias.name and alias.name != "*"
        )
        return modules
    return []


def build_effective_tool_catalog(
    *,
    task_id: str,
    builtin_allowed_tool_names: list[str],
    tool_runtime: ToolRuntime,
) -> TaskEffectiveToolCatalog:
    tool_specs_by_name: dict[str, ToolSpec] = {}
    generated_tool_names = tool_runtime.generated_tool_names()
    for name in [*builtin_allowed_tool_names, *generated_tool_names]:
        spec = tool_runtime._get_effective_spec(name)
        if spec is not None:
            tool_specs_by_name[name] = spec
    provenance_by_name = {
        name: tool_runtime.generated_tool_provenance(name)
        for name in generated_tool_names
    }
    return TaskEffectiveToolCatalog(
        task_id=task_id,
        builtin_allowed_tool_names=builtin_allowed_tool_names,
        generated_tool_names=generated_tool_names,
        tool_specs_by_name=tool_specs_by_name,
        provenance_by_name=provenance_by_name,
    )


def _registered_tool_name(task_id: str, run_ref: str, requested: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", requested).strip("_") or "tool"
    task_slug = re.sub(r"[^A-Za-z0-9_]+", "_", task_id).strip("_")[:24] or "task"
    run_slug = re.sub(r"[^A-Za-z0-9_]+", "_", run_ref).strip("_")[:24] or "run"
    return f"gt_{task_slug}_{run_slug}_{slug}"


def _unique_registered_tool_name(base_name: str, existing_names: list[str]) -> str:
    existing = set(existing_names)
    if base_name not in existing:
        return base_name
    index = 2
    while f"{base_name}_{index}" in existing:
        index += 1
    return f"{base_name}_{index}"


def _known_tool_names(tool_runtime: ToolRuntime) -> list[str]:
    names = set(tool_runtime.generated_tool_names())
    registry = getattr(tool_runtime, "_registry", None)
    specs = getattr(registry, "_specs", None)
    if isinstance(specs, dict):
        names.update(str(name) for name in specs)
    runtime_specs = getattr(tool_runtime, "_runtime_specs", None)
    if isinstance(runtime_specs, dict):
        names.update(str(name) for name in runtime_specs)
    return sorted(names)


def _default_grant(package_dir: Path) -> GeneratedToolCapabilityGrant:
    return GeneratedToolCapabilityGrant(
        allowed_read_roots=[str(package_dir)],
        allowed_write_root=str(package_dir),
    )


def _effective_grant(
    grant: GeneratedToolCapabilityGrant,
    *,
    policy: RuntimePolicy,
    package_dir: Path | None = None,
) -> GeneratedToolCapabilityGrant:
    policy_env_keys = set(policy.generated_tool_allowed_env_keys)
    policy_imports = set(policy.generated_tool_allowed_imports)
    read_roots = list(grant.allowed_read_roots)
    if package_dir is not None:
        read_roots = _dedupe_strings([str(package_dir), *read_roots])
    return GeneratedToolCapabilityGrant(
        allowed_read_roots=read_roots,
        allowed_write_root=grant.allowed_write_root,
        allowed_env_keys=[key for key in grant.allowed_env_keys if key in policy_env_keys],
        allow_network=grant.allow_network and policy.generated_tool_allow_network,
        allow_subprocess=grant.allow_subprocess and policy.generated_tool_allow_subprocess,
        allowed_imports=[name for name in grant.allowed_imports if name in policy_imports],
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _fallback_spec(package: GeneratedToolPackage, registered_name: str) -> ToolSpec:
    manifest = dict(package.manifest)
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    parameters_schema = manifest.get("parameters_schema")
    return ToolSpec(
        name=registered_name,
        description=str(manifest.get("description") or package.reason),
        parameters_schema=_object_parameters_schema_or_default(parameters_schema),
        metadata={
            "generated_tool": True,
            "requested_tool_name": package.tool_name,
            **metadata,
        },
    )


def _tool_spec_from_loaded(
    loaded_spec: dict[str, Any],
    registered_name: str,
    package: GeneratedToolPackage,
) -> ToolSpec:
    metadata = loaded_spec.get("metadata") if isinstance(loaded_spec.get("metadata"), dict) else {}
    parameters_schema = loaded_spec.get("parameters_schema")
    schema_error = _parameters_schema_error(parameters_schema, source="parameters_schema")
    if schema_error is not None:
        raise TypeError(schema_error)
    metadata_errors = _private_reasoning_metadata_errors(metadata, source="generated tool TOOL_SPEC metadata")
    if metadata_errors:
        raise ValueError("; ".join(metadata_errors))
    return ToolSpec(
        name=registered_name,
        description=str(
            loaded_spec.get("description")
            or package.manifest.get("description")
            or package.reason
        ),
        parameters_schema=_object_parameters_schema_or_default(parameters_schema),
        metadata={
            "generated_tool": True,
            "requested_tool_name": package.tool_name,
            **metadata,
        },
    )


def _parameters_schema_error(parameters_schema: Any, *, source: str) -> str | None:
    if parameters_schema is None:
        return None
    if not isinstance(parameters_schema, dict):
        return f"{source} must be a dict"
    schema_type = parameters_schema.get("type")
    if schema_type is not None and schema_type != "object":
        return f"{source} must be an object schema"
    return None


def _object_parameters_schema_or_default(parameters_schema: Any) -> dict[str, Any]:
    if (
        isinstance(parameters_schema, dict)
        and _parameters_schema_error(parameters_schema, source="parameters_schema") is None
    ):
        return parameters_schema
    return {"type": "object"}


_PRIVATE_REASONING_METADATA_KEYS = {"chain_of_thought", "reasoning", "hidden_reasoning"}


def _private_reasoning_metadata_errors(metadata: dict[str, Any], *, source: str) -> list[str]:
    errors: list[str] = []
    for key, value in metadata.items():
        key_text = str(key)
        if key_text in _PRIVATE_REASONING_METADATA_KEYS:
            errors.append(f"{source} contains private reasoning metadata key {key_text!r}")
        if isinstance(value, dict):
            errors.extend(_private_reasoning_metadata_errors(value, source=source))
    return errors


def _capability_grant_errors(grant: GeneratedToolCapabilityGrant, *, policy: RuntimePolicy) -> list[str]:
    errors: list[str] = []
    if grant.allow_network and not policy.generated_tool_allow_network:
        errors.append("generated tool package capability grant requests network access disabled by policy")
    if grant.allow_subprocess and not policy.generated_tool_allow_subprocess:
        errors.append("generated tool package capability grant requests subprocess access disabled by policy")
    disallowed_env_keys = sorted(set(grant.allowed_env_keys).difference(policy.generated_tool_allowed_env_keys))
    if disallowed_env_keys:
        errors.append(
            "generated tool package capability grant requests environment keys disabled by policy: "
            + ", ".join(disallowed_env_keys)
        )
    disallowed_imports = sorted(set(grant.allowed_imports).difference(policy.generated_tool_allowed_imports))
    if disallowed_imports:
        errors.append(
            "generated tool package capability grant requests allowed imports disabled by policy: "
            + ", ".join(disallowed_imports)
        )
    return errors


def _static_python_errors(
    source: str,
    *,
    effective_grant: GeneratedToolCapabilityGrant,
    policy_allowed_imports: set[str] | None = None,
    package_dir: Path | None = None,
    grant: GeneratedToolCapabilityGrant | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"invalid Python syntax: {exc}"]

    forbidden: set[str] = set()
    if not effective_grant.allow_subprocess:
        forbidden.add("subprocess")
    if not effective_grant.allow_network:
        forbidden.update({"socket", "requests", "urllib", "httpx"})
    if not effective_grant.allowed_env_keys:
        forbidden.add("os")

    controlled_imports = set(policy_allowed_imports or ())
    granted_controlled_imports = set(effective_grant.allowed_imports)
    open_aliases = _static_open_aliases(tree)
    os_aliases, os_subprocess_aliases = _static_os_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "builtins":
                    errors.append("import 'builtins' is not allowed by generated tool policy")
                if root in forbidden:
                    errors.append(f"import {root!r} is not allowed by generated tool policy")
                elif root in controlled_imports and root not in granted_controlled_imports:
                    errors.append(f"import {root!r} requires generated tool package allowed_imports grant")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root == "builtins":
                errors.append("import 'builtins' is not allowed by generated tool policy")
            if root in forbidden:
                errors.append(f"import {root!r} is not allowed by generated tool policy")
            elif root in controlled_imports and root not in granted_controlled_imports:
                errors.append(f"import {root!r} requires generated tool package allowed_imports grant")
        elif isinstance(node, ast.Call):
            if _is_forbidden_dynamic_builtin_call(node):
                errors.append(f"call to {node.func.id!r} is not allowed by generated tool policy")
            if not effective_grant.allow_subprocess and _is_forbidden_os_subprocess_call(
                node,
                os_aliases=os_aliases,
                function_aliases=os_subprocess_aliases,
            ):
                errors.append("call to os subprocess API is not allowed by generated tool policy")
            errors.extend(_static_write_errors(node, package_dir=package_dir, grant=grant, open_aliases=open_aliases))
    return errors


def _static_open_aliases(tree: ast.AST) -> set[str]:
    aliases = {"open"}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "builtins":
                for alias in node.names:
                    if alias.name == "open":
                        changed |= _add_alias(aliases, alias.asname or alias.name)
            elif isinstance(node, ast.Assign) and _is_open_alias_value(node.value, aliases):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        changed |= _add_alias(aliases, target.id)
            elif isinstance(node, ast.AnnAssign) and node.value is not None and _is_open_alias_value(node.value, aliases):
                if isinstance(node.target, ast.Name):
                    changed |= _add_alias(aliases, node.target.id)
    return aliases


def _add_alias(aliases: set[str], name: str) -> bool:
    before = len(aliases)
    aliases.add(name)
    return len(aliases) != before


def _is_open_alias_value(node: ast.AST, aliases: set[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in aliases


def _is_forbidden_dynamic_builtin_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec", "compile"}


_OS_SUBPROCESS_ATTRS = {
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "startfile",
    "system",
}


def _static_os_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_aliases = {"os"}
    function_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        changed |= _add_alias(module_aliases, alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    if alias.name in _OS_SUBPROCESS_ATTRS:
                        changed |= _add_alias(function_aliases, alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                if _is_os_subprocess_alias_value(node.value, module_aliases, function_aliases):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            changed |= _add_alias(function_aliases, target.id)
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and _is_os_subprocess_alias_value(node.value, module_aliases, function_aliases)
            ):
                if isinstance(node.target, ast.Name):
                    changed |= _add_alias(function_aliases, node.target.id)
    return module_aliases, function_aliases


def _is_os_subprocess_alias_value(
    node: ast.AST,
    module_aliases: set[str],
    function_aliases: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in function_aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr in _OS_SUBPROCESS_ATTRS
        and isinstance(node.value, ast.Name)
        and node.value.id in module_aliases
    )


def _is_forbidden_os_subprocess_call(
    node: ast.Call,
    *,
    os_aliases: set[str],
    function_aliases: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in function_aliases
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _OS_SUBPROCESS_ATTRS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in os_aliases
    )


def _static_write_errors(
    node: ast.Call,
    *,
    package_dir: Path | None,
    grant: GeneratedToolCapabilityGrant | None,
    open_aliases: set[str],
) -> list[str]:
    if _is_open_write_call(node, open_aliases=open_aliases):
        return _write_target_errors(
            _call_arg_literal_string(node, 0),
            package_dir=package_dir,
            grant=grant,
            api_name="open()",
        )
    if _is_path_write_call(node):
        return _write_target_errors(
            _path_constructor_literal_string(node.func.value),
            package_dir=package_dir,
            grant=grant,
            api_name=f"Path.{node.func.attr}()",
        )
    return []


def _is_open_write_call(node: ast.Call, *, open_aliases: set[str]) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id not in open_aliases:
        return False
    mode = _call_arg_literal_string(node, 1)
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            mode = keyword.value.value
    mode = mode or "r"
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _is_path_write_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr in {"write_text", "write_bytes"}


def _call_arg_literal_string(node: ast.Call, index: int) -> str | None:
    if index >= len(node.args):
        return None
    value = node.args[index]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _path_constructor_literal_string(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Path":
        return _call_arg_literal_string(node, 0)
    if isinstance(func, ast.Attribute) and func.attr == "Path":
        return _call_arg_literal_string(node, 0)
    return None


def _write_target_errors(
    target: str | None,
    *,
    package_dir: Path | None,
    grant: GeneratedToolCapabilityGrant | None,
    api_name: str,
) -> list[str]:
    if grant is None or not grant.allowed_write_root:
        return [f"{api_name} write grant requires an allowed_write_root capability grant"]
    if target is None:
        return [f"{api_name} write target must be a literal path inside allowed_write_root"]
    if package_dir is None:
        return []
    allowed_root = Path(grant.allowed_write_root)
    if not allowed_root.is_absolute():
        allowed_root = package_dir / allowed_root
    try:
        resolved_root = allowed_root.resolve(strict=False)
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = package_dir / target_path
        resolved_target = target_path.resolve(strict=False)
        resolved_target.relative_to(resolved_root)
    except ValueError:
        return [f"{api_name} write target is outside allowed_write_root"]
    return []


def _subprocess_handler(
    *,
    module_path: Path,
    runner_path: Path,
    timeout_s: int,
    max_output_bytes: int,
    context: dict[str, Any],
    allowed_env_keys: list[str],
    allowed_read_roots: list[str],
    allowed_write_root: str | None,
    import_modules: list[dict[str, Any]],
):
    def handler(arguments: dict[str, Any]) -> ToolResult:
        payload = _run_generated_subprocess(
            runner_path=runner_path,
            module_path=module_path,
            action="run",
            arguments=arguments,
            context=context,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            allowed_env_keys=allowed_env_keys,
            allowed_read_roots=allowed_read_roots,
            allowed_write_root=allowed_write_root,
            import_modules=import_modules,
        )
        if not payload.get("ok"):
            return ToolResult(call_id="generated-local-call", status="error", content=str(payload.get("error") or ""))
        result = payload.get("result")
        if not isinstance(result, dict):
            return ToolResult(
                call_id="generated-local-call",
                status="error",
                content="generated tool runner returned invalid result",
            )
        return ToolResult(
            call_id="generated-local-call",
            status=result.get("status", "ok"),
            content=str(result.get("content", "")),
            metadata=result.get("metadata") if isinstance(result.get("metadata"), dict) else {},
        )

    return handler


def _run_generated_subprocess(
    *,
    runner_path: Path,
    module_path: Path,
    action: str,
    timeout_s: int,
    max_output_bytes: int,
    allowed_env_keys: list[str],
    allowed_read_roots: list[str] | None = None,
    allowed_write_root: str | None = None,
    import_modules: list[dict[str, Any]] | None = None,
    arguments: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = {
        "action": action,
        "module_path": str(module_path),
        "arguments": arguments or {},
        "context": context or {},
        "allowed_read_roots": allowed_read_roots or [],
        "allowed_write_root": allowed_write_root,
        "import_modules": import_modules or [],
    }
    try:
        process = subprocess.Popen(
            [sys.executable, str(runner_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(module_path.parent),
            env=_subprocess_env(allowed_env_keys),
        )
    except OSError as exc:
        return {"ok": False, "error": f"generated tool subprocess failed to start: {exc}"}

    request_bytes = json.dumps(request, ensure_ascii=True).encode("utf-8")
    if process.stdin is not None:
        try:
            process.stdin.write(request_bytes)
        except BrokenPipeError:
            pass
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

    completed = _read_limited_process_output(process, timeout_s=timeout_s, max_output_bytes=max_output_bytes)
    if completed.get("error") is not None:
        return {"ok": False, "error": completed["error"]}

    stdout = bytes(completed["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(completed["stderr"]).decode("utf-8", errors="replace")
    returncode = int(completed["returncode"])
    if returncode != 0:
        stderr_text = stderr.strip()
        return {"ok": False, "error": stderr_text or f"generated tool subprocess exited with {returncode}"}
    output = stdout.strip()
    if not output:
        return {"ok": False, "error": "generated tool subprocess produced no output"}
    last_line = output.splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"generated tool subprocess produced invalid JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "generated tool subprocess produced non-object JSON"}
    return payload


def _read_limited_process_output(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    selector = selectors.DefaultSelector()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    counts = {"stdout": 0, "stderr": 0}
    streams = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    for name, stream in streams.items():
        if stream is not None:
            selector.register(stream, selectors.EVENT_READ, data=name)

    deadline = time.monotonic() + timeout_s
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process(process)
                return {"error": f"generated tool subprocess timed out after {timeout_s}s"}
            events = selector.select(timeout=remaining)
            if not events:
                if process.poll() is not None:
                    continue
                _kill_process(process)
                return {"error": f"generated tool subprocess timed out after {timeout_s}s"}
            for key, _ in events:
                name = str(key.data)
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                counts[name] += len(chunk)
                if counts[name] > max_output_bytes:
                    _kill_process(process)
                    return {"error": f"generated tool {name} exceeded maximum bytes"}
                if name == "stdout":
                    stdout_chunks.append(chunk)
                else:
                    stderr_chunks.append(chunk)
    finally:
        selector.close()

    return {
        "stdout": b"".join(stdout_chunks),
        "stderr": b"".join(stderr_chunks),
        "returncode": process.wait(),
    }


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _subprocess_env(allowed_env_keys: list[str]) -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8"}
    for key in allowed_env_keys:
        if key in os.environ:
            env[key] = os.environ[key]
    return env
