from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.generated_tools import (
    GeneratedToolCapabilityGrant,
    GeneratedToolFile,
    GeneratedToolPackage,
    GeneratedToolRegistration,
    GeneratedToolSmokeTest,
    GeneratedToolValidationResult,
    TaskEffectiveToolCatalog,
)
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.contracts.repair import RepairPlan
from evolab.contracts.tools import ToolSpec
from evolab.lab.layout import LabLayout
from evolab.runtime.generated_tools import GeneratedToolBuilder, GeneratedToolRuntime, build_effective_tool_catalog
from evolab.tools.runtime import ToolRegistry, ToolRuntime


VALID_TOOL_CODE = """\
TOOL_SPEC = {
    "name": "extract_rows",
    "description": "Extract rows for this task.",
    "parameters_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
    "metadata": {"generated_tool": True}
}

def run(arguments, context):
    return {
        "status": "ok",
        "content": "value=" + str(arguments.get("value")),
        "metadata": {"task_id": context.get("task_id")}
    }
"""


def _package(code: str = VALID_TOOL_CODE, *, tool_name: str = "extract_rows") -> GeneratedToolPackage:
    return GeneratedToolPackage(
        tool_name=tool_name,
        reason="Need generated behavior.",
        manifest={
            "description": "Extract rows for this task.",
            "parameters_schema": {"type": "object"},
        },
        files=[GeneratedToolFile(path="tool.py", content=code)],
        smoke_tests=[GeneratedToolSmokeTest(name="basic", arguments={"value": "alpha"})],
    )


class ScriptedBuilderLLM:
    def __init__(self, response: LLMRuntimeResponse) -> None:
        self.response = response
        self.calls: list[tuple[list[Message], list[dict], object]] = []

    def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: object) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return self.response


def test_generated_tool_package_validates_primary_module_and_files():
    package = GeneratedToolPackage(
        tool_name="extract_rows",
        reason="Need task-specific row extraction.",
        manifest={
            "description": "Extract task rows.",
            "parameters_schema": {"type": "object"},
        },
        files=[
            GeneratedToolFile(
                path="tool.py",
                content="TOOL_SPEC = {'name': 'extract_rows', 'description': 'Extract.', 'parameters_schema': {'type': 'object'}}\n\ndef run(arguments, context):\n    return 'ok'\n",
            )
        ],
        smoke_tests=[GeneratedToolSmokeTest(name="basic", arguments={})],
    )

    assert package.primary_module == "tool.py"
    assert package.files[0].path == "tool.py"


def test_generated_tool_builder_parses_llm_package_response_and_records_stage(tmp_path: Path):
    payload = _package().model_dump(mode="json")
    llm = ScriptedBuilderLLM(
        LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(payload)))
    )

    package = GeneratedToolBuilder(llm).build(
        task_id="task-1",
        task_goal="Extract rows.",
        run_ref="run-1",
        built_in_tool_specs=[ToolSpec(name="read_text", description="Read.", parameters_schema={"type": "object"})],
        generated_tool_specs=[],
        role_pool_templates=[{"name": "solver", "allowed_tools": ["read_text"]}],
        artifact_root=tmp_path,
        capability_grant=GeneratedToolCapabilityGrant(),
        requested_tool_name="extract_rows",
    )

    assert package.tool_name == "extract_rows"
    assert package.files[0].content == VALID_TOOL_CODE
    assert llm.calls
    generation_config = llm.calls[0][2]
    assert generation_config.model == ""
    assert generation_config.temperature == 0
    assert generation_config.max_output_tokens == 4096
    assert generation_config.metadata["runtime_stage"] == "generated_tool_builder"
    prompt_payload = json.loads(llm.calls[0][0][-1].content)
    assert prompt_payload["task_id"] == "task-1"
    assert prompt_payload["requested_tool_name"] == "extract_rows"


def test_generated_tool_package_rejects_path_escape():
    with pytest.raises(ValueError, match="must be a relative path inside the package"):
        GeneratedToolPackage(
            tool_name="bad",
            reason="path escape",
            manifest={"description": "Bad.", "parameters_schema": {"type": "object"}},
            files=[GeneratedToolFile(path="../bad.py", content="")],
        )


def test_runtime_policy_exposes_generated_tool_limits():
    policy = RuntimePolicy()

    assert policy.allow_runtime_tool_creation is True
    assert policy.max_generated_tools_per_task == 8
    assert policy.max_generated_tool_files == 8
    assert policy.max_generated_tool_source_bytes == 64_000
    assert policy.generated_tool_validation_timeout_s == 5
    assert policy.generated_tool_execution_timeout_s == 10
    assert policy.generated_tool_max_output_bytes == 256_000
    assert policy.generated_tool_allow_network is False
    assert policy.generated_tool_allow_subprocess is False
    assert policy.generated_tool_allowed_env_keys == []


def test_repair_plan_accepts_typed_generated_runtime_tool():
    package = GeneratedToolPackage(
        tool_name="fix_rows",
        reason="The original call requested a missing task-specific tool.",
        manifest={"description": "Fix rows.", "parameters_schema": {"type": "object"}},
        files=[GeneratedToolFile(path="tool.py", content="TOOL_SPEC = {}\ndef run(arguments, context):\n    return 'ok'\n")],
    )

    plan = RepairPlan(
        repair_id="repair-1",
        failure_id="failure-1",
        diagnosis="missing_tool",
        repair_action="create_runtime_tool",
        rationale="Need generated tool.",
        new_runtime_tool=package,
    )

    assert plan.new_runtime_tool.tool_name == "fix_rows"


def test_effective_catalog_combines_builtin_and_generated_specs():
    builtin = ToolSpec(name="read_text", description="Read text.", parameters_schema={"type": "object"})
    generated = ToolSpec(
        name="generated_task_extract_rows",
        description="Generated rows.",
        parameters_schema={"type": "object"},
        metadata={"generated_tool": True},
    )

    catalog = TaskEffectiveToolCatalog(
        task_id="task-1",
        builtin_allowed_tool_names=["read_text"],
        generated_tool_names=["generated_task_extract_rows"],
        tool_specs_by_name={"read_text": builtin, "generated_task_extract_rows": generated},
        provenance_by_name={"generated_task_extract_rows": {"code_hash": "abc"}},
    )

    assert catalog.effective_allowed_tool_names == ["read_text", "generated_task_extract_rows"]
    assert catalog.tool_specs_by_name["generated_task_extract_rows"].metadata["generated_tool"] is True


def test_generated_tool_registration_records_validation_and_grants():
    grant = GeneratedToolCapabilityGrant(
        allowed_read_roots=["/tmp/input"],
        allowed_write_root="/tmp/lab/artifacts/generated",
    )
    registration = GeneratedToolRegistration(
        requested_tool_name="extract_rows",
        registered_tool_name="gt_task_extract_rows",
        task_id="task-1",
        run_ref="run-1",
        tool_spec=ToolSpec(name="gt_task_extract_rows", description="Generated.", parameters_schema={"type": "object"}),
        module_path="/tmp/lab/artifacts/generated/tool.py",
        code_hash="abc",
        validation=GeneratedToolValidationResult(valid=True, status="passed"),
        capability_grant=grant,
    )

    assert registration.validation.valid is True
    assert registration.capability_grant.allow_network is False


def test_generated_runtime_persists_validates_registers_and_executes(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        artifact_root=tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(),
    )

    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    assert registration.registered_tool_name.startswith("gt_task_1_run_1_extract_rows")
    assert Path(registration.module_path).exists()
    assert registration.code_hash

    bundle = tool_runtime.prepare(
        required_tools=[registration.registered_tool_name],
        allowed_tools=[registration.registered_tool_name],
        policy=RuntimePolicy(),
    )
    assert [spec.name for spec in bundle.tool_specs] == [registration.registered_tool_name]

    result = tool_runtime.execute_tool_name(
        "call-1",
        registration.registered_tool_name,
        {"value": "beta"},
    )
    assert result.status == "ok"
    assert result.content == "value=beta"
    assert result.metadata["generated_tool"]["code_hash"] == registration.code_hash


def test_generated_runtime_writes_packages_under_dot_evolab_layout(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(layout.state_root, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    module_path = Path(registration.module_path)
    assert module_path.relative_to(layout.generated_tools_dir)
    assert not (layout.root / "generated_tools").exists()


def test_generated_runtime_rejects_syntax_error_without_registration(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package("def broken(:\n    pass\n"),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("invalid Python syntax" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_forbidden_static_capabilities(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(
            "import subprocess\n"
            "TOOL_SPEC = {'name': 'x', 'description': 'x', 'parameters_schema': {'type': 'object'}}\n"
            "def run(arguments, context):\n"
            "    return 'x'\n"
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("subprocess" in error for error in registration.validation.errors)


def test_generated_runtime_policy_subprocess_allow_does_not_override_default_grant(tmp_path: Path):
    code = """\
import subprocess
import sys
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    subprocess.run([sys.executable, "-c", "pass"], check=True)
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(generated_tool_allow_subprocess=True),
    )

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("subprocess" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_policy_allowed_import_does_not_override_package_grant(tmp_path: Path):
    code = """\
import math
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    return str(math.sqrt(4))
"""
    policy = RuntimePolicy(generated_tool_allowed_imports=["math"])

    default_tool_runtime = ToolRuntime(ToolRegistry())
    default_tool_runtime.activate_generated_tool_scope("task-1")
    default_generated = GeneratedToolRuntime(tmp_path, tool_runtime=default_tool_runtime, policy=policy)

    default_registration = default_generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert default_registration.validation.valid is False
    assert any("allowed_imports grant" in error for error in default_registration.validation.errors)
    assert default_tool_runtime.generated_tool_names() == []

    granted_tool_runtime = ToolRuntime(ToolRegistry())
    granted_tool_runtime.activate_generated_tool_scope("task-1")
    granted_generated = GeneratedToolRuntime(tmp_path, tool_runtime=granted_tool_runtime, policy=policy)

    granted_registration = granted_generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_imports=["math"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert granted_registration.validation.valid is True
    result = granted_tool_runtime.execute_tool_name(
        "call-1",
        granted_registration.registered_tool_name,
        {},
    )
    assert result.status == "ok"
    assert result.content == "2.0"


def test_generated_runtime_intersects_env_keys_with_package_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SECRET_TOKEN", "task-secret")
    code = """\
import os
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    return os.environ.get("SECRET_TOKEN", "missing")
"""
    policy = RuntimePolicy(generated_tool_allowed_env_keys=["SECRET_TOKEN"])

    default_tool_runtime = ToolRuntime(ToolRegistry())
    default_tool_runtime.activate_generated_tool_scope("task-1")
    default_generated = GeneratedToolRuntime(tmp_path, tool_runtime=default_tool_runtime, policy=policy)

    default_registration = default_generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert default_registration.validation.valid is False
    assert any("'os'" in error for error in default_registration.validation.errors)
    assert default_tool_runtime.generated_tool_names() == []

    granted_tool_runtime = ToolRuntime(ToolRegistry())
    granted_tool_runtime.activate_generated_tool_scope("task-1")
    granted_generated = GeneratedToolRuntime(tmp_path, tool_runtime=granted_tool_runtime, policy=policy)

    granted_registration = granted_generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_env_keys=["SECRET_TOKEN"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert granted_registration.validation.valid is True
    result = granted_tool_runtime.execute_tool_name(
        "call-1",
        granted_registration.registered_tool_name,
        {},
    )
    assert result.status == "ok"
    assert result.content == "task-secret"


def test_generated_runtime_rejects_os_process_apis_when_only_env_is_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PATH", "/usr/bin")
    code = """\
from os import system as run_command
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    run_command("true")
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(generated_tool_allowed_env_keys=["PATH"]),
    )

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_env_keys=["PATH"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("os subprocess API" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_smoke_test_failure_rejects_tool(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    raise RuntimeError("boom")
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("smoke test" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_missing_run_without_smoke_tests(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code).model_copy(update={"smoke_tests": []}),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("callable run" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_invalid_manifest_parameters_schema(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package().model_copy(
            update={"manifest": {"description": "Extract rows.", "parameters_schema": "not-a-dict"}}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("manifest parameters_schema" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_manifest_array_parameters_schema(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package().model_copy(
            update={"manifest": {"description": "Extract rows.", "parameters_schema": {"type": "array"}}}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("manifest parameters_schema" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_loaded_spec_invalid_parameters_schema(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": "not-a-dict"}
def run(arguments, context):
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any(
        "TOOL_SPEC is invalid" in error and "parameters_schema" in error
        for error in registration.validation.errors
    )
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_loaded_spec_array_parameters_schema(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "array"}}
def run(arguments, context):
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any(
        "TOOL_SPEC is invalid" in error and "parameters_schema" in error
        for error in registration.validation.errors
    )
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_private_reasoning_metadata(tmp_path: Path):
    code = """\
TOOL_SPEC = {
    "name": "extract_rows",
    "description": "Extract.",
    "parameters_schema": {"type": "object"},
    "metadata": {"reasoning": "private notes"},
}
def run(arguments, context):
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    manifest_registration = generated.register_package(
        package=_package().model_copy(
            update={
                "manifest": {
                    "description": "Extract rows.",
                    "parameters_schema": {"type": "object"},
                    "metadata": {"chain_of_thought": "private notes"},
                }
            }
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )
    loaded_registration = generated.register_package(
        package=_package(code, tool_name="extract_rows_2"),
        task_id="task-1",
        run_ref="run-2",
        context={"task_id": "task-1"},
    )

    assert manifest_registration.validation.valid is False
    assert loaded_registration.validation.valid is False
    assert any("private reasoning metadata" in error for error in manifest_registration.validation.errors)
    assert any("private reasoning metadata" in error for error in loaded_registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_does_not_allow_package_import_grants_to_override_policy(tmp_path: Path):
    code = """\
import subprocess
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_imports=["subprocess"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("subprocess" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_dynamic_import_subprocess_bypass(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    __import__("subprocess").run(["true"])
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("__import__" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_package_grants_disabled_by_policy(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package().model_copy(
            update={
                "capability_grant": GeneratedToolCapabilityGrant(
                    allow_network=True,
                    allow_subprocess=True,
                    allowed_env_keys=["SECRET_TOKEN"],
                    allowed_imports=["subprocess"],
                )
            }
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("network" in error for error in registration.validation.errors)
    assert any("subprocess" in error for error in registration.validation.errors)
    assert any("environment" in error for error in registration.validation.errors)
    assert any("allowed imports" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_static_writes_without_write_grant(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    open("out.txt", "w").write("x")
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_write_root=None)}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("write grant" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_open_read_outside_read_roots(tmp_path: Path):
    outside_path = tmp_path / "outside-secret.txt"
    outside_path.write_text("secret", encoding="utf-8")
    code = f"""\
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    return open({str(outside_path)!r}).read()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("outside allowed_read_roots" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_path_read_outside_read_roots(tmp_path: Path):
    outside_path = tmp_path / "outside-secret.txt"
    outside_path.write_text("secret", encoding="utf-8")
    code = f"""\
from pathlib import Path
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    return Path({str(outside_path)!r}).read_text()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("outside allowed_read_roots" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_io_open_read_outside_read_roots(tmp_path: Path):
    outside_path = tmp_path / "outside-secret.txt"
    outside_path.write_text("secret", encoding="utf-8")
    code = f"""\
import io
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    return io.open({str(outside_path)!r}).read()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("outside allowed_read_roots" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_runner_original_open_bypass(tmp_path: Path):
    outside_path = tmp_path / "outside-secret.txt"
    outside_path.write_text("secret", encoding="utf-8")
    code = f"""\
import sys
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    original_open = sys.modules["__main__"].__dict__.get("_ORIGINAL_OPEN")
    if original_open is None:
        return "blocked"
    return original_open({str(outside_path)!r}).read()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "blocked"


def test_generated_runtime_allows_reads_inside_granted_read_roots(tmp_path: Path):
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source_path = input_root / "source.txt"
    source_path.write_text("allowed", encoding="utf-8")
    code = f"""\
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    return open({str(source_path)!r}).read()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={
                "capability_grant": GeneratedToolCapabilityGrant(
                    allowed_read_roots=[str(input_root)],
                    allowed_write_root=str(tmp_path / "generated-output"),
                )
            }
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "allowed"


def test_generated_runtime_allows_granted_stdlib_imports_with_read_guard(tmp_path: Path):
    code = """\
import csv
import io
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    rows = list(csv.reader(io.StringIO("a,b\\n1,2")))
    return rows[1][1]
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(generated_tool_allowed_imports=["csv"]),
    )

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_imports=["csv"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "2"


def test_generated_runtime_allows_granted_stdlib_from_import_submodule_with_read_guard(tmp_path: Path):
    code = """\
from xml.etree import ElementTree as ET
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    root = ET.fromstring("<root><value>ok</value></root>")
    return root.findtext("value")
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(generated_tool_allowed_imports=["xml"]),
    )

    registration = generated.register_package(
        package=_package(code).model_copy(
            update={"capability_grant": GeneratedToolCapabilityGrant(allowed_imports=["xml"])}
        ),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "ok"


def test_generated_runtime_preload_does_not_execute_local_stdlib_shadow_before_guard(tmp_path: Path):
    outside_path = tmp_path / "preload-shadow.txt"
    tool_code = """\
from email import message
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    return message.Message().__class__.__name__
"""
    shadow_code = f"""\
_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open
_open({str(outside_path)!r}, "w").write("shadowed")
"""
    package = GeneratedToolPackage(
        tool_name="extract_rows",
        reason="Need generated behavior.",
        manifest={"description": "Extract.", "parameters_schema": {"type": "object"}},
        files=[
            GeneratedToolFile(path="tool.py", content=tool_code),
            GeneratedToolFile(path="quopri.py", content=shadow_code),
        ],
        smoke_tests=[GeneratedToolSmokeTest(name="basic")],
        capability_grant=GeneratedToolCapabilityGrant(allowed_imports=["email"]),
    )
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(generated_tool_allowed_imports=["email"]),
    )

    registration = generated.register_package(
        package=package,
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert outside_path.exists() is False
    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "Message"


def test_generated_runtime_allows_local_helper_import_after_guard(tmp_path: Path):
    tool_code = """\
import helper
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    return helper.value()
"""
    helper_code = """\
def value():
    return "helper-ok"
"""
    package = GeneratedToolPackage(
        tool_name="extract_rows",
        reason="Need generated behavior.",
        manifest={"description": "Extract.", "parameters_schema": {"type": "object"}},
        files=[
            GeneratedToolFile(path="tool.py", content=tool_code),
            GeneratedToolFile(path="helper.py", content=helper_code),
        ],
        smoke_tests=[GeneratedToolSmokeTest(name="basic")],
    )
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=package,
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    result = tool_runtime.execute_tool_name(
        call_id="call-1",
        name=registration.registered_tool_name,
        arguments={},
    )
    assert result.status == "ok"
    assert result.content == "helper-ok"


def test_generated_runtime_rejects_builtins_open_write_alias_bypass(tmp_path: Path):
    outside_path = tmp_path / "outside.txt"
    code = f"""\
from builtins import open as o
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    o({str(outside_path)!r}, "w").write("x")
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("builtins" in error or "open()" in error for error in registration.validation.errors)
    assert outside_path.exists() is False
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_smoke_test_writes_outside_generated_tool_dir(tmp_path: Path):
    outside_path = tmp_path / "outside.txt"
    code = f"""\
from pathlib import Path
TOOL_SPEC = {{"name": "extract_rows", "description": "Extract.", "parameters_schema": {{"type": "object"}}}}
def run(arguments, context):
    Path({str(outside_path)!r}).write_text("x")
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("outside allowed_write_root" in error for error in registration.validation.errors)
    assert outside_path.exists() is False
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_enforces_task_scoped_count_across_instances(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    policy = RuntimePolicy(max_generated_tools_per_task=1)
    first_runtime = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=policy)
    second_runtime = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=policy)

    first = first_runtime.register_package(
        package=_package(tool_name="extract_rows"),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )
    second = second_runtime.register_package(
        package=_package(tool_name="extract_rows_2"),
        task_id="task-1",
        run_ref="run-2",
        context={"task_id": "task-1"},
    )

    assert first.validation.valid is True
    assert second.validation.valid is False
    assert any("maximum generated tools per task exceeded" in error for error in second.validation.errors)
    assert tool_runtime.generated_tool_names() == [first.registered_tool_name]


def test_generated_runtime_avoids_builtin_name_collisions(tmp_path: Path):
    registry = ToolRegistry()
    builtin_name = "gt_task_1_run_1_extract_rows"
    registry.register(
        ToolSpec(name=builtin_name, description="Builtin.", parameters_schema={"type": "object"}),
        lambda arguments: "builtin",
    )
    tool_runtime = ToolRuntime(registry)
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    assert registration.registered_tool_name == f"{builtin_name}_2"
    assert tool_runtime.generated_tool_names() == [registration.registered_tool_name]


def test_generated_runtime_returns_rejected_registration_when_collision_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = ToolRegistry()
    builtin_name = "gt_task_1_run_1_extract_rows"
    registry.register(
        ToolSpec(name=builtin_name, description="Builtin.", parameters_schema={"type": "object"}),
        lambda arguments: "builtin",
    )
    tool_runtime = ToolRuntime(registry)
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())
    monkeypatch.setattr("evolab.runtime.generated_tools._unique_registered_tool_name", lambda base_name, existing_names: base_name)

    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("would replace a built-in tool" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_enforces_stderr_output_limit_during_validation(tmp_path: Path):
    code = """\
import sys
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    while True:
        sys.stderr.write("x" * 2048)
        sys.stderr.flush()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    policy = RuntimePolicy(generated_tool_max_output_bytes=1024, generated_tool_validation_timeout_s=1)
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=policy)

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("stderr exceeded maximum bytes" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_enforces_stderr_output_limit_during_execution(tmp_path: Path):
    code = """\
import sys
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    sys.stderr.write("x" * 2048)
    return "ok"
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    policy = RuntimePolicy(generated_tool_max_output_bytes=1024)
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=policy)

    registration = generated.register_package(
        package=_package(code).model_copy(update={"smoke_tests": []}),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )
    result = tool_runtime.execute_tool_name("call-1", registration.registered_tool_name, {})

    assert registration.validation.valid is True
    assert result.status == "error"
    assert "stderr exceeded maximum bytes" in result.content


def test_generated_runtime_enforces_stdout_output_limit_during_validation(tmp_path: Path):
    code = """\
import sys
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    while True:
        sys.stdout.write("x" * 2048)
        sys.stdout.flush()
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    policy = RuntimePolicy(generated_tool_max_output_bytes=1024, generated_tool_validation_timeout_s=1)
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=policy)

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("stdout exceeded maximum bytes" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_build_effective_catalog_includes_registered_generated_tools(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="read_text", description="Read text.", parameters_schema={"type": "object"}),
        lambda arguments: "read",
    )
    tool_runtime = ToolRuntime(registry)
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())
    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    catalog = build_effective_tool_catalog(
        task_id="task-1",
        builtin_allowed_tool_names=["read_text"],
        tool_runtime=tool_runtime,
    )

    assert catalog.effective_allowed_tool_names == ["read_text", registration.registered_tool_name]
    assert catalog.tool_specs_by_name["read_text"].description == "Read text."
    assert catalog.provenance_by_name[registration.registered_tool_name]["code_hash"] == registration.code_hash
