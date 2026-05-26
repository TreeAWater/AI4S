from __future__ import annotations

import json
import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from evolab.contracts.common import Message
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.registries.trajectory import FileTrajectoryRegistry

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_SCRIPT = _ROOT / "dev" / "scripts" / "smoke_native_mem0_api.py"
_SMOKE_SPEC = importlib.util.spec_from_file_location("evolab_dev_smoke_native_mem0_api", _SMOKE_SCRIPT)
assert _SMOKE_SPEC is not None and _SMOKE_SPEC.loader is not None
_SMOKE_MODULE = importlib.util.module_from_spec(_SMOKE_SPEC)
sys.modules[_SMOKE_SPEC.name] = _SMOKE_MODULE
_SMOKE_SPEC.loader.exec_module(_SMOKE_MODULE)
build_smoke_config = _SMOKE_MODULE.build_smoke_config
credential_status = _SMOKE_MODULE.credential_status
verify_lab = _SMOKE_MODULE.verify_lab

_UNSET = object()


def _create_sqlite_store(path: Path, *, memory_records: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS smoke_check (id INTEGER PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                memory_id TEXT PRIMARY KEY,
                deleted_at TEXT
            )
            """
        )
        for index in range(memory_records):
            connection.execute(
                "INSERT INTO memory_records(memory_id, deleted_at) VALUES (?, NULL)",
                (f"memory-{index}",),
            )


def _save_smoke_subagent_run(
    lab_root: Path,
    *,
    agent_update: dict | None = None,
    task_update: dict | None = None,
    agent_bundle: dict | None | object = _UNSET,
    task_bundle: dict | None | object = _UNSET,
) -> None:
    metadata = {
        "agent_memory_update_result": agent_update
        or {
            "status": "updated",
            "state_ref": "method://mem0/17:mem0-agent-memory/5:agent/12:agent:solver/v1",
            "previous_state_ref": "method://mem0/17:mem0-agent-memory/5:agent/12:agent:solver/v0",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "agent",
                "memory_scope_id": "agent:solver",
            },
        },
        "task_memory_update_result": task_update
        or {
            "status": "updated",
            "state_ref": "method://mem0/16:mem0-task-memory/4:task/10:task:smoke/v1",
            "previous_state_ref": "method://mem0/16:mem0-task-memory/4:task/10:task:smoke/v0",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "task",
                "memory_scope_id": "task:smoke",
            },
        },
    }
    if agent_bundle is _UNSET:
        metadata["agent_memory_bundle"] = {
            "backend_id": "mem0-agent-memory",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "agent",
                "memory_scope_id": "agent:solver",
            },
        }
    elif agent_bundle is not None:
        metadata["agent_memory_bundle"] = agent_bundle
    if task_bundle is _UNSET:
        metadata["task_memory_bundle"] = {
            "backend_id": "mem0-task-memory",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "task",
                "memory_scope_id": "task:smoke",
            },
        }
    elif task_bundle is not None:
        metadata["task_memory_bundle"] = task_bundle

    trajectory_registry = FileTrajectoryRegistry(lab_root / "registries" / "trajectory")
    trajectory_registry.save_subagent_run(
        SubagentRunRecord(
            run_ref="subagent-1",
            task_id="smoke",
            task_origin=TaskOrigin.HUMAN,
            task_purpose=TaskPurpose.REGRESSION,
            stage_index=0,
            role="solver",
            instruction="Smoke.",
            retrieval_request=RetrievalRequest(task_id="smoke", role="solver", query="memory"),
            memory_bundle=MemoryBundle(backend_id="combined-memory"),
            skill_bundle=SkillBundle(backend_id="skill-local"),
            prompt_messages=[Message(role="user", content="Smoke.")],
            llm_backend_id="llm-api",
            metadata=metadata,
        )
    )


def _create_valid_smoke_lab(lab_root: Path) -> None:
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(lab_root)


def test_native_mem0_api_smoke_credentials_require_chat_and_embedding_keys(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "AIGOCODE_GPT_API=openai-responses",
                "AIGOCODE_GPT_API_KEY=chat-secret",
                "AIGOCODE_GPT_MODEL=gpt-test",
                "MEMORY_EMBEDDING_API=openai-embeddings",
                "MEMORY_EMBEDDING_MODEL=text-embedding-test",
            ]
        ),
        encoding="utf-8",
    )

    status = credential_status(
        dotenv_path,
        chat_env_ref="aigocode-gpt",
        embedding_env_ref="memory-embedding",
        chat_model=None,
        embedding_model=None,
    )

    assert not status.ready
    assert status.missing == ["MEMORY_EMBEDDING_API_KEY"]


def test_native_mem0_api_smoke_generates_real_service_memory_config(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    lab_root = tmp_path / "lab"

    config = build_smoke_config(
        lab_root=lab_root,
        dotenv_path=dotenv_path,
        chat_env_ref="aigocode-gpt",
        embedding_env_ref="memory-embedding",
        chat_model="gpt-test",
        embedding_model="text-embedding-test",
    )

    assert config["dotenv_path"] == str(dotenv_path)
    assert config["backends"]["llm"]["mem0-memory-llm"] == {
        "type": "api",
        "env_ref": "aigocode-gpt",
        "model": "gpt-test",
    }
    assert config["backends"]["embedding"]["mem0-memory-embedding"] == {
        "type": "api",
        "env_ref": "memory-embedding",
        "model": "text-embedding-test",
    }
    assert config["backends"]["memory"]["mem0-agent-memory"]["llm_backend"] == "mem0-memory-llm"
    assert config["backends"]["memory"]["mem0-agent-memory"]["embedding_backend"] == "mem0-memory-embedding"
    assert "client" not in str(config)
    assert "in_memory" not in str(config)


def test_native_mem0_api_smoke_script_runs_from_file_path(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "dev/scripts/smoke_native_mem0_api.py",
            "--dotenv",
            str(dotenv_path),
            "--lab-root",
            str(tmp_path / "lab"),
        ],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(_ROOT / "src")},
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "skipped"


def test_native_mem0_api_smoke_rejects_failed_memory_updates(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    failed_update = {
        "status": "failed",
        "state_ref": "method://mem0/10:mem0-test/5:agent/12:agent:solver/v0",
        "previous_state_ref": "method://mem0/10:mem0-test/5:agent/12:agent:solver/v0",
        "metadata": {
            "memory_method": "mem0",
            "memory_scope": "agent",
            "memory_scope_id": "agent:solver",
            "error_type": "api_error",
        },
    }
    _save_smoke_subagent_run(
        lab_root,
        agent_update=failed_update,
        task_update={
            **failed_update,
            "status": "skipped",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "task",
                "memory_scope_id": "task:smoke",
            },
        },
    )

    with pytest.raises(RuntimeError, match="unexpected agent_memory_update_result status 'failed'"):
        verify_lab(lab_root)


def test_native_mem0_api_smoke_requires_both_expected_memory_stores(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _save_smoke_subagent_run(lab_root)

    with pytest.raises(RuntimeError, match="missing SQLite memory store registries/memory/mem0-task.sqlite"):
        verify_lab(lab_root)


def test_native_mem0_api_smoke_rejects_non_sqlite_memory_store(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    placeholder = lab_root / "registries" / "memory" / "mem0-task.sqlite"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("sqlite placeholder", encoding="utf-8")
    _save_smoke_subagent_run(lab_root)

    with pytest.raises(RuntimeError, match="not a readable SQLite database"):
        verify_lab(lab_root)


def test_native_mem0_api_smoke_requires_stored_memory_records(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite", memory_records=0)
    _save_smoke_subagent_run(lab_root)

    with pytest.raises(RuntimeError, match="mem0-task.sqlite has no active memory_records"):
        verify_lab(lab_root)


def test_native_mem0_api_smoke_rejects_wrong_scope_metadata(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(
        lab_root,
        task_update={
            "status": "updated",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "agent",
                "memory_scope_id": "agent:solver",
            },
        },
    )

    with pytest.raises(RuntimeError, match="task_memory_update_result metadata memory_scope expected 'task'"):
        verify_lab(lab_root)


@pytest.mark.parametrize(
    "update_kwargs, expected_message",
    [
        (
            {
                "agent_update": {
                    "status": "updated",
                    "state_ref": "method://mem0/11:wrong-agent/5:agent/12:agent:solver/v1",
                    "previous_state_ref": "method://mem0/17:mem0-agent-memory/5:agent/12:agent:solver/v0",
                    "metadata": {
                        "memory_method": "mem0",
                        "memory_scope": "agent",
                        "memory_scope_id": "agent:solver",
                    },
                }
            },
            "agent_memory_update_result state_ref backend_id expected 'mem0-agent-memory'",
        ),
        (
            {
                "task_update": {
                    "status": "updated",
                    "state_ref": "method://mem0/10:wrong-task/4:task/10:task:smoke/v1",
                    "previous_state_ref": "method://mem0/16:mem0-task-memory/4:task/10:task:smoke/v0",
                    "metadata": {
                        "memory_method": "mem0",
                        "memory_scope": "task",
                        "memory_scope_id": "task:smoke",
                    },
                }
            },
            "task_memory_update_result state_ref backend_id expected 'mem0-task-memory'",
        ),
    ],
)
def test_native_mem0_api_smoke_rejects_wrong_update_state_ref_backend(
    tmp_path: Path,
    update_kwargs: dict,
    expected_message: str,
):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(lab_root, **update_kwargs)

    with pytest.raises(RuntimeError, match=expected_message):
        verify_lab(lab_root)


def test_native_mem0_api_smoke_rejects_wrong_skipped_update_previous_state_ref_backend(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(
        lab_root,
        agent_update={
            "status": "skipped",
            "previous_state_ref": "method://mem0/11:wrong-agent/5:agent/12:agent:solver/v1",
            "metadata": {
                "memory_method": "mem0",
                "memory_scope": "agent",
                "memory_scope_id": "agent:solver",
            },
        },
    )

    with pytest.raises(
        RuntimeError,
        match="agent_memory_update_result previous_state_ref backend_id expected 'mem0-agent-memory'",
    ):
        verify_lab(lab_root)


@pytest.mark.parametrize(
    "missing_bundle_kwargs, expected_message",
    [
        ({"agent_bundle": None}, "trajectory subagent-1 missing agent_memory_bundle"),
        ({"task_bundle": None}, "trajectory subagent-1 missing task_memory_bundle"),
    ],
)
def test_native_mem0_api_smoke_requires_agent_and_task_memory_bundles(
    tmp_path: Path,
    missing_bundle_kwargs: dict,
    expected_message: str,
):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(lab_root, **missing_bundle_kwargs)

    with pytest.raises(RuntimeError, match=expected_message):
        verify_lab(lab_root)


@pytest.mark.parametrize(
    "agent_bundle, expected_message",
    [
        (
            {"backend_id": "mem0-agent-memory"},
            "trajectory subagent-1 agent_memory_bundle missing diagnostics metadata",
        ),
        (
            {
                "backend_id": "mem0-agent-memory",
                "metadata": {"memory_scope": "agent", "memory_scope_id": "agent:solver"},
            },
            "trajectory subagent-1 agent_memory_bundle missing memory_method='mem0'",
        ),
        (
            {
                "backend_id": "mem0-agent-memory",
                "metadata": {
                    "memory_method": "legacy",
                    "memory_scope": "agent",
                    "memory_scope_id": "agent:solver",
                },
            },
            "trajectory subagent-1 agent_memory_bundle missing memory_method='mem0'",
        ),
        (
            {
                "backend_id": "mem0-agent-memory",
                "metadata": {
                    "memory_method": "mem0",
                    "memory_scope": "task",
                    "memory_scope_id": "agent:solver",
                },
            },
            "agent_memory_bundle metadata memory_scope expected 'agent'",
        ),
        (
            {
                "backend_id": "mem0-agent-memory",
                "metadata": {
                    "memory_method": "mem0",
                    "memory_scope": "agent",
                    "memory_scope_id": "task:smoke",
                },
            },
            "agent_memory_bundle metadata memory_scope_id expected 'agent:solver'",
        ),
    ],
)
def test_native_mem0_api_smoke_rejects_missing_or_wrong_bundle_metadata(
    tmp_path: Path,
    agent_bundle: dict,
    expected_message: str,
):
    lab_root = tmp_path / "lab"
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-agent.sqlite")
    _create_sqlite_store(lab_root / "registries" / "memory" / "mem0-task.sqlite")
    _save_smoke_subagent_run(lab_root, agent_bundle=agent_bundle)

    with pytest.raises(RuntimeError, match=expected_message):
        verify_lab(lab_root)


@pytest.mark.parametrize(
    "relative_path, text",
    [
        ("copied.yaml", "backends:\n  memory:\n    mem0:\n      client_type: in_memory\n"),
        ("compact.json", '{"backends":{"memory":{"mem0":{"client_type":"in_memory"}}}}\n'),
        ("nested.yaml", "outer:\n  client:\n    kind: ignored\n  memory:\n    client: 'in_memory'\n"),
    ],
)
def test_native_mem0_api_smoke_detects_copied_legacy_client_variants(
    tmp_path: Path,
    relative_path: str,
    text: str,
):
    lab_root = tmp_path / "lab"
    _create_valid_smoke_lab(lab_root)
    config_path = lab_root / "configs" / relative_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="copied config contains legacy in_memory Mem0 client"):
        verify_lab(lab_root)
