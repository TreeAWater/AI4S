from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolab.cli import run_clean_demo
from evolab.config.env import env_ref_prefix, lookup_env_value, parse_dotenv
from evolab.registries.trajectory import FileTrajectoryRegistry


@dataclass(frozen=True)
class CredentialStatus:
    ready: bool
    missing: list[str]
    dotenv_path: Path


def credential_status(
    dotenv_path: Path,
    *,
    chat_env_ref: str,
    embedding_env_ref: str,
    chat_model: str | None,
    embedding_model: str | None,
) -> CredentialStatus:
    values = parse_dotenv(dotenv_path)
    missing: list[str] = []
    missing.extend(
        _missing_api_backend_env(
            values,
            env_ref=chat_env_ref,
            explicit_model=chat_model,
            label="chat LLM",
        )
    )
    missing.extend(
        _missing_api_backend_env(
            values,
            env_ref=embedding_env_ref,
            explicit_model=embedding_model,
            label="embedding",
        )
    )
    return CredentialStatus(ready=not missing, missing=missing, dotenv_path=dotenv_path)


def build_smoke_config(
    *,
    lab_root: Path,
    dotenv_path: Path,
    chat_env_ref: str,
    embedding_env_ref: str,
    chat_model: str,
    embedding_model: str,
) -> dict[str, Any]:
    task_id = "native-mem0-api-smoke"
    return {
        "lab_root": str(lab_root),
        "dotenv_path": str(dotenv_path),
        "task": {
            "task_id": task_id,
            "origin": "human",
            "purpose": "science",
            "goal": "Run native mem0 real API smoke for scoped memory updates.",
            "task_config_ref": "generated-native-mem0-api-smoke.json",
        },
        "task_config": {
            "task_id": task_id,
            "goal": "Run native mem0 real API smoke for scoped memory updates.",
            "task_memory_backend": {"backend_id": "mem0-task-memory"},
            "roles": {
                "solver": {
                    "name": "solver",
                    "system_prompt": "Return the configured smoke result.",
                    "llm_backend": {"backend_id": "fake-task-llm"},
                    "agent_memory_backend": {"backend_id": "mem0-agent-memory"},
                    "allowed_tools": [],
                }
            },
            "runtime_policy": {
                "max_tool_steps": 0,
                "allow_human_tools": False,
                "enable_workflow_planning": False,
            },
        },
        "backends": {
            "llm": {
                "fake-task-llm": {
                    "type": "fake",
                    "responses": [
                        {
                            "action": {
                                "action": "final_answer",
                                "content": (
                                    "Remember this native mem0 real API smoke fact: "
                                    "the solver completed scoped agent and task memory updates."
                                ),
                            }
                        }
                    ],
                },
                "mem0-memory-llm": {
                    "type": "api",
                    "env_ref": chat_env_ref,
                    "model": chat_model,
                },
            },
            "embedding": {
                "mem0-memory-embedding": {
                    "type": "api",
                    "env_ref": embedding_env_ref,
                    "model": embedding_model,
                }
            },
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "store_path": "registries/memory/mem0-agent.sqlite",
                    "llm_backend": "mem0-memory-llm",
                    "embedding_backend": "mem0-memory-embedding",
                },
                "mem0-task-memory": {
                    "type": "method",
                    "method": "mem0",
                    "store_path": "registries/memory/mem0-task.sqlite",
                    "llm_backend": "mem0-memory-llm",
                    "embedding_backend": "mem0-memory-embedding",
                },
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }


def run_smoke(
    *,
    config_path: Path,
    lab_root: Path,
) -> dict[str, Any]:
    result = run_clean_demo(config_path, lab_root)
    checks = verify_lab(lab_root)
    return {"status": "ran", "task_result": result, "checks": checks}


def verify_lab(lab_root: Path) -> dict[str, Any]:
    sqlite_paths, memory_record_counts = _validate_expected_sqlite_stores(lab_root)

    registry = FileTrajectoryRegistry(lab_root / "registries" / "trajectory")
    runs = registry.list_subagent_runs()
    if not runs:
        raise RuntimeError("no subagent trajectories recorded")

    update_statuses: dict[str, str] = {}
    for run in runs:
        agent_status = _validate_memory_update_result(
            run_ref=run.run_ref,
            key="agent_memory_update_result",
            update=run.metadata.get("agent_memory_update_result"),
            expected_backend_id="mem0-agent-memory",
            expected_scope="agent",
            expected_scope_id=f"agent:{run.role}",
        )
        task_status = _validate_memory_update_result(
            run_ref=run.run_ref,
            key="task_memory_update_result",
            update=run.metadata.get("task_memory_update_result"),
            expected_backend_id="mem0-task-memory",
            expected_scope="task",
            expected_scope_id=f"task:{run.task_id}",
        )
        _validate_memory_bundle(
            run_ref=run.run_ref,
            key="agent_memory_bundle",
            bundle=run.metadata.get("agent_memory_bundle"),
            expected_backend_id="mem0-agent-memory",
            expected_scope="agent",
            expected_scope_id=f"agent:{run.role}",
        )
        _validate_memory_bundle(
            run_ref=run.run_ref,
            key="task_memory_bundle",
            bundle=run.metadata.get("task_memory_bundle"),
            expected_backend_id="mem0-task-memory",
            expected_scope="task",
            expected_scope_id=f"task:{run.task_id}",
        )
        update_statuses["agent_memory_update_result"] = agent_status
        update_statuses["task_memory_update_result"] = task_status

    copied_config_hits = _copied_config_legacy_client_hits(lab_root)
    if copied_config_hits:
        raise RuntimeError("copied config contains legacy in_memory Mem0 client: " + ", ".join(copied_config_hits))

    return {
        "sqlite_stores": sqlite_paths,
        "memory_record_counts": memory_record_counts,
        "subagent_run_count": len(runs),
        "update_statuses": update_statuses,
        "copied_config_legacy_client_hits": copied_config_hits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native mem0 memory against real API LLM and embedding services.")
    parser.add_argument("--dotenv", default=".env", help="Path to the .env file to inspect and pass to clean-run.")
    parser.add_argument("--lab-root", default="/tmp/evolab-native-mem0-api-smoke", help="Lab root for the smoke run.")
    parser.add_argument("--config-out", default=None, help="Optional path for the generated smoke config.")
    parser.add_argument("--chat-env-ref", default="aigocode-gpt", help="env_ref for the chat LLM used by mem0 extraction.")
    parser.add_argument("--embedding-env-ref", default="memory-embedding", help="env_ref for the embedding backend.")
    parser.add_argument("--chat-model", default=None, help="Chat model override; defaults to <CHAT_PREFIX>_MODEL.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model override; defaults to <EMBED_PREFIX>_MODEL.")
    parser.add_argument(
        "--require-credentials",
        action="store_true",
        help="Return exit code 2 instead of skipping when required .env entries are missing.",
    )
    args = parser.parse_args(argv)

    dotenv_path = Path(args.dotenv).expanduser().resolve()
    status = credential_status(
        dotenv_path,
        chat_env_ref=args.chat_env_ref,
        embedding_env_ref=args.embedding_env_ref,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
    )
    if not status.ready:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "missing real API credentials or model config",
                    "dotenv_path": str(dotenv_path),
                    "missing": status.missing,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if args.require_credentials else 0

    values = parse_dotenv(dotenv_path)
    chat_model = args.chat_model or _env_model(values, args.chat_env_ref)
    embedding_model = args.embedding_model or _env_model(values, args.embedding_env_ref)
    if chat_model is None or embedding_model is None:
        raise RuntimeError("credential readiness unexpectedly passed without both models")

    lab_root = Path(args.lab_root).expanduser().resolve()
    config_path = Path(args.config_out).expanduser().resolve() if args.config_out else lab_root.parent / "native_mem0_api_smoke_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_smoke_config(
        lab_root=lab_root,
        dotenv_path=dotenv_path,
        chat_env_ref=args.chat_env_ref,
        embedding_env_ref=args.embedding_env_ref,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = run_smoke(config_path=config_path, lab_root=lab_root)
    report["config_path"] = str(config_path)
    report["dotenv_path"] = str(dotenv_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _missing_api_backend_env(
    values: dict[str, str],
    *,
    env_ref: str,
    explicit_model: str | None,
    label: str,
) -> list[str]:
    prefix = env_ref_prefix(env_ref)
    missing: list[str] = []
    if not _has_prefixed_entry(values, prefix):
        missing.append(f"{prefix}_* entry for {label}")
    if not _env_api_key(values, prefix):
        missing.append(f"{prefix}_API_KEY")
    if not explicit_model and not _env_model(values, env_ref):
        missing.append(f"{prefix}_MODEL")
    return missing


def _has_prefixed_entry(values: dict[str, str], prefix: str) -> bool:
    prefix_with_separator = f"{prefix}_"
    return any(key.upper().startswith(prefix_with_separator) for key in values)


def _env_api_key(values: dict[str, str], prefix: str) -> str | None:
    return lookup_env_value(values, f"{prefix}_API_KEY") or lookup_env_value(values, f"{prefix}_APIKEY")


def _env_model(values: dict[str, str], env_ref: str) -> str | None:
    prefix = env_ref_prefix(env_ref)
    return lookup_env_value(values, f"{prefix}_MODEL")


def _validate_expected_sqlite_stores(lab_root: Path) -> tuple[list[str], dict[str, int]]:
    expected_paths = [
        lab_root / "registries" / "memory" / "mem0-agent.sqlite",
        lab_root / "registries" / "memory" / "mem0-task.sqlite",
    ]
    relative_paths = [str(path.relative_to(lab_root)) for path in expected_paths]
    memory_record_counts: dict[str, int] = {}
    for path, relative_path in zip(expected_paths, relative_paths, strict=True):
        if not path.is_file():
            raise RuntimeError(f"missing SQLite memory store {relative_path}")
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
                row = connection.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE deleted_at IS NULL"
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"memory store {relative_path} is not a readable SQLite database") from exc
        count = int(row[0]) if row is not None else 0
        if count <= 0:
            raise RuntimeError(f"memory store {relative_path} has no active memory_records")
        memory_record_counts[relative_path] = count
    return relative_paths, memory_record_counts


def _validate_memory_update_result(
    *,
    run_ref: str,
    key: str,
    update: Any,
    expected_backend_id: str,
    expected_scope: str,
    expected_scope_id: str,
) -> str:
    if not isinstance(update, dict):
        raise RuntimeError(f"trajectory {run_ref} missing {key}")
    status = update.get("status")
    if status not in {"updated", "skipped"}:
        raise RuntimeError(f"trajectory {run_ref} has unexpected {key} status {status!r}")
    metadata = update.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise RuntimeError(f"trajectory {run_ref} {key} missing diagnostics metadata")
    if metadata.get("memory_method") != "mem0":
        raise RuntimeError(f"trajectory {run_ref} {key} missing memory_method='mem0'")
    if "backend_id" in metadata:
        _validate_metadata_field(
            run_ref=run_ref,
            key=f"{key} metadata",
            metadata=metadata,
            field="backend_id",
            expected=expected_backend_id,
        )
    _validate_metadata_field(
        run_ref=run_ref,
        key=f"{key} metadata",
        metadata=metadata,
        field="memory_scope",
        expected=expected_scope,
    )
    _validate_metadata_field(
        run_ref=run_ref,
        key=f"{key} metadata",
        metadata=metadata,
        field="memory_scope_id",
        expected=expected_scope_id,
    )
    _validate_update_state_ref_backend(
        run_ref=run_ref,
        key=key,
        field="state_ref",
        state_ref=update.get("state_ref"),
        expected_backend_id=expected_backend_id,
    )
    _validate_update_state_ref_backend(
        run_ref=run_ref,
        key=key,
        field="previous_state_ref",
        state_ref=update.get("previous_state_ref"),
        expected_backend_id=expected_backend_id,
    )
    return str(status)


def _validate_update_state_ref_backend(
    *,
    run_ref: str,
    key: str,
    field: str,
    state_ref: Any,
    expected_backend_id: str,
) -> None:
    if state_ref is None:
        return
    if not isinstance(state_ref, str) or not state_ref:
        raise RuntimeError(f"trajectory {run_ref} {key} {field} must be a non-empty string when present")
    backend_id = _mem0_state_ref_backend_id(state_ref)
    if backend_id is None:
        raise RuntimeError(f"trajectory {run_ref} {key} {field} missing native mem0 backend_id")
    if backend_id != expected_backend_id:
        raise RuntimeError(
            f"trajectory {run_ref} {key} {field} backend_id expected {expected_backend_id!r}, got {backend_id!r}"
        )


def _mem0_state_ref_backend_id(state_ref: str) -> str | None:
    parsed = urlparse(state_ref)
    if parsed.scheme != "method" or parsed.netloc != "mem0":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 4 or not parts[3].startswith("v"):
        return None
    return _parse_scope_ref_part(parts[0])


def _parse_scope_ref_part(value: str) -> str | None:
    length_text, separator, encoded_value = value.partition(":")
    if not separator or not length_text.isdigit():
        return None
    decoded_value = unquote(encoded_value)
    if len(decoded_value) != int(length_text):
        return None
    return decoded_value


def _validate_memory_bundle(
    *,
    run_ref: str,
    key: str,
    bundle: Any,
    expected_backend_id: str,
    expected_scope: str,
    expected_scope_id: str,
) -> None:
    if bundle is None:
        raise RuntimeError(f"trajectory {run_ref} missing {key}")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"trajectory {run_ref} {key} must be an object")
    backend_id = bundle.get("backend_id")
    if backend_id != expected_backend_id:
        raise RuntimeError(f"trajectory {run_ref} {key} backend_id expected {expected_backend_id!r}, got {backend_id!r}")
    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise RuntimeError(f"trajectory {run_ref} {key} missing diagnostics metadata")
    if metadata.get("memory_method") != "mem0":
        raise RuntimeError(f"trajectory {run_ref} {key} missing memory_method='mem0'")
    _validate_metadata_field(
        run_ref=run_ref,
        key=f"{key} metadata",
        metadata=metadata,
        field="memory_scope",
        expected=expected_scope,
    )
    _validate_metadata_field(
        run_ref=run_ref,
        key=f"{key} metadata",
        metadata=metadata,
        field="memory_scope_id",
        expected=expected_scope_id,
    )


def _validate_metadata_field(
    *,
    run_ref: str,
    key: str,
    metadata: dict[str, Any],
    field: str,
    expected: str,
) -> None:
    actual = metadata.get(field)
    if actual != expected:
        raise RuntimeError(f"trajectory {run_ref} {key} {field} expected {expected!r}, got {actual!r}")


def _copied_config_legacy_client_hits(lab_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted((lab_root / "configs").glob("**/*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if _contains_legacy_in_memory_client(text):
            hits.append(str(path.relative_to(lab_root)))
    return hits


def _contains_legacy_in_memory_client(text: str) -> bool:
    for parser in (json.loads, yaml.safe_load):
        try:
            payload = parser(text)
        except (json.JSONDecodeError, yaml.YAMLError):
            continue
        if _payload_contains_legacy_in_memory_client(payload):
            return True
    return bool(
        re.search(
            r"""(?ix)
            ["']?
            (?:client|client_type)
            ["']?
            \s*:
            \s*
            ["']?
            in_memory
            ["']?
            """,
            text,
        )
    )


def _payload_contains_legacy_in_memory_client(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"client", "client_type"} and value == "in_memory":
                return True
            if _payload_contains_legacy_in_memory_client(value):
                return True
    if isinstance(payload, list):
        return any(_payload_contains_legacy_in_memory_client(item) for item in payload)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
