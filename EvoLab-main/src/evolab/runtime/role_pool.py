from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from pydantic import BaseModel, ValidationError

from evolab.config.agents import agents_markdown_revision, parse_agents_markdown, render_agents_markdown
from evolab.config.task_config import RoleSpec

ROLE_POOL_UPDATE_KEYS = ("role_pool_update", "agent_config_update", "agents_update", "subagent_config_update")

_FORBIDDEN_REASONING_KEYS = frozenset({"chain_of_thought", "reasoning", "hidden_reasoning"})
_PROMPT_APPEND_KEYS = ("system_prompt_append", "prompt_append")
_HISTORY_SUFFIX = ".updates.jsonl"
_MAX_HISTORY_STRING_CHARS = 2000


@dataclass(frozen=True)
class RolePoolUpdateResult:
    status: Literal["updated", "no_op", "rejected"]
    before_revision: str
    after_revision: str
    active_roles: list[str] = field(default_factory=list)
    added_roles: list[str] = field(default_factory=list)
    modified_roles: list[str] = field(default_factory=list)
    removed_roles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    history_ref: str | None = None
    reason: str | None = None

    @property
    def revisions(self) -> dict[str, str]:
        return {"before": self.before_revision, "after": self.after_revision}

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["revisions"] = self.revisions
        return _json_compatible(payload)


def role_pool_update_payload(metadata: Mapping[str, Any] | None) -> Any | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in ROLE_POOL_UPDATE_KEYS:
        value = metadata.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def apply_role_pool_update(
    *,
    agents_path: Path | str,
    payload: Any,
    task_id: str,
    run_ref: str | None,
    known_llm_backend_ids: Iterable[str],
    allowed_tool_names: Iterable[str] | None = None,
    history_path: Path | str | None = None,
) -> RolePoolUpdateResult:
    agents_path = Path(agents_path)
    history_path = Path(history_path) if history_path is not None else Path(f"{agents_path}{_HISTORY_SUFFIX}")
    known_backend_ids = set(known_llm_backend_ids)
    allowed_tools = None if allowed_tool_names is None else set(allowed_tool_names)
    if _paths_alias(agents_path, history_path):
        return RolePoolUpdateResult(
            status="rejected",
            before_revision="",
            after_revision="",
            errors=["history_path must not alias agents_path"],
            reason=_payload_reason(payload),
        )
    compact_payload = _compact_update_payload(payload)
    update_hash = _update_hash(compact_payload)

    try:
        original_text = agents_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _record_rejection(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            before_revision="",
            active_roles=[],
            reason=_payload_reason(payload),
            errors=[f"failed to read {agents_path}: {exc}"],
        )

    before_revision = agents_markdown_revision(original_text)
    try:
        current_roles = parse_agents_markdown(original_text, source=str(agents_path))
    except (TypeError, ValueError, ValidationError) as exc:
        return _record_rejection(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            before_revision=before_revision,
            active_roles=[],
            reason=_payload_reason(payload),
            errors=[f"failed to parse current role pool: {exc}"],
        )

    reason = _payload_reason(payload)
    warnings: list[str] = []
    errors = _find_forbidden_reasoning_fields(payload)
    if not isinstance(payload, Mapping):
        errors.append("role pool update payload must be a mapping")
        return _record_rejection(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            before_revision=before_revision,
            active_roles=list(current_roles),
            reason=reason,
            errors=errors,
        )

    remove_roles = _parse_remove_roles(payload.get("remove_roles"), errors=errors)
    role_updates = _parse_update_roles(payload.get("roles", payload.get("agents")), errors=errors)
    removed_update_names = set(remove_roles)
    for name, _role_payload in role_updates:
        if name in removed_update_names:
            errors.append(f"role {name!r} cannot be both updated and removed")

    if errors:
        return _record_rejection(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            before_revision=before_revision,
            active_roles=list(current_roles),
            reason=reason,
            errors=errors,
        )

    candidate_roles = dict(current_roles)
    for name in remove_roles:
        if name not in candidate_roles:
            warnings.append(f"role {name!r} was not present")
            continue
        del candidate_roles[name]

    for name, update_payload in role_updates:
        try:
            candidate_roles[name] = _apply_role_payload(name, update_payload, current=candidate_roles.get(name))
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"role {name!r} is invalid: {exc}")

    errors.extend(_validate_active_roles(candidate_roles, known_backend_ids, allowed_tools))
    if errors:
        return _record_rejection(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            before_revision=before_revision,
            active_roles=list(current_roles),
            reason=reason,
            errors=errors,
            warnings=warnings,
        )

    added_roles, modified_roles, removed_roles = _role_diffs(current_roles, candidate_roles)
    if not added_roles and not modified_roles and not removed_roles:
        return _record_result(
            agents_path=agents_path,
            history_path=history_path,
            task_id=task_id,
            run_ref=run_ref,
            compact_payload=compact_payload,
            update_hash=update_hash,
            result=RolePoolUpdateResult(
                status="no_op",
                before_revision=before_revision,
                after_revision=before_revision,
                active_roles=list(candidate_roles),
                warnings=warnings,
                reason=reason,
            ),
        )

    updated_text = render_agents_markdown(candidate_roles)
    after_revision = agents_markdown_revision(updated_text)
    result = RolePoolUpdateResult(
        status="updated",
        before_revision=before_revision,
        after_revision=after_revision,
        active_roles=list(candidate_roles),
        added_roles=added_roles,
        modified_roles=modified_roles,
        removed_roles=removed_roles,
        warnings=warnings,
        reason=reason,
    )
    try:
        result = _with_history_ref(result, _next_history_ref(history_path))

        def append_success_history() -> None:
            _append_history(
                history_path=history_path,
                agents_path=agents_path,
                task_id=task_id,
                run_ref=run_ref,
                update_hash=update_hash,
                compact_payload=compact_payload,
                result=result,
            )

        _atomic_write_text(agents_path, updated_text, before_replace=append_success_history)
        return result
    except OSError as exc:
        return RolePoolUpdateResult(
            status="rejected",
            before_revision=before_revision,
            after_revision=before_revision,
            active_roles=list(current_roles),
            errors=[f"failed to record role pool update history: {exc}"],
            reason=reason,
        )


def _record_rejection(
    *,
    agents_path: Path,
    history_path: Path,
    task_id: str,
    run_ref: str | None,
    compact_payload: Any,
    update_hash: str,
    before_revision: str,
    active_roles: list[str],
    reason: str | None,
    errors: list[str],
    warnings: list[str] | None = None,
) -> RolePoolUpdateResult:
    return _record_result(
        agents_path=agents_path,
        history_path=history_path,
        task_id=task_id,
        run_ref=run_ref,
        compact_payload=compact_payload,
        update_hash=update_hash,
        result=RolePoolUpdateResult(
            status="rejected",
            before_revision=before_revision,
            after_revision=before_revision,
            active_roles=active_roles,
            warnings=list(warnings or []),
            errors=errors,
            reason=reason,
        ),
    )


def _record_result(
    *,
    agents_path: Path,
    history_path: Path,
    task_id: str,
    run_ref: str | None,
    compact_payload: Any,
    update_hash: str,
    result: RolePoolUpdateResult,
) -> RolePoolUpdateResult:
    try:
        result = _with_history_ref(result, _next_history_ref(history_path))
        _append_history(
            history_path=history_path,
            agents_path=agents_path,
            task_id=task_id,
            run_ref=run_ref,
            update_hash=update_hash,
            compact_payload=compact_payload,
            result=result,
        )
        return result
    except OSError as exc:
        return RolePoolUpdateResult(
            status="rejected",
            before_revision=result.before_revision,
            after_revision=result.before_revision,
            active_roles=list(result.active_roles),
            added_roles=list(result.added_roles),
            modified_roles=list(result.modified_roles),
            removed_roles=list(result.removed_roles),
            warnings=list(result.warnings),
            errors=[*result.errors, f"failed to record role pool update history: {exc}"],
            reason=result.reason,
        )


def _with_history_ref(result: RolePoolUpdateResult, history_ref: str) -> RolePoolUpdateResult:
    return RolePoolUpdateResult(
        status=result.status,
        before_revision=result.before_revision,
        after_revision=result.after_revision,
        active_roles=list(result.active_roles),
        added_roles=list(result.added_roles),
        modified_roles=list(result.modified_roles),
        removed_roles=list(result.removed_roles),
        warnings=list(result.warnings),
        errors=list(result.errors),
        history_ref=history_ref,
        reason=result.reason,
    )


def _parse_update_roles(raw_roles: Any, *, errors: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if raw_roles is None:
        return []
    parsed: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_roles, Mapping):
        for name, role_payload in raw_roles.items():
            if not isinstance(name, str) or not name.strip():
                errors.append("role update mapping contains an empty role name")
                continue
            if not isinstance(role_payload, Mapping):
                errors.append(f"role {name!r} update must be a mapping")
                continue
            parsed.append((name, {**dict(role_payload), "name": name}))
        return parsed
    if isinstance(raw_roles, list):
        for index, role_payload in enumerate(raw_roles):
            if not isinstance(role_payload, Mapping):
                errors.append(f"role update #{index + 1} must be a mapping")
                continue
            name = role_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"role update #{index + 1} requires non-empty name")
                continue
            parsed.append((name, dict(role_payload)))
        return parsed
    errors.append("roles must be a mapping or list")
    return []


def _parse_remove_roles(raw_remove_roles: Any, *, errors: list[str]) -> list[str]:
    if raw_remove_roles is None:
        return []
    if isinstance(raw_remove_roles, str):
        raw_values = [raw_remove_roles]
    elif isinstance(raw_remove_roles, list):
        raw_values = raw_remove_roles
    else:
        errors.append("remove_roles must be a list of role names")
        return []

    roles: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_values):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"remove_roles entry #{index + 1} must be a non-empty string")
            continue
        if value in seen:
            continue
        seen.add(value)
        roles.append(value)
    return roles


def _apply_role_payload(name: str, raw_payload: Mapping[str, Any], *, current: RoleSpec | None) -> RoleSpec:
    payload = _normalise_role_payload(raw_payload)
    payload["name"] = name
    append_text = _pop_prompt_append(payload)
    if current is not None:
        merged = current.model_dump(mode="json", exclude_none=True)
        merged.update(payload)
        payload = merged
    if append_text:
        system_prompt = payload.get("system_prompt", "")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string when appending")
        payload["system_prompt"] = _append_system_prompt(system_prompt, append_text)
    return RoleSpec.model_validate(payload)


def _normalise_role_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "prompt" in result and "system_prompt" not in result:
        result["system_prompt"] = result.pop("prompt")
    if "toolset" in result and "allowed_tools" not in result:
        result["allowed_tools"] = result.pop("toolset")
    if "skillset" in result and "required_skills" not in result:
        result["required_skills"] = result.pop("skillset")
    for key in ("llm_backend", "agent_memory_backend"):
        if isinstance(result.get(key), str):
            result[key] = {"backend_id": result[key]}
    return result


def _pop_prompt_append(payload: dict[str, Any]) -> str:
    append_values: list[str] = []
    for key in _PROMPT_APPEND_KEYS:
        value = payload.pop(key, None)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if value.strip():
            append_values.append(value.strip())
    return "\n\n".join(append_values)


def _append_system_prompt(system_prompt: str, append_text: str) -> str:
    if not system_prompt.strip():
        return append_text
    return f"{system_prompt.rstrip()}\n\n{append_text}"


def _validate_active_roles(
    roles: Mapping[str, RoleSpec],
    known_llm_backend_ids: set[str],
    allowed_tool_names: set[str] | None,
) -> list[str]:
    errors: list[str] = []
    if not roles:
        errors.append("role pool update must leave at least one active role")
        return errors

    for name, role in roles.items():
        role_errors = _find_forbidden_reasoning_fields(role.model_dump(mode="json", exclude_none=True), path=f"roles.{name}")
        errors.extend(role_errors)
        if not role.system_prompt.strip():
            errors.append(f"active role {name!r} requires non-empty system_prompt")
        if role.llm_backend.backend_id not in known_llm_backend_ids:
            errors.append(f"active role {name!r} uses unknown llm_backend.backend_id {role.llm_backend.backend_id!r}")
        if allowed_tool_names is not None:
            unknown_tools = sorted(set(role.allowed_tools) - allowed_tool_names)
            if unknown_tools:
                errors.append(f"active role {name!r} uses unknown allowed_tools {unknown_tools!r}")
    return errors


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        if left.exists() and right.exists() and left.samefile(right):
            return True
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def _role_diffs(
    before_roles: Mapping[str, RoleSpec],
    after_roles: Mapping[str, RoleSpec],
) -> tuple[list[str], list[str], list[str]]:
    added_roles = [name for name in after_roles if name not in before_roles]
    modified_roles = [
        name
        for name in after_roles
        if name in before_roles and _role_json(after_roles[name]) != _role_json(before_roles[name])
    ]
    removed_roles = [name for name in before_roles if name not in after_roles]
    return added_roles, modified_roles, removed_roles


def _role_json(role: RoleSpec) -> dict[str, Any]:
    return role.model_dump(mode="json", exclude_none=True)


def _payload_reason(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    reason = payload.get("reason")
    return reason if isinstance(reason, str) else None


def _find_forbidden_reasoning_fields(value: Any, *, path: str = "payload") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_REASONING_KEYS:
                errors.append(f"private reasoning field {child_path!r} is not allowed")
            errors.extend(_find_forbidden_reasoning_fields(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_forbidden_reasoning_fields(child, path=f"{path}[{index}]"))
    return errors


def _compact_update_payload(payload: Any) -> Any:
    return _json_compatible(payload, redact_private=True, max_string_chars=_MAX_HISTORY_STRING_CHARS)


def _update_hash(compact_payload: Any) -> str:
    encoded = json.dumps(compact_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _next_history_ref(history_path: Path) -> str:
    line_number = 1
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            line_number = sum(1 for _line in handle) + 1
    return f"{history_path}:{line_number}"


def _append_history(
    *,
    history_path: Path,
    agents_path: Path,
    task_id: str,
    run_ref: str | None,
    update_hash: str,
    compact_payload: Any,
    result: RolePoolUpdateResult,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": task_id,
        "run_ref": run_ref,
        "agents_path": str(agents_path),
        "update_hash": update_hash,
        "update": compact_payload,
        "result": result.to_json(),
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str, *, before_replace: Callable[[], None] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _json_compatible(
    value: Any,
    *,
    redact_private: bool = False,
    max_string_chars: int | None = None,
) -> Any:
    if isinstance(value, BaseModel):
        return _json_compatible(
            value.model_dump(mode="json", exclude_none=True),
            redact_private=redact_private,
            max_string_chars=max_string_chars,
        )
    if is_dataclass(value):
        return _json_compatible(
            asdict(value),
            redact_private=redact_private,
            max_string_chars=max_string_chars,
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if redact_private and key_text in _FORBIDDEN_REASONING_KEYS:
                result[key_text] = "[REDACTED]"
                continue
            result[key_text] = _json_compatible(
                child,
                redact_private=redact_private,
                max_string_chars=max_string_chars,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(child, redact_private=redact_private, max_string_chars=max_string_chars)
            for child in value
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(
            _json_compatible(child, redact_private=redact_private, max_string_chars=max_string_chars)
            for child in value
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and max_string_chars is not None and len(value) > max_string_chars:
            return f"{value[:max_string_chars]}...<truncated>"
        return value
    return repr(value)
