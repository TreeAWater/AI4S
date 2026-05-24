from __future__ import annotations

import posixpath
from pathlib import Path
from urllib.parse import unquote, urlparse

from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest, LLMEvolutionResult


def expected_roles_for(mode: LLMEvolutionMode) -> set[str]:
    if mode == LLMEvolutionMode.BASICS:
        return {"solver"}
    if mode == LLMEvolutionMode.CONSOLIDATION:
        return {"skill_distilled"}
    return set()


def validate_promotion(result: LLMEvolutionResult, request: LLMEvolutionRequest) -> list[str]:
    if not result.recommend_for_promotion:
        return []

    errors: list[str] = []
    if not result.new_state_ref:
        errors.append("new_state_ref is empty")

    if not any(_artifact_under_root(artifact.uri, request.artifact_root_uri) for artifact in result.artifact_refs):
        errors.append("no artifact under artifact_root_uri")

    if request.previous_state_ref is not None and result.standard_metrics.eval_score_after is None:
        errors.append("eval_score_after is missing for promotion with previous_state_ref")

    expected_roles = expected_roles_for(request.mode)
    if result.lora_role not in expected_roles:
        errors.append(
            f"lora_role {result.lora_role!r} does not match expected roles {sorted(expected_roles)!r}"
        )

    return errors


def _artifact_under_root(artifact_uri: str, root_uri: str) -> bool:
    artifact_path = _local_path_from_uri(artifact_uri)
    root_path = _local_path_from_uri(root_uri)
    if artifact_path is not None and root_path is not None:
        try:
            artifact_path.resolve(strict=False).relative_to(root_path.resolve(strict=False))
        except ValueError:
            return False
        return True

    return _remote_uri_under_root(artifact_uri, root_uri)


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        path = unquote(parsed.path if parsed.scheme == "file" else uri)
        return Path(path)
    return None


def _remote_uri_under_root(artifact_uri: str, root_uri: str) -> bool:
    artifact = urlparse(artifact_uri)
    root = urlparse(root_uri)
    if not artifact.scheme or not root.scheme:
        return False
    if artifact.scheme != root.scheme or artifact.netloc != root.netloc:
        return False

    artifact_path = _normalize_uri_path(artifact.path)
    root_path = _normalize_uri_path(root.path)
    return artifact_path == root_path or artifact_path.startswith(f"{root_path.rstrip('/')}/")


def _normalize_uri_path(path: str) -> str:
    normalized = posixpath.normpath(unquote(path))
    if normalized in ("", "."):
        return "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized
