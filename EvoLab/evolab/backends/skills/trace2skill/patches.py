from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json
from typing import Any

from evolab.backends.skills.trace2skill.schema import SkillPatchBundle, SkillPatchProposal


def build_patch_bundle(patches: Iterable[SkillPatchProposal]) -> SkillPatchBundle:
    patch_list = sorted(patches, key=lambda patch: patch.patch_id)
    target_skill_ids = _dedupe(
        patch.target_skill_id for patch in patch_list if isinstance(patch.target_skill_id, str)
    )
    source_trace_ids = _dedupe(trace_id for patch in patch_list for trace_id in patch.source_trace_ids)
    stats = {
        "patch_count": len(patch_list),
        "target_skill_count": len(target_skill_ids),
        "source_trace_count": len(source_trace_ids),
        "patch_types": sorted({patch.patch_type for patch in patch_list}),
    }
    return SkillPatchBundle(
        bundle_id=_stable_id("patch-bundle", [patch.patch_id for patch in patch_list]),
        local_patches=patch_list,
        target_skill_ids=target_skill_ids,
        source_trace_ids=source_trace_ids,
        stats=stats,
    )


def patch_similarity_key(patch: SkillPatchProposal) -> tuple[str, str, str]:
    subject = patch.target_skill_id or patch.candidate_skill_id or _candidate_task_key(patch.patch_content)
    content_key = json.dumps(_normalized_content(patch.patch_content), sort_keys=True, separators=(",", ":"))
    return (patch.patch_type, subject or "", content_key)


def merge_patch_content(patches: Iterable[SkillPatchProposal]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for patch in patches:
        for key, value in patch.patch_content.items():
            if isinstance(value, list):
                merged[key] = _dedupe([*merged.get(key, []), *value])
            elif key not in merged or not merged[key]:
                merged[key] = value
            elif merged[key] != value:
                merged.setdefault("alternatives", []).append({key: value})
    return merged


def _candidate_task_key(content: dict[str, Any]) -> str:
    return str(content.get("missing_capability_description") or content.get("proposed_name") or "")


def _normalized_content(content: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in content.items():
        if isinstance(value, list):
            normalized[key] = sorted(str(item) for item in value)
        else:
            normalized[key] = value
    return normalized


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
