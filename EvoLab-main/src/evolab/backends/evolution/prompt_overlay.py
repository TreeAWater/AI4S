from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.common import ArtifactRef
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult, StandardEvolutionMetrics


class PromptOverlayEvolutionTrainer(LLMTrainer):
    """Create role-specific prompt overlays from post-run evaluator feedback.

    This trainer does not mutate remote API model weights. It converts sanitized
    reflector feedback into a bounded system-prompt append that can be activated
    for the same backend/role in later runs.
    """

    trainer_id = "prompt_overlay"

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        feedback = request.metadata.get("reflector_feedback")
        if not isinstance(feedback, dict):
            return LLMEvolutionResult(
                status="skipped",
                metadata={"reason": "missing reflector_feedback", "trainer": self.trainer_id},
            )
        role = request.metadata.get("role")
        if not isinstance(role, str) or not role:
            return LLMEvolutionResult(
                status="skipped",
                metadata={"reason": "missing target role for prompt overlay", "trainer": self.trainer_id},
            )
        instructions = _specific_instructions(feedback)
        if not instructions:
            return LLMEvolutionResult(
                status="not_recommended",
                standard_metrics=StandardEvolutionMetrics(eval_metric_name="reflector_feedback_specificity"),
                metadata={"reason": "reflector feedback had no specific evolution instructions", "trainer": self.trainer_id},
            )
        prompt_append = _render_prompt_append(feedback, instructions)
        overlay = {
            "schema_version": "v1",
            "role": role,
            "backend_id": request.backend_id,
            "system_prompt_append": prompt_append,
            "source": "reflector_feedback",
            "instruction_count": len(instructions),
            "metric_source": feedback.get("metric_source"),
            "metrics": feedback.get("metrics"),
        }
        digest = sha256(json.dumps(overlay, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        state_ref = f"prompt-overlay://{request.backend_id}/{_slug(role)}/{digest}"
        artifact_uri = _write_prompt_overlay(request.artifact_root_uri, overlay)
        return LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref=state_ref,
            lora_role="solver",
            standard_metrics=StandardEvolutionMetrics(
                eval_metric_name="reflector_feedback_specificity",
                eval_score_after=min(1.0, len(instructions) / 3),
            ),
            artifact_refs=[
                ArtifactRef(
                    uri=artifact_uri,
                    type="other",
                    metadata={"artifact_kind": "prompt_overlay", "role": role, "trainer": self.trainer_id},
                )
            ],
            metadata={"trainer": self.trainer_id, "prompt_overlay": overlay},
        )


def _specific_instructions(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    raw = feedback.get("specific_evolution_instructions")
    if not isinstance(raw, list):
        return []
    instructions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction") or item.get("recommendation") or item.get("action")
        if isinstance(instruction, str) and instruction.strip():
            instructions.append({**item, "instruction": instruction.strip()})
    return instructions[:8]


def _render_prompt_append(feedback: dict[str, Any], instructions: list[dict[str, Any]]) -> str:
    metrics = feedback.get("metrics") if isinstance(feedback.get("metrics"), dict) else {}
    lines = [
        "Reflector-guided prompt evolution notes:",
        "- These notes come from post-run evaluation feedback. They are reusable process guidance, not task answers.",
    ]
    if metrics:
        lines.append(
            "- Prior evaluation metrics: "
            f"precision={metrics.get('precision')}, recall={metrics.get('recall')}, f1={metrics.get('f1')}."
        )
    for index, item in enumerate(instructions, start=1):
        stage = item.get("stage")
        priority = item.get("priority")
        prefix = f"- Instruction {index}"
        details = []
        if isinstance(stage, str) and stage:
            details.append(f"stage={stage}")
        if isinstance(priority, str) and priority:
            details.append(f"priority={priority}")
        if details:
            prefix += f" ({', '.join(details)})"
        lines.append(f"{prefix}: {item['instruction'][:800]}")
    lines.append("- Do not copy, memorize, or reveal ground-truth answers. Improve evidence gathering, validation, and artifact handoff behavior.")
    return "\n".join(lines)


def _write_prompt_overlay(root_uri: str, overlay: dict[str, Any]) -> str:
    root = _local_path_from_uri(root_uri)
    if root is None:
        return f"{root_uri.rstrip('/')}/prompt_overlay.json"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prompt_overlay.json"
    path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return slug or "role"
