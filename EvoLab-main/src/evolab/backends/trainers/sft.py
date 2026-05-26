from __future__ import annotations

import json
from inspect import signature
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import Field

from evolab.backends.llm.local import write_local_trainable_state_manifest
from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.common import ArtifactRef, StrictBaseModel
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult, StandardEvolutionMetrics
from evolab.contracts.local_trainable import new_local_trainable_state_ref
from evolab.registries.trajectory import TrajectoryRegistry
from evolab.runtime.sft_exporter import SFTExportConfig, export_sft_dataset


class SFTTrainerConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    training_backend: Literal["dry_run", "transformers"] = "dry_run"
    base_model_ref: str | None = None
    max_length: int = Field(default=2048, ge=1)
    min_train_samples: int = Field(default=1, ge=0)
    promote_dry_run: bool = False
    dry_run_eval_score_after: float = 1.0
    export: SFTExportConfig = Field(default_factory=SFTExportConfig)
    training_args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SFTTrainer(LLMTrainer):
    trainer_id = "sft"

    def __init__(
        self,
        *,
        trajectory_registry: TrajectoryRegistry,
        config: SFTTrainerConfig | None = None,
        trainer_id: str | None = None,
    ) -> None:
        if trainer_id is not None:
            self.trainer_id = trainer_id
        self.trajectory_registry = trajectory_registry
        self.config = config or SFTTrainerConfig()

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        artifact_root = _require_local_artifact_root(request.artifact_root_uri)
        artifact_root.mkdir(parents=True, exist_ok=True)
        export_config = self._export_config_for_request(request)
        export_result = export_sft_dataset(
            trajectory_registry=self.trajectory_registry,
            output_dir=artifact_root / "dataset",
            config=export_config,
        )
        artifacts = [
            ArtifactRef(uri=str(export_result.train_path), type="dataset", metadata={"role": "sft_train"}),
            ArtifactRef(uri=str(export_result.val_path), type="dataset", metadata={"role": "sft_val"}),
            ArtifactRef(uri=str(export_result.manifest_path), type="dataset", metadata={"role": "sft_manifest"}),
        ]
        if export_result.manifest.train_count < self.config.min_train_samples:
            return LLMEvolutionResult(
                status="skipped",
                artifact_refs=artifacts,
                metadata={
                    "reason": "not enough SFT train samples",
                    "trainer": self.trainer_id,
                    "sample_count": export_result.manifest.sample_count,
                    "train_count": export_result.manifest.train_count,
                    "manifest_uri": str(export_result.manifest_path),
                },
            )

        if self.config.training_backend == "transformers":
            return self._train_with_transformers(request, export_result.manifest_path, artifacts)
        return self._dry_run_result(request, export_result.manifest_path, artifacts)

    def _export_config_for_request(self, request: LLMEvolutionRequest) -> SFTExportConfig:
        source_run_refs = list(self.config.export.source_run_refs)
        if not source_run_refs and not self.config.export.source_llm_call_refs:
            source_run_refs = _training_trajectory_refs(request)
        teacher_backend_ids = list(self.config.export.teacher_backend_ids)
        if not teacher_backend_ids:
            teacher_backend_ids = [request.backend_id]
        max_samples = _merge_max_samples(
            configured=self.config.export.max_samples,
            budget=request.budget.max_train_samples,
        )
        return self.config.export.model_copy(
            update={
                "source_run_refs": source_run_refs,
                "teacher_backend_ids": teacher_backend_ids,
                "max_samples": max_samples,
                "metadata": {
                    **self.config.export.metadata,
                    "evolution_backend_id": request.backend_id,
                    "trigger_trajectory_ref": request.trigger_trajectory_ref,
                },
            },
            deep=True,
        )

    def _dry_run_result(
        self,
        request: LLMEvolutionRequest,
        manifest_path: Path,
        artifacts: list[ArtifactRef],
    ) -> LLMEvolutionResult:
        adapter_path = _write_dry_run_adapter(
            _require_local_artifact_root(request.artifact_root_uri) / "adapter",
            {
                "trainer": self.trainer_id,
                "training_backend": "dry_run",
                "backend_id": request.backend_id,
                "previous_state_ref": request.previous_state_ref,
                "base_model_ref": self.config.base_model_ref,
                "manifest_uri": str(manifest_path),
                "training_args": self.config.training_args,
            },
        )
        artifacts = [
            *artifacts,
            ArtifactRef(uri=str(adapter_path), type="model_adapter", metadata={"role": "sft_dry_run_adapter"}),
        ]
        if not self.config.promote_dry_run:
            return LLMEvolutionResult(
                status="not_recommended",
                artifact_refs=artifacts,
                standard_metrics=StandardEvolutionMetrics(
                    n_train_samples=_train_count(manifest_path),
                    eval_score_after=self.config.dry_run_eval_score_after,
                    eval_metric_name="dry_run",
                ),
                metadata={
                    "trainer": self.trainer_id,
                    "training_backend": "dry_run",
                    "manifest_uri": str(manifest_path),
                    "reason": "dry-run SFT does not promote by default",
                },
            )
        state_ref = new_local_trainable_state_ref(request.backend_id)
        state_manifest_path = write_local_trainable_state_manifest(
            _require_local_artifact_root(request.artifact_root_uri) / "state" / "local_trainable_state.json",
            backend_id=request.backend_id,
            state_ref=state_ref,
            parent_state_ref=request.previous_state_ref,
            created_by_trainer=self.trainer_id,
            adapter_uri=str(adapter_path),
            dataset_manifest_uri=str(manifest_path),
            default_content=f"local trainable {self.trainer_id} dry-run response",
            metadata={
                "training_backend": "dry_run",
                "training_args": self.config.training_args,
                "base_model_ref": self.config.base_model_ref,
            },
        )
        artifacts = [
            *artifacts,
            ArtifactRef(
                uri=str(state_manifest_path),
                type="model_adapter",
                metadata={"role": "local_trainable_state", "trainer": self.trainer_id},
            ),
        ]
        return LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref=state_ref,
            lora_role="solver",
            artifact_refs=artifacts,
            standard_metrics=StandardEvolutionMetrics(
                n_train_samples=_train_count(manifest_path),
                eval_score_before=0.0 if request.previous_state_ref else None,
                eval_score_after=self.config.dry_run_eval_score_after,
                eval_metric_name="dry_run",
            ),
            metadata={
                "trainer": self.trainer_id,
                "training_backend": "dry_run",
                "manifest_uri": str(manifest_path),
                "output_snapshot_refs": [state_ref],
            },
        )

    def _train_with_transformers(
        self,
        request: LLMEvolutionRequest,
        manifest_path: Path,
        artifacts: list[ArtifactRef],
    ) -> LLMEvolutionResult:
        if not self.config.base_model_ref:
            return LLMEvolutionResult(
                status="failed",
                artifact_refs=artifacts,
                metadata={
                    "trainer": self.trainer_id,
                    "training_backend": "transformers",
                    "manifest_uri": str(manifest_path),
                    "reason": "SFT transformers backend requires base_model_ref",
                },
            )

        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                "SFT transformers backend requires optional dependencies. Install with "
                "`pip install -e .[sft]` before using training_backend='transformers'."
            ) from exc

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_path = Path(manifest["train_path"])
        val_path = Path(manifest["val_path"])
        train_samples = _read_sft_samples(train_path)
        val_samples = _read_sft_samples(val_path) if val_path.is_file() else []

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_ref)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        train_examples = [
            _tokenize_sft_sample(sample, tokenizer, max_length=self.config.max_length)
            for sample in train_samples
        ]
        val_examples = [
            _tokenize_sft_sample(sample, tokenizer, max_length=self.config.max_length)
            for sample in val_samples
        ]

        class _SFTTorchDataset(Dataset):
            def __init__(self, rows: list[dict[str, list[int]]]) -> None:
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, list[int]]:
                return self.rows[index]

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_ref)
        input_embeddings = model.get_input_embeddings()
        if getattr(input_embeddings, "num_embeddings", len(tokenizer)) < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        artifact_root = _require_local_artifact_root(request.artifact_root_uri)
        trainer_output_dir = artifact_root / "transformers-run"
        model_dir = artifact_root / "model"
        args_payload = {
            "per_device_train_batch_size": 1,
            "num_train_epochs": 1,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": [],
            **self.config.training_args,
            "output_dir": str(trainer_output_dir),
        }
        training_args = TrainingArguments(**args_payload)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=_SFTTorchDataset(train_examples),
            eval_dataset=_SFTTorchDataset(val_examples) if val_examples else None,
            data_collator=lambda rows: _collate_sft_batch(
                rows,
                pad_token_id=tokenizer.pad_token_id,
                torch_module=torch,
            ),
            **_trainer_tokenizer_kwargs(Trainer, tokenizer),
        )
        train_output = trainer.train()
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        train_manifest_path = artifact_root / "sft_training_manifest.json"
        train_manifest_path.write_text(
            json.dumps(
                {
                    "trainer": self.trainer_id,
                    "training_backend": "transformers",
                    "backend_id": request.backend_id,
                    "previous_state_ref": request.previous_state_ref,
                    "base_model_ref": self.config.base_model_ref,
                    "manifest_uri": str(manifest_path),
                    "model_dir": str(model_dir),
                    "training_args": args_payload,
                    "train_metrics": getattr(train_output, "metrics", {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        state_ref = new_local_trainable_state_ref(request.backend_id)
        artifacts = [
            *artifacts,
            ArtifactRef(uri=str(model_dir), type="model_adapter", metadata={"role": "sft_model"}),
            ArtifactRef(uri=str(train_manifest_path), type="log", metadata={"role": "sft_training_manifest"}),
        ]
        return LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref=state_ref,
            lora_role="solver",
            artifact_refs=artifacts,
            standard_metrics=StandardEvolutionMetrics(
                n_train_samples=len(train_examples),
                eval_score_before=0.0 if request.previous_state_ref else None,
                eval_score_after=float(len(train_examples)),
                eval_metric_name="sft_train_samples",
            ),
            metadata={
                "trainer": self.trainer_id,
                "training_backend": "transformers",
                "manifest_uri": str(manifest_path),
                "model_dir": str(model_dir),
                "output_snapshot_refs": [state_ref],
            },
        )


def _training_trajectory_refs(request: LLMEvolutionRequest) -> list[str]:
    refs: list[str] = []
    if request.trigger_trajectory_ref is not None:
        refs.append(request.trigger_trajectory_ref)
    for ref in request.proposer_input_refs:
        if ref.ref_type == "trajectory" and ref.ref_id not in refs:
            refs.append(ref.ref_id)
    return refs


def _merge_max_samples(*, configured: int | None, budget: int | None) -> int | None:
    if configured is None:
        return budget
    if budget is None:
        return configured
    return min(configured, budget)


def _require_local_artifact_root(uri: str) -> Path:
    path = _local_path_from_uri(uri)
    if path is None:
        raise ValueError("SFTTrainer requires a local artifact_root_uri")
    return path


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
    return None


def _write_dry_run_adapter(root: Path, payload: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "adapter_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _train_count(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = payload.get("train_count")
    return value if isinstance(value, int) else 0


def _read_sft_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def _trainer_tokenizer_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def _tokenize_sft_sample(sample: dict[str, Any], tokenizer: Any, *, max_length: int) -> dict[str, list[int]]:
    messages = sample.get("messages", [])
    prompt_messages, _completion_messages = _split_prompt_completion(messages)
    full_text = _messages_to_training_text(messages, tokenizer)
    prompt_text = _messages_to_training_text(prompt_messages, tokenizer)
    input_ids = list(tokenizer(full_text, add_special_tokens=False).input_ids)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = _prompt_prefix_len(input_ids, prompt_ids)
    if tokenizer.eos_token_id is not None and (not input_ids or input_ids[-1] != tokenizer.eos_token_id):
        input_ids.append(tokenizer.eos_token_id)
    input_ids, prompt_len = _fit_full_sequence(input_ids, prompt_len, max_length=max_length)
    labels = [-100 for _ in input_ids[:prompt_len]] + list(input_ids[prompt_len:])
    return {
        "input_ids": input_ids,
        "attention_mask": [1 for _ in input_ids],
        "labels": labels,
    }


def _split_prompt_completion(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return messages[:index], messages[index:]
    if not messages:
        return [], []
    return messages[:-1], messages[-1:]


def _prompt_prefix_len(input_ids: list[int], prompt_ids: list[int]) -> int:
    if input_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    common = 0
    for left, right in zip(input_ids, prompt_ids):
        if left != right:
            break
        common += 1
    return common


def _fit_full_sequence(
    input_ids: list[int],
    prompt_len: int,
    *,
    max_length: int,
) -> tuple[list[int], int]:
    overflow = len(input_ids) - max_length
    if overflow <= 0:
        return input_ids, prompt_len
    return input_ids[overflow:], max(0, prompt_len - overflow)


def _messages_to_training_text(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    if not messages:
        return ""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                [_chat_template_message(message) for message in messages],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return "\n".join(_message_line(message) for message in messages if _message_line(message)) + "\n"


def _chat_template_message(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(message.get("role") or "user"),
        "content": _message_content(message),
    }


def _message_line(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message").upper()
    content = _message_content(message)
    return f"{role}: {content}" if content else role + ":"


def _message_content(message: dict[str, Any]) -> str:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        tool_call = metadata.get("tool_call")
        if isinstance(tool_call, dict):
            return "TOOL_CALL " + json.dumps(tool_call, sort_keys=True)
    role = message.get("role")
    if role == "tool":
        label = message.get("name") or message.get("tool_call_id") or "tool"
        return f"TOOL_RESULT {label}: {message.get('content') or ''}"
    return str(message.get("content") or "")


def _collate_sft_batch(
    rows: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    torch_module: Any,
) -> dict[str, Any]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    labels = []
    for row in rows:
        pad_count = max_length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id for _ in range(pad_count)])
        attention_mask.append(row["attention_mask"] + [0 for _ in range(pad_count)])
        labels.append(row["labels"] + [-100 for _ in range(pad_count)])
    return {
        "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
        "attention_mask": torch_module.tensor(attention_mask, dtype=torch_module.long),
        "labels": torch_module.tensor(labels, dtype=torch_module.long),
    }
