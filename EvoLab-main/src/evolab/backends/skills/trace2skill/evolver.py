from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised when PyYAML is installed.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from evolab.backends.skills.evolution import CandidateSkillRecord, SkillEvolutionPolicy, SkillUpdateDecision, SkillUpdateProposal
from evolab.backends.skills.trace2skill.adapter import Trace2SkillSkillBackendAdapter
from evolab.backends.skills.trace2skill.consolidator import HierarchicalPatchConsolidator
from evolab.backends.skills.trace2skill.llm_extractor import Trace2SkillLLMExtractor
from evolab.backends.skills.trace2skill.report import Trace2SkillReportWriter
from evolab.backends.skills.trace2skill.regression import BenchmarkTask, SkillEvolutionRegressionGate
from evolab.backends.skills.trace2skill.runner import ParallelAnalystRunner
from evolab.backends.skills.trace2skill.schema import (
    PatchConsolidationResult,
    PatchValidationResult,
    SkillLibraryUpdateTransaction,
    Trace2SkillRunConfig,
    Trace2SkillRunResult,
    TracePool,
)
from evolab.backends.skills.trace2skill.trace_pool import TracePoolBuilder
from evolab.backends.skills.trace2skill.validator import SkillPatchValidator


class Trace2SkillEvolver:
    def __init__(
        self,
        *,
        graph_backend: Any | None = None,
        tool_registry: Any | None = None,
        policy: SkillEvolutionPolicy | None = None,
        output_dir: str | Path | None = None,
        llm_client: Any | None = None,
        benchmark_runner: Any | None = None,
        observations: Iterable[Any] | None = None,
        run_records: Iterable[Any] | None = None,
        trace_pool: TracePool | None = None,
    ):
        self.graph_backend = graph_backend
        self.tool_registry = tool_registry
        self.policy = policy or getattr(graph_backend, "evolution_policy", None) or SkillEvolutionPolicy()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.observations = list(observations) if observations is not None else None
        self.run_records = list(run_records) if run_records is not None else None
        self.trace_pool = trace_pool
        self.trace_pool_builder = TracePoolBuilder()
        self.llm_extractor = Trace2SkillLLMExtractor(llm_client)
        self.benchmark_runner = benchmark_runner

    def run(
        self,
        config: Trace2SkillRunConfig,
        *,
        observations: Iterable[Any] | None = None,
        run_records: Iterable[Any] | None = None,
        trace_pool: TracePool | None = None,
    ) -> Trace2SkillRunResult:
        run_id = _stable_id("trace2skill-run", [config.model_dump(mode="json"), _utc_now()])
        before_hash = self._graph_hash()
        pool = trace_pool or self.trace_pool or self._build_pool(
            config,
            observations=observations if observations is not None else self.observations,
            run_records=run_records if run_records is not None else self.run_records,
        )
        analysis = ParallelAnalystRunner(
            execution_mode=config.analyst_execution_mode,
            max_workers=config.analyst_max_workers,
            timeout_seconds=config.analyst_timeout_seconds,
            llm_extractor=self.llm_extractor,
        ).run(pool, config=config)
        lessons, local_patches = analysis.lessons, analysis.patches
        if not lessons and not local_patches and config.enable_deterministic_fallback:
            lessons, local_patches = self.llm_extractor.extract_lessons_and_patches(
            pool,
                config=config.model_copy(update={"enable_llm_analysts": False}),
            )
        consolidation = HierarchicalPatchConsolidator(
            min_support_count=config.min_support_count,
            min_confidence=config.min_confidence,
        ).consolidate(local_patches)
        validator = SkillPatchValidator(graph_backend=self.graph_backend, tool_registry=self.tool_registry)
        validation = (
            validator.validate_patch_bundle(consolidation.consolidated_patches)
            if config.enable_validation_gate
            else PatchValidationResult(valid_patches=consolidation.consolidated_patches)
        )
        adapter = Trace2SkillSkillBackendAdapter(
            backend_id=getattr(self.graph_backend, "backend_id", "graph_skill"),
            graph_version_ref=self._graph_version(),
        )
        proposals = adapter.to_skill_update_proposals(validation.valid_patches)
        decisions = [self.policy.decide(proposal) for proposal in proposals]
        regression_result = self._run_regression_gate_if_needed(
            config=config,
            pool=pool,
            proposals=proposals,
            decisions=decisions,
            before_hash=before_hash,
        )
        blocked_by_regression = regression_result is not None and regression_result.status == "fail_regression"
        if blocked_by_regression:
            decisions = [
                decision.model_copy(
                    update={
                        "decision": "rejected",
                        "reason": f"blocked by regression gate: {regression_result.reason}",
                    }
                )
                if decision.decision == "applied"
                else decision
                for decision in decisions
            ]
        transaction = self._apply_policy_gated_transaction(
            proposals=proposals,
            decisions=decisions,
            dry_run=config.dry_run,
            before_graph_hash=before_hash,
            config=config,
        )
        after_hash = transaction.after_graph_hash or self._graph_hash()
        retrieval_change_summary = _retrieval_change_summary(transaction)
        report_dir = self._output_dir(config, run_id)
        result = Trace2SkillRunResult(
            run_id=run_id,
            config=config,
            trace_pool_summary=pool.stats,
            lessons=lessons,
            local_patches=local_patches,
            consolidation_result=consolidation,
            validation_result=validation,
            converted_skill_update_proposals=[proposal.model_dump(mode="json") for proposal in proposals],
            policy_decisions=[decision.model_dump(mode="json") for decision in decisions],
            applied_transactions=[transaction],
            staged_updates=transaction.staged_updates,
            rejected_updates=[
                *transaction.rejected_updates,
                *[patch.model_dump(mode="json") for patch in validation.invalid_patches],
            ],
            regression_gate_result=None if regression_result is None else regression_result.model_dump(mode="json"),
            before_metrics={} if regression_result is None else regression_result.before_metrics,
            after_metrics={} if regression_result is None else regression_result.after_metrics,
            blocked_by_regression=blocked_by_regression,
            before_graph_hash=before_hash,
            after_graph_hash=after_hash,
            retrieval_change_summary=retrieval_change_summary,
            changed_library=transaction.changed_library,
            report_paths={},
            status="dry_run" if config.dry_run else ("partial" if validation.invalid_patches else "completed"),
            metadata={
                "analysis": analysis.metadata,
                "analysis_failures": analysis.failures,
                "llm_audit_events": list(self.llm_extractor.audit_events),
            },
        )
        report_paths = Trace2SkillReportWriter(report_dir).write_run_report(result, trace_pool=pool)
        return result.model_copy(update={"report_paths": report_paths})

    def _build_pool(
        self,
        config: Trace2SkillRunConfig,
        *,
        observations: Iterable[Any] | None,
        run_records: Iterable[Any] | None,
    ) -> TracePool:
        if observations is not None:
            return self.trace_pool_builder.build_from_observations(
                observations,
                task_type=config.task_type,
                target_skill_ids=config.target_skill_ids,
                max_traces=config.max_traces,
            )
        if run_records is not None:
            return self.trace_pool_builder.build_from_run_records(
                run_records,
                task_type=config.task_type,
                target_skill_ids=config.target_skill_ids,
                max_traces=config.max_traces,
            )
        metadata = config.metadata
        if "observations" in metadata:
            return self.trace_pool_builder.build_from_observations(
                metadata["observations"],
                task_type=config.task_type,
                target_skill_ids=config.target_skill_ids,
                max_traces=config.max_traces,
            )
        if "run_records" in metadata:
            return self.trace_pool_builder.build_from_run_records(
                metadata["run_records"],
                task_type=config.task_type,
                target_skill_ids=config.target_skill_ids,
                max_traces=config.max_traces,
            )
        return self.trace_pool_builder.build_trace_pool(
            [],
            task_type=config.task_type,
            target_skill_ids=config.target_skill_ids,
            max_traces=config.max_traces,
        )

    def _apply_policy_gated_transaction(
        self,
        *,
        proposals: list[SkillUpdateProposal],
        decisions: list[SkillUpdateDecision],
        dry_run: bool,
        before_graph_hash: str | None,
        config: Trace2SkillRunConfig | None = None,
    ) -> SkillLibraryUpdateTransaction:
        config = config or Trace2SkillRunConfig(dry_run=dry_run)
        decisions_by_proposal = {decision.proposal_id: decision for decision in decisions}
        applied: list[dict[str, Any]] = []
        staged: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        loaded_raw_graph = self._load_raw_graph()
        raw_graph = deepcopy(loaded_raw_graph) if loaded_raw_graph is not None else None
        package_updates: dict[Path, dict[str, Any]] = {}
        changed = False

        for proposal in proposals:
            decision = decisions_by_proposal.get(proposal.proposal_id)
            if decision is None:
                continue
            record = {
                "proposal": proposal.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
            if dry_run:
                staged.append({**record, "reason": "dry_run"})
                continue
            if decision.decision == "rejected":
                rejected.append(record)
                continue
            if decision.decision == "staged":
                staged.append(record)
                continue
            mutation = self._apply_proposal(raw_graph, proposal, config=config, package_updates=package_updates)
            if mutation["changed"]:
                changed = True
                applied.append({**record, "mutation": mutation})
            elif mutation.get("rejected"):
                rejected.append({**record, "reason": mutation["reason"], "mutation": mutation})
            else:
                staged.append({**record, "reason": mutation["reason"]})

        write_error = None
        if changed and self.graph_backend is not None and raw_graph is not None:
            try:
                _validate_pending_library(self.graph_backend, raw_graph, package_updates)
                _write_pending_library(self.graph_backend, raw_graph, package_updates)
                # Force a load so schema/package errors surface in the transaction path.
                self.graph_backend.store.load_graph()
            except Exception as exc:  # pragma: no cover - defensive rollback boundary
                write_error = str(exc)
                _restore_pending_library(package_updates)
                if loaded_raw_graph is not None:
                    _write_graph_atomic(Path(self.graph_backend.graph_path), loaded_raw_graph)
                changed = False
                rejected.extend(applied)
                applied = []
        after_hash = self._graph_hash() if self.graph_backend is not None else before_graph_hash
        status = "dry_run" if dry_run else ("applied" if changed else ("staged" if staged else "no_op"))
        return SkillLibraryUpdateTransaction(
            transaction_id=_stable_id("trace2skill-transaction", [proposal.proposal_id for proposal in proposals]),
            proposal_ids=[proposal.proposal_id for proposal in proposals],
            decision_ids=[decision.decision_id for decision in decisions],
            applied_updates=applied,
            staged_updates=staged,
            rejected_updates=rejected,
            before_graph_hash=before_graph_hash,
            after_graph_hash=after_hash,
            changed_library=changed,
            status=status,  # type: ignore[arg-type]
            created_at=_utc_now(),
            metadata={
                "package_paths": [str(path) for path in sorted(package_updates)],
                "write_error": write_error,
                "changed_skill_ids": _changed_skill_ids(applied),
            },
        )

    def _apply_proposal(
        self,
        raw_graph: dict[str, Any] | None,
        proposal: SkillUpdateProposal,
        *,
        config: Trace2SkillRunConfig,
        package_updates: dict[Path, dict[str, Any]],
    ) -> dict[str, Any]:
        if raw_graph is None:
            return {"changed": False, "reason": "no graph backend configured"}
        if proposal.proposal_type == "candidate_skill_creation":
            return self._promote_candidate(raw_graph, proposal, package_updates=package_updates)
        if proposal.proposal_type == "relationship_update_proposal":
            return self._apply_relationship_update(raw_graph, proposal)
        changed = False
        touched: list[str] = []
        targets = self._skill_targets(raw_graph, proposal.related_skill_ids, package_updates)
        if not targets:
            return {"changed": False, "reason": "no matching mutable skill entries"}
        for target in targets:
            skill = target["payload"]
            if proposal.proposal_type == "required_tools_update_proposal":
                existing = [tool for tool in skill.get("required_tools", []) if isinstance(tool, str)]
                additions = [
                    tool
                    for tool in _strings(proposal.payload.get("required_tools") or proposal.payload.get("missing_required_tools"))
                    if tool not in existing
                ]
                if additions:
                    skill["required_tools"] = [*existing, *additions]
                    changed = True
                    touched.append(target["skill_id"])
            elif proposal.proposal_type == "failure_note_update":
                note = str(proposal.payload.get("failure_reason") or proposal.summary)[:240]
                evolution = _evolution_metadata(skill)
                if _append_bounded(evolution, "failure_cases", [note], limit=config.max_failure_cases_per_skill, max_chars=config.max_evolution_text_chars):
                    changed = True
                    touched.append(target["skill_id"])
            elif proposal.proposal_type == "example_trace_memory_update":
                example = str(proposal.payload.get("example_summary") or "")
                if _append_field_list(skill, "examples", [example], limit=config.max_examples_per_skill, max_chars=config.max_evolution_text_chars):
                    _append_bounded(
                        _evolution_metadata(skill),
                        "examples",
                        [example],
                        limit=config.max_examples_per_skill,
                        max_chars=config.max_evolution_text_chars,
                    )
                    changed = True
                    touched.append(target["skill_id"])
            elif proposal.proposal_type == "usage_stats_update":
                stats = _evolution_metadata(skill).setdefault("usage_stats", {})
                before = json.dumps(stats, sort_keys=True)
                stats["usage_count"] = _int(stats.get("usage_count")) + _int(proposal.payload.get("increment_usage_count"))
                stats["success_count"] = _int(stats.get("success_count")) + _int(proposal.payload.get("increment_success_count"))
                stats["failure_count"] = _int(stats.get("failure_count")) + _int(proposal.payload.get("increment_failure_count"))
                stats["last_status"] = str(proposal.payload.get("status") or "unknown")
                stats["last_observed_at"] = proposal.created_at
                if json.dumps(stats, sort_keys=True) != before:
                    changed = True
                    touched.append(target["skill_id"])
            elif proposal.proposal_type == "metadata_update":
                mutation = _apply_trace2skill_metadata_patch(skill, proposal, config=config)
                if mutation["changed"]:
                    changed = True
                    touched.append(target["skill_id"])
        return {
            "changed": changed,
            "reason": "applied" if changed else "no matching mutable skill entries",
            "touched_skill_ids": [skill_id for skill_id in touched if skill_id],
        }

    def _skill_targets(
        self,
        raw_graph: dict[str, Any],
        skill_ids: Iterable[str],
        package_updates: dict[Path, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        targets = []
        for raw_skill in _raw_skill_entries(raw_graph, skill_ids):
            skill_id = _raw_skill_id(raw_skill)
            if "skill_id" in raw_skill:
                targets.append({"skill_id": skill_id, "payload": raw_skill, "kind": "embedded"})
                continue
            package_path = _package_metadata_path(self.graph_backend, raw_skill)
            if package_path is None:
                continue
            pending = _pending_package_payload(package_path, package_updates)
            targets.append({"skill_id": skill_id, "payload": pending["payload"], "kind": "package", "path": package_path})
        return targets

    def _promote_candidate(
        self,
        raw_graph: dict[str, Any],
        proposal: SkillUpdateProposal,
        *,
        package_updates: dict[Path, dict[str, Any]],
    ) -> dict[str, Any]:
        record_payload = proposal.payload.get("candidate_record")
        if not isinstance(record_payload, dict):
            return {"changed": False, "rejected": True, "reason": "candidate payload missing candidate_record"}
        try:
            candidate = CandidateSkillRecord.model_validate(record_payload)
        except Exception as exc:
            return {"changed": False, "rejected": True, "reason": f"invalid candidate record: {exc}"}
        if not candidate.proposed_name or not candidate.missing_capability_description:
            return {"changed": False, "rejected": True, "reason": "candidate requires name and missing capability"}
        if self.graph_backend is None:
            return {"changed": False, "reason": "no graph backend configured"}

        existing_ids = {_raw_skill_id(skill) for skill in raw_graph.get("skills", []) if isinstance(skill, dict)}
        skill_id = _candidate_skill_id(candidate, existing_ids)
        package_ref = _candidate_package_ref(skill_id)
        metadata_path = Path(self.graph_backend.store.repo_root) / package_ref / "metadata.json"
        if metadata_path in package_updates or metadata_path.exists():
            skill_id = _candidate_skill_id(candidate, existing_ids | {skill_id})
            package_ref = _candidate_package_ref(skill_id)
            metadata_path = Path(self.graph_backend.store.repo_root) / package_ref / "metadata.json"

        package_payload = _candidate_package_payload(candidate, skill_id=skill_id, proposal=proposal)
        package_updates[metadata_path] = {
            "payload": package_payload,
            "original_text": metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else None,
            "created": not metadata_path.exists(),
            "skill_markdown_text": f"# {candidate.proposed_name}\n\n{candidate.missing_capability_description}\n",
        }
        raw_graph.setdefault("skills", []).append(
            {
                "id": skill_id,
                "name": candidate.proposed_name,
                "summary": candidate.missing_capability_description,
                "package_ref": package_ref,
                "status": "active",
                "tags": ["trace2skill"],
                "metadata": {
                    "trace2skill_candidate": candidate.model_dump(mode="json"),
                    "source_proposal_id": proposal.proposal_id,
                },
            }
        )
        edges_added = []
        if candidate.proposed_category and _category_exists(raw_graph, candidate.proposed_category):
            edge = {
                "source_id": skill_id,
                "target_id": candidate.proposed_category,
                "relation": "belongs_to_category",
            }
            raw_graph.setdefault("edges", []).append(edge)
            edges_added.append(edge)
        return {
            "changed": True,
            "reason": "candidate promoted",
            "touched_skill_ids": [skill_id],
            "package_ref": package_ref,
            "edges_added": edges_added,
        }

    def _apply_relationship_update(self, raw_graph: dict[str, Any], proposal: SkillUpdateProposal) -> dict[str, Any]:
        from evolab.backends.skills.trace2skill.conflicts import SUPPORTED_RELATIONS

        payload = proposal.payload.get("relationship_update")
        if not isinstance(payload, dict):
            return {"changed": False, "rejected": True, "reason": "relationship payload missing relationship_update"}
        relations = _relations(payload)
        if not relations:
            return {"changed": False, "rejected": True, "reason": "relationship update has no relations"}
        skill_ids = {_raw_skill_id(skill) for skill in raw_graph.get("skills", []) if isinstance(skill, dict)}
        existing = {
            (edge.get("source_id"), edge.get("target_id"), edge.get("relation"))
            for edge in raw_graph.get("edges", [])
            if isinstance(edge, dict)
        }
        added = []
        for relation in relations:
            relation_type = relation.get("relation")
            source_id = relation.get("source_skill_id") or relation.get("source_id")
            target_id = relation.get("target_skill_id") or relation.get("target_id")
            if relation_type not in SUPPORTED_RELATIONS:
                return {"changed": False, "rejected": True, "reason": f"unsupported relation type: {relation_type}"}
            if source_id not in skill_ids or target_id not in skill_ids:
                return {"changed": False, "rejected": True, "reason": "relationship references missing skill id"}
            key = (source_id, target_id, relation_type)
            if key in existing:
                continue
            edge = {"source_id": source_id, "target_id": target_id, "relation": relation_type}
            raw_graph.setdefault("edges", []).append(edge)
            existing.add(key)
            added.append(edge)
        return {
            "changed": bool(added),
            "reason": "relationship applied" if added else "relationship already exists",
            "touched_skill_ids": sorted({edge["source_id"] for edge in added} | {edge["target_id"] for edge in added}),
            "edges_added": added,
        }

    def _load_raw_graph(self) -> dict[str, Any] | None:
        if self.graph_backend is None:
            return None
        return self.graph_backend._load_raw_graph()

    def _graph_hash(self) -> str | None:
        raw_graph = self._load_raw_graph()
        if raw_graph is None:
            return None
        return _library_hash(raw_graph, self.graph_backend)

    def _graph_version(self) -> str | None:
        raw_graph = self._load_raw_graph()
        version = raw_graph.get("version") if raw_graph else None
        return version if isinstance(version, str) else None

    def _run_regression_gate_if_needed(
        self,
        *,
        config: Trace2SkillRunConfig,
        pool: TracePool,
        proposals: list[SkillUpdateProposal],
        decisions: list[SkillUpdateDecision],
        before_hash: str | None,
    ):
        if not config.enable_regression_gate:
            return None
        if not any(decision.decision == "applied" for decision in decisions):
            return None
        after_ref = _stable_id(
            "proposed-library",
            [
                before_hash,
                [proposal.model_dump(mode="json") for proposal in proposals],
                [decision.model_dump(mode="json") for decision in decisions],
            ],
        )
        tasks = _benchmark_tasks_from_config_or_pool(config, pool)
        return SkillEvolutionRegressionGate(
            benchmark_runner=self.benchmark_runner,
            metrics=config.regression_metrics,
            no_regression_threshold=config.regression_no_regression_threshold,
            graph_backend=self.graph_backend,
        ).evaluate(
            tasks,
            before_snapshot_ref="before" if self.benchmark_runner is not None else str(before_hash),
            after_snapshot_ref="after" if self.benchmark_runner is not None else after_ref,
            graph_backend=self.graph_backend,
        )

    def _output_dir(self, config: Trace2SkillRunConfig, run_id: str) -> Path:
        if config.output_dir:
            return Path(config.output_dir)
        if self.output_dir is not None:
            return self.output_dir / run_id
        if self.graph_backend is not None and hasattr(self.graph_backend, "evolution_root"):
            return Path(self.graph_backend.evolution_root) / "trace2skill" / run_id
        return Path("backend_state") / "skill_evolution" / "trace2skill" / run_id


def _raw_skill_entries(raw_graph: dict[str, Any], skill_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(skill_ids)
    entries = []
    for raw_skill in raw_graph.get("skills", []):
        if not isinstance(raw_skill, dict):
            continue
        skill_id = _raw_skill_id(raw_skill)
        if skill_id in wanted:
            entries.append(raw_skill)
    return entries


def _raw_skill_id(raw_skill: dict[str, Any]) -> str:
    value = raw_skill.get("skill_id") or raw_skill.get("id")
    return value if isinstance(value, str) else ""


def _package_metadata_path(graph_backend: Any | None, raw_skill: dict[str, Any]) -> Path | None:
    if graph_backend is None:
        return None
    package_ref = raw_skill.get("package_ref")
    if not isinstance(package_ref, str) or not package_ref:
        return None
    package_dir = Path(package_ref)
    if not package_dir.is_absolute():
        package_dir = Path(graph_backend.store.repo_root) / package_dir
    json_path = package_dir / "metadata.json"
    yaml_path = package_dir / "metadata.yaml"
    if json_path.exists():
        return json_path
    if yaml_path.exists():
        return yaml_path
    return json_path


def _pending_package_payload(path: Path, package_updates: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    if path in package_updates:
        return package_updates[path]
    if path.exists():
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix == ".json" else _load_yaml_or_json(text, path)
    else:
        text = None
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"skill package metadata must be an object: {path}")
    package_updates[path] = {"payload": payload, "original_text": text, "created": not path.exists()}
    return package_updates[path]


def _metadata(raw_skill: dict[str, Any]) -> dict[str, Any]:
    metadata = raw_skill.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        raw_skill["metadata"] = metadata
    return metadata


def _evolution_metadata(raw_skill: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(raw_skill)
    evolution = metadata.setdefault("evolution", {})
    if not isinstance(evolution, dict):
        evolution = {}
        metadata["evolution"] = evolution
    return evolution


def _apply_trace2skill_metadata_patch(
    skill: dict[str, Any],
    proposal: SkillUpdateProposal,
    *,
    config: Trace2SkillRunConfig,
) -> dict[str, Any]:
    patch = proposal.payload.get("trace2skill_patch")
    if not isinstance(patch, dict):
        evolution = _evolution_metadata(skill)
        changed = _append_bounded(
            evolution,
            "provenance",
            [json.dumps(proposal.payload, sort_keys=True)],
            limit=10,
            max_chars=config.max_evolution_text_chars,
        )
        return {"changed": changed}
    patch_type = patch.get("patch_type")
    content = patch.get("merged_content")
    content = content if isinstance(content, dict) else {}
    evolution = _evolution_metadata(skill)
    changed = False
    if patch_type == "procedure_step_patch":
        steps = _strings(content.get("procedure_steps"))
        changed = _append_field_list(
            skill,
            "procedure",
            steps,
            limit=config.max_procedure_notes_per_skill,
            max_chars=config.max_evolution_text_chars,
        ) or changed
        changed = _append_bounded(
            evolution,
            "procedure_notes",
            steps,
            limit=config.max_procedure_notes_per_skill,
            max_chars=config.max_evolution_text_chars,
        ) or changed
    elif patch_type == "precondition_patch":
        changed = _append_bounded(
            evolution,
            "preconditions",
            _strings(content.get("preconditions")),
            limit=config.max_preconditions_per_skill,
            max_chars=config.max_evolution_text_chars,
        )
    elif patch_type == "validation_rule_patch":
        changed = _append_bounded(
            evolution,
            "validation_rules",
            _strings(content.get("validation_rules")),
            limit=config.max_validation_rules_per_skill,
            max_chars=config.max_evolution_text_chars,
        )
    elif patch_type == "metadata_patch":
        changed = _append_bounded(
            evolution,
            "provenance",
            [json.dumps(content, sort_keys=True)],
            limit=10,
            max_chars=config.max_evolution_text_chars,
        )
    elif patch_type == "skill_deepen_patch":
        changed = _append_bounded(
            evolution,
            "procedure_notes",
            _strings(content.get("procedure_steps")) or [json.dumps(content, sort_keys=True)],
            limit=config.max_procedure_notes_per_skill,
            max_chars=config.max_evolution_text_chars,
        )
    return {"changed": changed}


def _append_field_list(
    payload: dict[str, Any],
    key: str,
    values: list[str],
    *,
    limit: int,
    max_chars: int,
) -> bool:
    current = payload.get(key)
    existing = current if isinstance(current, list) else []
    before = list(existing)
    merged = _dedupe([*(item for item in existing if isinstance(item, str)), *[_bounded(value, max_chars) for value in values if value]])
    payload[key] = merged[-limit:]
    return payload[key] != before


def _append_bounded(
    payload: dict[str, Any],
    key: str,
    values: list[str],
    *,
    limit: int,
    max_chars: int,
) -> bool:
    return _append_field_list(payload, key, values, limit=limit, max_chars=max_chars)


def _validate_pending_library(
    graph_backend: Any,
    raw_graph: dict[str, Any],
    package_updates: dict[Path, dict[str, Any]],
) -> None:
    from evolab.backends.skills.graph_schema import SkillGraph
    from evolab.backends.skills.package_schema import SkillPackage

    SkillGraph.model_validate(raw_graph)
    repo_root = Path(graph_backend.store.repo_root)
    for path, pending in package_updates.items():
        payload = pending["payload"]
        skill_markdown_path = path.parent / "SKILL.md"
        skill_markdown = pending.get("skill_markdown_text")
        if skill_markdown is None and skill_markdown_path.exists():
            skill_markdown = skill_markdown_path.read_text(encoding="utf-8")
        SkillPackage.model_validate(
            {
                **payload,
                "package_path": _display_path(path.parent, repo_root),
                "skill_markdown": skill_markdown,
            }
        )


def _write_pending_library(
    graph_backend: Any,
    raw_graph: dict[str, Any],
    package_updates: dict[Path, dict[str, Any]],
) -> None:
    for path, pending in package_updates.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_metadata_atomic(path, pending["payload"])
        skill_markdown = pending.get("skill_markdown_text")
        if isinstance(skill_markdown, str) and skill_markdown:
            skill_path = path.parent / "SKILL.md"
            if not skill_path.exists():
                skill_path.write_text(skill_markdown, encoding="utf-8")
    _write_graph_atomic(Path(graph_backend.graph_path), raw_graph)


def _restore_pending_library(package_updates: dict[Path, dict[str, Any]]) -> None:
    for path, pending in package_updates.items():
        original = pending.get("original_text")
        if original is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(original, encoding="utf-8")


def _write_metadata_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json" or yaml is None:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _write_graph_atomic(path: Path, raw_graph: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(raw_graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _library_hash(raw_graph: dict[str, Any], graph_backend: Any | None) -> str:
    payload: dict[str, Any] = {"graph": raw_graph, "packages": {}}
    if graph_backend is not None:
        for raw_skill in raw_graph.get("skills", []):
            if not isinstance(raw_skill, dict):
                continue
            path = _package_metadata_path(graph_backend, raw_skill)
            if path is not None and path.exists():
                payload["packages"][str(path)] = path.read_text(encoding="utf-8")
    encoded = json.dumps(_json_compatible(payload), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _graph_hash(raw_graph: dict[str, Any] | None) -> str | None:
    if raw_graph is None:
        return None
    encoded = json.dumps(_json_compatible(raw_graph), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _relations(content: dict[str, Any]) -> list[dict[str, Any]]:
    relation = content.get("relation")
    relation_updates = content.get("relations") or content.get("relation_updates")
    values = []
    if isinstance(relation, dict):
        values.append(relation)
    if isinstance(relation_updates, list):
        values.extend(item for item in relation_updates if isinstance(item, dict))
    return values


def _candidate_skill_id(candidate: CandidateSkillRecord, existing_ids: set[str]) -> str:
    base_text = candidate.proposed_name or candidate.missing_capability_description
    slug = _slug(base_text) or "generated_skill"
    digest = sha256(
        json.dumps(candidate.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    value = f"skill.trace2skill.{slug}.{digest}.v1"
    if value not in existing_ids:
        return value
    suffix = 1
    while f"skill.trace2skill.{slug}.{digest}.{suffix}.v1" in existing_ids:
        suffix += 1
    return f"skill.trace2skill.{slug}.{digest}.{suffix}.v1"


def _candidate_package_ref(skill_id: str) -> str:
    return f"skills/trace2skill/generated/{skill_id.replace('.', '_')}"


def _candidate_package_payload(
    candidate: CandidateSkillRecord,
    *,
    skill_id: str,
    proposal: SkillUpdateProposal,
) -> dict[str, Any]:
    task_types = [candidate.proposed_task] if candidate.proposed_task else ["generic"]
    required_inputs = candidate.suggested_required_inputs or ["task goal", "available evidence"]
    expected_outputs = candidate.suggested_expected_outputs or ["validated task output"]
    return {
        "schema_version": "v1",
        "skill_id": skill_id,
        "name": candidate.proposed_name,
        "version": "v1",
        "summary": candidate.missing_capability_description,
        "source_type": "package",
        "source_uri": f"trace2skill://{candidate.candidate_id}",
        "provenance": {
            "source": "trace2skill",
            "source_observation_id": candidate.source_observation_id,
            "source_proposal_id": proposal.proposal_id,
            "source_trace_ids": candidate.metadata.get("source_trace_ids", []),
        },
        "domain_tags": ["generic"],
        "task_types": task_types,
        "target_category": candidate.proposed_category,
        "scope": candidate.missing_capability_description,
        "applicability": [candidate.missing_capability_description],
        "limitations": ["Generated from bounded Trace2Skill evidence; validate on future runs."],
        "required_inputs": required_inputs,
        "expected_outputs": expected_outputs,
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": ["Apply the reusable pattern described by the source traces."],
        "required_tools": candidate.suggested_required_tools,
        "scripts": [],
        "resources": [],
        "examples": [candidate.evidence_summary],
        "tests": {"smoke": [], "synthetic": [], "system": [], "benchmark": []},
        "validation_signals": ["trace2skill_candidate_validation"],
        "confidence": 0.5,
        "metadata": {
            "evolution": {
                "provenance": [candidate.evidence_summary],
                "source_candidate": candidate.model_dump(mode="json"),
            }
        },
    }


def _category_exists(raw_graph: dict[str, Any], category_id: str) -> bool:
    return any(
        isinstance(category, dict) and category.get("category_id") == category_id
        for category in raw_graph.get("categories", [])
    )


def _changed_skill_ids(applied: list[dict[str, Any]]) -> list[str]:
    values = []
    for update in applied:
        mutation = update.get("mutation", {})
        if isinstance(mutation, dict):
            values.extend(_strings(mutation.get("touched_skill_ids")))
    return _dedupe(values)


def _retrieval_change_summary(transaction: SkillLibraryUpdateTransaction) -> dict[str, Any]:
    return {
        "changed_skill_ids": _changed_skill_ids(transaction.applied_updates),
        "applied_update_count": len(transaction.applied_updates),
        "staged_update_count": len(transaction.staged_updates),
        "rejected_update_count": len(transaction.rejected_updates),
    }


def _benchmark_tasks_from_config_or_pool(config: Trace2SkillRunConfig, pool: TracePool) -> list[BenchmarkTask]:
    raw_tasks = config.metadata.get("benchmark_tasks")
    if isinstance(raw_tasks, list):
        tasks = []
        for item in raw_tasks:
            if isinstance(item, BenchmarkTask):
                tasks.append(item)
            elif isinstance(item, dict):
                tasks.append(BenchmarkTask.model_validate(item))
        return tasks
    return [
        BenchmarkTask(
            task_id=trace.task_id or trace.trace_id,
            query=trace.task_summary or trace.compact_execution_summary or trace.error_summary or trace.trace_id,
            expected_skill_ids=trace.selected_skill_ids,
        )
        for trace in pool.traces
    ]


def _load_yaml_or_json(text: str, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ModuleNotFoundError(f"PyYAML is required to parse non-JSON YAML metadata: {path}")
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"skill package metadata must be an object: {path}")
    return payload


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    chars = []
    last_was_sep = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_")[:48]


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _bounded(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
