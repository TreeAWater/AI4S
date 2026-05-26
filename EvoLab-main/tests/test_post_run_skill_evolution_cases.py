from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.evolution import SkillEvolutionPolicy
from evolab.backends.skills.trace2skill import (
    HierarchicalPatchConsolidator,
    SkillPatchProposal,
    SkillPatchValidator,
    Trace2SkillEvolver,
    Trace2SkillLLMExtractor,
    Trace2SkillRunConfig,
    Trace2SkillSkillBackendAdapter,
    TracePoolBuilder,
    TraceRecord,
    TrajectoryLesson,
)
from evolab.contracts.retrieval import RetrievalRequest, SkillObservationRequest
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolSpec, ToolTrace
from evolab.tools.runtime import ToolRegistry


REPORT_ROOT = Path("/tmp/evolab-post-run-skill-evolution-cases")
TABLE_SKILL_ID = "skill.generic_table_extraction.v1"
VALIDATION_SKILL_ID = "skill.generic_schema_validation.v1"


@dataclass
class PipelineResult:
    backend: GraphSkillBackend
    trace: TraceRecord
    lessons: list[TrajectoryLesson]
    patches: list[SkillPatchProposal]
    consolidation: Any
    validation: Any
    proposals: list[Any]
    decisions: list[Any]
    transaction: Any
    before_graph: dict[str, Any]
    after_graph: dict[str, Any]


def test_controlled_post_run_skill_evolution_cases(tmp_path: Path):
    """Demonstrate controlled trajectory-to-skill evolution without external LLM calls."""

    if REPORT_ROOT.exists():
        shutil.rmtree(REPORT_ROOT)
    REPORT_ROOT.mkdir(parents=True)

    report = _CaseReport(REPORT_ROOT)

    direct_result = _run_direct_look_at_missing_tool_case(tmp_path / "direct-look-at")
    report.add(
        case="A1 direct look_at",
        trace_summary="Tool trace read_table_slice ok, detect_table_header failed, profile_table missing.",
        lesson="Missing required tool surfaced from post-run observation.",
        patch_type="required_tools_update_proposal",
        decision=_decision_for_proposal_type(
            direct_result.metadata["proposals"],
            direct_result.metadata["decisions"],
            "required_tools_update_proposal",
        ),
        result="Conservative look_at staged profile_table and left required_tools unchanged.",
        passed=True,
        details={
            "trace_id": "obs-missing-tool",
            "source_skill_id": TABLE_SKILL_ID,
            "before_state": {"required_tools": ["read_table_slice", "detect_table_header"]},
            "generated_proposals": direct_result.metadata["proposals"],
            "policy_decisions": direct_result.metadata["decisions"],
            "after_state": {"required_tools": _raw_skill(tmp_path / "direct-look-at" / "skills.json", TABLE_SKILL_ID)["required_tools"]},
            "staged_proposal_ids": [
                item["proposal_id"]
                for item in direct_result.metadata["staged_updates"]
                if item["proposal_type"] == "required_tools_update_proposal"
            ],
            "audit_paths": direct_result.metadata["output_paths"],
        },
    )

    case_a_conservative = _run_case_a_missing_tool(
        tmp_path / "case-a-conservative",
        policy=_conservative_policy(),
        policy_name="conservative",
    )
    case_a_validated = _run_case_a_missing_tool(
        tmp_path / "case-a-validated-tools",
        policy=_validated_tools_policy(),
        policy_name="auto_apply_validated_tools",
    )
    assert _proposal_types(case_a_conservative) >= {"required_tools_update_proposal", "metadata_update", "failure_note_update"}
    assert _skill_required_tools(case_a_conservative.after_graph, TABLE_SKILL_ID) == [
        "read_table_slice",
        "detect_table_header",
    ]
    assert "profile_table" in _skill_required_tools(case_a_validated.after_graph, TABLE_SKILL_ID)
    reloaded_bundle = GraphSkillBackend(case_a_validated.backend.graph_path, repo_root=case_a_validated.backend.store.repo_root).get(
        RetrievalRequest(task_id="reload", role="solver", query="structured document table extraction")
    )
    assert any(skill.skill_id == TABLE_SKILL_ID and "profile_table" in skill.required_tools for skill in reloaded_bundle.skills)
    report.add_pipeline_case(
        "A2 Trace2Skill missing tool",
        case_a_validated,
        "When header detection fails, profile the table before rejecting it.",
        "required_tools_patch + procedure_step_patch + failure_case_patch",
        "auto_apply_validated_tools applied required_tools; conservative staged it.",
        "profile_table persisted in required_tools after reload.",
        True,
        extra={
            "conservative_decisions": [decision.model_dump(mode="json") for decision in case_a_conservative.decisions],
            "before_required_tools": _skill_required_tools(case_a_validated.before_graph, TABLE_SKILL_ID),
            "after_required_tools": _skill_required_tools(case_a_validated.after_graph, TABLE_SKILL_ID),
        },
    )

    case_b = _run_case_b_success_example(tmp_path / "case-b-examples")
    examples = _raw_skill(case_b.backend.graph_path, TABLE_SKILL_ID).get("examples", [])
    evolution_examples = _raw_skill(case_b.backend.graph_path, TABLE_SKILL_ID)["metadata"]["evolution"]["examples"]
    assert _proposal_types(case_b) >= {"example_trace_memory_update", "metadata_update"}
    assert any("Normalize table before schema validation" in example for example in examples)
    assert any("Normalize table before schema validation" in example for example in evolution_examples)
    assert len(examples) <= 2
    report.add_pipeline_case(
        "B successful reusable strategy",
        case_b,
        "Normalize table before schema validation and JSONL writing.",
        "example_memory_patch + procedure_step_patch",
        "auto_apply_bounded_examples applied bounded example memory.",
        "Example appended to skill examples and metadata.evolution.examples with max_examples=2.",
        True,
        extra={"after_examples": examples, "after_evolution_examples": evolution_examples},
    )

    case_c = _run_case_c_schema_failure(tmp_path / "case-c-validation-rule")
    validation_rules = _raw_skill(case_c.backend.graph_path, TABLE_SKILL_ID)["metadata"]["evolution"]["validation_rules"]
    failure_cases = _raw_skill(case_c.backend.graph_path, TABLE_SKILL_ID)["metadata"]["evolution"]["failure_cases"]
    assert "metadata_update" in _proposal_types(case_c)
    assert any("Validate output against schema before final write" in rule for rule in validation_rules)
    assert failure_cases
    report.add_pipeline_case(
        "C output schema failure",
        case_c,
        "Validate output against schema before final write.",
        "validation_rule_patch + failure_case_patch",
        "bounded metadata policy applied validation rule and failure note.",
        "Validation rule persisted under metadata.evolution.validation_rules.",
        True,
        extra={"validation_rules": validation_rules, "failure_cases": failure_cases},
    )

    case_d_conservative = _run_case_d_candidate(
        tmp_path / "case-d-conservative",
        policy=_conservative_policy(),
        policy_name="conservative",
    )
    case_d_promoted = _run_case_d_candidate(
        tmp_path / "case-d-promoted",
        policy=SkillEvolutionPolicy(auto_apply_valid_candidates=True),
        policy_name="auto_apply_valid_candidates",
    )
    promoted_bundle = GraphSkillBackend(
        case_d_promoted.backend.graph_path,
        repo_root=case_d_promoted.backend.store.repo_root,
    ).get(
        RetrievalRequest(
            task_id="retrieval",
            role="solver",
            query="cross artifact entity linking across multiple artifacts",
            metadata={"top_k": 5},
        )
    )
    promoted_ids = [skill.skill_id for skill in promoted_bundle.skills]
    assert "candidate_skill_creation" in _proposal_types(case_d_conservative)
    assert case_d_conservative.transaction.changed_library is False
    assert case_d_conservative.transaction.before_graph_hash == case_d_conservative.transaction.after_graph_hash
    assert case_d_promoted.transaction.changed_library is True
    assert case_d_promoted.transaction.before_graph_hash != case_d_promoted.transaction.after_graph_hash
    assert any(skill_id.startswith("skill.trace2skill.") for skill_id in promoted_ids)
    report.add_pipeline_case(
        "D low retrieval coverage",
        case_d_promoted,
        "A new skill is needed for cross-artifact entity linking.",
        "skill_create_patch",
        "conservative staged candidate; auto_apply_valid_candidates promoted it.",
        "Promoted candidate skill is retrievable by cross-artifact linking query.",
        True,
        extra={
            "conservative_transaction": case_d_conservative.transaction.model_dump(mode="json"),
            "promoted_skill_packages": _package_payloads_from_transaction(case_d_promoted.transaction),
            "retrieval_result_skill_ids": promoted_ids,
        },
    )

    case_e_conservative = _run_case_e_relationship(
        tmp_path / "case-e-conservative",
        policy=_conservative_policy(),
        policy_name="conservative",
    )
    case_e_applied = _run_case_e_relationship(
        tmp_path / "case-e-applied",
        policy=SkillEvolutionPolicy(auto_apply_relationship_updates=True),
        policy_name="auto_apply_relationship_updates",
    )
    duplicate_transaction = _apply_existing_pipeline_patches(
        case_e_applied.backend,
        case_e_applied.patches,
        policy=SkillEvolutionPolicy(auto_apply_relationship_updates=True),
        config=_run_config(),
    ).transaction
    invalid_relation = _run_invalid_relationship_case(tmp_path / "case-e-invalid")
    edges = _raw_graph(case_e_applied.backend.graph_path)["edges"]
    assert "relationship_update_proposal" in _proposal_types(case_e_applied)
    assert case_e_conservative.transaction.changed_library is False
    assert edges == [{"source_id": TABLE_SKILL_ID, "target_id": VALIDATION_SKILL_ID, "relation": "validates"}]
    assert duplicate_transaction.changed_library is False
    assert len(_raw_graph(case_e_applied.backend.graph_path)["edges"]) == 1
    assert invalid_relation.validation.invalid_patches
    report.add_pipeline_case(
        "E relationship update",
        case_e_applied,
        "Extraction skill should be related to validation skill.",
        "relationship_patch",
        "conservative staged; explicit relation policy applied valid relation.",
        "validates edge added once, duplicate deduped, invalid relation rejected safely.",
        True,
        extra={
            "conservative_transaction": case_e_conservative.transaction.model_dump(mode="json"),
            "duplicate_transaction": duplicate_transaction.model_dump(mode="json"),
            "invalid_relation_warnings": invalid_relation.validation.warnings,
            "edges": edges,
        },
    )

    fake_llm_result, fake_runtime = _run_optional_fake_llm_case(tmp_path / "case-f-fake-llm")
    assert len(fake_runtime.requests) == 1
    assert fake_llm_result.transaction.changed_library is True
    report.add_pipeline_case(
        "F fake LLM extraction",
        fake_llm_result,
        "LLM returned a schema-first validation lesson as JSON.",
        "validation_rule_patch",
        "bounded metadata policy applied parsed fake-LLM validation rule.",
        "Fake LLM request used EvoLab LLMRuntime without network/API calls.",
        True,
        extra={"fake_llm_request_count": len(fake_runtime.requests)},
    )

    report.write()
    _assert_jsonl_loads(REPORT_ROOT / "lessons.jsonl")
    _assert_jsonl_loads(REPORT_ROOT / "patches.jsonl")
    _assert_jsonl_loads(REPORT_ROOT / "proposals.jsonl")
    _assert_jsonl_loads(REPORT_ROOT / "decisions.jsonl")
    _assert_jsonl_loads(REPORT_ROOT / "transactions.jsonl")
    assert (REPORT_ROOT / "report.md").exists()
    assert "Trajectory summary" in (REPORT_ROOT / "report.md").read_text(encoding="utf-8")
    assert not Path("skills/seed").exists()


def _run_direct_look_at_missing_tool_case(root: Path):
    root.mkdir(parents=True)
    graph_path = _write_graph(root, skills=[_table_skill()])
    backend = GraphSkillBackend(graph_path, repo_root=root, evolution_root=root / "skill_evolution")
    bundle = backend.get(RetrievalRequest(task_id="task-a", role="solver", query="structured document table extraction"))
    observation = SkillObservationRequest(
        task_id="task-a",
        run_ref="run-a",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-a", role="solver", query="structured document table extraction"),
        skill_bundle=bundle,
        graph_version_ref=bundle.graph_version_ref,
        skill_state_ref=bundle.skill_state_ref,
        tool_trace=ToolTrace(
            run_ref="run-a",
            calls=[
                ToolCallRecord(
                    tool_call=ToolCall(call_id="tool-1", name="read_table_slice", arguments={"path": "table.csv"}),
                    result=ToolResult(call_id="tool-1", status="ok", content="slice read"),
                ),
                ToolCallRecord(
                    tool_call=ToolCall(call_id="tool-2", name="detect_table_header", arguments={"path": "table.csv"}),
                    result=ToolResult(
                        call_id="tool-2",
                        status="error",
                        content="header ambiguous",
                        metadata={"error_type": "missing_required_tool", "missing_tools": ["profile_table"]},
                    ),
                ),
            ],
        ),
        metadata={
            "status": "failed",
            "missing_required_tools": ["profile_table"],
            "failure_reason": "missing/unused profiling step caused early rejection of a recoverable table",
        },
    )

    result = backend.look_at(observation)
    required_tool_proposals = [
        proposal for proposal in result.metadata["proposals"] if proposal["proposal_type"] == "required_tools_update_proposal"
    ]
    assert required_tool_proposals
    assert result.metadata["output_paths"]["proposals"].endswith("proposals.jsonl")
    _assert_jsonl_loads(Path(result.metadata["output_paths"]["proposals"]))
    assert "profile_table" not in _raw_skill(graph_path, TABLE_SKILL_ID)["required_tools"]
    return result


def _run_case_a_missing_tool(root: Path, *, policy: SkillEvolutionPolicy, policy_name: str) -> PipelineResult:
    trace = _trace(
        trace_id=f"trace-a-{policy_name}",
        task_summary="Inspect a structured document table and identify relevant columns.",
        selected_skill_ids=[TABLE_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID],
        tools_used=["read_table_slice", "detect_table_header"],
        missing_tools=["profile_table"],
        final_status="runtime_failure",
        error_summary="missing/unused profiling step caused early rejection of a recoverable table",
    )
    lesson = _lesson(
        "lesson-a-tool",
        trace,
        lesson_type="tool_lesson",
        target_skill_id=TABLE_SKILL_ID,
        principle="When header detection fails, profile the table before rejecting it.",
        delta={"missing_tools": ["profile_table"]},
    )
    patches = [
        _patch("patch-a-tools", "required_tools_patch", lesson, {"required_tools": ["profile_table"]}, risk_level="high"),
        _patch(
            "patch-a-procedure",
            "procedure_step_patch",
            lesson,
            {"procedure_steps": ["When header detection fails, profile the table before rejecting it."]},
        ),
        _patch(
            "patch-a-failure",
            "failure_case_patch",
            lesson,
            {"failure_reason": "Header detection failure should trigger profiling before rejection."},
        ),
    ]
    return _run_pipeline(root, trace=trace, lessons=[lesson], patches=patches, policy=policy)


def _run_case_b_success_example(root: Path) -> PipelineResult:
    trace = _trace(
        trace_id="trace-b-success",
        task_summary="Extract structured records from document text plus table artifacts.",
        selected_skill_ids=[TABLE_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID],
        tools_used=["inspect_file_metadata", "inspect_table", "normalize_table", "json_schema_validate", "write_jsonl"],
        final_status="runtime_success",
        evaluation_metrics={"recall": 0.96, "json_valid": True},
        compact_execution_summary="Normalized table before schema validation and JSONL writing.",
    )
    lesson = _lesson(
        "lesson-b-success",
        trace,
        lesson_type="success_lesson",
        target_skill_id=TABLE_SKILL_ID,
        principle="Normalize table before schema validation and JSONL writing.",
        delta={
            "example_summary": "Normalize table before schema validation and JSONL writing.",
            "procedure_steps": ["Normalize table before schema validation and JSONL writing."],
        },
    )
    patches = [
        _patch(
            "patch-b-example",
            "example_memory_patch",
            lesson,
            {"example_summary": "Normalize table before schema validation and JSONL writing."},
            risk_level="low",
        ),
        _patch(
            "patch-b-procedure",
            "procedure_step_patch",
            lesson,
            {"procedure_steps": ["Normalize table before schema validation and JSONL writing."]},
        ),
    ]
    return _run_pipeline(
        root,
        trace=trace,
        lessons=[lesson],
        patches=patches,
        policy=_bounded_examples_policy(),
        skills=[_table_skill(examples=["Existing bounded example."])],
        config=_run_config(max_examples_per_skill=2),
    )


def _run_case_c_schema_failure(root: Path) -> PipelineResult:
    trace = _trace(
        trace_id="trace-c-schema",
        task_summary="Produce structured extraction output.",
        selected_skill_ids=[TABLE_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID],
        tools_used=["inspect_table", "write_jsonl", "json_schema_validate"],
        final_status="runtime_failure",
        error_summary="json_schema_validate failed because output was missing a required field",
    )
    lesson = _lesson(
        "lesson-c-validation",
        trace,
        lesson_type="validation_lesson",
        target_skill_id=TABLE_SKILL_ID,
        principle="Validate output against schema before final write.",
        delta={"validation_rules": ["Validate output against schema before final write."]},
    )
    patches = [
        _patch(
            "patch-c-validation",
            "validation_rule_patch",
            lesson,
            {"validation_rules": ["Validate output against schema before final write."]},
        ),
        _patch(
            "patch-c-failure",
            "failure_case_patch",
            lesson,
            {"failure_reason": "Output missing required field or wrong JSON shape."},
        ),
    ]
    return _run_pipeline(root, trace=trace, lessons=[lesson], patches=patches, policy=_bounded_metadata_policy())


def _run_case_d_candidate(root: Path, *, policy: SkillEvolutionPolicy, policy_name: str) -> PipelineResult:
    trace = _trace(
        trace_id=f"trace-d-{policy_name}",
        task_summary="Link entity IDs across multiple artifacts.",
        retrieved_skill_ids=[],
        selected_skill_ids=[],
        tools_used=["inspect_file_metadata", "read_text", "inspect_table"],
        final_status="partial_success",
        error_summary="No existing skill covered cross-artifact entity linking.",
        metadata={"coverage": "low"},
    )
    lesson = _lesson(
        "lesson-d-coverage",
        trace,
        lesson_type="coverage_lesson",
        target_skill_id=None,
        principle="A new skill is needed for cross-artifact entity linking.",
        delta={
            "missing_capability": "cross artifact entity linking across multiple artifacts",
            "suggested_required_tools": ["inspect_file_metadata", "read_text", "inspect_table"],
        },
    )
    patch = _patch(
        "patch-d-candidate",
        "skill_create_patch",
        lesson,
        {
            "candidate_id": "candidate.cross_artifact_entity_linking.v1",
            "proposed_name": "Cross Artifact Entity Linking",
            "missing_capability_description": "cross artifact entity linking across multiple artifacts",
            "suggested_required_inputs": ["multiple artifacts", "entity identifiers"],
            "suggested_expected_outputs": ["linked entity records"],
            "suggested_required_tools": ["inspect_file_metadata", "read_text", "inspect_table"],
        },
        candidate_skill_id="candidate.cross_artifact_entity_linking.v1",
        risk_level="high",
    )
    return _run_pipeline(
        root,
        trace=trace,
        lessons=[lesson],
        patches=[patch],
        policy=policy,
        skills=[_table_skill(), _validation_skill()],
    )


def _run_case_e_relationship(root: Path, *, policy: SkillEvolutionPolicy, policy_name: str) -> PipelineResult:
    trace = _trace(
        trace_id=f"trace-e-{policy_name}",
        task_summary="Extraction should be followed by validation.",
        selected_skill_ids=[TABLE_SKILL_ID, VALIDATION_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID, VALIDATION_SKILL_ID],
        tools_used=["inspect_table", "json_schema_validate"],
        final_status="runtime_success",
        compact_execution_summary="Validation was needed immediately after extraction.",
    )
    lesson = _lesson(
        "lesson-e-relation",
        trace,
        lesson_type="success_lesson",
        target_skill_id=TABLE_SKILL_ID,
        principle="Extraction skill should be related to validation skill.",
        delta={"relation": "validates"},
    )
    patch = _patch(
        "patch-e-relation",
        "relationship_patch",
        lesson,
        {
            "relation": {
                "source_skill_id": TABLE_SKILL_ID,
                "target_skill_id": VALIDATION_SKILL_ID,
                "relation": "validates",
            }
        },
        risk_level="medium",
    )
    return _run_pipeline(
        root,
        trace=trace,
        lessons=[lesson],
        patches=[patch],
        policy=policy,
        skills=[_table_skill(), _validation_skill()],
    )


def _run_invalid_relationship_case(root: Path) -> PipelineResult:
    trace = _trace(
        trace_id="trace-e-invalid",
        task_summary="Invalid relation should fail safely.",
        selected_skill_ids=[TABLE_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID],
        tools_used=["inspect_table"],
        final_status="runtime_failure",
    )
    lesson = _lesson(
        "lesson-e-invalid",
        trace,
        lesson_type="error_lesson",
        target_skill_id=TABLE_SKILL_ID,
        principle="Invalid relation references must be rejected.",
        delta={"relation": "validates"},
    )
    patch = _patch(
        "patch-e-invalid",
        "relationship_patch",
        lesson,
        {"relation": {"source_skill_id": TABLE_SKILL_ID, "target_skill_id": "skill.missing.v1", "relation": "validates"}},
    )
    return _run_pipeline(
        root,
        trace=trace,
        lessons=[lesson],
        patches=[patch],
        policy=SkillEvolutionPolicy(auto_apply_relationship_updates=True),
        skills=[_table_skill(), _validation_skill()],
    )


def _run_optional_fake_llm_case(root: Path) -> tuple[PipelineResult, FakeLLMRuntime]:
    root.mkdir(parents=True)
    graph_path = _write_graph(root, skills=[_table_skill()])
    backend = GraphSkillBackend(graph_path, repo_root=root, evolution_root=root / "state")
    trace = _trace(
        trace_id="trace-f-fake-llm",
        task_summary="Produce schema-constrained output from extracted rows.",
        selected_skill_ids=[TABLE_SKILL_ID],
        retrieved_skill_ids=[TABLE_SKILL_ID],
        tools_used=["inspect_table", "write_jsonl"],
        final_status="runtime_failure",
        error_summary="Final output failed schema validation.",
    )
    fake_payload = json.dumps(
        {
            "lessons": [
                {
                    "lesson_type": "validation_lesson",
                    "target_skill_id": TABLE_SKILL_ID,
                    "evidence_summary": "The trace failed because final output was not schema valid.",
                    "reusable_principle": "Validate candidate records against the requested schema before writing.",
                    "proposed_delta": {"validation_rules": ["Validate candidate records against the requested schema before writing."]},
                    "confidence": 0.86,
                    "source_trace_ids": [trace.trace_id],
                }
            ],
            "patches": [
                {
                    "patch_type": "validation_rule_patch",
                    "target_skill_id": TABLE_SKILL_ID,
                    "evidence_summary": "The trace failed because final output was not schema valid.",
                    "proposed_delta": {"validation_rules": ["Validate candidate records against the requested schema before writing."]},
                    "confidence": 0.84,
                    "risk_level": "medium",
                    "source_trace_ids": [trace.trace_id],
                }
            ],
        }
    )
    fake_runtime = FakeLLMRuntime(default_content=fake_payload)
    extractor = Trace2SkillLLMExtractor(llm_client=fake_runtime)
    lessons, patches = extractor.extract_lessons_and_patches(
        TracePoolBuilder().build_trace_pool([trace]),
        config=Trace2SkillRunConfig(enable_llm_analysts=True, dry_run=False, max_llm_retries=0),
    )
    result = _run_pipeline(
        root,
        trace=trace,
        lessons=lessons,
        patches=patches,
        policy=_bounded_metadata_policy(),
        backend=backend,
    )
    return result, fake_runtime


def _run_pipeline(
    root: Path,
    *,
    trace: TraceRecord,
    lessons: list[TrajectoryLesson],
    patches: list[SkillPatchProposal],
    policy: SkillEvolutionPolicy,
    skills: list[dict[str, Any]] | None = None,
    backend: GraphSkillBackend | None = None,
    config: Trace2SkillRunConfig | None = None,
) -> PipelineResult:
    root.mkdir(parents=True, exist_ok=True)
    if backend is None:
        graph_path = _write_graph(root, skills=deepcopy(skills or [_table_skill(), _validation_skill()]))
        backend = GraphSkillBackend(graph_path, repo_root=root, evolution_root=root / "state", evolution_policy=policy)
    before_graph = _raw_graph(backend.graph_path)
    return _apply_existing_pipeline_patches(backend, patches, policy=policy, config=config, trace=trace, lessons=lessons, before_graph=before_graph)


def _apply_existing_pipeline_patches(
    backend: GraphSkillBackend,
    patches: list[SkillPatchProposal],
    *,
    policy: SkillEvolutionPolicy,
    config: Trace2SkillRunConfig | None = None,
    trace: TraceRecord | None = None,
    lessons: list[TrajectoryLesson] | None = None,
    before_graph: dict[str, Any] | None = None,
) -> PipelineResult:
    config = config or _run_config()
    trace = trace or _trace(trace_id="trace-reapply")
    lessons = lessons or []
    before_graph = before_graph or _raw_graph(backend.graph_path)
    consolidation = HierarchicalPatchConsolidator(min_support_count=1, min_confidence=0.25).consolidate(patches)
    validation = SkillPatchValidator(graph_backend=backend, tool_registry=_tool_registry()).validate_patch_bundle(
        consolidation.consolidated_patches
    )
    proposals = Trace2SkillSkillBackendAdapter(
        backend_id=backend.backend_id,
        graph_version_ref="graph-v1",
    ).to_skill_update_proposals(validation.valid_patches)
    decisions = [policy.decide(proposal) for proposal in proposals]
    evolver = Trace2SkillEvolver(graph_backend=backend, tool_registry=_tool_registry(), policy=policy)
    transaction = evolver._apply_policy_gated_transaction(
        proposals=proposals,
        decisions=decisions,
        dry_run=False,
        before_graph_hash=evolver._graph_hash(),
        config=config,
    )
    return PipelineResult(
        backend=backend,
        trace=trace,
        lessons=lessons,
        patches=patches,
        consolidation=consolidation,
        validation=validation,
        proposals=proposals,
        decisions=decisions,
        transaction=transaction,
        before_graph=before_graph,
        after_graph=_raw_graph(backend.graph_path),
    )


def _run_config(**overrides: Any) -> Trace2SkillRunConfig:
    data = {
        "dry_run": False,
        "max_examples_per_skill": 5,
        "max_procedure_notes_per_skill": 10,
        "max_failure_cases_per_skill": 10,
        "max_validation_rules_per_skill": 10,
    }
    data.update(overrides)
    return Trace2SkillRunConfig(**data)


def _conservative_policy() -> SkillEvolutionPolicy:
    return SkillEvolutionPolicy()


def _validated_tools_policy() -> SkillEvolutionPolicy:
    return SkillEvolutionPolicy(
        auto_apply_proposal_types=[
            "usage_stats_update",
            "failure_note_update",
            "required_tools_update_proposal",
        ]
    )


def _bounded_examples_policy() -> SkillEvolutionPolicy:
    return SkillEvolutionPolicy(
        auto_apply_proposal_types=[
            "usage_stats_update",
            "failure_note_update",
            "example_trace_memory_update",
            "metadata_update",
        ]
    )


def _bounded_metadata_policy() -> SkillEvolutionPolicy:
    return SkillEvolutionPolicy(
        auto_apply_proposal_types=[
            "usage_stats_update",
            "failure_note_update",
            "metadata_update",
        ]
    )


def _write_graph(root: Path, *, skills: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "skills.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": skills,
                "categories": [],
                "edges": edges or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _table_skill(**overrides: Any) -> dict[str, Any]:
    data = _embedded_skill(
        TABLE_SKILL_ID,
        name="Generic Scientific Table Extraction",
        description="Inspect structured document tables and extract schema-constrained records from evidence.",
        task_types=["table_extraction", "scientific_extraction"],
        scope="Generic structured table extraction for scientific documents.",
        required_inputs=["document inventory", "candidate source table"],
        expected_outputs=["candidate records with source provenance"],
        procedure=["Read table slice.", "Detect table header.", "Extract records with provenance."],
        required_tools=["read_table_slice", "detect_table_header"],
        examples=[],
        metadata={"evolution": {}},
    )
    data.update(overrides)
    return data


def _validation_skill(**overrides: Any) -> dict[str, Any]:
    data = _embedded_skill(
        VALIDATION_SKILL_ID,
        name="Generic Schema Validation",
        description="Validate structured extraction records against a schema and source evidence.",
        task_types=["validation", "scientific_extraction"],
        scope="Generic schema and evidence validation for extracted records.",
        required_inputs=["candidate records", "schema", "source references"],
        expected_outputs=["validated records", "validation report"],
        procedure=["Validate required fields.", "Check source references."],
        required_tools=["json_schema_validate"],
        metadata={"evolution": {}},
    )
    data.update(overrides)
    return data


def _embedded_skill(skill_id: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "skill_id": skill_id,
        "name": skill_id.replace(".", " ").title(),
        "description": "Solve generic scientific tasks.",
        "source_type": "human",
        "source_uri": "seed://test/generic",
        "provenance": {},
        "domain_tags": ["generic", "scientific"],
        "task_types": ["generic"],
        "target_category": None,
        "scope": "generic scientific task handling",
        "applicability": ["generic scientific tasks"],
        "limitations": ["No domain-specific hard-coded rules."],
        "required_inputs": ["task goal"],
        "expected_outputs": ["validated task output"],
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": ["Read the task."],
        "required_tools": ["read_text"],
        "scripts": [],
        "resources": [],
        "examples": [],
        "smoke_tests": [],
        "synthetic_tests": [],
        "system_tests": [],
        "benchmark_tests": [],
        "validation_signals": ["source evidence"],
        "confidence": 0.8,
        "metadata": {},
    }
    data.update(overrides)
    return data


def _trace(**overrides: Any) -> TraceRecord:
    data: dict[str, Any] = {
        "trace_id": "trace-generic",
        "task_id": "task-generic",
        "task_summary": "generic scientific extraction task",
        "task_type": "scientific_extraction",
        "target_skill_ids": [],
        "retrieved_skill_ids": [TABLE_SKILL_ID],
        "selected_skill_ids": [TABLE_SKILL_ID],
        "tools_used": [],
        "missing_tools": [],
        "artifacts": [],
        "final_status": "runtime_failure",
        "evaluation_metrics": {},
        "error_summary": None,
        "compact_execution_summary": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }
    data.update(overrides)
    return TraceRecord(**data)


def _lesson(
    lesson_id: str,
    trace: TraceRecord,
    *,
    lesson_type: str,
    target_skill_id: str | None,
    principle: str,
    delta: dict[str, Any],
) -> TrajectoryLesson:
    return TrajectoryLesson(
        lesson_id=lesson_id,
        source_trace_ids=[trace.trace_id],
        lesson_type=lesson_type,  # type: ignore[arg-type]
        target_skill_id=target_skill_id,
        evidence_summary=trace.error_summary or trace.compact_execution_summary or trace.task_summary or principle,
        reusable_principle=principle,
        proposed_delta=delta,
        confidence=0.82,
        support_count=1,
    )


def _patch(
    patch_id: str,
    patch_type: str,
    lesson: TrajectoryLesson,
    content: dict[str, Any],
    *,
    candidate_skill_id: str | None = None,
    risk_level: str = "medium",
) -> SkillPatchProposal:
    return SkillPatchProposal(
        patch_id=patch_id,
        patch_type=patch_type,  # type: ignore[arg-type]
        target_skill_id=lesson.target_skill_id,
        candidate_skill_id=candidate_skill_id,
        source_lesson_ids=[lesson.lesson_id],
        source_trace_ids=lesson.source_trace_ids,
        patch_content=content,
        evidence_summary=lesson.evidence_summary,
        confidence=lesson.confidence,
        support_count=1,
        risk_level=risk_level,  # type: ignore[arg-type]
        created_at="2026-01-01T00:00:00+00:00",
    )


def _tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in [
        "read_text",
        "read_table_slice",
        "detect_table_header",
        "profile_table",
        "inspect_file_metadata",
        "inspect_table",
        "normalize_table",
        "json_schema_validate",
        "write_jsonl",
    ]:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return registry


def _raw_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_skill(graph_path: Path, skill_id: str) -> dict[str, Any]:
    raw = _raw_graph(graph_path)
    for skill in raw["skills"]:
        if skill.get("skill_id") == skill_id or skill.get("id") == skill_id:
            return skill
    raise AssertionError(f"missing skill {skill_id}")


def _package_payloads_from_transaction(transaction: Any) -> list[dict[str, Any]]:
    payloads = []
    for raw_path in transaction.metadata.get("package_paths", []):
        metadata_path = Path(raw_path)
        if not metadata_path.exists():
            continue
        skill_md_path = metadata_path.parent / "SKILL.md"
        payloads.append(
            {
                "metadata_path": str(metadata_path),
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                "skill_markdown_path": str(skill_md_path),
                "skill_markdown": skill_md_path.read_text(encoding="utf-8") if skill_md_path.exists() else None,
            }
        )
    return payloads


def _skill_required_tools(raw_graph: dict[str, Any], skill_id: str) -> list[str]:
    for skill in raw_graph["skills"]:
        if skill.get("skill_id") == skill_id or skill.get("id") == skill_id:
            return skill.get("required_tools", [])
    return []


def _proposal_types(result: PipelineResult) -> set[str]:
    return {proposal.proposal_type for proposal in result.proposals}


def _decision_for_proposal_type(
    proposals: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    proposal_type: str,
) -> str:
    proposal_ids = {proposal["proposal_id"] for proposal in proposals if proposal.get("proposal_type") == proposal_type}
    for decision in decisions:
        if decision.get("proposal_id") in proposal_ids:
            return str(decision.get("decision"))
    return "unknown"


def _assert_jsonl_loads(path: Path) -> None:
    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


class _CaseReport:
    def __init__(self, root: Path):
        self.root = root
        self.cases: list[dict[str, Any]] = []
        self.lessons: list[dict[str, Any]] = []
        self.patches: list[dict[str, Any]] = []
        self.proposals: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.transactions: list[dict[str, Any]] = []

    def add_pipeline_case(
        self,
        case: str,
        result: PipelineResult,
        lesson: str,
        patch_type: str,
        decision: str,
        evolved_result: str,
        passed: bool,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        for item in result.lessons:
            self.lessons.append(item.model_dump(mode="json"))
        for item in result.patches:
            self.patches.append(item.model_dump(mode="json"))
        for item in result.proposals:
            self.proposals.append(item.model_dump(mode="json"))
        for item in result.decisions:
            self.decisions.append(item.model_dump(mode="json"))
        self.transactions.append(result.transaction.model_dump(mode="json"))
        self.add(
            case=case,
            trace_summary=result.trace.task_summary or "",
            lesson=lesson,
            patch_type=patch_type,
            decision=decision,
            result=evolved_result,
            passed=passed,
            details={
                "trace_id": result.trace.trace_id,
                "source_skill_id": result.trace.selected_skill_ids or result.trace.retrieved_skill_ids,
                "before_state": result.before_graph,
                "generated_lessons": [item.model_dump(mode="json") for item in result.lessons],
                "generated_patches": [item.model_dump(mode="json") for item in result.patches],
                "consolidated_patches": [item.model_dump(mode="json") for item in result.consolidation.consolidated_patches],
                "skill_update_proposals": [item.model_dump(mode="json") for item in result.proposals],
                "policy_decisions": [item.model_dump(mode="json") for item in result.decisions],
                "after_state": result.after_graph,
                "transaction_id": result.transaction.transaction_id,
                "staged_proposal_ids": [
                    item["proposal"]["proposal_id"] for item in result.transaction.staged_updates if "proposal" in item
                ],
                "applied_transaction": result.transaction.model_dump(mode="json"),
                **(extra or {}),
            },
        )

    def add(
        self,
        *,
        case: str,
        trace_summary: str,
        lesson: str,
        patch_type: str,
        decision: str,
        result: str,
        passed: bool,
        details: dict[str, Any],
    ) -> None:
        self.cases.append(
            {
                "case": case,
                "trace_summary": trace_summary,
                "extracted_lesson": lesson,
                "patch_type": patch_type,
                "policy_decision": decision,
                "evolved_skill_result": result,
                "passed": passed,
                "details": details,
            }
        )

    def write(self) -> None:
        (self.root / "cases.json").write_text(json.dumps(self.cases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_jsonl(self.root / "lessons.jsonl", self.lessons)
        _write_jsonl(self.root / "patches.jsonl", self.patches)
        _write_jsonl(self.root / "proposals.jsonl", self.proposals)
        _write_jsonl(self.root / "decisions.jsonl", self.decisions)
        _write_jsonl(self.root / "transactions.jsonl", self.transactions)
        lines = [
            "# EvoLab Post-Run Skill Evolution Cases",
            "",
            "All cases use synthetic trajectories, temporary skill graphs, deterministic policies, and no external LLM/API calls.",
            "",
            "Policy mapping used by this demo:",
            "- conservative: default SkillEvolutionPolicy; safe direct metadata only, risky Trace2Skill proposals staged.",
            "- auto_apply_bounded_examples: example_trace_memory_update and metadata_update are auto-applied.",
            "- auto_apply_validated_tools: required_tools_update_proposal is auto-applied after tool validation.",
            "- auto_apply_valid_candidates: SkillEvolutionPolicy(auto_apply_valid_candidates=True).",
            "- auto_apply_relationship_updates: SkillEvolutionPolicy(auto_apply_relationship_updates=True).",
            "",
            "| Case | Trajectory summary | Extracted lesson | Patch type | Policy decision | Evolved skill result | Passed? |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.cases:
            lines.append(
                "| {case} | {trace_summary} | {extracted_lesson} | {patch_type} | {policy_decision} | {evolved_skill_result} | {passed} |".format(
                    case=_cell(item["case"]),
                    trace_summary=_cell(item["trace_summary"]),
                    extracted_lesson=_cell(item["extracted_lesson"]),
                    patch_type=_cell(item["patch_type"]),
                    policy_decision=_cell(item["policy_decision"]),
                    evolved_skill_result=_cell(item["evolved_skill_result"]),
                    passed="yes" if item["passed"] else "no",
                )
            )
        lines.extend(["", "## Case Details", ""])
        for item in self.cases:
            details = item["details"]
            lines.extend(
                [
                    f"### {item['case']}",
                    "",
                    f"- trace id: `{details.get('trace_id')}`",
                    f"- source skill id: `{details.get('source_skill_id')}`",
                    f"- transaction id: `{details.get('transaction_id')}`",
                    f"- staged proposal id: `{(details.get('staged_proposal_ids') or [None])[0]}`",
                    f"- retrieval result: `{details.get('retrieval_result_skill_ids')}`",
                    "",
                ]
            )
        (self.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.root / "generated_skill_inspection.md").write_text(
            "\n".join(self._inspection_lines()) + "\n",
            encoding="utf-8",
        )

    def _inspection_lines(self) -> list[str]:
        lines = [
            "# Generated Skill Evolution Inspection",
            "",
            "This report expands the controlled synthetic post-run evolution cases into inspectable generated artifacts.",
            "The run used temporary skill graphs and FakeLLMRuntime only; no production skill library or external LLM was used.",
            "",
            "## Update Type Summary",
            "",
            "- Formal skill creation: Case D under `auto_apply_valid_candidates` promoted one Trace2Skill package-backed skill.",
            "- Existing formal skill updates: Cases A2, B, C, and F modified the temporary `skill.generic_table_extraction.v1` entry.",
            "- Candidate/staged skill: Case D under conservative policy staged the same candidate skill without enabling it.",
            "- Memory items: Case B added bounded example memory; Case C and F added validation memory; Case A2 added failure memory.",
            "- Tool proposal: Case A1/A2 proposed `profile_table`; only A2 validated-tools policy applied it.",
            "- Relation update: Case E added a `validates` relation from extraction to validation skill.",
            "- Rejected update: Case E invalid relation was rejected by validation before transaction mutation.",
            "",
            "## Why Most Cases Do Not Create New Formal Skills",
            "",
            "Cases A, B, C, E, and F are skill-deepening or relationship updates against already-retrieved skills. They intentionally do not create new formal skills because the trajectory evidence points to a missing tool contract, reusable memory, validation rule, or relation, not a missing capability. Case D is the only coverage-gap case, so it is the only one that exercises candidate creation and formal promotion.",
            "",
        ]
        for item in self.cases:
            lines.extend(self._case_inspection_lines(item))
        return lines

    def _case_inspection_lines(self, item: dict[str, Any]) -> list[str]:
        details = item["details"]
        lines = [
            f"## {item['case']}",
            "",
            f"- case name: {item['case']}",
            f"- source trajectory/signal: {item['trace_summary']}",
            f"- extracted lesson: {item['extracted_lesson']}",
            f"- patch/proposal type: {item['patch_type']}",
            f"- policy decision: {item['policy_decision']}",
            f"- passed: {item['passed']}",
            "",
        ]
        lessons = details.get("generated_lessons")
        if lessons:
            lines.extend(["### Lessons", "", "```json", json.dumps(lessons, indent=2, sort_keys=True), "```", ""])
        patches = details.get("generated_patches") or details.get("generated_proposals")
        if patches:
            lines.extend(["### Patches / Direct Proposals", "", "```json", json.dumps(patches, indent=2, sort_keys=True), "```", ""])
        proposals = details.get("skill_update_proposals")
        if proposals:
            lines.extend(["### SkillUpdateProposals", "", "```json", json.dumps(proposals, indent=2, sort_keys=True), "```", ""])
        decisions = details.get("policy_decisions")
        if decisions:
            lines.extend(["### Policy Decisions", "", "```json", json.dumps(decisions, indent=2, sort_keys=True), "```", ""])

        transaction = details.get("applied_transaction")
        if transaction:
            lines.extend(
                [
                    "### Graph Transaction",
                    "",
                    f"- transaction id: `{transaction.get('transaction_id')}`",
                    f"- status: `{transaction.get('status')}`",
                    f"- changed library: `{transaction.get('changed_library')}`",
                    f"- before graph hash: `{transaction.get('before_graph_hash')}`",
                    f"- after graph hash: `{transaction.get('after_graph_hash')}`",
                    f"- changed skill ids: `{transaction.get('metadata', {}).get('changed_skill_ids')}`",
                    "",
                    "```json",
                    json.dumps(
                        {
                            "applied_updates": transaction.get("applied_updates", []),
                            "staged_updates": transaction.get("staged_updates", []),
                            "rejected_updates": transaction.get("rejected_updates", []),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                ]
            )
        conservative_transaction = details.get("conservative_transaction")
        if conservative_transaction:
            lines.extend(["### Conservative/Staged Transaction", "", "```json", json.dumps(conservative_transaction, indent=2, sort_keys=True), "```", ""])
        duplicate_transaction = details.get("duplicate_transaction")
        if duplicate_transaction:
            lines.extend(["### Duplicate Relation Transaction", "", "```json", json.dumps(duplicate_transaction, indent=2, sort_keys=True), "```", ""])
        if details.get("invalid_relation_warnings"):
            lines.extend(["### Rejected Invalid Relation", "", "```json", json.dumps(details["invalid_relation_warnings"], indent=2, sort_keys=True), "```", ""])

        after_state = details.get("after_state")
        changed_ids = []
        if transaction:
            changed_ids = transaction.get("metadata", {}).get("changed_skill_ids") or []
        if after_state:
            changed_skills = [
                skill
                for skill in after_state.get("skills", [])
                if not changed_ids or (skill.get("skill_id") or skill.get("id")) in changed_ids
            ]
            lines.extend(["### Formal Skill Library State", ""])
            if changed_skills:
                lines.extend(["Changed or relevant graph skill entries:", "", "```json", json.dumps(changed_skills, indent=2, sort_keys=True), "```", ""])
            edges = after_state.get("edges") or []
            if edges:
                lines.extend(["Related skills / relations:", "", "```json", json.dumps(edges, indent=2, sort_keys=True), "```", ""])

        package_payloads = details.get("promoted_skill_packages") or []
        if package_payloads:
            lines.extend(["### Full Promoted Skill Package Content", ""])
            for package in package_payloads:
                metadata = package["metadata"]
                lines.extend(
                    [
                        f"#### {metadata.get('skill_id')}",
                        "",
                        f"- skill id: `{metadata.get('skill_id')}`",
                        f"- name: {metadata.get('name')}",
                        f"- category: `{metadata.get('target_category')}`",
                        f"- task/capability: `{metadata.get('task_types')}` / {metadata.get('summary')}",
                        f"- description: {metadata.get('summary')}",
                        f"- usage conditions: `{metadata.get('applicability')}`",
                        f"- inputs: `{metadata.get('required_inputs')}`",
                        f"- outputs: `{metadata.get('expected_outputs')}`",
                        f"- step-by-step instructions: `{metadata.get('procedure')}`",
                        f"- required tools: `{metadata.get('required_tools')}`",
                        f"- resources: `{metadata.get('resources')}`",
                        f"- evaluation/validation criteria: `{metadata.get('validation_signals')}`",
                        f"- provenance: `{metadata.get('provenance')}`",
                        "",
                        "metadata.json:",
                        "",
                        "```json",
                        json.dumps(metadata, indent=2, sort_keys=True),
                        "```",
                        "",
                        "SKILL.md:",
                        "",
                        "```markdown",
                        package.get("skill_markdown") or "",
                        "```",
                        "",
                    ]
                )

        for label, key in [
            ("Memory Items / Examples", "after_evolution_examples"),
            ("Validation Rules", "validation_rules"),
            ("Failure Cases", "failure_cases"),
            ("Relation Edges", "edges"),
        ]:
            if details.get(key):
                lines.extend([f"### {label}", "", "```json", json.dumps(details[key], indent=2, sort_keys=True), "```", ""])
        return lines


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
