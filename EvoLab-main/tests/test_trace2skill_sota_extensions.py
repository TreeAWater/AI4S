import json
from pathlib import Path
from typing import Any

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.evolution import SkillEvolutionPolicy, SkillUpdateDecision, SkillUpdateProposal
from evolab.backends.skills.trace2skill import (
    Trace2SkillEvolver,
    Trace2SkillLLMExtractor,
    Trace2SkillRunConfig,
    TracePoolBuilder,
)
from evolab.backends.skills.trace2skill.adapter import Trace2SkillSkillBackendAdapter
from evolab.backends.skills.trace2skill.analysts import ErrorAnalyst
from evolab.backends.skills.trace2skill.regression import (
    BenchmarkRunResult,
    BenchmarkTask,
    SkillEvolutionRegressionGate,
)
from evolab.backends.skills.trace2skill.runner import ParallelAnalystRunner
from evolab.backends.skills.trace2skill.schema import ConsolidatedSkillPatch, TraceRecord
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest
from evolab.contracts.tools import ToolSpec
from evolab.tools.runtime import ToolRegistry


def _embedded_skill(skill_id: str = "skill.generic.v1", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "skill_id": skill_id,
        "name": skill_id.replace(".", " ").title(),
        "description": "Solve generic tasks.",
        "source_type": "human",
        "source_uri": "seed://test",
        "provenance": {},
        "domain_tags": ["generic"],
        "task_types": ["generic"],
        "target_category": None,
        "scope": "generic task handling",
        "applicability": ["generic tasks"],
        "limitations": [],
        "required_inputs": ["task goal"],
        "expected_outputs": ["validated result"],
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
        "validation_signals": [],
        "confidence": 0.8,
        "metadata": {},
    }
    data.update(overrides)
    return data


def _package_metadata(skill_id: str = "skill.package.v1", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "v1",
        "skill_id": skill_id,
        "name": "Package Skill",
        "version": "v1",
        "summary": "Solve package-backed generic tasks.",
        "source_type": "human",
        "source_uri": "seed://test/package",
        "provenance": {},
        "domain_tags": ["generic"],
        "task_types": ["generic"],
        "target_category": None,
        "scope": "package-backed generic task handling",
        "applicability": ["generic tasks"],
        "limitations": [],
        "required_inputs": ["task goal"],
        "expected_outputs": ["validated result"],
        "dependencies": [],
        "environment_assumptions": [],
        "procedure": ["Read the task."],
        "required_tools": ["read_text"],
        "scripts": [],
        "resources": [],
        "examples": [],
        "tests": {"smoke": [], "synthetic": [], "system": [], "benchmark": []},
        "validation_signals": [],
        "confidence": 0.8,
        "metadata": {},
    }
    data.update(overrides)
    return data


def _write_package_graph(tmp_path: Path, *, skill_id: str = "skill.package.v1") -> tuple[Path, Path]:
    package_dir = tmp_path / "skills" / "package_skill"
    package_dir.mkdir(parents=True)
    (package_dir / "metadata.json").write_text(json.dumps(_package_metadata(skill_id), indent=2), encoding="utf-8")
    (package_dir / "SKILL.md").write_text("# Package Skill\n", encoding="utf-8")
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": [
                    {
                        "id": skill_id,
                        "name": "Package Skill",
                        "summary": "Solve package-backed generic tasks.",
                        "package_ref": package_dir.relative_to(tmp_path).as_posix(),
                        "status": "active",
                        "metadata": {},
                    }
                ],
                "categories": [],
                "edges": [],
                "metadata": {"storage_format": "package_ref_v1"},
            }
        ),
        encoding="utf-8",
    )
    return graph_path, package_dir


def _write_embedded_graph(tmp_path: Path, *, skills: list[dict[str, Any]] | None = None) -> Path:
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v1",
                "skills": skills if skills is not None else [_embedded_skill()],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def _trace(status: str = "runtime_failure", **overrides: Any) -> TraceRecord:
    data: dict[str, Any] = {
        "trace_id": "trace-1",
        "task_id": "task-1",
        "task_summary": "generic task",
        "selected_skill_ids": ["skill.package.v1"],
        "retrieved_skill_ids": ["skill.package.v1"],
        "tools_used": ["read_text"],
        "missing_tools": ["inspect_table"] if "failure" in status else [],
        "final_status": status,
        "error_summary": "missing table inspection tool" if "failure" in status else None,
        "compact_execution_summary": "completed" if "success" in status else None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return TraceRecord(**data)


def _pool(*traces: TraceRecord):
    return TracePoolBuilder().build_trace_pool(list(traces))


def _registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return registry


def _config(**overrides: Any) -> Trace2SkillRunConfig:
    data = {
        "mode": "combined",
        "enable_llm_analysts": True,
        "dry_run": True,
        "llm_temperature": 0,
        "max_llm_retries": 1,
    }
    data.update(overrides)
    return Trace2SkillRunConfig(**data)


def _llm_payload(*, include_chain_of_thought: bool = False, patch_type: str = "failure_case_patch") -> str:
    lesson: dict[str, Any] = {
        "lesson_type": "error_lesson",
        "target_skill_id": "skill.package.v1",
        "evidence_summary": "The trace failed because table inspection was unavailable.",
        "reusable_principle": "Surface missing operational constraints as reusable failure guidance.",
        "proposed_delta": {"failure_case": "Require table inspection before table-heavy tasks."},
        "confidence": 0.86,
        "risk_level": "low",
        "source_trace_ids": ["trace-1"],
    }
    patch: dict[str, Any] = {
        "patch_type": patch_type,
        "target_skill_id": "skill.package.v1",
        "evidence_summary": "The trace failed because table inspection was unavailable.",
        "proposed_delta": {"failure_reason": "Require table inspection before table-heavy tasks."},
        "confidence": 0.82,
        "risk_level": "low",
        "source_trace_ids": ["trace-1"],
    }
    if include_chain_of_thought:
        lesson["chain_of_thought"] = "private reasoning must not be stored"
        patch["chain_of_thought"] = "private reasoning must not be stored"
    return json.dumps({"lessons": [lesson], "patches": [patch]})


def _policy_for_mutation(**overrides: Any) -> SkillEvolutionPolicy:
    data = {
        "auto_apply_proposal_types": [
            "required_tools_update_proposal",
            "example_trace_memory_update",
            "failure_note_update",
            "metadata_update",
        ],
        "staged_proposal_types": ["candidate_skill_creation", "relationship_update_proposal"],
    }
    data.update(overrides)
    return SkillEvolutionPolicy(**data)


def _patch(patch_type: str, **overrides: Any) -> ConsolidatedSkillPatch:
    data: dict[str, Any] = {
        "consolidated_patch_id": f"consolidated-{patch_type}",
        "patch_type": patch_type,
        "target_skill_id": "skill.package.v1",
        "merged_content": {},
        "source_patch_ids": [f"patch-{patch_type}"],
        "source_lesson_ids": ["lesson-1"],
        "source_trace_ids": ["trace-1"],
        "confidence": 0.8,
        "risk_level": "medium",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return ConsolidatedSkillPatch(**data)


def _apply_patches(
    backend: GraphSkillBackend,
    patches: list[ConsolidatedSkillPatch],
    *,
    policy: SkillEvolutionPolicy,
    config: Trace2SkillRunConfig | None = None,
):
    evolver = Trace2SkillEvolver(graph_backend=backend, policy=policy)
    proposals = Trace2SkillSkillBackendAdapter(
        backend_id=backend.backend_id,
        graph_version_ref="graph-v1",
    ).to_skill_update_proposals(patches)
    decisions = [policy.decide(proposal) for proposal in proposals]
    return evolver._apply_policy_gated_transaction(
        proposals=proposals,
        decisions=decisions,
        dry_run=False,
        before_graph_hash=evolver._graph_hash(),
        config=config or Trace2SkillRunConfig(dry_run=False),
    )


def test_llm_extractor_uses_evolab_runtime_and_parses_valid_json():
    runtime = FakeLLMRuntime(default_content=_llm_payload())
    extractor = Trace2SkillLLMExtractor(llm_client=runtime)

    lessons, patches = extractor.extract_lessons_and_patches(
        _pool(_trace()),
        config=_config(),
    )

    assert lessons[0].lesson_type == "error_lesson"
    assert patches[0].patch_type == "failure_case_patch"
    assert runtime.requests[0].generation_config.temperature == 0
    assert runtime.requests[0].generation_config.response_json_schema is not None


def test_llm_extractor_falls_back_on_invalid_json_schema_invalid_exception_and_missing_client():
    invalid_json = Trace2SkillLLMExtractor(FakeLLMRuntime(default_content="not json"))
    schema_invalid = Trace2SkillLLMExtractor(FakeLLMRuntime(default_content=json.dumps({"lessons": [{}], "patches": []})))

    class RaisingRuntime:
        def generate(self, messages, tool_specs, generation_config: LLMGenerationConfig):
            raise RuntimeError("model unavailable")

    for extractor in [
        invalid_json,
        schema_invalid,
        Trace2SkillLLMExtractor(RaisingRuntime()),
        Trace2SkillLLMExtractor(None),
    ]:
        lessons, patches = extractor.extract_lessons_and_patches(_pool(_trace()), config=_config())
        assert lessons
        assert patches
        assert patches[0].patch_type == "required_tools_patch"


def test_llm_extractor_rejects_chain_of_thought_and_unsupported_patch_type():
    extractor = Trace2SkillLLMExtractor(
        FakeLLMRuntime(default_content=_llm_payload(include_chain_of_thought=True, patch_type="unsafe_patch"))
    )

    lessons, patches = extractor.extract_lessons_and_patches(_pool(_trace()), config=_config())

    assert lessons
    assert patches
    assert "chain_of_thought" not in json.dumps([lesson.model_dump(mode="json") for lesson in lessons])
    assert all(patch.patch_type != "unsafe_patch" for patch in patches)
    assert extractor.audit_events


def test_parallel_and_sequential_analyst_outputs_are_equivalent_and_stable():
    pool = _pool(_trace("runtime_failure"), _trace("runtime_success", trace_id="trace-2", missing_tools=[]))
    config = Trace2SkillRunConfig(mode="combined", dry_run=True)

    sequential = ParallelAnalystRunner(execution_mode="sequential").run(pool, config=config)
    parallel = ParallelAnalystRunner(execution_mode="thread", max_workers=2).run(pool, config=config)

    assert [lesson.lesson_id for lesson in parallel.lessons] == [lesson.lesson_id for lesson in sequential.lessons]
    assert [patch.patch_id for patch in parallel.patches] == [patch.patch_id for patch in sequential.patches]
    assert parallel.metadata["max_workers"] == 2


def test_parallel_analyst_runner_isolates_failed_trace_and_continues():
    class FailingAnalyst:
        name = "failing"

        def analyze(self, traces):
            raise RuntimeError("bad trace")

    pool = _pool(_trace("runtime_failure"))
    result = ParallelAnalystRunner(
        execution_mode="thread",
        max_workers=2,
        lesson_analysts=[ErrorAnalyst(), FailingAnalyst()],
    ).run(pool, config=Trace2SkillRunConfig(mode="error_only", dry_run=True))

    assert result.lessons
    assert result.failures[0]["analyst"] == "failing"
    assert result.failures[0]["trace_id"] == "trace-1"


def test_policy_approved_package_mutations_update_stable_files_and_retrieval(tmp_path):
    graph_path, package_dir = _write_package_graph(tmp_path)
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    policy = _policy_for_mutation()
    config = Trace2SkillRunConfig(dry_run=False, max_examples_per_skill=5)

    transaction = _apply_patches(
        backend,
        [
            _patch("required_tools_patch", merged_content={"required_tools": ["inspect_table"]}),
            _patch("example_memory_patch", merged_content={"example_summary": "worked example"}),
            _patch("procedure_step_patch", merged_content={"procedure_steps": ["Inspect available inputs first."]}),
            _patch("precondition_patch", merged_content={"preconditions": ["Inputs must be readable."]}),
            _patch("failure_case_patch", merged_content={"failure_reason": "Input was unreadable."}),
            _patch("validation_rule_patch", merged_content={"validation_rules": ["Output must cite evidence."]}),
        ],
        policy=policy,
        config=config,
    )

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state").get(
        RetrievalRequest(task_id="task-1", role="solver", query="package generic")
    )
    evolution = metadata["metadata"]["evolution"]
    assert transaction.changed_library is True
    assert "inspect_table" in metadata["required_tools"]
    assert "worked example" in metadata["examples"]
    assert "Inspect available inputs first." in metadata["procedure"]
    assert evolution["preconditions"] == ["Inputs must be readable."]
    assert evolution["failure_cases"] == ["Input was unreadable."]
    assert evolution["validation_rules"] == ["Output must cite evidence."]
    assert "inspect_table" in bundle.skills[0].required_tools
    assert "worked example" in bundle.skills[0].content


def test_package_mutation_prunes_bounded_memory_and_staged_patch_does_not_mutate(tmp_path):
    graph_path, package_dir = _write_package_graph(tmp_path)
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    policy = _policy_for_mutation()

    _apply_patches(
        backend,
        [
            _patch(
                "example_memory_patch",
                consolidated_patch_id=f"example-{index}",
                merged_content={"example_summary": f"example-{index}"},
            )
            for index in range(4)
        ],
        policy=policy,
        config=Trace2SkillRunConfig(dry_run=False, max_examples_per_skill=2),
    )
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    before = metadata["required_tools"]

    staged = _apply_patches(
        backend,
        [_patch("required_tools_patch", merged_content={"required_tools": ["write_report"]})],
        policy=SkillEvolutionPolicy(),
    )
    after = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["examples"] == ["example-2", "example-3"]
    assert staged.changed_library is False
    assert after["required_tools"] == before


def test_invalid_mutation_fails_safely_and_graph_hash_remains_unchanged(tmp_path):
    graph_path, package_dir = _write_package_graph(tmp_path)
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    before_hash = Trace2SkillEvolver(graph_backend=backend)._graph_hash()
    before_package = (package_dir / "metadata.json").read_text(encoding="utf-8")

    transaction = _apply_patches(
        backend,
        [_patch("required_tools_patch", target_skill_id="skill.missing.v1", merged_content={"required_tools": ["x"]})],
        policy=_policy_for_mutation(),
    )

    assert transaction.changed_library is False
    assert transaction.after_graph_hash == before_hash
    assert (package_dir / "metadata.json").read_text(encoding="utf-8") == before_package


def test_auto_apply_valid_candidate_promotes_skill_and_conservative_stages(tmp_path):
    graph_path = _write_embedded_graph(tmp_path, skills=[])
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    low_coverage = SkillObservationRequest(
        task_id="task-1",
        run_ref="run-1",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="summarize generic audit logs"),
        skill_bundle=SkillBundle(
            backend_id="graph_skill",
            graph_version_ref="graph-v1",
            skills=[],
            required_tools=[],
            metadata={"graph_context_summary": {"coverage_report": {"sufficient": False}}},
        ),
        metadata={"status": "failed"},
    )

    staged = Trace2SkillEvolver(graph_backend=backend).run(
        Trace2SkillRunConfig(mode="skill_creation_from_scratch", dry_run=False, output_dir=str(tmp_path / "staged")),
        observations=[low_coverage],
    )
    applied = Trace2SkillEvolver(
        graph_backend=backend,
        policy=SkillEvolutionPolicy(auto_apply_valid_candidates=True),
    ).run(
        Trace2SkillRunConfig(mode="skill_creation_from_scratch", dry_run=False, output_dir=str(tmp_path / "applied")),
        observations=[low_coverage],
    )
    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state").get(
        RetrievalRequest(task_id="task-2", role="solver", query="summarize generic audit logs")
    )

    assert staged.changed_library is False
    assert any(update["proposal"]["proposal_type"] == "candidate_skill_creation" for update in staged.staged_updates)
    assert applied.changed_library is True
    assert bundle.skills
    assert bundle.skills[0].skill_id.startswith("skill.trace2skill.")


def test_candidate_id_collision_is_promoted_with_collision_safe_id(tmp_path):
    graph_path = _write_embedded_graph(tmp_path, skills=[_embedded_skill("skill.package.v1")])
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    policy = SkillEvolutionPolicy(auto_apply_valid_candidates=True)
    proposal = SkillUpdateProposal(
        proposal_id="proposal-candidate",
        proposal_type="candidate_skill_creation",
        observation_id="obs-1",
        backend_id="graph_skill",
        related_skill_ids=[],
        summary="candidate",
        payload={
            "candidate_record": {
                "candidate_id": "skill.package.v1",
                "proposed_name": "Generic Audit Log Summarizer",
                "proposed_task": "generic",
                "proposed_category": None,
                "source_observation_id": "obs-1",
                "evidence_summary": "coverage gap",
                "missing_capability_description": "summarize generic audit logs",
                "suggested_required_inputs": ["audit log"],
                "suggested_expected_outputs": ["summary"],
                "suggested_required_tools": [],
                "status": "staged",
                "created_at": "2026-01-01T00:00:00+00:00",
                "metadata": {},
            }
        },
        created_at="2026-01-01T00:00:00+00:00",
    )
    transaction = Trace2SkillEvolver(graph_backend=backend, policy=policy)._apply_policy_gated_transaction(
        proposals=[proposal],
        decisions=[policy.decide(proposal)],
        dry_run=False,
        before_graph_hash=Trace2SkillEvolver(graph_backend=backend)._graph_hash(),
        config=Trace2SkillRunConfig(dry_run=False),
    )

    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    ids = {skill.get("skill_id") or skill.get("id") for skill in raw["skills"]}
    assert transaction.changed_library is True
    assert "skill.package.v1" in ids
    assert any(skill_id.startswith("skill.trace2skill.") for skill_id in ids)


def test_policy_approved_relationship_update_adds_deduped_relation_and_affects_retrieval(tmp_path):
    graph_path = _write_embedded_graph(
        tmp_path,
        skills=[
            _embedded_skill("skill.alpha.v1", name="Alpha Skill", description="Alpha query handler."),
            _embedded_skill("skill.beta.v1", name="Beta Skill", description="Beta related helper."),
        ],
    )
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    policy = SkillEvolutionPolicy(auto_apply_relationship_updates=True)
    relation_patch = _patch(
        "relationship_patch",
        target_skill_id="skill.alpha.v1",
        merged_content={
            "relations": [
                {
                    "source_skill_id": "skill.alpha.v1",
                    "target_skill_id": "skill.beta.v1",
                    "relation": "related_to",
                },
                {
                    "source_skill_id": "skill.alpha.v1",
                    "target_skill_id": "skill.beta.v1",
                    "relation": "related_to",
                },
            ]
        },
    )

    transaction = _apply_patches(backend, [relation_patch], policy=policy)
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state").get(
        RetrievalRequest(task_id="task-1", role="solver", query="alpha", metadata={"top_k": 5})
    )

    assert transaction.changed_library is True
    assert raw["edges"] == [{"source_id": "skill.alpha.v1", "target_id": "skill.beta.v1", "relation": "related_to"}]
    assert "skill.beta.v1" in [skill.skill_id for skill in bundle.skills]
    assert bundle.metadata["retrieval_trace"]["relation_expansion_steps"]


def test_invalid_relationship_update_is_blocked(tmp_path):
    graph_path = _write_embedded_graph(tmp_path, skills=[_embedded_skill("skill.alpha.v1")])
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    before = json.loads(graph_path.read_text(encoding="utf-8"))

    transaction = _apply_patches(
        backend,
        [
            _patch(
                "relationship_patch",
                target_skill_id="skill.alpha.v1",
                merged_content={
                    "relations": [
                        {
                            "source_skill_id": "skill.alpha.v1",
                            "target_skill_id": "skill.missing.v1",
                            "relation": "related_to",
                        }
                    ]
                },
            )
        ],
        policy=SkillEvolutionPolicy(auto_apply_relationship_updates=True),
    )

    assert transaction.changed_library is False
    assert json.loads(graph_path.read_text(encoding="utf-8")) == before


class _StaticBenchmarkRunner:
    def __init__(self, before: dict[str, float], after: dict[str, float]):
        self.before = before
        self.after = after

    def run(self, tasks: list[BenchmarkTask], *, snapshot_ref: str, graph_backend=None) -> BenchmarkRunResult:
        metrics = self.before if snapshot_ref == "before" else self.after
        return BenchmarkRunResult(status="completed", task_count=len(tasks), metrics=metrics)


def test_regression_gate_passes_fails_and_skips_without_benchmark():
    tasks = [BenchmarkTask(task_id="bench-1", query="generic")]

    passed = SkillEvolutionRegressionGate(
        benchmark_runner=_StaticBenchmarkRunner({"accuracy": 0.7}, {"accuracy": 0.75}),
        metrics=["accuracy"],
        no_regression_threshold=0.01,
    ).evaluate(tasks, before_snapshot_ref="before", after_snapshot_ref="after")
    failed = SkillEvolutionRegressionGate(
        benchmark_runner=_StaticBenchmarkRunner({"accuracy": 0.7}, {"accuracy": 0.6}),
        metrics=["accuracy"],
        no_regression_threshold=0.01,
    ).evaluate(tasks, before_snapshot_ref="before", after_snapshot_ref="after")
    skipped = SkillEvolutionRegressionGate().evaluate(tasks, before_snapshot_ref="before", after_snapshot_ref="after")

    assert passed.status == "pass"
    assert failed.status == "fail_regression"
    assert skipped.status == "skipped_no_benchmark"


def test_regression_failure_blocks_policy_approved_mutation(tmp_path):
    graph_path = _write_embedded_graph(tmp_path, skills=[])
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    observation = SkillObservationRequest(
        task_id="task-1",
        run_ref="run-1",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="summarize generic audit logs"),
        skill_bundle=SkillBundle(backend_id="graph_skill", graph_version_ref="graph-v1", skills=[], required_tools=[]),
        metadata={"status": "failed"},
    )

    result = Trace2SkillEvolver(
        graph_backend=backend,
        policy=SkillEvolutionPolicy(auto_apply_valid_candidates=True),
        benchmark_runner=_StaticBenchmarkRunner({"accuracy": 0.7}, {"accuracy": 0.5}),
    ).run(
        Trace2SkillRunConfig(
            mode="skill_creation_from_scratch",
            dry_run=False,
            enable_regression_gate=True,
            regression_metrics=["accuracy"],
            output_dir=str(tmp_path / "blocked"),
        ),
        observations=[observation],
    )

    assert result.blocked_by_regression is True
    assert result.changed_library is False
    assert json.loads(graph_path.read_text(encoding="utf-8"))["skills"] == []


def test_report_includes_regression_hashes_and_changed_skill_ids(tmp_path):
    graph_path, _package_dir = _write_package_graph(tmp_path)
    backend = GraphSkillBackend(graph_path, repo_root=tmp_path, evolution_root=tmp_path / "state")
    result = Trace2SkillEvolver(
        graph_backend=backend,
        tool_registry=_registry("read_text", "inspect_table"),
        policy=_policy_for_mutation(),
    ).run(
        Trace2SkillRunConfig(mode="error_only", dry_run=False, output_dir=str(tmp_path / "report")),
        trace_pool=_pool(_trace()),
    )

    summary = json.loads(Path(result.report_paths["trace2skill_run_summary"]).read_text(encoding="utf-8"))
    assert summary["before_graph_hash"] == result.before_graph_hash
    assert summary["after_graph_hash"] == result.after_graph_hash
    assert "regression_gate_result" in summary
    assert "skill.package.v1" in summary["changed_skill_ids"]
