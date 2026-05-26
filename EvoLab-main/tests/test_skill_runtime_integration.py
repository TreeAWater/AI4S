import json

import pytest

from evolab.backends.skills import GraphSkillBackend
from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.contracts.common import RuntimePolicy
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillRef
from evolab.contracts.tools import ToolSpec
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.skill_retrieval import MissingRequiredToolError, prepare_skill_runtime_context
from evolab.runtime.task_runtime import _generic_role_retrieval_metadata
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def _tool_spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters_schema={"type": "object"},
    )


def _skill_graph_payload() -> dict:
    return {
        "schema_version": "v1",
        "version": "skill-tree-v1",
        "categories": [
            {
                "category_id": "cap-literature",
                "name": "Literature",
                "description": "Understand scientific literature.",
                "layer": "scientific_process_capability",
            },
            {
                "category_id": "task-retrieval",
                "name": "Retrieval",
                "description": "Find relevant scientific papers.",
                "parent_category_id": "cap-literature",
                "layer": "scientific_task",
            },
        ],
        "skills": [
            {
                "schema_version": "v1",
                "skill_id": "skill-pubmed-retrieval",
                "name": "PubMed retrieval",
                "description": "Retrieve biomedical papers from PubMed.",
                "source_type": "human",
                "source_uri": "human://seed/pubmed-retrieval",
                "provenance": {"author": "seed"},
                "domain_tags": ["literature", "biomedical"],
                "task_types": ["retrieval"],
                "target_category": "task-retrieval",
                "scope": "Literature retrieval for biomedical reviews",
                "applicability": ["research question needs literature evidence"],
                "limitations": ["May miss papers outside PubMed."],
                "required_inputs": ["research question"],
                "expected_outputs": ["ranked paper list"],
                "dependencies": [],
                "environment_assumptions": [],
                "procedure": ["Translate the question into PubMed search terms.", "Rank returned papers."],
                "required_tools": ["pubmed", "rg"],
                "scripts": [],
                "resources": [],
                "examples": [],
                "smoke_tests": [],
                "synthetic_tests": [],
                "system_tests": [],
                "benchmark_tests": [],
                "validation_signals": ["manual_relevance_check"],
                "confidence": 0.9,
                "metadata": {},
            }
        ],
        "edges": [],
    }


def _backend(tmp_path) -> GraphSkillBackend:
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_skill_graph_payload()), encoding="utf-8")
    return GraphSkillBackend(graph_path)


def _request() -> RetrievalRequest:
    return RetrievalRequest(
        task_id="task-1",
        role="solver",
        query="biomedical literature retrieval",
        metadata={"scientific_process_capability": "Literature"},
    )


def _tool_runtime(*tool_names: str) -> ToolRuntime:
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(_tool_spec(name), lambda arguments: "ok")
    return ToolRuntime(registry)


def test_prepare_skill_runtime_context_retrieves_skills_tools_and_prompt_context(tmp_path):
    prepared = prepare_skill_runtime_context(
        retrieval_request=_request(),
        skill_backend=_backend(tmp_path),
        tool_runtime=_tool_runtime("pubmed", "rg"),
        allowed_tools=["pubmed", "rg"],
        policy=RuntimePolicy(),
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == ["skill-pubmed-retrieval"]
    assert prepared.skill_bundle.required_tools == ["pubmed", "rg"]
    assert [spec.name for spec in prepared.tool_bundle.tool_specs] == ["pubmed", "rg"]
    assert prepared.skill_context["selected_skills"] == [
        {
            "skill_id": "skill-pubmed-retrieval",
            "name": "PubMed retrieval",
            "required_tools": ["pubmed", "rg"],
            "retrieval": prepared.skill_bundle.skills[0].metadata["retrieval"],
        }
    ]
    assert prepared.skill_context["tree_paths"] == [
        {
            "category_ids": ["cap-literature", "task-retrieval"],
            "endpoint_category_id": "task-retrieval",
            "category_path": "Literature > Retrieval",
        }
    ]
    assert prepared.skill_context["graph_context_summary"]["graph_version"] == "skill-tree-v1"
    assert prepared.skill_context["required_tools"] == ["pubmed", "rg"]


def test_prepare_skill_runtime_context_raises_clear_error_for_missing_tool(tmp_path):
    with pytest.raises(
        MissingRequiredToolError,
        match="missing required tools for skill retrieval: rg",
    ):
        prepare_skill_runtime_context(
            retrieval_request=_request(),
            skill_backend=_backend(tmp_path),
            tool_runtime=_tool_runtime("pubmed"),
            allowed_tools=["pubmed", "rg"],
            policy=RuntimePolicy(),
        )


def test_prepare_skill_runtime_context_filters_incompatible_tools_for_node_scoped_preparation(tmp_path):
    prepared = prepare_skill_runtime_context(
        retrieval_request=_request(),
        skill_backend=_backend(tmp_path),
        tool_runtime=_tool_runtime("pubmed"),
        allowed_tools=["pubmed"],
        policy=RuntimePolicy(
            metadata={
                "subagent_policy": {
                    "tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill",
                }
            }
        ),
    )

    assert [spec.name for spec in prepared.tool_bundle.tool_specs] == ["pubmed"]
    assert prepared.skill_bundle.required_tools == ["pubmed"]
    assert prepared.skill_context["required_tools"] == ["pubmed"]
    assert prepared.skill_context["unavailable_required_tools"] == ["rg"]
    assert prepared.skill_context["incompatible_skills"] == [
        {
            "skill_id": "skill-pubmed-retrieval",
            "name": "PubMed retrieval",
            "missing_tools": ["rg"],
        }
    ]


def test_prepare_skill_runtime_context_scopes_skills_to_generic_subagent_role():
    class BroadSkillRuntime:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["list_files", "json_schema_validate", "write_report"],
                skills=[
                    SkillRef(
                        skill_id="skill.scientific_document_intake.v1",
                        name="Scientific Document Intake",
                        content="Survey files.",
                        required_tools=["list_files"],
                        metadata={"retrieval": {"matched_category_path": "Literature > Document Intake"}},
                    ),
                    SkillRef(
                        skill_id="skill.extraction_result_validation.v1",
                        name="Extraction Result Validation",
                        content="Validate records.",
                        required_tools=["json_schema_validate"],
                        metadata={"retrieval": {"matched_category_path": "Validation > Extraction Result Validation"}},
                    ),
                    SkillRef(
                        skill_id="skill.ground_truth_based_evaluation.v1",
                        name="Ground Truth Based Evaluation",
                        content="Evaluate records.",
                        required_tools=["json_schema_validate"],
                        metadata={"retrieval": {"matched_category_path": "Validation > Ground Truth Based Evaluation"}},
                    ),
                    SkillRef(
                        skill_id="skill.trajectory_pattern_mining.v1",
                        name="Trajectory Pattern Mining",
                        content="Mine traces.",
                        required_tools=["write_report"],
                        metadata={"retrieval": {"matched_category_path": "Execution > Trajectory Pattern Mining"}},
                    ),
                ],
                metadata={"retrieval_trace": {"returned_skill_ids": [
                    "skill.scientific_document_intake.v1",
                    "skill.extraction_result_validation.v1",
                    "skill.ground_truth_based_evaluation.v1",
                    "skill.trajectory_pattern_mining.v1",
                ]}},
            )

    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="SurveyAgent",
            query="Survey all article packages and report source inventory.",
        ),
        skill_backend=BroadSkillRuntime(),
        tool_runtime=_tool_runtime("list_files", "json_schema_validate", "write_report"),
        allowed_tools=["list_files", "json_schema_validate", "write_report"],
        policy=RuntimePolicy(
            metadata={
                "subagent_policy": {
                    "skill_retrieval_scope": "per_internal_dag_node",
                }
            }
        ),
        role_name="SurveyAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == [
        "skill.scientific_document_intake.v1",
    ]
    assert prepared.skill_bundle.required_tools == ["list_files"]
    assert prepared.skill_context["filtered_out_skill_ids"] == [
        "skill.extraction_result_validation.v1",
        "skill.ground_truth_based_evaluation.v1",
        "skill.trajectory_pattern_mining.v1",
    ]
    assert prepared.skill_context["skill_scope"]["generic_agent_type"] == "SurveyAgent"


def test_assignment_scope_filter_ignores_metadata_field_names_when_matching_exclusions():
    class GraphLikeSkillRuntime:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=[],
                skills=[
                    SkillRef(
                        skill_id="skill.scientific_document_intake.v1",
                        name="Scientific Document Intake",
                        content=(
                            "Description:\nMap scientific document packages into source inventory.\n\n"
                            "Validation Signals:\n- manual inventory check"
                        ),
                        metadata={
                            "domain_tags": ["artifact", "document"],
                            "task_types": ["intake", "document"],
                            "validation_signals": ["manual_check"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.extraction_result_validation.v1",
                        name="Extraction Result Validation",
                        content="Description:\nValidate candidate records.",
                        metadata={
                            "domain_tags": ["record"],
                            "task_types": ["validation"],
                            "validation_signals": ["schema_check"],
                        },
                    ),
                ],
            )

    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="SurveyAgent",
            query="Survey article packages.",
        ),
        skill_backend=GraphLikeSkillRuntime(),
        tool_runtime=_tool_runtime(),
        allowed_tools=[],
        policy=RuntimePolicy(
            metadata={"subagent_policy": {"skill_retrieval_scope": "per_internal_dag_node"}}
        ),
        role_name="SurveyAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == [
        "skill.scientific_document_intake.v1"
    ]
    assert prepared.skill_context["filtered_out_skill_ids"] == [
        "skill.extraction_result_validation.v1"
    ]


def test_survey_scope_keeps_intake_and_discovery_skills_even_with_generic_extraction_words():
    class ScientificIESurveySkillRuntime:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["list_files"],
                skills=[
                    SkillRef(
                        skill_id="skill.scientific_document_intake.v1",
                        name="Scientific Document Intake",
                        content=(
                            "Description:\nMap a scientific document package into main text, "
                            "supplementary files, table-like artifacts, unreadable files, "
                            "and recommended next actions.\n\n"
                            "Scope:\nReusable document package intake for scientific information extraction."
                        ),
                        required_tools=["list_files"],
                        metadata={
                            "domain_tags": ["package", "paper", "article", "files"],
                            "task_types": ["intake", "document", "extract"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.supplementary_artifact_discovery.v1",
                        name="Supplementary Artifact Discovery",
                        content="Description:\nDiscover supplementary artifacts likely to contain evidence.",
                        required_tools=["list_files"],
                        metadata={
                            "domain_tags": ["artifact", "supplementary", "table"],
                            "task_types": ["discovery", "artifact"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.structured_record_construction.v1",
                        name="Structured Record Construction",
                        content="Description:\nConstruct candidate structured records.",
                        required_tools=[],
                    ),
                ],
            )

    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="SurveyAgent",
            query="Survey dataset files and resources.",
        ),
        skill_backend=ScientificIESurveySkillRuntime(),
        tool_runtime=_tool_runtime("list_files"),
        allowed_tools=["list_files"],
        policy=RuntimePolicy(
            metadata={"subagent_policy": {"skill_retrieval_scope": "per_internal_dag_node"}}
        ),
        role_name="SurveyAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == [
        "skill.scientific_document_intake.v1",
        "skill.supplementary_artifact_discovery.v1",
    ]
    assert prepared.skill_bundle.required_tools == ["list_files"]


def test_write_scope_keeps_only_final_artifact_writing_skills():
    class ScientificIEWritingSkillRuntime:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["write_jsonl", "write_report", "read_table_slice", "json_schema_validate"],
                skills=[
                    SkillRef(
                        skill_id="skill.final_artifact_writing.v1",
                        name="Final Artifact Writing",
                        content="Description:\nWrite final JSONL records and report artifacts from validated content.",
                        required_tools=["write_jsonl", "write_report"],
                        metadata={
                            "domain_tags": ["artifact", "report", "output"],
                            "task_types": ["writing", "reporting", "finalization"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.scientific_table_structure_understanding.v1",
                        name="Scientific Table Structure Understanding",
                        content="Description:\nInspect and understand scientific tables.",
                        required_tools=["read_table_slice"],
                        metadata={
                            "domain_tags": ["table", "spreadsheet"],
                            "task_types": ["table", "extraction"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.extraction_result_validation.v1",
                        name="Extraction Result Validation",
                        content="Description:\nValidate candidate records.",
                        required_tools=["json_schema_validate"],
                        metadata={
                            "domain_tags": ["record", "validation"],
                            "task_types": ["validation", "extraction"],
                        },
                    ),
                ],
            )

    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="WriteAgent",
            query="Write final biology_component_records.jsonl and biology_component_report.md from upstream content.",
        ),
        skill_backend=ScientificIEWritingSkillRuntime(),
        tool_runtime=_tool_runtime("write_jsonl", "write_report", "read_table_slice", "json_schema_validate"),
        allowed_tools=["write_jsonl", "write_report", "read_table_slice", "json_schema_validate"],
        policy=RuntimePolicy(
            metadata={"subagent_policy": {"skill_retrieval_scope": "per_internal_dag_node"}}
        ),
        role_name="WriteAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == [
        "skill.final_artifact_writing.v1",
    ]
    assert prepared.skill_bundle.required_tools == ["write_jsonl", "write_report"]
    assert prepared.skill_context["filtered_out_skill_ids"] == [
        "skill.scientific_table_structure_understanding.v1",
        "skill.extraction_result_validation.v1",
    ]


def test_write_scope_keeps_final_artifact_skill_with_negative_limitations():
    class FinalArtifactSkillRuntime:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["write_jsonl", "write_report"],
                skills=[
                    SkillRef(
                        skill_id="skill.final_artifact_writing.v1",
                        name="Final Artifact Writing",
                        content=(
                            "Description:\nWrite final JSONL records and report artifacts.\n\n"
                            "Limitations:\n- Does not extract new evidence.\n"
                            "- Does not validate unsupported records by invention.\n\n"
                            "Procedure:\n1. Write final records.\n"
                            "2. Do not call table-reading, extraction, validation, or ontology tools."
                        ),
                        required_tools=["write_jsonl", "write_report"],
                        metadata={
                            "domain_tags": ["artifact", "report", "output"],
                            "task_types": ["writing", "reporting", "finalization"],
                        },
                    ),
                    SkillRef(
                        skill_id="skill.scientific_table_structure_understanding.v1",
                        name="Scientific Table Structure Understanding",
                        content="Description:\nInspect and understand scientific tables.",
                        required_tools=["read_table_slice"],
                        metadata={
                            "domain_tags": ["table", "spreadsheet"],
                            "task_types": ["table", "extraction"],
                        },
                    ),
                ],
            )

    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="WriteAgent",
            query="Write final records and report artifacts.",
        ),
        skill_backend=FinalArtifactSkillRuntime(),
        tool_runtime=_tool_runtime("write_jsonl", "write_report", "read_table_slice"),
        allowed_tools=["write_jsonl", "write_report", "read_table_slice"],
        policy=RuntimePolicy(
            metadata={"subagent_policy": {"skill_retrieval_scope": "per_internal_dag_node"}}
        ),
        role_name="WriteAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == [
        "skill.final_artifact_writing.v1",
    ]
    assert prepared.skill_context["filtered_out_skill_ids"] == [
        "skill.scientific_table_structure_understanding.v1"
    ]


def test_agent_config_required_skills_filters_retrieved_bundle():
    class TwoSkillBackend:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                skills=[
                    SkillRef(
                        skill_id="skill.keep",
                        name="Keep skill",
                        content="Use this skill.",
                        required_tools=["read_text"],
                    ),
                    SkillRef(
                        skill_id="skill.drop",
                        name="Drop skill",
                        content="Do not use this skill.",
                        required_tools=["write_report"],
                    ),
                ],
                required_tools=["read_text", "write_report"],
                metadata={"retrieval_trace": {"returned_skill_ids": ["skill.keep", "skill.drop"]}},
            )

    registry = ToolRegistry()
    registry.register(_tool_spec("read_text"), lambda arguments: "ok")
    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(
            task_id="task-1",
            role="ExecAgent",
            query="extract",
            metadata={"agent_config_required_skills": ["skill.keep"]},
        ),
        skill_backend=TwoSkillBackend(),
        tool_runtime=ToolRuntime(registry),
        allowed_tools=["read_text", "write_report"],
        policy=RuntimePolicy(),
        role_name="ExecAgent",
    )

    assert [skill.skill_id for skill in prepared.skill_bundle.skills] == ["skill.keep"]
    assert prepared.skill_bundle.required_tools == ["read_text"]
    assert prepared.skill_context["agent_config_filtered_out_skill_ids"] == ["skill.drop"]


def test_generic_role_retrieval_metadata_steers_survey_to_literature_intake():
    metadata = _generic_role_retrieval_metadata("SurveyAgent")

    assert metadata["scientific_process_capability"] == "Literature"
    assert metadata["target_category"] == "task.scientific_document_intake"
    assert "intake" in metadata["task_types"]


def test_prompt_builder_can_inject_skill_context(tmp_path):
    prepared = prepare_skill_runtime_context(
        retrieval_request=_request(),
        skill_backend=_backend(tmp_path),
        tool_runtime=_tool_runtime("pubmed", "rg"),
        allowed_tools=["pubmed", "rg"],
        policy=RuntimePolicy(),
    )
    role = RoleSpec(
        name="solver",
        system_prompt="Solve scientific tasks.",
        llm_backend=BackendBinding(backend_id="llm-local"),
    )

    messages = PromptBuilder.build(
        role,
        "Find relevant papers.",
        memory=None,
        skills=prepared.skill_bundle,
        skill_context=prepared.skill_context,
    )

    prompt_text = "\n".join(message.content for message in messages)
    assert "Skill Context:" in prompt_text
    assert "skill-pubmed-retrieval" in prompt_text
    assert "Literature > Retrieval" in prompt_text
    assert '"required_tools": ["pubmed", "rg"]' in prompt_text
