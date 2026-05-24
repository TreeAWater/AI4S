import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evolab.backends.skills import CandidateSkill, GraphSkillBackend, SkillBackend, SkillCategoryNode, SkillGraph
from evolab.backends.skills.base import SkillBackend as BaseModuleSkillBackend
from evolab.backends.skills.graph import (
    GraphSkillBackend as GraphModuleSkillBackend,
    _build_category_index,
    _get_ancestors,
    _get_descendants,
    _get_root_for_category,
    _get_subtree_category_ids,
    _match_root_capabilities,
    _match_scientific_tasks,
    _parse_query_info,
)
from evolab.backends.skills.graph_schema import SCIENTIFIC_PROCESS_CAPABILITIES
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillUpdateResult


def _request(query: str, **overrides) -> RetrievalRequest:
    data = {"task_id": "task-1", "role": "solver", "query": query}
    data.update(overrides)
    return RetrievalRequest(**data)


def _candidate_skill(**overrides):
    data = {
        "skill_id": "skill-1",
        "name": "Pytest failure triage",
        "description": "Use pytest output to isolate failing behavior.",
        "source_type": "human",
        "source_uri": "human://seed/pytest-triage",
        "provenance": {"author": "seed"},
        "domain_tags": ["testing", "debugging"],
        "task_types": ["regression"],
        "target_category": "testing",
        "scope": "Python test failure diagnosis",
        "applicability": ["pytest output is available"],
        "limitations": ["Does not fix production bugs automatically"],
        "required_inputs": ["pytest failure output"],
        "expected_outputs": ["root cause hypothesis", "focused test command"],
        "dependencies": ["pytest"],
        "environment_assumptions": ["repository has pytest configured"],
        "procedure": [
            "Run the failing pytest node with verbose output.",
            "Read the assertion failure and traceback.",
            "Inspect the smallest code path that explains the failure.",
        ],
        "required_tools": ["pytest", "rg"],
        "scripts": ["pytest {test_node} -v"],
        "resources": ["docs/testing.md"],
        "examples": ["pytest tests/test_example.py::test_failure -v"],
        "smoke_tests": ["pytest --version"],
        "synthetic_tests": ["pytest tests/test_example.py -q"],
        "system_tests": ["pytest -q"],
        "benchmark_tests": ["pytest tests/test_regressions.py -q"],
        "validation_signals": ["human_seeded"],
        "confidence": 0.8,
        "metadata": {"priority": "high"},
    }
    data.update(overrides)
    return data


def _phase2_graph_payload():
    return {
        "schema_version": "v1",
        "version": "phase2-v1",
        "categories": [
            {
                "category_id": "cap-literature",
                "name": "Literature",
                "description": "Understand scientific literature.",
                "layer": "scientific_process_capability",
            },
            {
                "category_id": "cap-analysis",
                "name": "Analysis",
                "description": "Analyze and model scientific data.",
                "layer": "scientific_process_capability",
            },
            {
                "category_id": "task-retrieval",
                "name": "Retrieval",
                "description": "Find relevant scientific papers.",
                "parent_category_id": "cap-literature",
                "layer": "scientific_task",
            },
            {
                "category_id": "task-extraction",
                "name": "Table Extraction",
                "description": "Extract table data from papers.",
                "parent_category_id": "cap-literature",
                "layer": "scientific_task",
            },
            {
                "category_id": "task-evaluation",
                "name": "Model Evaluation",
                "description": "Evaluate model predictions.",
                "parent_category_id": "cap-analysis",
                "layer": "scientific_task",
            },
        ],
        "skills": [
            _candidate_skill(
                skill_id="skill-retrieve",
                name="Biomedical paper retrieval",
                description="Retrieve biomedical papers from PubMed for a review objective.",
                domain_tags=["biomedical", "literature"],
                task_types=["retrieval"],
                target_category="task-retrieval",
                scope="Literature retrieval for biomedical reviews",
                required_inputs=["research question", "search terms"],
                expected_outputs=["ranked paper list"],
                procedure=["Translate the question into search terms.", "Run PubMed queries.", "Rank papers."],
                required_tools=["pubmed", "rg"],
                limitations=["May miss papers outside indexed databases."],
                validation_signals=["manual_relevance_check"],
                confidence=0.92,
            ),
            _candidate_skill(
                skill_id="skill-extract",
                name="Paper table extractor",
                description="Extract structured numeric tables from scientific papers.",
                domain_tags=["literature"],
                task_types=["extraction"],
                target_category="task-extraction",
                scope="PDF table extraction",
                required_tools=["python"],
                confidence=0.7,
            ),
            _candidate_skill(
                skill_id="skill-evaluate",
                name="Prediction model evaluation",
                description="Compute AUROC and calibration metrics for predictions.",
                domain_tags=["modeling"],
                task_types=["evaluation"],
                target_category="task-evaluation",
                scope="Model evaluation",
                required_tools=["python"],
                confidence=0.85,
            ),
        ],
        "edges": [],
        "metadata": {"graph_context_summary": "legacy summary"},
    }


def _multi_level_graph_payload():
    return {
        "schema_version": "v1",
        "version": "tree-v1",
        "categories": [
            {
                "category_id": "cap-literature",
                "name": "Literature",
                "description": "Understand scientific literature.",
                "layer": "scientific_process_capability",
            },
            {
                "category_id": "task-supplementary",
                "name": "Supplementary Material Understanding",
                "description": "Understand supplementary files and appendices.",
                "parent_category_id": "cap-literature",
                "layer": "scientific_task",
                "metadata": {"domain_tags": ["literature"], "task_types": ["understanding"]},
            },
            {
                "category_id": "task-supplementary-table",
                "name": "Supplementary Table Extraction",
                "description": "Extract supplementary tables from scientific papers.",
                "parent_category_id": "task-supplementary",
                "layer": "scientific_task",
                "metadata": {"domain_tags": ["literature"], "task_types": ["extraction"]},
            },
            {
                "category_id": "task-table",
                "name": "Table Extraction",
                "description": "Extract tables from scientific papers.",
                "parent_category_id": "cap-literature",
                "layer": "scientific_task",
                "metadata": {"domain_tags": ["literature"], "task_types": ["extraction"]},
            },
            {
                "category_id": "cap-analysis",
                "name": "Analysis",
                "description": "Analyze scientific data.",
                "layer": "scientific_process_capability",
            },
            {
                "category_id": "task-model-evaluation",
                "name": "Model Evaluation",
                "description": "Evaluate predictive models.",
                "parent_category_id": "cap-analysis",
                "layer": "scientific_task",
            },
        ],
        "skills": [
            _candidate_skill(
                skill_id="skill-supp-table",
                name="Supplementary table extractor",
                description="Extract supplementary tables from research paper appendices.",
                domain_tags=["literature"],
                task_types=["extraction"],
                target_category="task-supplementary-table",
                scope="Supplementary table extraction",
                required_tools=["python"],
                confidence=0.9,
            ),
            _candidate_skill(
                skill_id="skill-shallow-table",
                name="General table extractor",
                description="Extract tables from research papers.",
                domain_tags=["literature"],
                task_types=["extraction"],
                target_category="task-table",
                scope="General paper table extraction",
                required_tools=["python"],
                confidence=0.95,
            ),
            _candidate_skill(
                skill_id="skill-edge-supp-table",
                name="Edge supplementary extractor",
                description="Extract supplementary table data through an edge category assignment.",
                domain_tags=["literature"],
                task_types=["extraction"],
                target_category=None,
                scope="Edge-assigned supplementary table extraction",
                required_tools=["python"],
                confidence=0.8,
            ),
        ],
        "edges": [
            {
                "source_id": "skill-edge-supp-table",
                "target_id": "task-supplementary-table",
                "relation": "belongs_to_category",
            }
        ],
    }


def test_skill_backend_package_exports_base_and_graph_modules():
    assert SkillBackend is BaseModuleSkillBackend
    assert GraphSkillBackend is GraphModuleSkillBackend


def test_graph_skill_backend_inherits_skill_backend_base(tmp_path):
    assert issubclass(GraphSkillBackend, SkillBackend)
    assert isinstance(GraphSkillBackend(tmp_path / "skills.json"), SkillBackend)


def test_skill_backend_base_requires_contract_methods():
    class MissingGet(SkillBackend):
        backend_id = "missing_get"

        def look_at(self, event):
            return event

    class MissingLookAt(SkillBackend):
        backend_id = "missing_look_at"

        def get(self, request):
            return SkillBundle(skills=[], required_tools=[], backend_id=self.backend_id)

    with pytest.raises(TypeError, match="abstract"):
        MissingGet()
    with pytest.raises(TypeError, match="abstract"):
        MissingLookAt()


def test_candidate_skill_accepts_docx_aligned_definition():
    candidate = CandidateSkill(**_candidate_skill())

    assert candidate.skill_id == "skill-1"
    assert candidate.source_type == "human"
    assert candidate.domain_tags == ["testing", "debugging"]
    assert candidate.procedure[0] == "Run the failing pytest node with verbose output."
    assert candidate.confidence == 0.8


def test_candidate_skill_rejects_invalid_source_type_and_confidence():
    with pytest.raises(ValidationError):
        CandidateSkill(**_candidate_skill(source_type="invalid"))

    with pytest.raises(ValidationError):
        CandidateSkill(**_candidate_skill(confidence=1.5))


def test_skill_graph_accepts_candidate_skill_nodes():
    graph = SkillGraph(
        version="graph-v2",
        skills=[_candidate_skill()],
        categories=[
            {
                "category_id": "testing",
                "name": "Testing",
                "description": "Testing workflows",
            }
        ],
        edges=[
            {
                "source_id": "skill-1",
                "target_id": "testing",
                "relation": "belongs_to_category",
                "weight": 0.9,
            }
        ],
        metadata={"owner": "skills"},
    )

    assert graph.schema_version == "v1"
    assert graph.skills[0].skill_id == "skill-1"
    assert graph.categories[0].category_id == "testing"
    assert graph.edges[0].relation == "belongs_to_category"


def test_scientific_process_capability_names_are_validated():
    assert SCIENTIFIC_PROCESS_CAPABILITIES == (
        "Research",
        "Literature",
        "Data Preparation",
        "Analysis",
        "Validation",
        "Execution",
        "Writing",
    )
    category = SkillCategoryNode(
        category_id="cap-research",
        name="Research",
        layer="scientific_process_capability",
    )

    assert category.layer == "scientific_process_capability"

    with pytest.raises(ValidationError, match="scientific_process_capability"):
        SkillCategoryNode(
            category_id="cap-invalid",
            name="Research design",
            layer="scientific_process_capability",
        )


def test_category_index_construction_splits_roots_tasks_and_children():
    graph = SkillGraph.model_validate(_phase2_graph_payload())

    index = _build_category_index(graph)

    assert set(index.by_id) == {
        "cap-literature",
        "cap-analysis",
        "task-retrieval",
        "task-extraction",
        "task-evaluation",
    }
    assert [node.category_id for node in index.process_capabilities] == ["cap-literature", "cap-analysis"]
    assert [node.category_id for node in index.scientific_tasks] == [
        "task-retrieval",
        "task-extraction",
        "task-evaluation",
    ]
    assert [node.category_id for node in index.children_by_parent["cap-literature"]] == [
        "task-retrieval",
        "task-extraction",
    ]


def test_root_capability_matching_uses_query_and_metadata():
    graph = SkillGraph.model_validate(_phase2_graph_payload())
    index = _build_category_index(graph)

    query_match = _match_root_capabilities(_parse_query_info(_request("literature retrieval")), index)
    metadata_match = _match_root_capabilities(
        _parse_query_info(
            _request(
                "evaluate predictions",
                metadata={"scientific_process_capability": "Analysis"},
            )
        ),
        index,
    )

    assert [node.category_id for node in query_match] == ["cap-literature"]
    assert [node.category_id for node in metadata_match] == ["cap-analysis"]


def test_scientific_task_matching_prefers_children_under_matched_roots():
    graph = SkillGraph.model_validate(_phase2_graph_payload())
    index = _build_category_index(graph)
    query = _parse_query_info(_request("literature model evaluation", metadata={"capability": "Literature"}))
    roots = _match_root_capabilities(query, index)

    tasks = _match_scientific_tasks(query, index, roots)

    assert [task.category_id for task in tasks] == ["task-retrieval", "task-extraction"]


def test_multi_level_category_index_builds_depths_and_root_mapping():
    graph = SkillGraph.model_validate(_multi_level_graph_payload())

    index = _build_category_index(graph)

    assert index.parent_by_child["task-supplementary-table"] == "task-supplementary"
    assert index.root_capability_ids == {"cap-literature", "cap-analysis"}
    assert index.task_category_ids == {
        "task-supplementary",
        "task-supplementary-table",
        "task-table",
        "task-model-evaluation",
    }
    assert index.depth_by_id["cap-literature"] == 0
    assert index.depth_by_id["task-supplementary"] == 1
    assert index.depth_by_id["task-supplementary-table"] == 2
    assert index.root_by_category_id["task-supplementary-table"] == "cap-literature"
    assert _get_ancestors(index, "task-supplementary-table") == ["cap-literature", "task-supplementary"]
    assert _get_descendants(index, "cap-literature", max_depth=1) == ["task-supplementary", "task-table"]
    assert _get_subtree_category_ids(index, "task-supplementary") == {
        "task-supplementary",
        "task-supplementary-table",
    }
    assert _get_root_for_category(index, "task-supplementary-table") == "cap-literature"


def test_recursive_task_matching_finds_third_level_task_node():
    graph = SkillGraph.model_validate(_multi_level_graph_payload())
    index = _build_category_index(graph)
    query = _parse_query_info(_request("extract supplementary tables from literature"))
    roots = _match_root_capabilities(query, index)

    tasks = _match_scientific_tasks(query, index, roots)

    assert tasks[0].category_id == "task-supplementary-table"


def test_deeper_category_wins_over_shallow_category_when_scores_are_comparable():
    graph = SkillGraph.model_validate(_multi_level_graph_payload())
    index = _build_category_index(graph)
    query = _parse_query_info(_request("table extraction", metadata={"domain_tags": ["literature"]}))
    roots = _match_root_capabilities(query, index)

    tasks = _match_scientific_tasks(query, index, roots)

    assert [task.category_id for task in tasks[:2]] == ["task-supplementary-table", "task-table"]


def test_matching_high_level_category_does_not_expand_unmatched_descendants(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_multi_level_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(
        _request(
            "supplementary material understanding",
            metadata={"scientific_task": "Supplementary Material Understanding"},
        )
    )

    assert bundle.skills == []
    summary = bundle.metadata["graph_context_summary"]
    assert summary["retrieval_paths"] == [
        {
            "category_ids": ["cap-literature", "task-supplementary"],
            "endpoint_category_id": "task-supplementary",
            "category_path": "Literature > Supplementary Material Understanding",
        }
    ]
    assert "task-supplementary-table" not in summary["retrieved_from_tree_category_ids"]


def test_multi_branch_request_retrieves_skills_from_multiple_tree_paths(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(
        _request(
            "retrieve papers and extract paper tables from literature",
            metadata={"scientific_process_capability": "Literature", "top_k": 5},
        )
    )

    assert {skill.skill_id for skill in bundle.skills} == {"skill-retrieve", "skill-extract"}
    summary = bundle.metadata["graph_context_summary"]
    assert summary["retrieval_paths"] == [
        {
            "category_ids": ["cap-literature", "task-retrieval"],
            "endpoint_category_id": "task-retrieval",
            "category_path": "Literature > Retrieval",
        },
        {
            "category_ids": ["cap-literature", "task-extraction"],
            "endpoint_category_id": "task-extraction",
            "category_path": "Literature > Table Extraction",
        },
    ]


def test_target_category_can_point_to_deep_task_node(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_multi_level_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("supplementary table extraction"))

    assert bundle.skills[0].skill_id == "skill-supp-table"
    assert bundle.skills[0].metadata["target_category"] == "task-supplementary-table"
    assert bundle.skills[0].metadata["retrieval"]["retrieved_by"] == "direct"


def test_belongs_to_category_edge_can_point_to_deep_task_node(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _multi_level_graph_payload()
    payload["skills"] = [skill for skill in payload["skills"] if skill["skill_id"] == "skill-edge-supp-table"]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("supplementary table extraction"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-edge-supp-table"]
    assert bundle.skills[0].metadata["retrieval"]["source"] == "belongs_to_category"
    assert bundle.skills[0].metadata["retrieval"]["matched_category_id"] == "task-supplementary-table"


def test_graph_context_summary_includes_category_paths(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_multi_level_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("supplementary table extraction"))

    summary = bundle.metadata["graph_context_summary"]
    assert summary["matched_category_paths"][0] == (
        "Literature > Supplementary Material Understanding > Supplementary Table Extraction"
    )
    assert summary["deepest_matched_categories"] == [
        {"category_id": "task-supplementary-table", "name": "Supplementary Table Extraction"}
    ]
    assert "task-supplementary-table" in summary["retrieved_from_subtree_category_ids"]


def test_skill_ref_metadata_includes_matched_category_path(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_multi_level_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("supplementary table extraction"))

    retrieval = bundle.skills[0].metadata["retrieval"]
    assert retrieval["matched_category_id"] == "task-supplementary-table"
    assert retrieval["matched_category_path"] == (
        "Literature > Supplementary Material Understanding > Supplementary Table Extraction"
    )
    assert retrieval["retrieved_by"] == "direct"


def test_get_returns_matching_candidate_skill_from_json_graph(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v2",
                "skills": [
                    _candidate_skill(required_tools=["pytest"]),
                    _candidate_skill(
                        skill_id="skill-2",
                        name="Release notes",
                        description="Write a concise changelog.",
                        domain_tags=["docs"],
                        task_types=["documentation"],
                        target_category="documentation",
                        scope="Release note drafting",
                        applicability=["merged changes are available"],
                        procedure=["Collect merged changes.", "Group user-visible changes."],
                        required_tools=["git"],
                        scripts=["git log --oneline"],
                        resources=["docs/releases.md"],
                        examples=["Draft release notes from merged pull requests."],
                    ),
                ],
                "categories": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("triage a pytest regression"))

    assert isinstance(bundle, SkillBundle)
    assert bundle.backend_id == "graph_skill"
    assert bundle.graph_version_ref == "graph-v2"
    assert bundle.skill_state_ref == "graph-v2"
    assert [skill.skill_id for skill in bundle.skills] == ["skill-1"]
    assert bundle.skills[0].name == "Pytest failure triage"
    assert bundle.skills[0].metadata["domain_tags"] == ["testing", "debugging"]
    assert bundle.required_tools == ["pytest"]


def test_get_retrieves_candidate_skill_by_target_category(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("find relevant biomedical papers for literature retrieval"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-retrieve"]
    retrieval = bundle.skills[0].metadata["retrieval"]
    assert retrieval["source"] == "target_category"
    assert "matched scientific task Retrieval" in retrieval["reasons"]
    assert retrieval["relation_path"] == ["task-retrieval"]


def test_get_retrieves_candidate_skill_by_belongs_to_category_edge(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"] = [
        _candidate_skill(
            skill_id="skill-edge-retrieve",
            name="Edge-only retrieval",
            description="Retrieve studies for a literature review.",
            domain_tags=["literature"],
            task_types=["retrieval"],
            target_category=None,
            required_tools=["pubmed"],
        )
    ]
    payload["edges"] = [
        {
            "source_id": "skill-edge-retrieve",
            "target_id": "task-retrieval",
            "relation": "belongs_to_category",
        }
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("literature retrieval"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-edge-retrieve"]
    assert bundle.skills[0].metadata["retrieval"]["source"] == "belongs_to_category"
    assert bundle.skills[0].metadata["retrieval"]["relation_path"] == ["task-retrieval"]


def test_seed_scoring_uses_lexical_domain_task_tool_and_category_signals(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(
        _request(
            "biomedical literature retrieval with pubmed",
            metadata={
                "domain_tags": ["biomedical"],
                "task_types": ["retrieval"],
                "required_tools": ["pubmed"],
                "target_category": "task-retrieval",
            },
        )
    )

    assert bundle.skills[0].skill_id == "skill-retrieve"
    retrieval = bundle.skills[0].metadata["retrieval"]
    assert retrieval["score"] > 0
    assert {
        "lexical overlap",
        "domain tag match: biomedical",
        "task type match: retrieval",
        "required tool match: pubmed",
        "target category match: task-retrieval",
    }.issubset(set(retrieval["reasons"]))


def test_one_hop_relationship_expansion_includes_supported_neighbors(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"].append(
        _candidate_skill(
            skill_id="skill-refined-retrieve",
            name="Advanced PubMed retrieval",
            description="Refine PubMed query expansion for systematic reviews.",
            domain_tags=["biomedical", "literature"],
            task_types=["retrieval"],
            target_category="task-evaluation",
            confidence=0.6,
        )
    )
    payload["edges"] = [
        {
            "source_id": "skill-retrieve",
            "target_id": "skill-refined-retrieve",
            "relation": "refines",
            "weight": 0.5,
        }
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval", metadata={"top_k": 5}))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-retrieve", "skill-refined-retrieve"]
    neighbor_retrieval = bundle.skills[1].metadata["retrieval"]
    assert neighbor_retrieval["source"] == "relationship:refines"
    assert neighbor_retrieval["relation_path"] == ["skill-retrieve", "refines", "skill-refined-retrieve"]
    assert bundle.metadata["graph_context_summary"]["expanded_relationships"] == [
        {
            "source_skill_id": "skill-retrieve",
            "relation": "refines",
            "target_skill_id": "skill-refined-retrieve",
        }
    ]


def test_deprecated_relationship_edges_are_ignored(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"].append(
        _candidate_skill(
            skill_id="skill-deprecated-neighbor",
            name="Deprecated neighbor",
            description="Deprecated related skill.",
            target_category="task-evaluation",
            confidence=1.0,
        )
    )
    payload["edges"] = [
        {
            "source_id": "skill-retrieve",
            "target_id": "skill-deprecated-neighbor",
            "relation": "related_to",
            "deprecated": True,
        }
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval", metadata={"top_k": 5}))

    assert "skill-deprecated-neighbor" not in [skill.skill_id for skill in bundle.skills]
    assert bundle.metadata["graph_context_summary"]["expanded_relationships"] == []


def test_conflicts_with_edges_record_warning_without_adding_neighbor(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"].append(
        _candidate_skill(
            skill_id="skill-conflicting-retrieve",
            name="Conflicting approach",
            description="Conflicting approach.",
            target_category="task-evaluation",
        )
    )
    payload["edges"] = [
        {
            "source_id": "skill-retrieve",
            "target_id": "skill-conflicting-retrieve",
            "relation": "conflicts_with",
        }
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval", metadata={"top_k": 5}))

    assert "skill-conflicting-retrieve" not in [skill.skill_id for skill in bundle.skills]
    assert bundle.metadata["graph_context_summary"]["warnings"] == [
        "skill-retrieve conflicts_with skill-conflicting-retrieve"
    ]


def test_top_k_ranking_uses_score_confidence_then_name(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"] = [
        _candidate_skill(
            skill_id="skill-a",
            name="Beta retrieval",
            description="Literature retrieval",
            domain_tags=["literature"],
            task_types=["retrieval"],
            target_category="task-retrieval",
            confidence=0.6,
        ),
        _candidate_skill(
            skill_id="skill-b",
            name="Alpha retrieval",
            description="Literature retrieval",
            domain_tags=["literature"],
            task_types=["retrieval"],
            target_category="task-retrieval",
            confidence=0.6,
        ),
        _candidate_skill(
            skill_id="skill-c",
            name="Gamma retrieval",
            description="Literature retrieval pubmed biomedical",
            domain_tags=["biomedical", "literature"],
            task_types=["retrieval"],
            target_category="task-retrieval",
            confidence=0.5,
        ),
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    all_ranked = backend.get(_request("biomedical literature retrieval", metadata={"top_k": 3}))
    top_one = backend.get(_request("biomedical literature retrieval", metadata={"top_k": 1}))

    assert [skill.skill_id for skill in all_ranked.skills] == ["skill-c", "skill-b", "skill-a"]
    assert [skill.skill_id for skill in top_one.skills] == ["skill-c"]


def test_constructor_initializes_missing_graph(tmp_path):
    graph_path = tmp_path / "nested" / "skills.json"

    backend = GraphSkillBackend(graph_path)

    assert backend.graph_path == graph_path
    assert json.loads(graph_path.read_text(encoding="utf-8")) == {
        "schema_version": "v1",
        "version": "v1",
        "skills": [],
        "categories": [],
        "edges": [],
        "metadata": {},
    }


def test_unmatched_query_returns_missing_skill_report(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "v1",
                "skills": [_candidate_skill()],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("database migration"))

    assert bundle.skills == []
    assert bundle.required_tools == []
    assert bundle.metadata["matched_skill_ids"] == []
    assert bundle.metadata["missing_skill_report"] == {
        "schema_version": "v1",
        "missing_capability": "database migration",
        "reason": "No CandidateSkill matched the retrieval query.",
        "can_be_solved_by_existing_tools": False,
        "risk_level": "medium",
        "on_demand_synthesis_allowed": False,
        "metadata": {},
    }


def test_coverage_report_marks_sufficient_matches(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval"))

    coverage = bundle.metadata["graph_context_summary"]["coverage_report"]
    assert coverage["sufficient"] is True
    assert coverage["covered"] == [
        "objective",
        "input_output_contract",
        "required_tools",
        "domain_tags",
        "procedure",
        "failure_modes",
    ]
    assert coverage["missing"] == []
    assert "missing_skill_report" not in bundle.metadata


def test_coverage_report_marks_insufficient_matches(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"] = [
        _candidate_skill(
            skill_id="skill-thin",
            name="Thin retrieval",
            description="Literature retrieval.",
            domain_tags=[],
            task_types=["retrieval"],
            target_category="task-retrieval",
            scope="",
            required_inputs=[],
            expected_outputs=[],
            procedure=[],
            required_tools=[],
            limitations=[],
            smoke_tests=[],
            synthetic_tests=[],
            system_tests=[],
            benchmark_tests=[],
            validation_signals=[],
        )
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("literature retrieval"))

    coverage = bundle.metadata["graph_context_summary"]["coverage_report"]
    assert coverage["sufficient"] is False
    assert coverage["missing"] == [
        "input_output_contract",
        "required_tools",
        "domain_tags",
        "procedure",
        "failure_modes",
    ]


def test_missing_skill_report_is_generated_from_insufficient_coverage(tmp_path):
    graph_path = tmp_path / "skills.json"
    payload = _phase2_graph_payload()
    payload["skills"] = [
        _candidate_skill(
            skill_id="skill-thin",
            name="Thin retrieval",
            description="Literature retrieval.",
            domain_tags=[],
            task_types=["retrieval"],
            target_category="task-retrieval",
            scope="",
            required_inputs=[],
            expected_outputs=[],
            procedure=[],
            required_tools=[],
            limitations=[],
            validation_signals=[],
        )
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("literature retrieval"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-thin"]
    assert bundle.metadata["missing_skill_report"]["missing_capability"] == "literature retrieval"
    assert bundle.metadata["missing_skill_report"]["reason"] == (
        "Matched CandidateSkill coverage is insufficient: input_output_contract, "
        "required_tools, domain_tags, procedure, failure_modes."
    )
    assert bundle.metadata["missing_skill_report"]["risk_level"] == "medium"


def test_graph_context_summary_contains_three_layer_trace(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval"))

    summary = bundle.metadata["graph_context_summary"]
    assert summary["graph_version"] == "phase2-v1"
    assert summary["counts"] == {
        "skills": 3,
        "categories": 5,
        "edges": 0,
        "process_capabilities": 2,
        "scientific_tasks": 3,
        "returned_specific_abilities": 1,
    }
    assert summary["matched_root_capabilities"] == [
        {"category_id": "cap-literature", "name": "Literature"}
    ]
    assert summary["matched_scientific_tasks"] == [
        {"category_id": "task-retrieval", "name": "Retrieval", "parent_category_id": "cap-literature"}
    ]
    assert summary["returned_specific_abilities"] == [
        {"skill_id": "skill-retrieve", "name": "Biomedical paper retrieval"}
    ]
    assert summary["warnings"] == []


def test_skill_ref_metadata_contains_retrieval_info(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload()), encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("biomedical literature retrieval"))

    retrieval = bundle.skills[0].metadata["retrieval"]
    assert set(retrieval) == {
        "score",
        "source",
        "reasons",
        "relation_path",
        "matched_category_id",
        "matched_category_path",
        "retrieved_by",
    }
    assert isinstance(retrieval["score"], float)
    assert retrieval["source"] == "target_category"
    assert retrieval["reasons"]
    assert retrieval["relation_path"] == ["task-retrieval"]


def test_get_does_not_mutate_graph_file(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(json.dumps(_phase2_graph_payload(), sort_keys=True), encoding="utf-8")
    before = graph_path.read_text(encoding="utf-8")
    backend = GraphSkillBackend(graph_path)

    backend.get(_request("biomedical literature retrieval"))

    assert graph_path.read_text(encoding="utf-8") == before


def test_required_tools_are_deduplicated_and_sorted(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "v1",
                "skills": [
                    _candidate_skill(
                        skill_id="skill-1",
                        name="Pytest helper",
                        description="pytest assertions",
                        domain_tags=["testing"],
                        required_tools=["pytest", "rg"],
                    ),
                    _candidate_skill(
                        skill_id="skill-2",
                        name="Regression workflow",
                        description="pytest regression tests",
                        domain_tags=["testing"],
                        required_tools=["git", "pytest"],
                    ),
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("pytest testing"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-1", "skill-2"]
    assert bundle.required_tools == ["git", "pytest", "rg"]


def test_get_renders_candidate_skill_operational_contract(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v2",
                "skills": [
                    _candidate_skill(
                        skill_id="skill-1",
                        procedure=["Run pytest with verbose output.", "Inspect the failing assertion."],
                        scripts=["pytest {test_node} -v"],
                        resources=["docs/testing.md"],
                        examples=["pytest tests/test_example.py::test_failure -v"],
                    )
                ],
                "edges": [],
                "metadata": {"graph_context_summary": "Testing graph"},
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("pytest regression"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-1"]
    assert bundle.graph_version_ref == "graph-v2"
    assert bundle.metadata["graph_context_summary"]["source_summary"] == "Testing graph"
    assert bundle.metadata["matched_skill_ids"] == ["skill-1"]
    assert "Description:\nUse pytest output to isolate failing behavior." in bundle.skills[0].content
    assert "Procedure:\n1. Run pytest with verbose output.\n2. Inspect the failing assertion." in bundle.skills[0].content
    assert "Required Inputs:\n- pytest failure output" in bundle.skills[0].content
    assert bundle.skills[0].metadata["scripts"] == ["pytest {test_node} -v"]
    assert bundle.skills[0].metadata["resources"] == ["docs/testing.md"]
    assert bundle.skills[0].metadata["confidence"] == 0.8
    assert bundle.required_tools == ["pytest", "rg"]


def test_look_at_writes_update_summary_jsonl_with_versions(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "graph-v2",
                "skills": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)
    event = {
        "source_run_id": "run-1",
        "candidate_skill_id": "skill-1",
        "update_type": "skipped",
        "affected_skill_ids": ["skill-1"],
        "affected_edges": [{"source_id": "skill-1", "target_id": "testing"}],
        "decision_rationale": "Observation recorded without graph mutation.",
        "validation_signals": ["human_feedback"],
        "provenance": {"observer": "test"},
    }

    result = backend.look_at(event)

    update_log = tmp_path / "skills.updates.jsonl"
    logged = json.loads(update_log.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(result, SkillUpdateResult)
    assert result.status == "recorded"
    assert result.graph_version_ref == "graph-v2"
    assert result.skill_state_ref == "graph-v2"
    assert result.update_summary["graph_version_before"] == "graph-v2"
    assert result.update_summary["graph_version_after"] == "graph-v2"
    assert result.update_summary["output_paths"]["legacy_update_log"] == str(update_log)
    assert result.metadata["update_log"] == str(update_log)
    assert result.update_summary["before_graph_version"] == "graph-v2"
    assert result.update_summary["after_graph_version"] == "graph-v2"
    assert logged["source_run_id"] == "run-1"
    assert logged["candidate_skill_id"] == "skill-1"
    assert logged["update_type"] == "skipped"
    assert logged["affected_skill_ids"] == ["skill-1"]
    assert logged["graph_version_before"] == "graph-v2"
    assert logged["graph_version_after"] == "graph-v2"
    assert logged["provenance"]["observer"] == "test"
    assert logged["provenance"]["proposal_ids"] == []
    assert logged["provenance"]["observation_id"].startswith("skill-observation-")
    assert "timestamp" in logged


def test_invalid_skill_entries_are_skipped_and_reported(tmp_path):
    graph_path = tmp_path / "skills.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "version": "v1",
                "skills": [
                    {"skill_id": "bad-1", "name": "Missing CandidateSkill fields"},
                    "not-a-skill",
                    _candidate_skill(
                        skill_id="skill-1",
                        name="Pytest helper",
                        description="Use pytest for focused tests.",
                        domain_tags=["testing"],
                    ),
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    backend = GraphSkillBackend(graph_path)

    bundle = backend.get(_request("pytest"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill-1"]
    assert len(bundle.metadata["skipped_skills"]) == 2
    assert bundle.metadata["skipped_skills"][0]["index"] == 0
    assert "Field required" in bundle.metadata["skipped_skills"][0]["reason"]
    assert bundle.metadata["skipped_skills"][1] == {"index": 1, "reason": "skill entry must be an object"}


def test_seed_skill_graph_contains_candidate_skills():
    graph = SkillGraph.model_validate_json(
        Path("docs/superpowers/fixtures/simple-skill-graph.json").read_text(encoding="utf-8")
    )

    assert graph.version == "seed-v1"
    assert [skill.skill_id for skill in graph.skills] == [
        "skill-empty-literature-review",
        "skill-empty-experiment-planning",
        "skill-empty-result-summarization",
    ]


def test_blank_extension_points_raise_clear_errors(tmp_path):
    backend = GraphSkillBackend(tmp_path / "skills.json")

    with pytest.raises(NotImplementedError, match="mine_resources is not implemented"):
        backend.mine_resources()
    with pytest.raises(NotImplementedError, match="rewire_edges is not implemented"):
        backend.rewire_edges()
