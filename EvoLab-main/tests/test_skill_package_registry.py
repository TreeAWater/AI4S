import json
from pathlib import Path

import pytest

from evolab.backends.skills import GraphSkillBackend, SkillGraph, SkillGraphSkillNode
from evolab.backends.skills.package_loader import SkillPackageLoader
from evolab.backends.skills.registry import SkillRegistry
from evolab.backends.skills.store import GraphSkillStore
from evolab.contracts.retrieval import RetrievalRequest


DEV_ROOT = Path(__file__).resolve().parents[1] / "dev"
DEV_SEED_GRAPH = DEV_ROOT / "configs" / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json"
DEV_SKILL_GROUP = DEV_ROOT / "configs" / "skills" / "groups" / "scientific_ie_v1.yaml"
DEV_DOCUMENT_INTAKE_METADATA = DEV_ROOT / "skills" / "scientific_ie" / "scientific_document_intake" / "metadata.yaml"
DEV_DOCUMENT_INTAKE_SKILL = DEV_ROOT / "skills" / "scientific_ie" / "scientific_document_intake" / "SKILL.md"


def _request(query: str, **metadata) -> RetrievalRequest:
    return RetrievalRequest(task_id="task-1", role="solver", query=query, metadata=metadata)


def _package_metadata(skill_id: str = "skill.test_deep_table.v1", **overrides) -> dict:
    data = {
        "schema_version": "v1",
        "skill_id": skill_id,
        "name": "Deep Table Reader",
        "version": "v1",
        "summary": "Read deeply nested scientific table evidence.",
        "source_type": "human",
        "source_uri": "seed://test/v1",
        "provenance": {"author": "test"},
        "domain_tags": ["literature", "table"],
        "task_types": ["extraction"],
        "target_category": "task.deep.table",
        "scope": "Scientific table evidence extraction",
        "applicability": ["A scientific paper has table evidence."],
        "limitations": ["Does not infer missing headers."],
        "required_inputs": ["paper text", "table artifact"],
        "expected_outputs": ["normalized table evidence"],
        "dependencies": ["document intake"],
        "environment_assumptions": ["Table artifacts are readable."],
        "procedure": ["Inspect the table.", "Read relevant rows.", "Validate values."],
        "required_tools": ["inspect_table", "read_table_slice"],
        "scripts": [],
        "resources": ["resources/table-policy.md"],
        "examples": ["Extract table evidence from supplementary workbook."],
        "tests": {
            "smoke": ["Load a package fixture."],
            "synthetic": ["Extract a synthetic table row."],
            "system": [],
            "benchmark": [],
        },
        "validation_signals": ["schema_valid", "evidence_present"],
        "confidence": 0.84,
        "metadata": {"stable_skill": True, "domain_specific": False},
    }
    data.update(overrides)
    return data


def _write_package(root: Path, slug: str = "deep_table_reader", **metadata_overrides) -> Path:
    package_dir = root / slug
    (package_dir / "resources").mkdir(parents=True)
    (package_dir / "tests").mkdir()
    (package_dir / "SKILL.md").write_text("# Deep Table Reader\n\nUse table evidence.\n", encoding="utf-8")
    metadata = _package_metadata(**metadata_overrides)
    (package_dir / "metadata.yaml").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return package_dir


def _lightweight_graph(package_ref: str, *, skill_id: str = "skill.test_deep_table.v1") -> dict:
    return {
        "schema_version": "v1",
        "version": "package-graph-v1",
        "categories": [
            {
                "category_id": "cap.literature",
                "name": "Literature",
                "layer": "scientific_process_capability",
                "description": "Read scientific literature.",
            },
            {
                "category_id": "task.artifacts",
                "name": "Artifact Understanding",
                "layer": "scientific_task",
                "parent_category_id": "cap.literature",
                "metadata": {"task_types": ["extraction"]},
            },
            {
                "category_id": "task.artifact.tables",
                "name": "Scientific Table Evidence",
                "layer": "scientific_task",
                "parent_category_id": "task.artifacts",
                "metadata": {"task_types": ["table extraction"], "domain_tags": ["table"]},
            },
            {
                "category_id": "task.deep.table",
                "name": "Deep Table Extraction",
                "layer": "scientific_task",
                "parent_category_id": "task.artifact.tables",
                "metadata": {"task_types": ["table extraction"], "domain_tags": ["table"]},
            },
        ],
        "skills": [
            {
                "id": skill_id,
                "name": "Deep Table Reader",
                "summary": "Read deeply nested scientific table evidence.",
                "package_ref": package_ref,
                "group": "test_group",
                "status": "active",
                "tags": ["table", "scientific-ie"],
                "metadata": {"stable_skill": True, "domain_specific": False},
            }
        ],
        "edges": [
            {
                "source_id": skill_id,
                "target_id": "task.deep.table",
                "relation": "belongs_to_category",
            }
        ],
        "metadata": {"storage_format": "package_ref_v1"},
    }


def test_skill_package_loader_hydrates_candidate_skill(tmp_path):
    package_dir = _write_package(tmp_path / "skills")

    package = SkillPackageLoader(repo_root=tmp_path).load(package_dir)
    candidate = package.to_candidate_skill()

    assert candidate.skill_id == "skill.test_deep_table.v1"
    assert candidate.required_tools == ["inspect_table", "read_table_slice"]
    assert candidate.resources == ["resources/table-policy.md"]
    assert candidate.smoke_tests == ["Load a package fixture."]
    assert candidate.synthetic_tests == ["Extract a synthetic table row."]
    assert candidate.metadata["package_path"].endswith("deep_table_reader")
    assert "Use table evidence" in candidate.metadata["skill_markdown"]


def test_skill_registry_scans_roots_and_loads_group_config(tmp_path):
    skill_root = tmp_path / "skills"
    _write_package(skill_root)
    group_config = tmp_path / "group.yaml"
    group_config.write_text(
        json.dumps(
            {
                "group_name": "test_group",
                "description": "Test group",
                "graph": "graph.json",
                "skill_roots": ["skills"],
                "domain_packages": ["domain_packages/test"],
                "default_active_status": "active",
            }
        ),
        encoding="utf-8",
    )

    registry = SkillRegistry(repo_root=tmp_path)
    group = registry.load_group(group_config)

    assert group.group_name == "test_group"
    assert sorted(registry.packages_by_id) == ["skill.test_deep_table.v1"]
    assert registry.group_skill_ids["test_group"] == ["skill.test_deep_table.v1"]
    assert registry.category_skill_ids["task.deep.table"] == ["skill.test_deep_table.v1"]


def test_skill_registry_duplicate_ids_are_deterministic(tmp_path):
    skill_root = tmp_path / "skills"
    first = _write_package(skill_root, "a_reader", summary="First package")
    second = _write_package(skill_root, "b_reader", summary="Second package")

    registry = SkillRegistry(repo_root=tmp_path)
    registry.scan(["skills"], group_name="test_group")

    assert registry.get("skill.test_deep_table.v1").package_path == first.relative_to(tmp_path).as_posix()
    assert second.relative_to(tmp_path).as_posix() in registry.warnings[0]
    assert "duplicate skill_id skill.test_deep_table.v1" in registry.warnings[0]


def test_lightweight_graph_validates_with_skill_nodes():
    payload = json.loads(DEV_SEED_GRAPH.read_text(encoding="utf-8"))

    graph = SkillGraph.model_validate(payload)

    assert graph.version == "scientific-ie-v1"
    assert all(isinstance(skill, SkillGraphSkillNode) for skill in graph.skills)
    assert all(skill.package_ref for skill in graph.skills)
    assert not any("scope" in raw_skill for raw_skill in payload["skills"])


def test_graph_backend_hydrates_candidate_from_package_ref(tmp_path):
    package_dir = _write_package(tmp_path / "skills")
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_lightweight_graph(package_dir.relative_to(tmp_path).as_posix())), encoding="utf-8")

    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path).get(_request("literature table extraction"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill.test_deep_table.v1"]
    assert bundle.skills[0].metadata["candidate_metadata"]["package_path"].endswith("deep_table_reader")
    assert {"inspect_table", "read_table_slice"}.issubset(bundle.required_tools)


def test_missing_package_ref_warns_or_raises_in_strict_mode(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_lightweight_graph("skills/missing")), encoding="utf-8")

    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path).get(_request("literature table extraction"))

    assert bundle.skills == []
    assert bundle.metadata["package_warnings"] == ["missing skill package for skill.test_deep_table.v1: skills/missing"]
    with pytest.raises(ValueError, match="missing skill package"):
        GraphSkillBackend(graph_path, repo_root=tmp_path, strict_packages=True).get(
            _request("literature table extraction")
        )


def test_recursive_package_retrieval_supports_more_than_three_category_levels(tmp_path):
    package_dir = _write_package(tmp_path / "skills")
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_lightweight_graph(package_dir.relative_to(tmp_path).as_posix())), encoding="utf-8")

    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path).get(_request("deep table extraction from literature"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill.test_deep_table.v1"]
    assert bundle.metadata["graph_context_summary"]["matched_category_paths"] == [
        "Literature > Artifact Understanding > Scientific Table Evidence > Deep Table Extraction"
    ]


def test_adding_new_package_requires_no_backend_code_change(tmp_path):
    package_dir = _write_package(
        tmp_path / "skills",
        "new_record_builder",
        skill_id="skill.new_record_builder.v1",
        name="New Record Builder",
        summary="Construct new structured records.",
        target_category="task.deep.table",
        required_tools=["json_schema_validate"],
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(_lightweight_graph(package_dir.relative_to(tmp_path).as_posix(), skill_id="skill.new_record_builder.v1")),
        encoding="utf-8",
    )

    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path).get(_request("construct records from deep table extraction"))

    assert [skill.skill_id for skill in bundle.skills] == ["skill.new_record_builder.v1"]
    assert bundle.required_tools == ["json_schema_validate"]


def test_retrieval_trace_and_top_k_are_deterministic_for_seed_graph():
    backend = GraphSkillBackend(DEV_SEED_GRAPH)

    bundle = backend.get(
        _request(
            "extract biological component records from scientific literature with supplementary tables and sequences",
            top_k=3,
        )
    )

    trace = bundle.metadata["retrieval_trace"]
    assert len(bundle.skills) == 3
    assert trace["graph_version"] == "scientific-ie-v1"
    assert trace["returned_skill_ids"] == [skill.skill_id for skill in bundle.skills]
    assert trace["selected_category_paths"]
    assert trace["coverage_report"]["sufficient"] is True


def test_depends_on_completion_adds_schema_interpretation_prerequisite():
    backend = GraphSkillBackend(DEV_SEED_GRAPH)

    bundle = backend.get(
        _request(
            "field mapping",
            target_category="task.schema_guided_field_mapping",
            top_k=2,
        )
    )

    returned = [skill.skill_id for skill in bundle.skills]
    assert returned == [
        "skill.extraction_schema_interpretation.v1",
        "skill.schema_guided_field_mapping.v1",
    ]
    trace = bundle.metadata["retrieval_trace"]
    assert trace["dependency_added_skill_ids"] == ["skill.extraction_schema_interpretation.v1"]
    assert trace["relation_expansion_steps"] == [
        {
            "source_skill_id": "skill.schema_guided_field_mapping.v1",
            "target_skill_id": "skill.extraction_schema_interpretation.v1",
            "relation": "depends_on",
            "reason": "dependency_added",
        }
    ]
    assert "json_schema_validate" in bundle.required_tools
    summary = bundle.metadata["graph_context_summary"]
    assert summary["directly_matched_skill_ids"] == ["skill.schema_guided_field_mapping.v1"]
    assert summary["dependency_completed_skill_ids"] == ["skill.extraction_schema_interpretation.v1"]
    assert summary["optional_relationship_expanded_skill_ids"] == []


def test_validation_dependency_completion_records_prerequisites():
    backend = GraphSkillBackend(DEV_SEED_GRAPH)

    bundle = backend.get(
        _request(
            "result validation",
            target_category="task.extraction_result_validation",
            top_k=6,
        )
    )

    returned = [skill.skill_id for skill in bundle.skills]
    assert "skill.structured_record_construction.v1" in returned
    assert "skill.evidence_source_attribution.v1" in returned
    assert "skill.extraction_result_validation.v1" in returned
    assert returned.index("skill.structured_record_construction.v1") < returned.index(
        "skill.extraction_result_validation.v1"
    )
    assert returned.index("skill.evidence_source_attribution.v1") < returned.index(
        "skill.extraction_result_validation.v1"
    )
    trace = bundle.metadata["retrieval_trace"]
    assert {
        "skill.structured_record_construction.v1",
        "skill.evidence_source_attribution.v1",
    }.issubset(set(trace["dependency_added_skill_ids"]))
    assert {
        (step["source_skill_id"], step["target_skill_id"], step["relation"], step["reason"])
        for step in trace["relation_expansion_steps"]
    } >= {
        (
            "skill.extraction_result_validation.v1",
            "skill.structured_record_construction.v1",
            "depends_on",
            "dependency_added",
        ),
        (
            "skill.extraction_result_validation.v1",
            "skill.evidence_source_attribution.v1",
            "depends_on",
            "dependency_added",
        ),
    }


def test_optional_related_expansion_is_traced_separately_from_dependencies():
    backend = GraphSkillBackend(DEV_SEED_GRAPH)

    bundle = backend.get(
        _request(
            "discover supplementary artifacts",
            target_category="task.supplementary_artifact_discovery",
            top_k=4,
        )
    )

    returned = [skill.skill_id for skill in bundle.skills]
    assert "skill.supplementary_artifact_discovery.v1" in returned
    assert "skill.multi_format_artifact_reading.v1" in returned
    trace = bundle.metadata["retrieval_trace"]
    assert "skill.multi_format_artifact_reading.v1" in trace["optional_expanded_skill_ids"]
    assert any(
        step == {
            "source_skill_id": "skill.supplementary_artifact_discovery.v1",
            "target_skill_id": "skill.multi_format_artifact_reading.v1",
            "relation": "related_to",
            "reason": "optional_relationship_added",
        }
        for step in trace["relation_expansion_steps"]
    )


def test_conflict_and_deprecated_relationships_warn_without_adding_target(tmp_path):
    source_package = _write_package(
        tmp_path / "skills",
        "source",
        skill_id="skill.source.v1",
        name="Source Skill",
        summary="Selected source skill.",
    )
    target_package = _write_package(
        tmp_path / "skills",
        "target",
        skill_id="skill.target.v1",
        name="Target Skill",
        summary="Conflicting target skill.",
        target_category=None,
    )
    graph = _lightweight_graph(source_package.relative_to(tmp_path).as_posix(), skill_id="skill.source.v1")
    graph["skills"].append(
        {
            "id": "skill.target.v1",
            "name": "Target Skill",
            "summary": "Conflicting target skill.",
            "package_ref": target_package.relative_to(tmp_path).as_posix(),
            "group": "test_group",
            "status": "active",
            "tags": ["table"],
            "metadata": {"stable_skill": True, "domain_specific": False},
        }
    )
    graph["edges"].extend(
        [
            {
                "source_id": "skill.source.v1",
                "target_id": "skill.target.v1",
                "relation": "conflicts_with",
            },
            {
                "source_id": "skill.source.v1",
                "target_id": "skill.target.v1",
                "relation": "deprecated_by",
            },
        ]
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    bundle = GraphSkillBackend(graph_path, repo_root=tmp_path).get(_request("literature table extraction", top_k=5))

    assert [skill.skill_id for skill in bundle.skills] == ["skill.source.v1"]
    trace = bundle.metadata["retrieval_trace"]
    assert trace["conflict_warnings"] == [
        "skill.source.v1 conflicts_with skill.target.v1",
        "skill.source.v1 deprecated_by skill.target.v1",
    ]
    assert trace["skipped_relationships"] == [
        {
            "source_skill_id": "skill.source.v1",
            "target_skill_id": "skill.target.v1",
            "relation": "conflicts_with",
            "reason": "warning_only",
        },
        {
            "source_skill_id": "skill.source.v1",
            "target_skill_id": "skill.target.v1",
            "relation": "deprecated_by",
            "reason": "warning_only",
        },
    ]


def test_relationship_expansion_order_and_trace_are_deterministic():
    backend = GraphSkillBackend(DEV_SEED_GRAPH)
    request = _request(
        "field mapping",
        target_category="task.schema_guided_field_mapping",
        top_k=4,
    )

    first = backend.get(request)
    second = backend.get(request)

    assert [skill.skill_id for skill in first.skills] == [skill.skill_id for skill in second.skills]
    assert first.metadata["retrieval_trace"]["relation_expansion_steps"] == (
        second.metadata["retrieval_trace"]["relation_expansion_steps"]
    )


def test_retrieval_does_not_mutate_graph_group_or_package_files():
    paths = [
        DEV_SEED_GRAPH,
        DEV_SKILL_GROUP,
        DEV_DOCUMENT_INTAKE_METADATA,
        DEV_DOCUMENT_INTAKE_SKILL,
    ]
    before = {path: path.read_text(encoding="utf-8") for path in paths}

    GraphSkillBackend(paths[0]).get(_request("scientific document intake for supplementary tables"))

    assert {path: path.read_text(encoding="utf-8") for path in paths} == before


def test_graph_skill_store_loads_registry_for_seed_group():
    loaded = GraphSkillStore(DEV_SEED_GRAPH).load_graph()

    assert loaded.group_configs[0].group_name == "scientific_ie_v1"
    assert loaded.registry is not None
    assert len(loaded.registry.packages_by_id) >= 19
    assert "skill.scientific_document_intake.v1" in loaded.registry.group_skill_ids["scientific_ie_v1"]
