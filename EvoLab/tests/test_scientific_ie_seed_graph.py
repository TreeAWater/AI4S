import json
from pathlib import Path

from evolab.backends.skills import GraphSkillBackend
from evolab.backends.skills.graph_schema import SkillGraph
from evolab.contracts.retrieval import RetrievalRequest


SEED_GRAPH_PATH = Path("configs/skills/graphs/scientific_ie_seed_graph_v1.json")
BIOLOGY_PACKAGE_DIR = Path("domain_packages/biology_component_extraction_v1")

FORBIDDEN_STABLE_ID_TERMS = (
    "promoter",
    "rbs",
    "terminator",
    "grna",
    "plasmid_extraction",
    "microbe_trait",
    "chemical_reaction",
    "material_property",
)


def _load_seed_graph_payload() -> dict:
    return json.loads(SEED_GRAPH_PATH.read_text(encoding="utf-8"))


def _backend() -> GraphSkillBackend:
    return GraphSkillBackend(SEED_GRAPH_PATH)


def _request(query: str, **metadata) -> RetrievalRequest:
    return RetrievalRequest(
        task_id="seed-graph-test",
        role="solver",
        query=query,
        metadata=metadata,
    )


def _skill_suffixes(bundle) -> set[str]:
    return {skill.skill_id.removeprefix("skill.").removesuffix(".v1") for skill in bundle.skills}


def test_scientific_ie_seed_graph_loads_with_existing_schema():
    payload = _load_seed_graph_payload()

    graph = SkillGraph.model_validate(payload)

    assert graph.schema_version == "v1"
    assert graph.version == "scientific-ie-v1"
    assert {category.name for category in graph.categories if category.layer == "scientific_process_capability"} == {
        "Literature",
        "Analysis",
        "Validation",
        "Execution",
        "Writing",
    }
    assert len(graph.skills) >= 19
    assert all(skill.metadata["stable_skill"] is True for skill in graph.skills)
    assert all(getattr(skill, "package_ref", None) for skill in graph.skills)


def test_seed_graph_contains_reusable_final_artifact_writing_skill():
    bundle = _backend().get(_request("write final JSONL records artifact and report output"))

    assert "final_artifact_writing" in _skill_suffixes(bundle)
    writing_skill = next(skill for skill in bundle.skills if skill.skill_id == "skill.final_artifact_writing.v1")
    assert writing_skill.required_tools == ["write_jsonl", "write_report"]
    assert writing_skill.metadata["candidate_metadata"]["stable_skill"] is True
    assert writing_skill.metadata["candidate_metadata"]["domain_specific"] is False


def test_biological_component_query_returns_reusable_scientific_ie_skills():
    bundle = _backend().get(
        _request("extract biological component records from scientific literature with supplementary tables and sequences")
    )

    returned = _skill_suffixes(bundle)
    assert {
        "scientific_document_intake",
        "supplementary_artifact_discovery",
        "scientific_table_structure_understanding",
        "extraction_schema_interpretation",
        "schema_guided_field_mapping",
        "structured_record_construction",
        "domain_entity_validation",
        "extraction_result_validation",
    }.issubset(returned)
    assert not any(term in skill.skill_id.casefold() for skill in bundle.skills for term in FORBIDDEN_STABLE_ID_TERMS)


def test_chemical_reaction_query_reuses_general_scientific_ie_skills():
    biological = _backend().get(
        _request("extract biological component records from scientific literature with supplementary tables and sequences")
    )
    chemical = _backend().get(_request("extract chemical reaction conditions from a paper and supplementary tables"))

    biological_skills = _skill_suffixes(biological)
    chemical_skills = _skill_suffixes(chemical)
    expected_overlap = {
        "scientific_document_intake",
        "supplementary_artifact_discovery",
        "scientific_table_structure_understanding",
        "extraction_schema_interpretation",
        "schema_guided_field_mapping",
        "structured_record_construction",
        "extraction_result_validation",
    }
    assert expected_overlap.issubset(chemical_skills)
    assert expected_overlap.issubset(biological_skills & chemical_skills)


def test_seed_graph_required_tools_are_aggregated_for_table_and_schema_tasks():
    bundle = _backend().get(
        _request("extract biological component records from scientific literature with supplementary tables and sequences")
    )

    assert "scientific_table_structure_understanding" in _skill_suffixes(bundle)
    assert "schema_guided_field_mapping" in _skill_suffixes(bundle)
    assert {"inspect_table", "read_table_slice", "json_schema_validate"}.issubset(set(bundle.required_tools))


def test_seed_graph_context_summary_contains_retrieval_metadata():
    bundle = _backend().get(
        _request("extract biological component records from scientific literature with supplementary tables and sequences")
    )

    summary = bundle.metadata["graph_context_summary"]
    assert summary["graph_version"] == "scientific-ie-v1"
    assert summary["matched_category_paths"]
    assert summary["returned_specific_abilities"]
    assert summary["coverage_report"]["sufficient"] is True


def test_seed_graph_top_k_limits_returned_skills():
    bundle = _backend().get(
        _request(
            "extract biological component records from scientific literature with supplementary tables and sequences",
            top_k=3,
        )
    )

    assert len(bundle.skills) == 3


def test_seed_graph_has_no_domain_specific_stable_candidate_skills():
    graph = SkillGraph.model_validate(_load_seed_graph_payload())

    for skill in graph.skills:
        assert skill.metadata["stable_skill"] is True
        assert skill.metadata["domain_specific"] is False
        assert not any(term in skill.skill_id.casefold() for term in FORBIDDEN_STABLE_ID_TERMS)


def test_biology_domain_package_exists_as_resources_not_stable_skills():
    expected_files = {
        "biology_component_schema.json",
        "biological_component_ontology.yaml",
        "biological_sequence_policy.yaml",
        "biological_evidence_policy.yaml",
        "biological_negative_patterns.yaml",
        "task_config.yaml",
        "README.md",
    }

    assert {path.name for path in BIOLOGY_PACKAGE_DIR.iterdir() if path.is_file()} >= expected_files
    schema = json.loads((BIOLOGY_PACKAGE_DIR / "biology_component_schema.json").read_text(encoding="utf-8"))
    negative_patterns = (BIOLOGY_PACKAGE_DIR / "biological_negative_patterns.yaml").read_text(encoding="utf-8")
    ontology = (BIOLOGY_PACKAGE_DIR / "biological_component_ontology.yaml").read_text(encoding="utf-8")

    assert {"article_id", "component_name", "component_type", "sequence", "evidence_source", "status"}.issubset(
        schema["properties"]
    )
    assert "promoter_sequence_record_schema.json" not in (BIOLOGY_PACKAGE_DIR / "README.md").read_text(
        encoding="utf-8"
    )
    for term in ("primer", "barcode", "adapter", "restriction site"):
        assert term in negative_patterns
    for term in ("promoter", "RBS", "terminator", "gRNA"):
        assert term in ontology
