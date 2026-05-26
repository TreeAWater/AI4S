from evolab.contracts.retrieval import SkillBundle, SkillRef
from evolab.contracts.workflow import WorkflowEdge, WorkflowPlan
from evolab.runtime.workflow_planner import SkillWorkflowPlanner, topologically_order_workflow_nodes


def _skill(
    skill_id: str,
    name: str,
    *,
    required_tools: list[str] | None = None,
    required_inputs: list[str] | None = None,
    expected_outputs: list[str] | None = None,
) -> SkillRef:
    return SkillRef(
        skill_id=skill_id,
        name=name,
        content=(
            f"Description:\n{name}\n\n"
            f"Required Inputs:\n" + "\n".join(f"- {item}" for item in required_inputs or []) + "\n\n"
            f"Expected Outputs:\n" + "\n".join(f"- {item}" for item in expected_outputs or [])
        ),
        required_tools=required_tools or [],
        metadata={
            "required_inputs": required_inputs or [],
            "expected_outputs": expected_outputs or [],
            "retrieval": {"score": 1.0},
        },
    )


def _bundle(*skills: SkillRef, relation_steps: list[dict] | None = None) -> SkillBundle:
    return SkillBundle(
        backend_id="skill-local",
        graph_version_ref="graph-v1",
        required_tools=sorted({tool for skill in skills for tool in skill.required_tools}),
        skills=list(skills),
        metadata={
            "graph_context_summary": {"graph_version": "graph-v1"},
            "retrieval_trace": {
                "returned_skill_ids": [skill.skill_id for skill in skills],
                "directly_matched_skill_ids": [skills[0].skill_id] if skills else [],
                "dependency_added_skill_ids": [],
                "optional_expanded_skill_ids": [],
                "relation_expansion_steps": relation_steps or [],
            },
        },
    )


def test_workflow_planner_creates_one_node_per_selected_skill_and_aggregates_tools():
    bundle = _bundle(
        _skill("skill.scientific_document_intake.v1", "Scientific Document Intake", required_tools=["read_text"]),
        _skill("skill.schema_guided_field_mapping.v1", "Schema Guided Field Mapping", required_tools=["inspect_table"]),
    )

    plan = SkillWorkflowPlanner().plan(task_id="task-1", task_goal="extract records", role="solver", skill_bundle=bundle)

    assert isinstance(plan, WorkflowPlan)
    assert [node.skill_id for node in plan.nodes] == [
        "skill.scientific_document_intake.v1",
        "skill.schema_guided_field_mapping.v1",
    ]
    assert plan.required_tools == ["inspect_table", "read_text"]
    assert plan.metadata["source_skill_bundle_backend_id"] == "skill-local"


def test_mandatory_dependency_relations_become_edges_with_prerequisite_direction():
    schema = _skill("skill.extraction_schema_interpretation.v1", "Extraction Schema Interpretation")
    mapping = _skill("skill.schema_guided_field_mapping.v1", "Schema Guided Field Mapping")
    bundle = _bundle(
        mapping,
        schema,
        relation_steps=[
            {
                "source_skill_id": mapping.skill_id,
                "target_skill_id": schema.skill_id,
                "relation": "depends_on",
                "reason": "dependency_added",
            }
        ],
    )

    plan = SkillWorkflowPlanner().plan(task_id="task-1", task_goal="map fields", role="solver", skill_bundle=bundle)

    schema_node = next(node for node in plan.nodes if node.skill_id == schema.skill_id)
    mapping_node = next(node for node in plan.nodes if node.skill_id == mapping.skill_id)
    assert any(
        edge.source_node_id == schema_node.node_id
        and edge.target_node_id == mapping_node.node_id
        and edge.relation == "depends_on"
        for edge in plan.edges
    )
    assert plan.metadata["topological_order"].index(schema.skill_id) < plan.metadata["topological_order"].index(
        mapping.skill_id
    )


def test_validation_relations_place_validation_after_construction():
    construction = _skill("skill.structured_record_construction.v1", "Structured Record Construction")
    validation = _skill("skill.extraction_result_validation.v1", "Extraction Result Validation")
    bundle = _bundle(
        validation,
        construction,
        relation_steps=[
            {
                "source_skill_id": validation.skill_id,
                "target_skill_id": construction.skill_id,
                "relation": "validates",
                "reason": "optional_relationship_added",
            }
        ],
    )

    plan = SkillWorkflowPlanner().plan(task_id="task-1", task_goal="validate records", role="solver", skill_bundle=bundle)

    assert plan.metadata["topological_order"].index(construction.skill_id) < plan.metadata["topological_order"].index(
        validation.skill_id
    )


def test_planner_falls_back_to_deterministic_scientific_ie_phase_order():
    mapping = _skill("skill.schema_guided_field_mapping.v1", "Schema Guided Field Mapping")
    intake = _skill("skill.scientific_document_intake.v1", "Scientific Document Intake")
    validation = _skill("skill.extraction_result_validation.v1", "Extraction Result Validation")

    plan = SkillWorkflowPlanner().plan(
        task_id="task-1",
        task_goal="extract scientific records",
        role="solver",
        skill_bundle=_bundle(mapping, validation, intake),
    )

    assert plan.metadata["topological_order"] == [intake.skill_id, mapping.skill_id, validation.skill_id]
    assert any(edge.relation == "phase_order" for edge in plan.edges)


def test_topological_order_is_deterministic_and_cycle_warnings_are_recorded():
    first = _skill("skill.first.v1", "First")
    second = _skill("skill.second.v1", "Second")
    nodes = [
        SkillWorkflowPlanner().plan(
            task_id="task-1",
            task_goal="cycle",
            role="solver",
            skill_bundle=_bundle(first, second),
        ).nodes[0],
        SkillWorkflowPlanner().plan(
            task_id="task-1",
            task_goal="cycle",
            role="solver",
            skill_bundle=_bundle(first, second),
        ).nodes[1],
    ]
    edges = [
        WorkflowEdge(
            source_node_id=nodes[0].node_id,
            target_node_id=nodes[1].node_id,
            relation="related_to",
            reason="test",
            metadata={"edge_strength": "optional"},
        ),
        WorkflowEdge(
            source_node_id=nodes[1].node_id,
            target_node_id=nodes[0].node_id,
            relation="related_to",
            reason="test",
            metadata={"edge_strength": "optional"},
        ),
    ]

    first_result = topologically_order_workflow_nodes(nodes, edges)
    second_result = topologically_order_workflow_nodes(nodes, edges)

    assert first_result.node_ids == second_result.node_ids
    assert first_result.warnings == second_result.warnings
    assert first_result.warnings


def test_workflow_planner_does_not_introduce_forbidden_domain_specific_skill_ids():
    forbidden = ["promoter", "rbs", "terminator", "grna", "microbe_trait", "chemical_reaction", "material_property"]
    bundle = _bundle(_skill("skill.domain_entity_validation.v1", "Domain Entity Validation"))

    plan = SkillWorkflowPlanner().plan(task_id="task-1", task_goal="extract records", role="solver", skill_bundle=bundle)

    stable_ids = [node.skill_id.casefold() for node in plan.nodes]
    assert not any(term in skill_id for term in forbidden for skill_id in stable_ids)

