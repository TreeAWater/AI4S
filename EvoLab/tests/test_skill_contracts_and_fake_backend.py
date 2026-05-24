from evolab.backends.skills import FakeSkillBackend
from evolab.contracts.common import ArtifactRef
from evolab.contracts.retrieval import (
    RetrievalRequest,
    SkillBundle,
    SkillItem,
    SkillObservationRequest,
    SkillRef,
    SkillUpdateResult,
)


def test_skill_contracts_include_skill_items_and_state_refs():
    skill = SkillItem(
        skill_id="skill-1",
        name="Lookup",
        content="Use lookup before answering.",
        required_tools=["lookup"],
        script_refs=[ArtifactRef(uri="file:///lab/skills/lookup.py", type="code")],
        resource_refs=[ArtifactRef(uri="file:///lab/skills/resource.txt", type="text")],
    )
    bundle = SkillBundle(
        backend_id="fake_skill",
        graph_version_ref="graph-v1",
        skill_state_ref="skill-state-v1",
        skills=[skill],
        required_tools=["lookup"],
    )
    observation = SkillObservationRequest(
        task_id="task-1",
        run_ref="run-1",
        role="solver",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="lookup"),
        skill_bundle=bundle,
        graph_version_ref=bundle.graph_version_ref,
        skill_state_ref=bundle.skill_state_ref,
        final_answer="done",
    )
    update = SkillUpdateResult(
        status="recorded",
        update_summary={"observed_runs": 1},
        graph_version_ref="graph-v2",
        skill_state_ref="skill-state-v2",
    )

    assert SkillRef is SkillItem
    assert bundle.skill_state_ref == "skill-state-v1"
    assert observation.skill_bundle.skills[0].script_refs[0].type == "code"
    assert update.update_summary == {"observed_runs": 1}


def test_fake_skill_backend_returns_deterministic_bundle_and_update_refs():
    skill = SkillItem(
        skill_id="skill-1",
        name="Lookup",
        content="Use lookup before answering.",
        required_tools=["lookup"],
    )
    backend = FakeSkillBackend(
        skills=[skill],
        graph_version_ref="graph-v1",
        skill_state_ref="skill-state-v1",
        next_skill_state_ref="skill-state-v2",
    )
    request = RetrievalRequest(task_id="task-1", role="solver", query="lookup")

    first = backend.get(request)
    second = backend.get(request)
    update = backend.look_at(
        SkillObservationRequest(
            task_id="task-1",
            run_ref="run-1",
            role="solver",
            retrieval_request=request,
            skill_bundle=first,
            graph_version_ref=first.graph_version_ref,
            skill_state_ref=first.skill_state_ref,
            final_answer="done",
        ).model_dump(mode="json")
    )

    assert first == second
    assert first.required_tools == ["lookup"]
    assert first.graph_version_ref == "graph-v1"
    assert first.skill_state_ref == "skill-state-v1"
    assert update == SkillUpdateResult(
        status="recorded",
        update_summary={"observed_runs": 1},
        graph_version_ref="graph-v1",
        skill_state_ref="skill-state-v2",
    )
    assert backend.get_requests == [request, request]
    assert backend.look_at_events[0]["run_ref"] == "run-1"
