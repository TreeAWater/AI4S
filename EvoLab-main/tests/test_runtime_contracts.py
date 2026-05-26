import pytest
from pydantic import ValidationError

from evolab.contracts.common import Message
from evolab.contracts.evolution import EvolutionRunEvent, EvolutionRunEventType, LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.evolution import LabSignals, LLMEvolutionResult, StandardEvolutionMetrics
from evolab.contracts.llm import LLMGenerationConfig, SubAgentAction
from evolab.contracts.records import EvolutionRunRecord, SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, MemoryItem, RetrievalRequest, SkillBundle
from evolab.contracts.task import ProposedTaskRelationType, TaskOrigin, TaskPurpose
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult


def test_retrieval_request_round_trips():
    request = RetrievalRequest(task_id="task-1", role="solver", query="prior failures")
    loaded = RetrievalRequest.model_validate_json(request.model_dump_json())
    assert loaded.query == "prior failures"


def test_tool_call_and_result_round_trip():
    call = ToolCall(call_id="call-1", name="read_file", arguments={"path": "x"})
    result = ToolResult(call_id="call-1", status="ok", content="done")
    assert ToolCall.model_validate(call.model_dump()).name == "read_file"
    assert ToolResult.model_validate(result.model_dump()).status == "ok"


def test_evolution_request_requires_mode():
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id="local",
        artifact_root_uri="lab/evolution/llm/run-1",
        trigger_trajectory_ref="traj-1",
    )
    assert request.mode == LLMEvolutionMode.BASICS


def test_evolution_run_event_uses_known_event_types():
    event = EvolutionRunEvent(run_ref="evo-1", event_type=EvolutionRunEventType.RUN_STARTED)
    loaded = EvolutionRunEvent.model_validate_json(event.model_dump_json())

    assert loaded.event_type == EvolutionRunEventType.RUN_STARTED

    with pytest.raises(ValidationError):
        EvolutionRunEvent(run_ref="evo-1", event_type="unknown_event")


def test_subagent_action_tool_call_requires_tool_call():
    with pytest.raises(ValidationError):
        SubAgentAction(action="tool_call")

    with pytest.raises(ValidationError):
        SubAgentAction(
            action="tool_call",
            content="ignored",
            tool_call=ToolCall(call_id="call-1", name="read_file"),
        )

    action = SubAgentAction(
        action="tool_call",
        tool_call=ToolCall(call_id="call-1", name="read_file"),
    )
    assert action.tool_call is not None
    assert action.tool_calls == [action.tool_call]


def test_subagent_action_tool_call_accepts_multiple_tool_calls():
    first = ToolCall(call_id="call-1", name="read_file")
    second = ToolCall(call_id="call-2", name="search")

    action = SubAgentAction(action="tool_call", tool_calls=[first, second])

    assert action.tool_call == first
    assert action.tool_calls == [first, second]


@pytest.mark.parametrize("action", ["final_answer", "ask_human", "abort"])
def test_subagent_action_text_actions_require_content_and_reject_tool_calls(action):
    with pytest.raises(ValidationError):
        SubAgentAction(action=action, content="")

    with pytest.raises(ValidationError):
        SubAgentAction(
            action=action,
            content="done",
            tool_call=ToolCall(call_id="call-1", name="read_file"),
        )
    with pytest.raises(ValidationError):
        SubAgentAction(
            action=action,
            content="done",
            tool_calls=[ToolCall(call_id="call-1", name="read_file")],
        )

    assert SubAgentAction(action=action, content="done").content == "done"


def test_tool_call_record_requires_matching_call_ids():
    with pytest.raises(ValidationError):
        ToolCallRecord(
            tool_call=ToolCall(call_id="call-1", name="read_file"),
            result=ToolResult(call_id="call-2", status="ok", content="done"),
        )


def test_evolution_run_record_requires_matching_result_status():
    with pytest.raises(ValidationError):
        EvolutionRunRecord(
            run_ref="evo-1",
            mode=LLMEvolutionMode.BASICS,
            backend_id="local",
            result_status="failed",
            result=LLMEvolutionResult(status="skipped"),
        )


def test_evolution_result_requires_promotion_fields_to_match_status():
    with pytest.raises(ValidationError):
        LLMEvolutionResult(
            status="not_recommended",
            recommend_for_promotion=True,
            new_state_ref="state-1",
        )

    with pytest.raises(ValidationError):
        LLMEvolutionResult(status="skipped", new_state_ref="state-1")

    with pytest.raises(ValidationError):
        LLMEvolutionResult(status="promoted_candidate")

    result = LLMEvolutionResult(
        status="promoted_candidate",
        recommend_for_promotion=True,
        new_state_ref="state-1",
    )
    assert result.recommend_for_promotion is True


def test_evolution_run_record_requires_matching_lora_role_when_result_has_one():
    with pytest.raises(ValidationError):
        EvolutionRunRecord(
            run_ref="evo-1",
            mode=LLMEvolutionMode.BASICS,
            backend_id="local",
            result_status="promoted_candidate",
            result=LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
                lora_role="solver",
            ),
            lora_role="composed",
        )

    with pytest.raises(ValidationError):
        EvolutionRunRecord(
            run_ref="evo-1",
            mode=LLMEvolutionMode.BASICS,
            backend_id="local",
            result_status="promoted_candidate",
            result=LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
            ),
            lora_role="unknown",
        )

    with pytest.raises(ValidationError):
        EvolutionRunRecord(
            run_ref="evo-1",
            mode=LLMEvolutionMode.BASICS,
            backend_id="local",
            result_status="promoted_candidate",
            result=LLMEvolutionResult(
                status="promoted_candidate",
                recommend_for_promotion=True,
                new_state_ref="state-1",
            ),
            lora_role="solver",
        )

    record = EvolutionRunRecord(
        run_ref="evo-1",
        mode=LLMEvolutionMode.BASICS,
        backend_id="local",
        result_status="promoted_candidate",
        result=LLMEvolutionResult(
            status="promoted_candidate",
            recommend_for_promotion=True,
            new_state_ref="state-1",
            lora_role="solver",
        ),
    )
    assert record.lora_role is None


def test_retrieval_request_rejects_invalid_task_origin_and_purpose():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="prior failures",
            task_origin="unknown",
        )

    with pytest.raises(ValidationError):
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="prior failures",
            task_purpose="unknown",
        )


def test_subagent_run_record_rejects_invalid_task_origin_and_purpose():
    base_kwargs = {
        "run_ref": "run-1",
        "task_id": "task-1",
        "task_origin": TaskOrigin.HUMAN,
        "task_purpose": TaskPurpose.SCIENCE,
        "stage_index": 0,
        "role": "solver",
        "instruction": "Solve it.",
        "retrieval_request": RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="prior failures",
            task_origin=TaskOrigin.HUMAN,
            task_purpose=TaskPurpose.SCIENCE,
        ),
        "memory_bundle": MemoryBundle(backend_id="memory-local"),
        "skill_bundle": SkillBundle(backend_id="skill-local"),
        "prompt_messages": [Message(role="user", content="Solve it.")],
        "llm_backend_id": "local",
        "proposed_relation_type": ProposedTaskRelationType.SUBPROBLEM,
    }

    with pytest.raises(ValidationError):
        SubagentRunRecord(**{**base_kwargs, "task_origin": "unknown"})

    with pytest.raises(ValidationError):
        SubagentRunRecord(**{**base_kwargs, "task_purpose": "unknown"})

    with pytest.raises(ValidationError):
        SubagentRunRecord(**{**base_kwargs, "proposed_relation_type": "unknown"})


def test_runtime_contracts_reject_invalid_numeric_values():
    with pytest.raises(ValidationError):
        MemoryItem(memory_id="memory-1", content="prior failure", score=-0.1)

    with pytest.raises(ValidationError):
        MemoryItem(memory_id="memory-1", content="prior failure", score=1.1)

    with pytest.raises(ValidationError):
        LLMGenerationConfig(model="local", temperature=-0.1)

    with pytest.raises(ValidationError):
        LLMGenerationConfig(model="local", max_output_tokens=0)

    with pytest.raises(ValidationError):
        LLMGenerationConfig(model="local", response_input_items=["not-a-dict"])

    with pytest.raises(ValidationError):
        LabSignals(solve_rate=1.1)

    with pytest.raises(ValidationError):
        StandardEvolutionMetrics(n_train_samples=-1)
