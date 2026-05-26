from evolab.backends.rewards import (
    CompositeRewardCalculator,
    NumToolCallRewardCalculator,
    RewardCalculationContext,
    RewardCalculationRequest,
    RewardComponent,
    RewardExample,
    RewardScore,
    RewardVerification,
    VerifierRewardCalculator,
)
from evolab.contracts.common import Message
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.snapshots import SnapshotRef, ToolsetSnapshot
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolSpec


def _tool_call(name: str, status: str = "ok") -> ToolCallRecord:
    call_id = f"{name}-{status}"
    return ToolCallRecord(
        tool_call=ToolCall(call_id=call_id, name=name),
        result=ToolResult(call_id=call_id, status=status, content="done"),
    )


def _example(sample_id: str, calls: list[ToolCallRecord]) -> RewardExample:
    return RewardExample(sample_id=sample_id, tool_calls=calls)


def test_reward_calculator_abstract_contracts():
    assert RewardCalculationRequest(examples=[]).examples == []
    assert NumToolCallRewardCalculator.__mro__[1].__name__ == "RewardCalculator"
    assert VerifierRewardCalculator.__abstractmethods__ == {"verify"}


def test_num_toolcall_reward_counts_calls_and_computes_advantages():
    request = RewardCalculationRequest(
        examples=[
            _example("sample-1", [_tool_call("read_file"), _tool_call("write_file", "error")]),
            _example("sample-2", [_tool_call("read_file")]),
        ]
    )

    result = NumToolCallRewardCalculator().calculate(request)

    assert [score.reward for score in result.scores] == [2.0, 1.0]
    assert result.aggregate_reward == 1.5
    assert [score.advantage for score in result.scores] == [0.5, -0.5]


def test_reward_request_carries_snapshot_and_reward_policy_refs():
    request = RewardCalculationRequest(
        examples=[_example("sample-1", [_tool_call("read_file")])],
        reward_policy_snapshot_ref="reward-policy-1",
        before_snapshot_refs=[SnapshotRef(snapshot_ref="toolset-before", kind="toolset")],
        after_snapshot_refs=[SnapshotRef(snapshot_ref="toolset-after", kind="toolset")],
        curriculum_state_ref="curriculum-1",
    )

    loaded = RewardCalculationRequest.model_validate_json(request.model_dump_json())

    assert loaded.reward_policy_snapshot_ref == "reward-policy-1"
    assert loaded.before_snapshot_refs[0].snapshot_ref == "toolset-before"
    assert loaded.after_snapshot_refs[0].snapshot_ref == "toolset-after"
    assert loaded.curriculum_state_ref == "curriculum-1"


def test_reward_calculator_can_use_snapshot_context():
    class StaticContext(RewardCalculationContext):
        def get_snapshot(self, snapshot_ref: str):
            if snapshot_ref == "toolset-after":
                return ToolsetSnapshot(
                    snapshot_ref="toolset-after",
                    tool_specs=[ToolSpec(name="read_file", description="Read a file.")],
                )
            return None

    class ToolsetSizeReward(NumToolCallRewardCalculator):
        calculator_id = "toolset_size"

        def calculate(self, request, context=None):
            snapshot_ref = request.after_snapshot_refs[0].snapshot_ref
            snapshot = context.get_snapshot(snapshot_ref) if context is not None else None
            size = len(snapshot.tool_specs) if snapshot is not None else 0
            return self._finalize(
                request,
                [RewardScore(sample_id=request.examples[0].sample_id, reward=float(size))],
            )

    request = RewardCalculationRequest(
        examples=[_example("sample-1", [])],
        after_snapshot_refs=[SnapshotRef(snapshot_ref="toolset-after", kind="toolset")],
    )

    result = ToolsetSizeReward().calculate(request, StaticContext())

    assert result.scores[0].reward == 1.0


def test_num_toolcall_reward_can_filter_tool_name_and_status():
    request = RewardCalculationRequest(
        examples=[
            _example("sample-1", [_tool_call("read_file"), _tool_call("write_file", "error")]),
            _example("sample-2", [_tool_call("read_file"), _tool_call("read_file", "error")]),
        ],
        advantage_baseline=0.0,
    )

    result = NumToolCallRewardCalculator(tool_name="read_file", status="ok").calculate(request)

    assert [score.reward for score in result.scores] == [1.0, 1.0]
    assert [score.raw_score for score in result.scores] == [1.0, 1.0]
    assert [score.advantage for score in result.scores] == [1.0, 1.0]


def test_composite_reward_calculator_combines_component_rewards():
    request = RewardCalculationRequest(
        examples=[
            _example("sample-1", [_tool_call("read_file"), _tool_call("write_file", "error")]),
            _example("sample-2", [_tool_call("read_file")]),
        ]
    )
    calculator = CompositeRewardCalculator(
        [
            RewardComponent(NumToolCallRewardCalculator(), weight=0.5),
            RewardComponent(
                NumToolCallRewardCalculator(status="ok", calculator_id="ok_toolcall"),
                weight=2.0,
            ),
        ],
        mode="weighted_sum",
    )

    result = calculator.calculate(request)

    assert [score.reward for score in result.scores] == [3.0, 2.5]
    assert result.aggregate_reward == 2.75
    assert [score.advantage for score in result.scores] == [0.25, -0.25]
    assert result.metadata["component_ids"] == ["num_toolcall", "ok_toolcall"]


def test_verifier_reward_calculator_maps_pass_fail_to_rewards():
    class HasToolVerifier(VerifierRewardCalculator):
        def __init__(self):
            super().__init__(calculator_id="has_tool", pass_reward=1.0, fail_reward=-1.0)

        def verify(self, example: RewardExample) -> RewardVerification:
            passed = bool(example.tool_calls)
            return RewardVerification(
                sample_id=example.sample_id,
                passed=passed,
                reason="has tool call" if passed else "no tool calls",
            )

    request = RewardCalculationRequest(
        examples=[
            _example("sample-1", [_tool_call("read_file")]),
            _example("sample-2", []),
        ],
        compute_advantages=False,
    )

    result = HasToolVerifier().calculate(request)

    assert [score.reward for score in result.scores] == [1.0, -1.0]
    assert [score.passed for score in result.scores] == [True, False]
    assert [score.advantage for score in result.scores] == [None, None]


def test_reward_example_can_be_built_from_subagent_run():
    record = SubagentRunRecord(
        run_ref="subagent-1",
        task_id="task-1",
        task_origin=TaskOrigin.HUMAN,
        task_purpose=TaskPurpose.SCIENCE,
        stage_index=0,
        role="solver",
        instruction="Solve.",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="q"),
        memory_bundle=MemoryBundle(backend_id="memory-local"),
        skill_bundle=SkillBundle(backend_id="skill-local"),
        prompt_messages=[Message(role="user", content="Solve.")],
        llm_backend_id="fake-llm",
        tool_calls=[_tool_call("read_file")],
    )

    example = RewardExample.from_subagent_run(record)

    assert example.sample_id == "subagent-1"
    assert example.trajectory_ref == "subagent-1"
    assert example.task_id == "task-1"
    assert example.role == "solver"
    assert example.tool_calls == record.tool_calls
