from pathlib import Path

from evolab.contracts.common import ArtifactRef
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.snapshots import (
    EnvironmentSnapshot,
    RewardPolicyComponent,
    RewardPolicySnapshot,
    SkillSnapshot,
    SnapshotRef,
    ToolsetSnapshot,
)
from evolab.contracts.tools import ToolSpec
from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver
from evolab.registries.snapshots import FileSnapshotRegistry, SnapshotRegistry


def test_snapshot_registry_round_trips_typed_snapshots(tmp_path: Path):
    registry = FileSnapshotRegistry(tmp_path / "snapshots")
    toolset = ToolsetSnapshot(
        snapshot_ref="toolset-1",
        tool_specs=[ToolSpec(name="read_file", description="Read a lab file.")],
        implementation_refs=[
            ArtifactRef(uri="git://repo/tools/read_file.py", type="code"),
        ],
    )
    skill = SkillSnapshot(
        snapshot_ref="skill-1",
        skill_backend_id="skill-local",
        skill_state_ref="skill-state-1",
        graph_version_ref="graph-1",
        skills=[SkillItem(skill_id="compare", name="Compare", content="Compare evidence.")],
        required_tools=["read_file"],
    )
    reward_policy = RewardPolicySnapshot(
        snapshot_ref="reward-policy-1",
        components=[
            RewardPolicyComponent(
                calculator_id="num_toolcall",
                weight=0.1,
                config={"tool_name": "read_file"},
            )
        ],
    )
    environment = EnvironmentSnapshot(
        snapshot_ref="env-1",
        task_config_ref="configs/demo.yaml",
        toolset_snapshot_ref=toolset.snapshot_ref,
        skill_snapshot_ref=skill.snapshot_ref,
        reward_policy_snapshot_ref=reward_policy.snapshot_ref,
        llm_state_refs={"fake-llm": "llm-state-1"},
    )

    assert registry.save_snapshot(toolset) == "toolset-1"
    registry.save_snapshot(skill)
    registry.save_snapshot(reward_policy)
    registry.save_snapshot(environment)

    assert isinstance(registry, SnapshotRegistry)
    assert registry.get_snapshot("toolset-1") == toolset
    assert [snapshot.snapshot_ref for snapshot in registry.list_snapshots("skill")] == ["skill-1"]
    assert [snapshot.kind for snapshot in registry.list_snapshots()] == [
        "toolset",
        "skill",
        "reward_policy",
        "environment",
    ]


def test_snapshot_ref_contract_supports_snapshot_inputs():
    ref = SnapshotRef(snapshot_ref="toolset-1", kind="toolset", uri="lab/snapshots/toolset-1.json")

    loaded = SnapshotRef.model_validate_json(ref.model_dump_json())

    assert loaded.snapshot_ref == "toolset-1"
    assert loaded.kind == "toolset"


def test_lab_layout_and_resolver_include_snapshot_registry(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    resolver = LabResolver(layout)

    resolver.ensure()
    registry = resolver.snapshot_registry()

    assert layout.snapshots_dir.exists()
    assert (layout.registries_dir / "snapshots").exists()
    assert isinstance(registry, FileSnapshotRegistry)
    assert registry.root == layout.registries_dir / "snapshots"
