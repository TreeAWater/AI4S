from pathlib import Path

from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver


def test_lab_layout_places_internal_state_under_dot_evolab(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")

    assert layout.root == tmp_path / "lab"
    assert layout.state_root == tmp_path / "lab" / ".evolab"
    assert layout.tasks_queue_dir == layout.state_root / "queues" / "tasks"
    assert layout.evolve_queue_dir == layout.state_root / "queues" / "evolve"
    assert layout.registries_dir == layout.state_root / "registries"
    assert layout.trajectory_dir == layout.state_root / "trajectories"
    assert layout.snapshots_dir == layout.state_root / "snapshots"
    assert layout.agents_path == layout.state_root / "AGENTS.md"
    assert layout.generated_tools_dir == layout.state_root / "generated_tools"
    assert layout.task_dir("task-1") == layout.state_root / "tasks" / "task-1"


def test_lab_resolver_ensure_does_not_create_internal_state_at_lab_root(tmp_path: Path):
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "input.txt").write_text("user data", encoding="utf-8")

    LabResolver(LabLayout(lab)).ensure()

    assert (lab / "input.txt").read_text(encoding="utf-8") == "user data"
    assert (lab / ".evolab" / "queues" / "tasks").is_dir()
    assert (lab / ".evolab" / "registries" / "task").is_dir()
    assert not (lab / "queues").exists()
    assert not (lab / "registries").exists()
    assert not (lab / "trajectories").exists()
    assert not (lab / "snapshots").exists()
