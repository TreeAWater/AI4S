from __future__ import annotations

from pathlib import Path


class LabLayout:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def tasks_queue_dir(self) -> Path:
        return self.root / "queues" / "tasks"

    @property
    def evolve_queue_dir(self) -> Path:
        return self.root / "queues" / "evolve"

    @property
    def trajectory_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def registries_dir(self) -> Path:
        return self.root / "registries"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    def task_dir(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id

    def evolution_run_dir(self, run_ref: str) -> Path:
        return self.root / "evolution" / "llm" / run_ref

    def ensure(self) -> None:
        dirs = [
            self.root / "configs",
            self.tasks_queue_dir,
            self.evolve_queue_dir,
            self.trajectory_dir / "meta_agent",
            self.trajectory_dir / "subagent",
            self.trajectory_dir / "llm_calls",
            self.trajectory_dir / "evolution",
            self.snapshots_dir,
            self.registries_dir / "trajectory",
            self.registries_dir / "backend_state",
            self.registries_dir / "lab_state",
            self.registries_dir / "snapshots",
            self.registries_dir / "task",
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
