from __future__ import annotations

from pathlib import Path


class LabLayout:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def state_root(self) -> Path:
        return self.root / ".evolab"

    @property
    def agents_path(self) -> Path:
        return self.state_root / "AGENTS.md"

    @property
    def configs_dir(self) -> Path:
        return self.state_root / "configs"

    @property
    def tools_dir(self) -> Path:
        return self.state_root / "tools"

    @property
    def skills_dir(self) -> Path:
        return self.state_root / "skills"

    @property
    def memory_dir(self) -> Path:
        return self.state_root / "memory"

    @property
    def generated_tools_dir(self) -> Path:
        return self.state_root / "generated_tools"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def user_artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def tasks_queue_dir(self) -> Path:
        return self.state_root / "queues" / "tasks"

    @property
    def evolve_queue_dir(self) -> Path:
        return self.state_root / "queues" / "evolve"

    @property
    def trajectory_dir(self) -> Path:
        return self.state_root / "trajectories"

    @property
    def registries_dir(self) -> Path:
        return self.state_root / "registries"

    @property
    def snapshots_dir(self) -> Path:
        return self.state_root / "snapshots"

    def task_dir(self, task_id: str) -> Path:
        return self.state_root / "tasks" / task_id

    def evolution_run_dir(self, run_ref: str) -> Path:
        return self.state_root / "evolution" / "llm" / run_ref

    def ensure(self) -> None:
        dirs = [
            self.root,
            self.state_root,
            self.configs_dir,
            self.tools_dir / "builtin",
            self.tools_dir / "generated",
            self.tools_dir / "task_local",
            self.skills_dir / "pool",
            self.skills_dir / "graph",
            self.skills_dir / "evolution",
            self.memory_dir / "task",
            self.memory_dir / "meta",
            self.memory_dir / "stores",
            self.generated_tools_dir,
            self.state_root / "logs",
            self.output_dir,
            self.user_artifacts_dir,
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
