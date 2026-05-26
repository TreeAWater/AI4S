from __future__ import annotations

from pathlib import Path

from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.snapshots import FileSnapshotRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry


class LabResolver:
    def __init__(self, layout: LabLayout | Path | str):
        self.layout = layout if isinstance(layout, LabLayout) else LabLayout(layout)

    def ensure(self) -> None:
        self.layout.ensure()

    def task_queue(self) -> FileWorkQueue:
        return FileWorkQueue(self.layout.tasks_queue_dir)

    def evolve_queue(self) -> FileWorkQueue:
        return FileWorkQueue(self.layout.evolve_queue_dir)

    def task_registry(self) -> FileTaskRegistry:
        return FileTaskRegistry(self.layout.registries_dir / "task")

    def trajectory_registry(self) -> FileTrajectoryRegistry:
        return FileTrajectoryRegistry(self.layout.registries_dir / "trajectory")

    def backend_state_registry(self) -> FileBackendStateRegistry:
        return FileBackendStateRegistry(self.layout.registries_dir / "backend_state")

    def lab_state_registry(self) -> FileLabStateRegistry:
        return FileLabStateRegistry(self.layout.registries_dir / "lab_state")

    def snapshot_registry(self) -> FileSnapshotRegistry:
        return FileSnapshotRegistry(self.layout.registries_dir / "snapshots")
