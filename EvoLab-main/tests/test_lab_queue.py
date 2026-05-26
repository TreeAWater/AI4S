from pathlib import Path

import pytest

from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.lab.resolver import LabResolver
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry


def test_lab_layout_creates_core_dirs(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    layout.ensure()
    assert layout.tasks_queue_dir.exists()
    assert layout.evolve_queue_dir.exists()
    assert layout.trajectory_dir.exists()
    assert (layout.registries_dir / "lab_state").exists()


def test_lab_resolver_returns_standard_components(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    resolver = LabResolver(layout)

    resolver.ensure()

    task_queue = resolver.task_queue()
    evolve_queue = resolver.evolve_queue()
    task_registry = resolver.task_registry()
    trajectory_registry = resolver.trajectory_registry()
    backend_state_registry = resolver.backend_state_registry()
    lab_state_registry = resolver.lab_state_registry()

    assert isinstance(task_queue, FileWorkQueue)
    assert task_queue.root == layout.tasks_queue_dir
    assert isinstance(evolve_queue, FileWorkQueue)
    assert evolve_queue.root == layout.evolve_queue_dir
    assert isinstance(task_registry, FileTaskRegistry)
    assert task_registry.root == layout.registries_dir / "task"
    assert isinstance(trajectory_registry, FileTrajectoryRegistry)
    assert trajectory_registry.root == layout.registries_dir / "trajectory"
    assert isinstance(backend_state_registry, FileBackendStateRegistry)
    assert backend_state_registry.root == layout.registries_dir / "backend_state"
    assert isinstance(lab_state_registry, FileLabStateRegistry)
    assert lab_state_registry.root == layout.registries_dir / "lab_state"


def test_file_queue_enqueue_claim_done(tmp_path: Path):
    queue = FileWorkQueue(tmp_path / "queue")
    queue.ensure()
    queue.enqueue("job-1", {"job_id": "job-1", "value": 1})
    claimed = queue.claim("worker-1")
    assert claimed is not None
    assert claimed.payload["job_id"] == "job-1"
    queue.mark_done(claimed)
    assert (tmp_path / "queue" / "done").exists()


def test_file_queue_claim_uses_enqueue_job_id_when_payload_omits_it(tmp_path: Path):
    queue = FileWorkQueue(tmp_path / "queue")
    queue.ensure()
    payload = {"value": 1}
    queue.enqueue("job-1", payload)
    claimed = queue.claim("worker-1")
    assert claimed is not None
    assert claimed.job_id == "job-1"
    assert claimed.payload["job_id"] == "job-1"
    assert payload == {"value": 1}


@pytest.mark.parametrize(
    "job_id",
    ["", ".", "..", "../outside", "nested/job", r"nested\\job", "/tmp/job", "job..1"],
)
def test_file_queue_rejects_unsafe_job_ids(tmp_path: Path, job_id: str):
    queue = FileWorkQueue(tmp_path / "queue")

    with pytest.raises(ValueError, match="unsafe job_id"):
        queue.enqueue(job_id, {"value": 1})

    assert not (tmp_path / "outside.json").exists()
