from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import TypeAdapter

from evolab.contracts.snapshots import SnapshotRecord

_SNAPSHOT_ADAPTER = TypeAdapter(SnapshotRecord)


class SnapshotRegistry(ABC):
    @abstractmethod
    def save_snapshot(self, snapshot: SnapshotRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_snapshot(self, snapshot_ref: str) -> SnapshotRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_snapshots(self, kind: str | None = None) -> list[SnapshotRecord]:
        raise NotImplementedError


class FileSnapshotRegistry(SnapshotRegistry):
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.root / "snapshots.jsonl"

    def save_snapshot(self, snapshot: SnapshotRecord) -> str:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_snapshot_json(snapshot) + "\n")
        return snapshot.snapshot_ref

    def get_snapshot(self, snapshot_ref: str) -> SnapshotRecord | None:
        for snapshot in self.list_snapshots():
            if snapshot.snapshot_ref == snapshot_ref:
                return snapshot
        return None

    def list_snapshots(self, kind: str | None = None) -> list[SnapshotRecord]:
        if not self.path.exists():
            return []
        snapshots = [
            _SNAPSHOT_ADAPTER.validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if kind is None:
            return snapshots
        return [snapshot for snapshot in snapshots if snapshot.kind == kind]


def _snapshot_json(snapshot: SnapshotRecord) -> str:
    dump = getattr(snapshot, "model_dump_json", None)
    if callable(dump):
        return dump()
    raise TypeError(f"unsupported snapshot record: {type(snapshot)!r}")
