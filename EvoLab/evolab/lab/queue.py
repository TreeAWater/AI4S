from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    path: Path
    payload: dict[str, Any]


class FileWorkQueue:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def ensure(self) -> None:
        for name in ["queued", "claimed", "done", "failed", "skipped", "interrupted"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def enqueue(self, job_id: str, payload: dict[str, Any]) -> Path:
        self.ensure()
        safe_job_id = _validate_job_id(job_id)
        path = self.root / "queued" / f"{safe_job_id}.json"
        tmp = path.with_suffix(".json.tmp")
        queued_payload = dict(payload)
        queued_payload["job_id"] = safe_job_id
        tmp.write_text(json.dumps(queued_payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def claim(self, worker_id: str) -> ClaimedJob | None:
        self.ensure()
        for path in sorted((self.root / "queued").glob("*.json")):
            claimed_path = self.root / "claimed" / path.name
            try:
                os.replace(path, claimed_path)
            except FileNotFoundError:
                continue
            payload = json.loads(claimed_path.read_text(encoding="utf-8"))
            payload["claimed_by"] = worker_id
            payload["claimed_at"] = datetime.now(timezone.utc).isoformat()
            claimed_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return ClaimedJob(job_id=payload["job_id"], path=claimed_path, payload=payload)
        return None

    def mark_done(self, job: ClaimedJob) -> None:
        os.replace(job.path, self.root / "done" / job.path.name)

    def mark_failed(self, job: ClaimedJob, error: str) -> None:
        payload = dict(job.payload)
        payload["error"] = error
        job.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(job.path, self.root / "failed" / job.path.name)

    def mark_interrupted(self, job: ClaimedJob, reason: str) -> None:
        payload = dict(job.payload)
        payload["interrupt_reason"] = reason
        job.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(job.path, self.root / "interrupted" / job.path.name)

    def mark_skipped(self, job: ClaimedJob, reason: str) -> None:
        payload = dict(job.payload)
        payload["skip_reason"] = reason
        job.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(job.path, self.root / "skipped" / job.path.name)


def _validate_job_id(job_id: str) -> str:
    path = Path(job_id)
    if (
        not job_id
        or job_id in {".", ".."}
        or path.is_absolute()
        or path.name != job_id
        or "/" in job_id
        or "\\" in job_id
        or ".." in job_id
    ):
        raise ValueError(f"unsafe job_id: {job_id!r}")
    return job_id
