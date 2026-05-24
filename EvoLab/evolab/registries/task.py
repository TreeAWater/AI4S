from __future__ import annotations

from pathlib import Path

from evolab.contracts.task import TaskOrigin, TaskRequest


class FileTaskRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        path = Path(task_id)
        if (
            not task_id
            or task_id in {".", ".."}
            or path.is_absolute()
            or path.name != task_id
            or "/" in task_id
            or "\\" in task_id
            or ".." in task_id
        ):
            raise ValueError(f"unsafe task_id: {task_id!r}")
        return self.root / f"{task_id}.json"

    def save_task_request(self, request: TaskRequest) -> Path:
        path = self._task_path(request.task_id)
        path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, task_id: str) -> TaskRequest:
        return TaskRequest.model_validate_json(self._task_path(task_id).read_text(encoding="utf-8"))

    def query_by_origin(self, origin: TaskOrigin) -> list[TaskRequest]:
        results = []
        for path in sorted(self.root.glob("*.json")):
            request = TaskRequest.model_validate_json(path.read_text(encoding="utf-8"))
            if request.origin == origin:
                results.append(request)
        return results

    def query_by_human_anchor(self, task_ref: str) -> list[TaskRequest]:
        results = []
        for path in sorted(self.root.glob("*.json")):
            request = TaskRequest.model_validate_json(path.read_text(encoding="utf-8"))
            relation = request.proposed_task_relation
            if relation and task_ref in relation.human_anchor_task_refs:
                results.append(request)
        return results
