from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_path_arguments(
    arguments: dict[str, Any],
    *,
    base_dir: str | Path | None,
    names: tuple[str, ...],
) -> dict[str, Any]:
    if base_dir is None:
        return arguments
    resolved = dict(arguments)
    root = Path(base_dir).expanduser()
    for name in names:
        value = resolved.get(name)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved[name] = str(path)
    return resolved
