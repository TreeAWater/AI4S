from __future__ import annotations

from fnmatch import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Iterable

from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.paths import resolve_path_arguments
from evolab.tools.runtime import ToolRegistry


def register_file_tools(
    registry: ToolRegistry,
    *,
    base_dir: str | Path | None = None,
    excluded_roots: Iterable[str | Path] = (),
) -> None:
    excluded = tuple(Path(path).expanduser().resolve() for path in excluded_roots)
    _register_if_missing(
        registry,
        _list_files_spec(),
        lambda arguments: list_files(
            resolve_path_arguments(arguments, base_dir=base_dir, names=("root",)),
            excluded_roots=excluded,
        ),
    )
    _register_if_missing(
        registry,
        _read_text_spec(),
        lambda arguments: read_text(
            resolve_path_arguments(arguments, base_dir=base_dir, names=("path",)),
            excluded_roots=excluded,
        ),
    )
    _register_if_missing(
        registry,
        _inspect_file_metadata_spec(),
        lambda arguments: inspect_file_metadata(
            resolve_path_arguments(arguments, base_dir=base_dir, names=("path",)),
            excluded_roots=excluded,
        ),
    )


def list_files(arguments: dict[str, Any], *, excluded_roots: Iterable[Path] = ()) -> ToolResult:
    root = _path_argument(arguments, "root")
    _reject_excluded_path(root, excluded_roots)
    recursive = bool(arguments.get("recursive", True))
    raw_include_patterns = _string_list(arguments.get("include_patterns"))
    include_patterns = _normalize_glob_patterns(root, raw_include_patterns) or ["*"]
    exclude_patterns = _normalize_glob_patterns(root, _string_list(arguments.get("exclude_patterns")))
    max_files = _optional_int(arguments.get("max_files"))
    warnings: list[str] = []
    files = []
    search_roots = _list_files_search_roots(root, raw_include_patterns)
    seen_paths: set[Path] = set()
    for path in sorted(
        _iter_file_candidates(search_roots, recursive=recursive, warnings=warnings, excluded_roots=excluded_roots),
        key=lambda item: item.as_posix(),
    ):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not _matches_patterns(relative_path, path.name, include_patterns):
            continue
        if _matches_patterns(relative_path, path.name, exclude_patterns):
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            warnings.append(f"skipped unreadable file {path}: {exc}")
            continue
        files.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "suffix": path.suffix,
                "size": stat.st_size,
            }
        )
        if max_files is not None and len(files) >= max_files:
            break
    return _json_result(
        "list_files",
        {"root": str(root), "file_count": len(files), "files": files, "warnings": warnings},
        f"listed {len(files)} files under {root}",
    )


def read_text(arguments: dict[str, Any], *, excluded_roots: Iterable[Path] = ()) -> ToolResult:
    path = _path_argument(arguments, "path")
    _reject_excluded_path(path, excluded_roots)
    encoding = str(arguments.get("encoding") or "utf-8")
    max_chars = _optional_int(arguments.get("max_chars"))
    text = path.read_text(encoding=encoding)
    char_count = len(text)
    truncated = max_chars is not None and char_count > max_chars
    if truncated:
        text = text[:max_chars]
    return _json_result(
        "read_text",
        {
            "path": str(path),
            "encoding": encoding,
            "text": text,
            "char_count": char_count,
            "returned_char_count": len(text),
            "truncated": truncated,
        },
        f"read {len(text)} characters from {path}",
    )


def inspect_file_metadata(arguments: dict[str, Any], *, excluded_roots: Iterable[Path] = ()) -> ToolResult:
    path = Path(str(arguments.get("path", ""))).expanduser()
    _reject_excluded_path(path, excluded_roots)
    exists = path.exists()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "suffix": path.suffix,
    }
    if exists:
        stat = path.stat()
        payload.update({"size": stat.st_size, "modified_time": stat.st_mtime})
    return _json_result("inspect_file_metadata", payload, f"inspected metadata for {path}")


def _list_files_spec() -> ToolSpec:
    return ToolSpec(
        name="list_files",
        description="List local files under a root directory with deterministic ordering.",
        parameters_schema={
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Directory to list."},
                "recursive": {"type": "boolean", "description": "Whether to include nested files."},
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns to include.",
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns to exclude.",
                },
                "max_files": {"type": "integer", "minimum": 1, "description": "Maximum files to return."},
            },
            "required": ["root"],
        },
    )


def _read_text_spec() -> ToolSpec:
    return ToolSpec(
        name="read_text",
        description="Read a local text file with optional truncation.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the text file to read."},
                "encoding": {"type": "string", "description": "Text encoding. Defaults to utf-8."},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional maximum number of characters to return.",
                },
            },
            "required": ["path"],
        },
    )


def _inspect_file_metadata_spec() -> ToolSpec:
    return ToolSpec(
        name="inspect_file_metadata",
        description="Inspect local file or directory metadata.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path to inspect."},
            },
            "required": ["path"],
        },
    )


def _register_if_missing(registry: ToolRegistry, spec: ToolSpec, handler: Any) -> None:
    if registry.get_spec(spec.name) is None:
        registry.register(spec, handler)


def _path_argument(arguments: dict[str, Any], name: str) -> Path:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return Path(value).expanduser()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"expected integer or null, got {value!r}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _normalize_glob_patterns(root: Path, patterns: list[str]) -> list[str]:
    normalized: list[str] = []
    root = root.resolve()
    for pattern in patterns:
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            normalized.append(pattern)
            continue
        try:
            relative_pattern = pattern_path.relative_to(root).as_posix()
        except ValueError:
            normalized.append(pattern)
            continue
        if relative_pattern.startswith("**/"):
            normalized.append(relative_pattern.removeprefix("**/"))
        normalized.append(relative_pattern)
    return normalized


def _list_files_search_roots(root: Path, include_patterns: list[str]) -> list[Path]:
    if not include_patterns:
        return [root]
    root = root.resolve()
    absolute_roots: list[Path] = []
    for pattern in include_patterns:
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            return [root]
        static_prefix = _absolute_glob_static_prefix(pattern_path)
        try:
            static_prefix.relative_to(root)
        except ValueError:
            continue
        search_root = static_prefix if not static_prefix.is_file() else static_prefix.parent
        absolute_roots.append(search_root)
    if not absolute_roots:
        return [root]
    return _dedupe_nested_paths(absolute_roots)


def _absolute_glob_static_prefix(path: Path) -> Path:
    static_parts: list[str] = []
    for part in path.parts:
        if _has_glob_magic(part):
            break
        static_parts.append(part)
    if not static_parts:
        return Path("/")
    return Path(*static_parts)


def _has_glob_magic(value: str) -> bool:
    return any(char in value for char in "*?[")


def _dedupe_nested_paths(paths: list[Path]) -> list[Path]:
    resolved = sorted({path.resolve() for path in paths}, key=lambda item: len(item.parts))
    kept: list[Path] = []
    for path in resolved:
        if any(_path_is_relative_to(path, existing) for existing in kept):
            continue
        kept.append(path)
    return kept


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_excluded_path(path: Path, excluded_roots: Iterable[Path]) -> None:
    if _path_is_excluded(path, excluded_roots):
        raise ValueError(f"path is inside an excluded EvoLab internal directory: {path}")


def _path_is_excluded(path: Path, excluded_roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve()
    for root in excluded_roots:
        root = root.expanduser().resolve()
        if resolved == root or _path_is_relative_to(resolved, root):
            return True
    return False


def _iter_file_candidates(
    search_roots: list[Path],
    *,
    recursive: bool,
    warnings: list[str],
    excluded_roots: Iterable[Path],
) -> list[Path]:
    candidates: list[Path] = []
    for search_root in search_roots:
        if _path_is_excluded(search_root, excluded_roots):
            continue
        if search_root.is_file():
            candidates.append(search_root)
            continue
        if not search_root.exists():
            continue
        if recursive:
            candidates.extend(_walk_files(search_root, warnings=warnings, excluded_roots=excluded_roots))
        else:
            candidates.extend(_direct_child_files(search_root, warnings=warnings, excluded_roots=excluded_roots))
    return candidates


def _walk_files(root: Path, *, warnings: list[str], excluded_roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []

    def on_error(error: OSError) -> None:
        warnings.append(f"skipped unreadable path: {error}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        dirnames.sort()
        filenames.sort()
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _path_is_excluded(Path(dirpath) / dirname, excluded_roots)
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if not _path_is_excluded(path, excluded_roots):
                files.append(path)
    return files


def _direct_child_files(root: Path, *, warnings: list[str], excluded_roots: Iterable[Path]) -> list[Path]:
    try:
        return sorted(
            (item for item in root.iterdir() if item.is_file() and not _path_is_excluded(item, excluded_roots)),
            key=lambda item: item.as_posix(),
        )
    except OSError as exc:
        warnings.append(f"skipped unreadable path {root}: {exc}")
        return []


def _matches_patterns(relative_path: str, name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch(relative_path, pattern) or fnmatch(name, pattern):
            return True
        if pattern.startswith("**/") and (
            fnmatch(relative_path, pattern.removeprefix("**/")) or fnmatch(name, pattern.removeprefix("**/"))
        ):
            return True
    return False


def _json_result(call_id: str, payload: dict[str, Any], content: str) -> ToolResult:
    return ToolResult(call_id=call_id, status="ok", content=content, metadata=payload)
