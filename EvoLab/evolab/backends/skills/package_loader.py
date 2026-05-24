from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when PyYAML is installed.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on optional environment package.
    yaml = None

from evolab.backends.skills.package_schema import SkillPackage


class SkillPackageLoader:
    def __init__(self, *, repo_root: Path | str | None = None):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()

    def load(self, package_path: Path | str) -> SkillPackage:
        resolved_path = self._resolve_path(package_path)
        if not resolved_path.exists() or not resolved_path.is_dir():
            raise FileNotFoundError(f"skill package directory not found: {resolved_path}")

        metadata_path = self._metadata_path(resolved_path)
        raw_metadata = self._read_metadata(metadata_path)
        skill_markdown_path = resolved_path / "SKILL.md"
        skill_markdown = (
            skill_markdown_path.read_text(encoding="utf-8") if skill_markdown_path.exists() else None
        )
        payload = {
            **raw_metadata,
            "package_path": self._display_path(resolved_path),
            "skill_markdown": skill_markdown,
        }
        return SkillPackage.model_validate(payload)

    def _resolve_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.repo_root / candidate).resolve()

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _metadata_path(package_path: Path) -> Path:
        yaml_path = package_path / "metadata.yaml"
        json_path = package_path / "metadata.json"
        if yaml_path.exists():
            return yaml_path
        if json_path.exists():
            return json_path
        raise FileNotFoundError(f"skill package metadata not found in: {package_path}")

    @staticmethod
    def _read_metadata(metadata_path: Path) -> dict[str, Any]:
        text = metadata_path.read_text(encoding="utf-8")
        if metadata_path.suffix == ".json":
            payload = json.loads(text)
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                if yaml is None:
                    raise ModuleNotFoundError(
                        f"PyYAML is required to parse non-JSON YAML metadata: {metadata_path}"
                    )
                payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"skill package metadata must be an object: {metadata_path}")
        return payload
