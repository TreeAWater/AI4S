from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
import json
from pathlib import Path

try:  # pragma: no cover - exercised only when PyYAML is installed.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on optional environment package.
    yaml = None

from evolab.backends.skills.package_loader import SkillPackageLoader
from evolab.backends.skills.package_schema import SkillGroupConfig, SkillPackage


@dataclass
class SkillRegistry:
    repo_root: Path | str | None = None
    loader: SkillPackageLoader | None = None
    packages_by_id: dict[str, SkillPackage] = field(default_factory=dict)
    group_skill_ids: dict[str, list[str]] = field(default_factory=dict)
    category_skill_ids: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve() if self.repo_root is not None else Path.cwd().resolve()
        if self.loader is None:
            self.loader = SkillPackageLoader(repo_root=self.repo_root)

    def scan(
        self,
        skill_roots: list[str | Path],
        *,
        group_name: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        include_patterns = include_patterns or ["*/metadata.yaml", "*/metadata.json"]
        exclude_patterns = exclude_patterns or []
        for root in skill_roots:
            root_path = self._resolve_path(root)
            if not root_path.exists():
                self.warnings.append(f"skill root not found: {self._display_path(root_path)}")
                continue
            for metadata_path in sorted(
                [
                    *root_path.glob("**/metadata.yaml"),
                    *root_path.glob("**/metadata.json"),
                ],
                key=lambda path: path.as_posix(),
            ):
                rel_path = self._display_path(metadata_path)
                if not self._is_included(rel_path, include_patterns, exclude_patterns):
                    continue
                self.register_package(metadata_path.parent, group_name=group_name)

    def load_group(self, group_config_path: str | Path) -> SkillGroupConfig:
        config_path = self._resolve_path(group_config_path)
        text = config_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if yaml is None:
                raise ModuleNotFoundError(f"PyYAML is required to parse non-JSON YAML config: {config_path}")
            payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"skill group config must be an object: {config_path}")
        config = SkillGroupConfig.model_validate(payload)
        self.scan(
            config.skill_roots,
            group_name=config.group_name,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
        )
        self.group_skill_ids.setdefault(config.group_name, [])
        return config

    def register_package(self, package_path: str | Path, *, group_name: str | None = None) -> SkillPackage | None:
        assert self.loader is not None
        try:
            package = self.loader.load(package_path)
        except Exception as exc:
            self.warnings.append(f"failed to load skill package {package_path}: {exc}")
            return None

        current = self.packages_by_id.get(package.skill_id)
        if current is not None:
            chosen = min([current, package], key=lambda item: item.package_path)
            skipped = package if chosen is current else current
            self.packages_by_id[package.skill_id] = chosen
            self.warnings.append(
                f"duplicate skill_id {package.skill_id}; using {chosen.package_path}, skipped {skipped.package_path}"
            )
        else:
            self.packages_by_id[package.skill_id] = package

        if group_name is not None:
            self._append_unique(self.group_skill_ids.setdefault(group_name, []), package.skill_id)
        if package.target_category:
            self._append_unique(self.category_skill_ids.setdefault(package.target_category, []), package.skill_id)
        return self.packages_by_id[package.skill_id]

    def get(self, skill_id: str) -> SkillPackage | None:
        return self.packages_by_id.get(skill_id)

    def load_package_ref(self, package_ref: str, *, group_name: str | None = None) -> SkillPackage | None:
        return self.register_package(package_ref, group_name=group_name)

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(self.repo_root) / candidate).resolve()

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(Path(self.repo_root)).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)
            values.sort()

    @staticmethod
    def _is_included(path: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
        if any(fnmatch(path, pattern) for pattern in exclude_patterns):
            return False
        if not include_patterns:
            return True
        return any(fnmatch(path, pattern) for pattern in include_patterns)
