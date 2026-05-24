from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evolab.backends.skills.candidates import CandidateSkill
from evolab.backends.skills.graph_schema import SkillGraph
from evolab.backends.skills.package_schema import SkillGraphSkillNode, SkillGroupConfig
from evolab.backends.skills.registry import SkillRegistry


_EMPTY_GRAPH = {
    "schema_version": "v1",
    "version": "v1",
    "skills": [],
    "categories": [],
    "edges": [],
    "metadata": {},
}


@dataclass(frozen=True)
class LoadedSkillGraph:
    graph: SkillGraph
    raw_graph: dict[str, Any]
    skill_nodes: list[SkillGraphSkillNode] = field(default_factory=list)
    skipped_skills: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    registry: SkillRegistry | None = None
    group_configs: list[SkillGroupConfig] = field(default_factory=list)


class GraphSkillStore:
    def __init__(
        self,
        graph_path: Path | str,
        *,
        repo_root: Path | str | None = None,
        registry: SkillRegistry | None = None,
        strict_packages: bool = False,
    ):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.exists():
            self.graph_path.write_text(json.dumps(_EMPTY_GRAPH), encoding="utf-8")
        self.repo_root = self._infer_repo_root(repo_root)
        self.strict_packages = strict_packages
        self.registry = registry or SkillRegistry(repo_root=self.repo_root)

    def load_raw_graph(self) -> dict[str, Any]:
        with self.graph_path.open(encoding="utf-8") as graph_file:
            graph = json.load(graph_file)
        if not isinstance(graph, dict):
            raise ValueError("skill graph must be a JSON object")
        for field_name in ("skills", "categories", "edges"):
            if field_name in graph and not isinstance(graph[field_name], list):
                raise ValueError(f"skill graph field '{field_name}' must be a list")
        return graph

    def load_graph(self) -> LoadedSkillGraph:
        raw_graph = self.load_raw_graph()
        registry = SkillRegistry(repo_root=self.repo_root)
        group_configs = self._load_group_configs(raw_graph, registry)
        valid_skills: list[CandidateSkill] = []
        skill_nodes: list[SkillGraphSkillNode] = []
        skipped_skills: list[dict[str, Any]] = []
        warnings: list[str] = [*registry.warnings]

        for index, raw_skill in enumerate(raw_graph.get("skills", [])):
            if not isinstance(raw_skill, dict):
                skipped_skills.append({"index": index, "reason": "skill entry must be an object"})
                continue

            candidate = self._validate_embedded_candidate(raw_skill)
            if candidate is not None:
                valid_skills.append(candidate)
                continue

            try:
                node = SkillGraphSkillNode.model_validate(raw_skill)
            except ValidationError as exc:
                skipped_skills.append({"index": index, "reason": str(exc)})
                continue

            skill_nodes.append(node)
            if node.status == "dormant":
                continue
            package = registry.get(node.id)
            if package is None and node.package_ref:
                package = registry.load_package_ref(node.package_ref, group_name=node.group)
            if package is None:
                message = f"missing skill package for {node.id}"
                if node.package_ref:
                    message = f"{message}: {node.package_ref}"
                if self.strict_packages:
                    raise ValueError(message)
                warnings.append(message)
                continue

            candidate_skill = package.to_candidate_skill()
            candidate_skill.metadata.update(node.metadata)
            candidate_skill.metadata.update(
                {
                    "package_ref": node.package_ref,
                    "skill_node_status": node.status,
                    "skill_node_tags": node.tags,
                    "skill_group": node.group,
                }
            )
            valid_skills.append(candidate_skill)

        graph_payload = {
            "schema_version": raw_graph.get("schema_version", "v1"),
            "version": raw_graph.get("version", "v1"),
            "skills": [skill.model_dump(mode="json") for skill in valid_skills],
            "categories": raw_graph.get("categories", []),
            "edges": raw_graph.get("edges", []),
            "metadata": raw_graph.get("metadata", {}),
        }
        try:
            graph = SkillGraph.model_validate(graph_payload)
        except ValidationError as exc:
            raise ValueError(f"invalid skill graph: {exc}") from exc

        return LoadedSkillGraph(
            graph=graph,
            raw_graph=raw_graph,
            skill_nodes=skill_nodes,
            skipped_skills=skipped_skills,
            warnings=warnings,
            registry=registry,
            group_configs=group_configs,
        )

    def _load_group_configs(self, raw_graph: dict[str, Any], registry: SkillRegistry) -> list[SkillGroupConfig]:
        group_names = set()
        metadata = raw_graph.get("metadata", {})
        if isinstance(metadata, dict):
            groups = metadata.get("skill_groups", [])
            if isinstance(groups, str):
                group_names.add(groups)
            elif isinstance(groups, list):
                group_names.update(group for group in groups if isinstance(group, str))
        for raw_skill in raw_graph.get("skills", []):
            if isinstance(raw_skill, dict) and isinstance(raw_skill.get("group"), str):
                group_names.add(raw_skill["group"])

        loaded: list[SkillGroupConfig] = []
        for group_name in sorted(group_names):
            config_path = self.repo_root / "configs" / "skills" / "groups" / f"{group_name}.yaml"
            if not config_path.exists():
                continue
            loaded.append(registry.load_group(config_path))
        return loaded

    @staticmethod
    def _validate_embedded_candidate(raw_skill: dict[str, Any]) -> CandidateSkill | None:
        try:
            return CandidateSkill.model_validate(raw_skill)
        except ValidationError:
            return None

    def _infer_repo_root(self, repo_root: Path | str | None) -> Path:
        if repo_root is not None:
            return Path(repo_root).resolve()
        for candidate in [Path.cwd(), *Path.cwd().parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate.resolve()
        for candidate in [self.graph_path.resolve().parent, *self.graph_path.resolve().parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate.resolve()
        return Path.cwd().resolve()
