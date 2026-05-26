from __future__ import annotations

from pathlib import Path

from evolab.tools.files import register_file_tools
from evolab.tools.human import HumanToolAdapter, register_human_tools
from evolab.tools.output import register_output_tools
from evolab.tools.runtime import ToolRegistry
from evolab.tools.schema import register_schema_tools
from evolab.tools.scientific_artifacts import register_scientific_artifact_tools
from evolab.tools.tables import register_table_tools
from evolab.tools.text import register_text_tools


def register_scientific_ie_tools(
    registry: ToolRegistry,
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    allow_shell: bool = False,
    include_human_tools: bool = True,
    human_adapter: HumanToolAdapter | None = None,
) -> None:
    if allow_shell:
        raise ValueError("scientific IE v1 tools do not support shell execution")
    register_file_tools(registry, base_dir=base_dir)
    register_text_tools(registry, base_dir=base_dir)
    register_table_tools(registry, base_dir=base_dir)
    register_schema_tools(registry, base_dir=base_dir)
    register_scientific_artifact_tools(registry, artifact_root=artifact_root, base_dir=base_dir)
    register_output_tools(registry, artifact_root=artifact_root)
    if include_human_tools:
        register_human_tools(registry, adapter=human_adapter)
