from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evolab import EvoLabSession, SessionConfig, TaskSpec


DEFAULT_LAB_DIR = Path("~/evolab-labs/biology-component-extraction")

SCIENTIFIC_IE_ALLOWED_TOOLS = [
    "list_files",
    "read_text",
    "inspect_file_metadata",
    "search_text",
    "extract_sections",
    "inspect_table",
    "read_table_slice",
    "inspect_excel_workbook",
    "read_excel_sheet",
    "detect_table_header",
    "normalize_table",
    "profile_table",
    "json_schema_validate",
    "build_document_inventory",
    "discover_candidate_source_files",
    "discover_candidate_tables",
    "extract_candidate_rows",
    "build_candidate_records",
    "validate_candidate_records",
    "serialize_final_records",
    "write_jsonl",
    "write_report",
]


def build_session_config(*, validate_lab: bool = True) -> SessionConfig:
    lab_dir = _path_env("EVOLAB_LAB_DIR", DEFAULT_LAB_DIR)
    state_dir = lab_dir / ".evolab"
    env_file = _path_env("EVOLAB_ENV_FILE", lab_dir / ".env")
    skill_repo_root = _path_env("EVOLAB_SKILL_REPO_ROOT", state_dir)
    skill_graph_path = _path_env(
        "EVOLAB_SKILL_GRAPH",
        skill_repo_root / "configs" / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json",
    )
    resource_dir = _path_env(
        "EVOLAB_BIOLOGY_RESOURCE_DIR",
        lab_dir / "resources" / "biology_component_extraction_v1",
    )
    source_root = _path_env("EVOLAB_SOURCE_ROOT", lab_dir / "papers")
    supplementary_root = _path_env("EVOLAB_SUPPLEMENTARY_ROOT", lab_dir / "supplementary")
    schema_path = _path_env("EVOLAB_BIOLOGY_SCHEMA_PATH", resource_dir / "biology_component_schema.json")
    ontology_path = _path_env(
        "EVOLAB_BIOLOGY_ONTOLOGY_PATH",
        resource_dir / "biological_component_ontology.yaml",
    )
    sequence_policy_path = _path_env(
        "EVOLAB_BIOLOGY_SEQUENCE_POLICY_PATH",
        resource_dir / "biological_sequence_policy.yaml",
    )
    evidence_policy_path = _path_env(
        "EVOLAB_BIOLOGY_EVIDENCE_POLICY_PATH",
        resource_dir / "biological_evidence_policy.yaml",
    )
    negative_patterns_path = _path_env(
        "EVOLAB_BIOLOGY_NEGATIVE_PATTERNS_PATH",
        resource_dir / "biological_negative_patterns.yaml",
    )

    paths = {
        "lab_dir": lab_dir,
        "env_file": env_file,
        "skill_repo_root": skill_repo_root,
        "skill_graph_path": skill_graph_path,
        "source_root": source_root,
        "supplementary_root": supplementary_root,
        "resource_dir": resource_dir,
        "schema_path": schema_path,
        "ontology_path": ontology_path,
        "sequence_policy_path": sequence_policy_path,
        "evidence_policy_path": evidence_policy_path,
        "negative_patterns_path": negative_patterns_path,
    }
    if validate_lab:
        validate_lab_layout(paths)

    return SessionConfig(
        env_file=env_file,
        lab_dir=lab_dir,
        task=TaskSpec(
            goal=(
                "Extract biological component information from the supplied source articles "
                "and supplementary artifacts. The target components include engineered "
                "biological parts, sequences, strains, plasmids, proteins, regulatory "
                "elements, and related constructs when the evidence supports extraction."
            ),
            resources=(
                f"Use source articles from {source_root}. Use supplementary tables, "
                f"workbooks, and text artifacts from {supplementary_root} when present. "
                f"Use the task schema at {schema_path}, ontology at {ontology_path}, "
                f"sequence policy at {sequence_policy_path}, evidence policy at "
                f"{evidence_policy_path}, and negative-pattern policy at "
                f"{negative_patterns_path}. Use only files in this Lab; do not read "
                "repository fixtures or development datasets."
            ),
            expected_outputs=(
                "Write biology_component_records.jsonl and biology_component_report.md "
                "at the Lab root. Put auditable intermediate manifests, candidate rows, "
                "candidate records, validation reports, and generated task-local tool "
                "artifacts under artifacts/."
            ),
            success_criteria=(
                "Every accepted JSONL record is grounded in a source article or "
                "supplementary artifact and includes source identifier, component name, "
                "component type, sequence when available, evidence location, extraction "
                "status, confidence, and uncertainty notes. The report must summarize "
                "coverage by source, validation checks, rejected candidates, unresolved "
                "ambiguities, and any generated tools or evolved roles used by the run."
            ),
            optional_context=(
                "This is a production-style scientific IE task, not a demo with canned "
                "responses. Use task-level memory through mem0 for durable extraction "
                "decisions and schema interpretations. Retrieve reusable scientific IE "
                "skills from the graph skill backend rooted in .evolab. The MetaAgent "
                "may evolve .evolab/AGENTS.md and generate task-local Python tools under "
                ".evolab when the current role or tool pool is not sufficient."
            ),
        ),
        llm={
            "default": _llm_backend(
                model_env="EVOLAB_MODEL",
                default_model="gpt-4.1",
                api="openai-responses",
                max_output_tokens=12000,
            ),
            "memory-extractor": _llm_backend(
                model_env="EVOLAB_MEMORY_MODEL",
                default_model="gpt-4.1-mini",
                api="openai-chat-completions",
                max_output_tokens=2000,
            ),
        },
        embeddings={
            "memory-embedding": _embedding_backend(
                model_env="EVOLAB_EMBEDDING_MODEL",
                default_model="text-embedding-3-large",
            )
        },
        memory={
            "mem0-task-memory": {
                "type": "method",
                "method": "mem0",
                "store_path": "memory/stores/mem0-task.sqlite",
                "audit_log_path": "memory/stores/mem0-task.audit.jsonl",
                "llm_backend": "memory-extractor",
                "embedding_backend": "memory-embedding",
                "top_k_existing": 12,
                "default_search_top_k": 8,
                "default_search_threshold": 0.12,
            }
        },
        skills={
            "scientific-ie-graph": {
                "type": "graph",
                "graph_path": str(skill_graph_path),
                "repo_root": str(skill_repo_root),
                "strict_packages": True,
            }
        },
        tools={"builtin": True, "allowed_tools": SCIENTIFIC_IE_ALLOWED_TOOLS},
        seed_roles=_seed_roles(),
        runtime={
            "task_id": "biology-component-extraction",
            "max_workflow_nodes": 48,
            "max_tool_steps": 120,
            "max_tool_steps_per_node": 24,
            "max_meta_dispatch_parse_retries": 3,
        },
        meta_agent={
            "system_prompt": (
                "You are EvoLab's MetaAgent for a production biological component "
                "extraction task. During preplanning, return JSON only. Inspect the "
                "current role pool and graph skill pool before dispatch. If the current "
                "role pool is not enough, update .evolab/AGENTS.md directly. If built-in "
                "tools are not specific enough for the Lab's files, generate task-local "
                "Python tools under .evolab and expose them to the workflow. Prefer "
                "task-level memory for durable decisions; do not create agent-specific "
                "memory assumptions. Final outputs must be written into the Lab root."
            )
        },
    )


def validate_lab_layout(paths: dict[str, Path]) -> None:
    problems: list[str] = []
    env_file = paths["env_file"]
    source_root = paths["source_root"]
    supplementary_root = paths["supplementary_root"]
    skill_repo_root = paths["skill_repo_root"]
    skill_graph_path = paths["skill_graph_path"]
    skill_group_path = skill_repo_root / "configs" / "skills" / "groups" / "scientific_ie_v1.yaml"
    skill_root = skill_repo_root / "skills" / "scientific_ie"

    api_key_env_name = os.environ.get("EVOLAB_API_KEY_ENV", "OPENAI_API_KEY")
    if not env_file.is_file() and not os.environ.get(api_key_env_name):
        problems.append(f"missing env file {env_file} or environment variable {api_key_env_name}")
    if not source_root.is_dir():
        problems.append(f"missing source article directory: {source_root}")
    elif not any(path.is_file() for path in source_root.rglob("*")):
        problems.append(f"source article directory has no files: {source_root}")
    if supplementary_root.exists() and not supplementary_root.is_dir():
        problems.append(f"supplementary path exists but is not a directory: {supplementary_root}")
    for key in (
        "schema_path",
        "ontology_path",
        "sequence_policy_path",
        "evidence_policy_path",
        "negative_patterns_path",
    ):
        if not paths[key].is_file():
            problems.append(f"missing task resource {key}: {paths[key]}")
    if not skill_graph_path.is_file():
        problems.append(f"missing graph skill file: {skill_graph_path}")
    if not skill_group_path.is_file():
        problems.append(f"missing graph skill group config: {skill_group_path}")
    if not skill_root.is_dir():
        problems.append(f"missing scientific IE skill package root: {skill_root}")
    elif not any(skill_root.glob("*/metadata.yaml")):
        problems.append(f"scientific IE skill package root has no metadata packages: {skill_root}")

    if problems:
        expected = _expected_layout(paths)
        raise RuntimeError(
            "Biology component extraction Lab is not ready:\n- "
            + "\n- ".join(problems)
            + "\n\nExpected Lab layout:\n"
            + expected
        )


def _expected_layout(paths: dict[str, Path]) -> str:
    lab_dir = paths["lab_dir"]
    return "\n".join(
        [
            f"{lab_dir}/",
            "  .env                            # or export OPENAI_API_KEY before running",
            "  papers/                         # user-provided source papers",
            "  supplementary/                  # optional user-provided supplementary files",
            "  resources/biology_component_extraction_v1/",
            "    biology_component_schema.json",
            "    biological_component_ontology.yaml",
            "    biological_sequence_policy.yaml",
            "    biological_evidence_policy.yaml",
            "    biological_negative_patterns.yaml",
            "  .evolab/",
            "    configs/skills/graphs/scientific_ie_seed_graph_v1.json",
            "    configs/skills/groups/scientific_ie_v1.yaml",
            "    skills/scientific_ie/*/{metadata.yaml,SKILL.md}",
        ]
    )


def _seed_roles() -> dict[str, dict[str, Any]]:
    memory_policy = {
        "scope": "task",
        "read": True,
        "write": True,
        "notes": "Use shared task-level mem0 memory; do not assume stable agent-level memory.",
    }
    return {
        "ScientificIntakeAgent": {
            "system_prompt": (
                "Map the Lab's article and supplementary file inventory. Identify main "
                "texts, table-like artifacts, unreadable files, and extraction priorities."
            ),
            "llm_backend": {"backend_id": "default"},
            "allowed_tools": [
                "list_files",
                "inspect_file_metadata",
                "read_text",
                "extract_sections",
                "build_document_inventory",
                "discover_candidate_source_files",
                "discover_candidate_tables",
            ],
            "required_skills": [
                "skill.scientific_document_intake.v1",
                "skill.supplementary_artifact_discovery.v1",
                "skill.multi_format_artifact_reading.v1",
            ],
            "memory_policy": memory_policy,
        },
        "EvidenceExtractionAgent": {
            "system_prompt": (
                "Extract evidence-backed biological component candidates from article "
                "text, supplementary tables, and workbooks. Preserve source provenance "
                "and uncertainty instead of guessing missing values."
            ),
            "llm_backend": {"backend_id": "default"},
            "allowed_tools": [
                "read_text",
                "search_text",
                "extract_sections",
                "inspect_table",
                "read_table_slice",
                "inspect_excel_workbook",
                "read_excel_sheet",
                "extract_candidate_rows",
                "build_candidate_records",
            ],
            "required_skills": [
                "skill.task_relevant_section_localization.v1",
                "skill.evidence_aware_claim_classification.v1",
                "skill.evidence_source_attribution.v1",
            ],
            "memory_policy": memory_policy,
        },
        "SchemaMappingAgent": {
            "system_prompt": (
                "Map extracted candidates into the biology component schema. Normalize "
                "component types, sequence fields, evidence fields, and status values "
                "using the Lab's task resources."
            ),
            "llm_backend": {"backend_id": "default"},
            "allowed_tools": [
                "read_text",
                "json_schema_validate",
                "normalize_table",
                "profile_table",
                "build_candidate_records",
            ],
            "required_skills": [
                "skill.extraction_schema_interpretation.v1",
                "skill.schema_guided_field_mapping.v1",
                "skill.structured_record_construction.v1",
                "skill.domain_ontology_alignment.v1",
            ],
            "memory_policy": memory_policy,
        },
        "ValidationAgent": {
            "system_prompt": (
                "Validate candidate records against schema, ontology, sequence policy, "
                "negative patterns, and source evidence. Separate accepted, rejected, "
                "and review-needed records with explicit reasons."
            ),
            "llm_backend": {"backend_id": "default"},
            "allowed_tools": [
                "read_text",
                "json_schema_validate",
                "validate_candidate_records",
                "search_text",
                "inspect_table",
                "read_table_slice",
            ],
            "required_skills": [
                "skill.domain_entity_validation.v1",
                "skill.domain_negative_pattern_filtering.v1",
                "skill.extraction_result_validation.v1",
                "skill.record_deduplication_and_conflict_resolution.v1",
            ],
            "memory_policy": memory_policy,
        },
        "FinalizationAgent": {
            "system_prompt": (
                "Write final JSONL records and a concise audit report at the Lab root. "
                "Summarize coverage, validation outcomes, unresolved ambiguities, and "
                "role/tool evolution performed during the run."
            ),
            "llm_backend": {"backend_id": "default"},
            "allowed_tools": [
                "serialize_final_records",
                "write_jsonl",
                "write_report",
                "list_files",
                "read_text",
            ],
            "required_skills": [
                "skill.final_artifact_writing.v1",
                "skill.record_deduplication_and_conflict_resolution.v1",
            ],
            "memory_policy": memory_policy,
        },
    }


def _llm_backend(
    *,
    model_env: str,
    default_model: str,
    api: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "api",
        "api": api,
        "model": os.environ.get(model_env, default_model),
        "api_key_env": os.environ.get("EVOLAB_API_KEY_ENV", "OPENAI_API_KEY"),
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": float(os.environ.get("EVOLAB_LLM_TIMEOUT_SECONDS", "120")),
        "max_retries": int(os.environ.get("EVOLAB_LLM_MAX_RETRIES", "3")),
    }
    base_url = _optional_env("EVOLAB_LLM_BASE_URL") or _optional_env("OPENAI_BASE_URL")
    if base_url:
        payload["base_url"] = base_url
    return payload


def _embedding_backend(*, model_env: str, default_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "api",
        "api": "openai-embeddings",
        "model": os.environ.get(model_env, default_model),
        "api_key_env": os.environ.get("EVOLAB_API_KEY_ENV", "OPENAI_API_KEY"),
        "timeout_seconds": float(os.environ.get("EVOLAB_EMBEDDING_TIMEOUT_SECONDS", "120")),
    }
    base_url = _optional_env("EVOLAB_EMBEDDING_BASE_URL") or _optional_env("OPENAI_BASE_URL")
    if base_url:
        payload["base_url"] = base_url
    return payload


def _path_env(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def main() -> None:
    session = EvoLabSession(build_session_config())
    session.run()


if __name__ == "__main__":
    main()
