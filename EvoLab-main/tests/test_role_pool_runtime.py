import json
from pathlib import Path

import evolab.runtime.role_pool as role_pool_runtime
from evolab.config.agents import render_agents_markdown
from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.runtime.role_pool import (
    ROLE_POOL_UPDATE_KEYS,
    RolePoolUpdateResult,
    apply_role_pool_update,
    role_pool_update_payload,
)


def _role(name: str, prompt: str = "Base prompt.") -> RoleSpec:
    return RoleSpec(
        name=name,
        system_prompt=prompt,
        llm_backend=BackendBinding(backend_id="planner-llm"),
        allowed_tools=["read_text", "write_report"],
    )


def test_role_pool_update_payload_prefers_canonical_key():
    metadata = {
        "agent_config_update": {"reason": "legacy"},
        "role_pool_update": {"reason": "canonical"},
    }

    assert role_pool_update_payload(metadata) == {"reason": "canonical"}


def test_role_pool_update_keys_are_exact_legacy_order():
    assert ROLE_POOL_UPDATE_KEYS == (
        "role_pool_update",
        "agent_config_update",
        "agents_update",
        "subagent_config_update",
    )


def test_role_pool_update_payload_skips_non_mapping_canonical_alias():
    metadata = {
        "role_pool_update": None,
        "agent_config_update": ["not a mapping"],
        "agents_update": {"reason": "legacy mapping"},
    }

    assert role_pool_update_payload(metadata) == {"reason": "legacy mapping"}


def test_apply_role_pool_update_adds_and_edits_roles(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(render_agents_markdown({"SurveyAgent": _role("SurveyAgent")}), encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "Need table triage.",
            "roles": {
                "SurveyAgent": {"system_prompt_append": "Also report missing files."},
                "TableEvidenceTriageAgent": {
                    "system_prompt": "Inspect tables before extraction.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "write_report"],
                    "required_skills": ["scientific_table_structure_understanding"],
                },
            },
        },
        task_id="task-1",
        run_ref="meta-1",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "updated"
    assert result.added_roles == ["TableEvidenceTriageAgent"]
    assert result.modified_roles == ["SurveyAgent"]
    assert result.removed_roles == []
    text = agents_path.read_text(encoding="utf-8")
    assert "TableEvidenceTriageAgent" in text
    assert "Also report missing files." in text
    history = (tmp_path / "agents.md.updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1
    history_record = json.loads(history[0])
    assert history_record["task_id"] == "task-1"
    assert history_record["run_ref"] == "meta-1"
    assert isinstance(history_record["update_hash"], str)
    assert len(history_record["update_hash"]) == 64
    assert history_record["update"]["reason"] == "Need table triage."
    assert history_record["result"]["status"] == "updated"
    assert history_record["result"]["added_roles"] == ["TableEvidenceTriageAgent"]


def test_apply_role_pool_update_removes_role(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(
        render_agents_markdown(
            {
                "SurveyAgent": _role("SurveyAgent"),
                "OldAgent": _role("OldAgent"),
            }
        ),
        encoding="utf-8",
    )

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={"reason": "Old role is obsolete.", "remove_roles": ["OldAgent"]},
        task_id="task-1",
        run_ref="meta-2",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "updated"
    assert result.removed_roles == ["OldAgent"]
    assert "OldAgent" not in agents_path.read_text(encoding="utf-8")


def test_apply_role_pool_update_rejects_delete_all_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={"reason": "bad update", "remove_roles": ["SurveyAgent"]},
        task_id="task-1",
        run_ref="meta-3",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert "at least one active role" in result.errors[0]
    assert agents_path.read_text(encoding="utf-8") == original
    history = (tmp_path / "agents.md.updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1
    assert json.loads(history[0])["result"]["status"] == "rejected"


def test_apply_role_pool_update_rejects_unknown_backend_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "bad backend",
            "roles": {
                "NewAgent": {
                    "system_prompt": "Do new work.",
                    "llm_backend": {"backend_id": "unknown-llm"},
                    "allowed_tools": ["read_text"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-4",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("unknown llm_backend.backend_id" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_unknown_tools_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "bad tool",
            "roles": {
                "NewAgent": {
                    "system_prompt": "Do new work.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "shell_exec"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-5",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("unknown allowed_tools" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_task_local_generated_tool_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "bad generated tool",
            "roles": {
                "GeneratedToolAgent": {
                    "system_prompt": "Use a task-local generated helper.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "gt_task_extract_rows"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-generated-tool",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("unknown allowed_tools" in error and "gt_task_extract_rows" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_tools_when_allowed_tool_names_is_empty(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "no tools are allowed",
            "roles": {
                "NewAgent": {
                    "system_prompt": "Do new work.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-empty-tools",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names=[],
    )

    assert result.status == "rejected"
    assert any("unknown allowed_tools" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_private_reasoning_metadata_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "bad metadata",
            "roles": {"SurveyAgent": {"metadata": {"chain_of_thought": "private"}}},
        },
        task_id="task-1",
        run_ref="meta-6",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("private reasoning field" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_result_to_json_includes_public_result_fields():
    result = RolePoolUpdateResult(
        status="rejected",
        before_revision="before",
        after_revision="after",
        active_roles=["SurveyAgent"],
        added_roles=["NewAgent"],
        modified_roles=["SurveyAgent"],
        removed_roles=["OldAgent"],
        warnings=["warning"],
        errors=["error"],
        history_ref="agents.md.updates.jsonl:1",
        reason="because",
    )

    payload = result.to_json()

    assert payload["status"] == "rejected"
    assert payload["revisions"] == {"before": "before", "after": "after"}
    assert payload["before_revision"] == "before"
    assert payload["after_revision"] == "after"
    assert payload["active_roles"] == ["SurveyAgent"]
    assert payload["added_roles"] == ["NewAgent"]
    assert payload["modified_roles"] == ["SurveyAgent"]
    assert payload["removed_roles"] == ["OldAgent"]
    assert payload["warnings"] == ["warning"]
    assert payload["errors"] == ["error"]
    assert payload["history_ref"] == "agents.md.updates.jsonl:1"
    assert payload["reason"] == "because"


def test_success_update_rejects_without_overwrite_when_history_append_fails(tmp_path: Path, monkeypatch):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    def fail_history_append(**kwargs):
        raise OSError("history path unavailable")

    monkeypatch.setattr(role_pool_runtime, "_append_history", fail_history_append)

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "Need table triage.",
            "roles": {
                "TableEvidenceTriageAgent": {
                    "system_prompt": "Inspect tables before extraction.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "write_report"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-7",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("history" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_when_history_path_aliases_agents_path(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        history_path=agents_path,
        payload={
            "reason": "Need table triage.",
            "roles": {
                "TableEvidenceTriageAgent": {
                    "system_prompt": "Inspect tables before extraction.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "write_report"],
                }
            },
        },
        task_id="task-1",
        run_ref="meta-history-alias",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("history_path" in error and "agents_path" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original


def test_apply_role_pool_update_rejects_when_history_path_is_hard_link_to_agents_path(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")
    history_path = tmp_path / "agents-history-link.jsonl"
    history_path.hardlink_to(agents_path)

    result = apply_role_pool_update(
        agents_path=agents_path,
        history_path=history_path,
        payload={"reason": "bad update", "remove_roles": ["SurveyAgent"]},
        task_id="task-1",
        run_ref="meta-history-hardlink",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert any("history_path" in error and "agents_path" in error for error in result.errors)
    assert agents_path.read_text(encoding="utf-8") == original
