import json
from pathlib import Path
from zipfile import ZipFile

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.lab_state import ArtifactIndexRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime as _TaskRuntime, _scientific_handoff_bootstrap_calls
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.scientific_artifacts import serialize_final_records
from evolab.tools.scientific_ie import register_scientific_ie_tools


def _runtime(tmp_path: Path) -> ToolRuntime:
    registry = ToolRegistry()
    register_scientific_ie_tools(registry, artifact_root=tmp_path / "artifacts")
    runtime = ToolRuntime(registry)
    tools = [
        "build_document_inventory",
        "discover_candidate_source_files",
        "discover_candidate_tables",
        "extract_candidate_rows",
        "build_candidate_records",
        "validate_candidate_records",
        "serialize_final_records",
    ]
    runtime.prepare(required_tools=tools, allowed_tools=tools, policy=RuntimePolicy(max_tool_steps=20))
    return runtime


class _DynamicPlanner:
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config

    def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
        agents = []
        nodes = []
        previous = None
        for index, role in enumerate(self.task_config.roles.values()):
            subagent_id = f"{role.name}-dynamic"
            node_id = f"node-{index}-{role.name}"
            agents.append(
                {
                    "subagent_id": subagent_id,
                    "role_name": role.name,
                    "goal": self.task_config.goal,
                    "system_prompt": role.system_prompt,
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "allowed_tools": list(role.allowed_tools),
                    "llm_backend_id": role.llm_backend.backend_id,
                }
            )
            nodes.append({"node_id": node_id, "subagent_id": subagent_id, "dependencies": [previous] if previous else []})
            previous = node_id
        payload = {
            "workflow_id": "wf-scientific-artifacts",
            "task_summary": self.task_config.goal,
            "article_context_summary": "unit test",
            "dynamic_subagents": agents,
            "workflow_nodes": nodes,
            "workflow_edges": [],
            "artifact_contracts": {},
            "validation_rules": [],
            "planner_rationale_summary": "Run configured roles in order.",
            "metadata": {"extraction_task": False},
        }
        return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(payload)))


def TaskRuntime(**kwargs) -> _TaskRuntime:
    task_config = kwargs.get("task_config")
    if (
        isinstance(task_config, TaskConfig)
        and task_config.dynamic_subagents is None
        and task_config.meta_agent is None
    ):
        roles = list(task_config.roles.values())
        kwargs["task_config"] = task_config.model_copy(
            update={
                "agents_ref": task_config.agents_ref or "configs/agents.md",
                "dynamic_subagents": DynamicSubagentsConfig(
                    enabled=True,
                    mode="dynamic",
                    planner_backend={"backend_id": "planner-local"},
                    default_worker_backend={"backend_id": roles[0].llm_backend.backend_id},
                    allowed_worker_backend_ids=sorted({role.llm_backend.backend_id for role in roles}),
                    allowed_tool_names=sorted({tool for role in roles for tool in role.allowed_tools}),
                    max_planner_retries=0,
                    require_output_schema=False,
                )
            }
        )
        kwargs["llm_runtimes"] = {"planner-local": _DynamicPlanner(task_config), **(kwargs.get("llm_runtimes") or {})}
    return _TaskRuntime(**kwargs)


def test_document_inventory_includes_main_text_and_supplementary_files(tmp_path):
    article = _write_article_fixture(tmp_path)
    runtime = _runtime(tmp_path)

    result = runtime.execute(
        ToolCall(
            call_id="inventory",
            name="build_document_inventory",
            arguments={"root": str(article), "work_item_id": "article_a"},
        )
    )

    assert result.status == "ok"
    inventory_path = Path(result.artifact_refs[0].uri)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    file_types = {item["file_type"] for item in inventory["files"]}
    assert "main_text" in {item["role"] for item in inventory["files"]}
    assert {"markdown", "spreadsheet", "csv_tsv"}.issubset(file_types)
    assert any(item["metadata"].get("sheet_count") == 1 for item in inventory["files"])


def test_candidate_source_files_are_ranked_and_written(tmp_path):
    article = _write_article_fixture(tmp_path)
    runtime = _runtime(tmp_path)
    inventory = runtime.execute(
        ToolCall(
            call_id="inventory",
            name="build_document_inventory",
            arguments={"root": str(article), "work_item_id": "article_a"},
        )
    )

    result = runtime.execute(
        ToolCall(
            call_id="sources",
            name="discover_candidate_source_files",
            arguments={
                "document_inventory_path": inventory.artifact_refs[0].uri,
                "work_item_id": "article_a",
                "task_goal": "extract sequence component records from scientific tables",
            },
        )
    )

    assert result.status == "ok"
    sources = json.loads(Path(result.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert sources["candidate_sources"]
    assert sources["candidate_sources"][0]["file_type"] in {"spreadsheet", "csv_tsv", "markdown"}
    assert all("reasons" in item for item in sources["candidate_sources"])


def test_candidate_tables_include_sheet_metadata_and_sample_rows(tmp_path):
    article = _write_article_fixture(tmp_path)
    runtime = _runtime(tmp_path)
    inventory = runtime.execute(
        ToolCall(
            call_id="inventory",
            name="build_document_inventory",
            arguments={"root": str(article), "work_item_id": "article_a"},
        )
    )
    sources = runtime.execute(
        ToolCall(
            call_id="sources",
            name="discover_candidate_source_files",
            arguments={"document_inventory_path": inventory.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )

    result = runtime.execute(
        ToolCall(
            call_id="tables",
            name="discover_candidate_tables",
            arguments={"candidate_source_files_path": sources.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )

    assert result.status == "ok"
    tables = json.loads(Path(result.artifact_refs[0].uri).read_text(encoding="utf-8"))
    spreadsheet_tables = [item for item in tables["candidate_tables"] if item["file_type"] == "spreadsheet"]
    assert spreadsheet_tables
    assert spreadsheet_tables[0]["sheet_name"] == "Sheet1"
    assert spreadsheet_tables[0]["headers"] == ["component_name", "sequence", "evidence"]
    assert spreadsheet_tables[0]["sample_rows"]


def test_candidate_rows_and_records_can_be_built_from_spreadsheet_fixture(tmp_path):
    article = _write_article_fixture(tmp_path)
    runtime = _runtime(tmp_path)
    inventory = runtime.execute(
        ToolCall(
            call_id="inventory",
            name="build_document_inventory",
            arguments={"root": str(article), "work_item_id": "article_a"},
        )
    )
    sources = runtime.execute(
        ToolCall(
            call_id="sources",
            name="discover_candidate_source_files",
            arguments={"document_inventory_path": inventory.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    tables = runtime.execute(
        ToolCall(
            call_id="tables",
            name="discover_candidate_tables",
            arguments={"candidate_source_files_path": sources.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )

    rows = runtime.execute(
        ToolCall(
            call_id="rows",
            name="extract_candidate_rows",
            arguments={"candidate_tables_path": tables.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    records = runtime.execute(
        ToolCall(
            call_id="records",
            name="build_candidate_records",
            arguments={
                "candidate_rows_path": rows.artifact_refs[0].uri,
                "article_id": "article_a",
                "work_item_id": "article_a",
            },
        )
    )

    assert rows.status == "ok"
    row_payload = json.loads(Path(rows.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert len(row_payload["candidate_rows"]) >= 2
    assert row_payload["candidate_rows"][0]["candidate_sequence_fields"]
    assert records.status == "ok"
    record_payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    sequences = {record["sequence"] for record in record_payload["records"]}
    assert {
        "TTGACAGCAATCTCAATCCTAGGTATAAT",
        "AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATAATGCGAGCA",
    }.issubset(sequences)


def test_promoter_profile_builds_records_from_promoter_sequence_fields_and_deduplicates(tmp_path):
    runtime = _runtime(tmp_path)
    rows_path = tmp_path / "candidate_rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "work_item_id": "article_a",
                "candidate_rows": [
                    {
                        "source_file": str(tmp_path / "supplement.xlsx"),
                        "sheet_name": "Sheet1",
                        "row_index": 2,
                        "row": {
                            "promoter_name": "P1",
                            "promoter_sequence": "TTGACAGCAATCTCAATCCTAGGTATAAT",
                            "pbc_seq": "AAAAAAAAAAAAAAAAAAAA",
                        },
                        "candidate_sequence_fields": [
                            {"field": "pbc_seq", "value": "AAAAAAAAAAAAAAAAAAAA"},
                            {"field": "promoter_sequence", "value": "TTGACAGCAATCTCAATCCTAGGTATAAT"},
                        ],
                        "candidate_label_fields": [{"field": "promoter_name", "value": "P1"}],
                    },
                    {
                        "source_file": str(tmp_path / "supplement.xlsx"),
                        "sheet_name": "Sheet1",
                        "row_index": 3,
                        "row": {
                            "promoter_name": "P1 duplicate",
                            "promoter_sequence": "TTGACAGCAATCTCAATCCTAGGTATAAT",
                        },
                        "candidate_sequence_fields": [
                            {"field": "promoter_sequence", "value": "TTGACAGCAATCTCAATCCTAGGTATAAT"},
                        ],
                        "candidate_label_fields": [{"field": "promoter_name", "value": "P1 duplicate"}],
                    },
                    {
                        "source_file": str(tmp_path / "generated.xlsx"),
                        "sheet_name": "seqs",
                        "row_index": 4,
                        "row": {"generated_sequences": "CCCCCCCCCCCCCCCCCCCC"},
                        "candidate_sequence_fields": [{"field": "generated_sequences", "value": "CCCCCCCCCCCCCCCCCCCC"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runtime.execute(
        ToolCall(
            call_id="records",
            name="build_candidate_records",
            arguments={
                "candidate_rows_path": str(rows_path),
                "article_id": "article_a",
                "work_item_id": "article_a",
                "sequence_extraction_profile": "promoter",
                "deduplicate_sequences": True,
            },
        )
    )

    assert result.status == "ok"
    payload = json.loads(Path(result.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    assert payload["records"][0]["sequence_field"] == "promoter_sequence"
    assert payload["records"][0]["component_name"] == "P1"
    assert payload["skipped_counts"]["duplicate_sequence"] == 1
    assert payload["skipped_counts"]["no_profile_compatible_sequence_field"] == 1


def test_validation_and_writer_serialize_final_records(tmp_path):
    article = _write_article_fixture(tmp_path)
    runtime = _runtime(tmp_path)
    records = _build_records(runtime, article)

    validated = runtime.execute(
        ToolCall(
            call_id="validate",
            name="validate_candidate_records",
            arguments={"candidate_records_path": records.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    final = runtime.execute(
        ToolCall(
            call_id="final",
            name="serialize_final_records",
            arguments={
                "records_path": validated.artifact_refs[0].uri,
                "artifact_name": "biology_component_records.jsonl",
                "also_write_final_records": True,
            },
        )
    )

    assert validated.status == "ok"
    validated_payload = json.loads(Path(validated.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert len(validated_payload["accepted_records"]) >= 2
    assert final.status == "ok"
    output_paths = {Path(ref.uri).name: Path(ref.uri) for ref in final.artifact_refs}
    assert set(output_paths) == {"biology_component_records.jsonl", "final_records.jsonl"}
    assert output_paths["biology_component_records.jsonl"].read_text(encoding="utf-8").count("\n") == len(
        validated_payload["accepted_records"]
    )


def test_serialize_final_records_accepts_jsonl_record_artifact(tmp_path):
    records_path = tmp_path / "validated_records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps({"component_name": "alpha", "sequence": "AACCGGTT"}),
                json.dumps({"component_name": "rejected", "sequence": "TTTT", "status": "rejected"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = serialize_final_records(
        {"records_path": str(records_path), "artifact_name": "final_records.jsonl"},
        artifact_root=tmp_path / "artifacts",
    )

    assert result.status == "ok"
    output = Path(result.artifact_refs[0].uri)
    assert output.read_text(encoding="utf-8").splitlines() == [
        json.dumps({"component_name": "alpha", "sequence": "AACCGGTT"}, sort_keys=True)
    ]
    assert result.metadata["record_count"] == 1


def test_runtime_bootstrap_preserves_scientific_handoff_artifacts(tmp_path):
    lab_root = tmp_path / "lab"
    article = _write_article_fixture(tmp_path)
    registry = ToolRegistry()
    register_scientific_ie_tools(registry, artifact_root=lab_root / "artifacts" / "tools")
    llm = _ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="exec done")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="write done")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal=(
            "Process one article.\n"
            "1. work_item_id: article_a\n"
            f"   article_package: {article}\n"
            "   exact_source_files:\n"
            f"     - {article / 'main_text.md'}\n"
            f"     - {article / 'supplementary' / 'table.xlsx'}\n"
        ),
        runtime_policy=RuntimePolicy(),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=[
                    "build_document_inventory",
                    "discover_candidate_source_files",
                    "discover_candidate_tables",
                    "extract_candidate_rows",
                    "build_candidate_records",
                ],
            ),
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["serialize_final_records"],
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(lab_root / "registries" / "trajectory"),
        lab_state_registry=FileLabStateRegistry(lab_root / "registries" / "lab_state"),
        tool_artifact_root_factory=lambda request, run_ref: lab_root / "tasks" / request.task_id / "artifacts" / run_ref,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": _MemoryRuntime()},
        skill_runtimes={"skill-local": _SkillRuntime()},
    )

    result = runtime.run(
        TaskRequest(
            task_id="task-1",
            origin=TaskOrigin.HUMAN,
            purpose=TaskPurpose.SCIENCE,
            goal=task_config.goal,
        )
    )

    assert result["runs"][0]["artifact_refs"]
    artifact_names = {
        ref.get("metadata", {}).get("filename") or Path(ref["uri"]).name
        for run in result["runs"]
        for ref in run.get("artifact_refs", [])
    }
    assert {
        "document_inventory.json",
        "candidate_source_files.json",
        "candidate_tables.json",
        "candidate_rows.json",
        "candidate_records.json",
        "biology_component_records.jsonl",
        "final_records.jsonl",
        "biology_component_report.md",
    }.issubset(artifact_names)
    final_path = lab_root / "artifacts" / "tools" / "biology_component_records.jsonl"
    assert final_path.exists()
    assert final_path.read_text(encoding="utf-8").count("\n") >= 2


def test_bootstrap_context_prefers_configured_work_item_block_over_trailing_instruction_text(tmp_path):
    article = _write_article_fixture(tmp_path)
    task_goal = (
        "Process one article.\n"
        "1. work_item_id: article_a\n"
        f"   article_package: {article}\n"
        "   exact_source_files:\n"
        f"     - {article / 'main_text.md'}\n"
        f"     - {article / 'supplementary' / 'table.xlsx'}\n"
    )
    instruction = (
        f"Extract work item article_a. Article package: {article}. "
        "Read all listed source files: manifest.json, main_text.md. Identify supported records."
    )

    calls = _scientific_handoff_bootstrap_calls(
        role="ExecAgent",
        role_instruction=instruction,
        task_goal=task_goal,
        dispatch_metadata={"work_item_id": "article_a"},
        lab_state_registry=None,
        task_id="task-1",
        artifact_root=tmp_path / "artifacts",
        runtime_metadata={"scientific_max_rows_per_table": 10000},
    )

    assert calls[0]["name"] == "build_document_inventory"
    assert calls[0]["arguments"]["root"] == str(article)
    assert calls[0]["arguments"]["source_files"] == [
        str(article / "main_text.md"),
        str(article / "supplementary" / "table.xlsx"),
    ]
    assert calls[3]["name"] == "extract_candidate_rows"
    assert calls[3]["arguments"]["max_rows_per_table"] == 10000
    assert calls[4]["arguments"]["sequence_extraction_profile"] is None


def test_bootstrap_context_trims_instruction_sentence_after_article_package_path(tmp_path):
    article = _write_article_fixture(tmp_path)
    instruction = (
        f"Extract records from the article package: {article}. "
        "Inspect all exact_source_files listed, including main_text.md and supplementary files."
    )

    calls = _scientific_handoff_bootstrap_calls(
        role="ExecAgent",
        role_instruction=instruction,
        task_goal="Process the assigned scientific article package.",
        dispatch_metadata={"work_item_id": "article_a"},
        lab_state_registry=None,
        task_id="task-1",
        artifact_root=tmp_path / "artifacts",
    )

    assert calls[0]["name"] == "build_document_inventory"
    assert calls[0]["arguments"]["root"] == str(article)


def test_write_bootstrap_serializes_records_and_writes_report(tmp_path):
    calls = _scientific_handoff_bootstrap_calls(
        role="WriteAgent",
        role_instruction="Write final artifacts.",
        task_goal="Write final_records.jsonl and biology_component_records.jsonl plus a report.",
        dispatch_metadata={},
        lab_state_registry=None,
        task_id="task-1",
        artifact_root=tmp_path / "artifacts",
    )

    assert [call["name"] for call in calls] == ["serialize_final_records", "write_report"]
    assert calls[0]["arguments"]["artifact_name"] == "biology_component_records.jsonl"
    assert calls[0]["arguments"]["also_write_final_records"] is True
    assert calls[1]["arguments"]["artifact_name"] == "biology_component_report.md"


def test_write_bootstrap_triggers_bounded_recovery_when_work_item_has_zero_records(tmp_path):
    article = _write_article_fixture(tmp_path)
    task_goal = (
        "Extract promoter sequence records.\n"
        "1. work_item_id: article_a\n"
        f"   article_package: {article}\n"
        "   exact_source_files:\n"
        f"     - {article / 'main_text.md'}\n"
        f"     - {article / 'supplementary' / 'table.xlsx'}\n"
    )

    calls = _scientific_handoff_bootstrap_calls(
        role="WriteAgent",
        role_instruction="Write final artifacts for work item article_a.",
        task_goal=task_goal,
        dispatch_metadata={"work_item_id": "article_a", "execution_mode": "dynamic"},
        lab_state_registry=None,
        task_id="task-1",
        artifact_root=tmp_path / "artifacts",
        runtime_metadata={"dynamic_zero_output_recovery_enabled": True},
    )

    names = [call["name"] for call in calls]
    assert names[:4] == [
        "write_report",
        "build_document_inventory",
        "discover_candidate_source_files",
        "discover_candidate_tables",
    ]
    assert "extract_candidate_rows" in names
    assert "build_candidate_records" in names
    assert names[-2:] == ["serialize_final_records", "write_report"]
    assert calls[-2]["arguments"]["work_item_id"] == "article_a"
    assert calls[-2]["arguments"]["records_path"].endswith("article_a/candidate_records.json")


def test_write_bootstrap_does_not_recover_when_current_work_item_records_exist(tmp_path):
    registry = FileLabStateRegistry(tmp_path / "lab_state")
    candidate_path = tmp_path / "candidate_records.json"
    candidate_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "P1",
                        "sequence": "TTGACAGCAATCTCAATCCTAGGTATAAT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="artifact-candidate",
            task_id="task-1",
            uri=str(candidate_path),
            artifact_type="dataset",
            metadata={
                "work_item_id": "article_a",
                "filename": "candidate_records.json",
                "artifact_kind": "candidate_records",
            },
        )
    )

    calls = _scientific_handoff_bootstrap_calls(
        role="WriteAgent",
        role_instruction="Write final artifacts for work item article_a.",
        task_goal="Extract promoter sequence records.",
        dispatch_metadata={"work_item_id": "article_a", "lab_path": str(tmp_path), "execution_mode": "dynamic"},
        lab_state_registry=registry,
        task_id="task-1",
        artifact_root=tmp_path / "artifacts",
        runtime_metadata={"dynamic_zero_output_recovery_enabled": True},
    )

    assert [call["name"] for call in calls] == ["serialize_final_records", "write_report"]
    assert calls[0]["arguments"]["records"][0]["component_name"] == "P1"


def test_invalid_work_item_lab_path_fails_before_subagent_dispatch_or_tool_calls(tmp_path):
    invalid_article = tmp_path / "missing-article"

    class ExecLLM:
        def __init__(self):
            self.calls = []

        def generate(self, messages, tool_specs, generation_config):
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="should not run"))

    registry = ToolRegistry()
    register_scientific_ie_tools(registry, artifact_root=tmp_path / "lab" / "artifacts" / "tools")
    task_goal = (
        "Process one article.\n"
        "1. work_item_id: article_a\n"
        f"   article_package: {invalid_article}\n"
        "   exact_source_files:\n"
        f"     - {invalid_article / 'main_text.md'}\n"
    )
    exec_llm = ExecLLM()
    lab_root = tmp_path / "lab"
    task_config = TaskConfig(
        task_id="task-1",
        goal=task_goal,
        agents_ref=str(lab_root / "configs" / "agents.md"),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
                allowed_tools=["build_document_inventory", "inspect_table"],
            )
        },
        dynamic_subagents=DynamicSubagentsConfig(
            enabled=True,
            mode="dynamic",
            scope="per_work_item",
            planner_backend={"backend_id": "planner-local"},
            default_worker_backend={"backend_id": "exec-llm"},
            allowed_worker_backend_ids=["exec-llm"],
            allowed_tool_names=["build_document_inventory", "inspect_table"],
            max_planner_retries=0,
            require_output_schema=False,
        ),
        runtime_policy=RuntimePolicy(),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(lab_root / "registries" / "trajectory"),
        lab_state_registry=FileLabStateRegistry(lab_root / "registries" / "lab_state"),
        llm_runtimes={"planner-local": _DynamicPlanner(task_config), "exec-llm": exec_llm},
        memory_runtimes={"memory-local": _MemoryRuntime()},
        skill_runtimes={"skill-local": _SkillRuntime()},
    )

    result = runtime.run(
        TaskRequest(
            task_id="task-1",
            origin=TaskOrigin.HUMAN,
            purpose=TaskPurpose.SCIENCE,
            goal=task_goal,
        )
    )

    assert result["status"] == "failed"
    assert "lab_path is not an existing directory" in result["failure_reason"]
    assert exec_llm.calls == []
    assert not (lab_root / "registries" / "trajectory" / "subagent.jsonl").exists()
    tool_records = lab_root / "registries" / "trajectory" / "tool_calls.jsonl"
    assert not tool_records.exists()
    work_item_record = lab_root / "registries" / "lab_state" / "work_items" / "task-1" / "article_a.json"
    payload = json.loads(work_item_record.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "lab_path is not an existing directory" in payload["history"][-1]["failure_reason"]


def _build_records(runtime: ToolRuntime, article: Path):
    inventory = runtime.execute(
        ToolCall(
            call_id="inventory",
            name="build_document_inventory",
            arguments={"root": str(article), "work_item_id": "article_a"},
        )
    )
    sources = runtime.execute(
        ToolCall(
            call_id="sources",
            name="discover_candidate_source_files",
            arguments={"document_inventory_path": inventory.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    tables = runtime.execute(
        ToolCall(
            call_id="tables",
            name="discover_candidate_tables",
            arguments={"candidate_source_files_path": sources.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    rows = runtime.execute(
        ToolCall(
            call_id="rows",
            name="extract_candidate_rows",
            arguments={"candidate_tables_path": tables.artifact_refs[0].uri, "work_item_id": "article_a"},
        )
    )
    return runtime.execute(
        ToolCall(
            call_id="records",
            name="build_candidate_records",
            arguments={
                "candidate_rows_path": rows.artifact_refs[0].uri,
                "article_id": "article_a",
                "work_item_id": "article_a",
            },
        )
    )


def _write_article_fixture(tmp_path: Path) -> Path:
    article = tmp_path / "article"
    supplementary = article / "supplementary"
    supplementary.mkdir(parents=True)
    (article / "main_text.md").write_text(
        "# Article\n\nThe study reports sequence-bearing component tables in supplementary data.\n",
        encoding="utf-8",
    )
    (supplementary / "notes.md").write_text(
        "| component_name | sequence | evidence |\n"
        "| --- | --- | --- |\n"
        "| md-row | CCCCCCCCGGGGGGGG | table evidence |\n",
        encoding="utf-8",
    )
    (supplementary / "table.csv").write_text(
        "component_name,sequence,evidence\ncsv-row,GGGGTTTTCCCCAAAA,csv evidence\n",
        encoding="utf-8",
    )
    _write_minimal_xlsx(supplementary / "table.xlsx")
    return article


def _write_minimal_xlsx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="worksheets/sheet1.xml"
                 Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
                </Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                <sheetData>
                <row r="1"><c r="A1" t="inlineStr"><is><t>component_name</t></is></c><c r="B1" t="inlineStr"><is><t>sequence</t></is></c><c r="C1" t="inlineStr"><is><t>evidence</t></is></c></row>
                <row r="2"><c r="A2" t="inlineStr"><is><t>alpha</t></is></c><c r="B2" t="inlineStr"><is><t>TTGACAGCAATCTCAATCCTAGGTATAAT</t></is></c><c r="C2" t="inlineStr"><is><t>validated in table</t></is></c></row>
                <row r="3"><c r="A3" t="inlineStr"><is><t>beta</t></is></c><c r="B3" t="inlineStr"><is><t>AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATAATGCGAGCA</t></is></c><c r="C3" t="inlineStr"><is><t>validated in table</t></is></c></row>
                </sheetData></worksheet>""",
        )


class _MemoryRuntime:
    def search(self, request: RetrievalRequest) -> MemoryBundle:
        return MemoryBundle(backend_id="memory-local", items=[])

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        return {"status": "updated", "state_ref": f"{role}-memory"}


class _SkillRuntime:
    def get(self, request: RetrievalRequest) -> SkillBundle:
        return SkillBundle(backend_id="skill-local", skills=[], required_tools=[])

    def look_at(self, event: dict) -> dict[str, str]:
        return {"status": "recorded"}


class _ScriptedLLM:
    def __init__(self, responses: list[LLMRuntimeResponse]):
        self.responses = responses
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tools, generation_config))
        if not self.responses:
            raise AssertionError("unexpected llm call")
        return self.responses.pop(0)
