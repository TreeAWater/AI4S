# Scientific IE Tool Coverage

Every reusable scientific IE skill package under `skills/scientific_ie/` declares `required_tools`. EvoLab v1 provides ToolSpec entries and executable handlers for the complete required set.

## Required Tool Set

File and document tools:

- `list_files`
- `read_text`
- `search_text`
- `inspect_file_metadata`
- `extract_sections`

Table and spreadsheet tools:

- `inspect_table`
- `read_table_slice`
- `inspect_excel_workbook`
- `read_excel_sheet`
- `detect_table_header`
- `normalize_table`
- `profile_table`

Schema and output tools:

- `json_schema_validate`
- `write_jsonl`
- `write_report`

Human tools:

- `ask_human`
- `request_human_review`
- `notify_human`

The human tools are optional runtime tools. They are not stable skill `required_tools`.

## Registration

`register_scientific_ie_tools(...)` composes:

- `register_file_tools(...)`
- `register_text_tools(...)`
- `register_table_tools(...)`
- `register_schema_tools(...)`
- `register_output_tools(...)`
- `register_human_tools(...)`

When `base_dir` is provided, local file-style arguments such as `path`,
`root`, `schema_path`, and `instance_path` resolve relative to that base
directory. `clean-run` passes the Lab root so seeded files such as
`inputs/biology_component_article.md` are readable by the same relative paths
that appear in the task goal.

Every registered `ToolSpec` includes a JSON object parameter schema with
properties and required fields for LLM tool calling. This is required for the
real `ApiLLMBackend` path; an empty object schema lets the model legally call
tools with `{}` and is not considered V1-complete.

## Boundaries

`GraphSkillBackend` retrieves skills and returns `SkillBundle` metadata. It does not execute tools.

`TaskRuntime` and `ToolRuntime` execute tools during flat or workflow-plan-aware runs.

The v1 handlers are generic and local:

- no shell execution
- no biology-specific stable skill IDs
- no domain-specific extraction algorithms
- output tools return `ArtifactRef`s through existing contracts

Production sandboxing, remote artifact stores, and richer table parsing are future hardening work.
