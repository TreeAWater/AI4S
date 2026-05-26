# Tools

`evolab.tools` contains built-in tools and the tool runtime registry. Tools are
the actions available to task workers during LLM/tool loops.

## Module Map

- `runtime.py`: `ToolRegistry` and `ToolRuntime`.
- `files.py`, `text.py`, `tables.py`, `schema.py`: generic file, text, table,
  and JSON schema tools.
- `output.py`: user-visible output writers such as `write_report` and
  `write_jsonl`.
- `scientific_artifacts.py` and `scientific_ie.py`: scientific artifact helpers
  and grouped scientific IE registration.
- `human.py`: human-in-the-loop tool specs and handlers.
- `paths.py`: path resolution helpers.

## Boundaries

Built-in tools should be generic and deterministic where possible. Task-specific
tools generated during a session belong under `.evolab/generated_tools`, not in
this package. Tools should never read or write `.evolab` unless they are
explicitly internal maintenance tools.

