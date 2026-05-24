# Skill Runtime Integration Design

Date: 2026-05-04

## Goal

Add the minimal runtime integration needed to turn a `RetrievalRequest` into selected skills, prepared tool specs, and prompt-injectable skill context.

## Architecture

The integration lives in `evolab/runtime/skill_retrieval.py` so `TaskRuntime` does not own graph retrieval, tool preparation, and prompt-context formatting directly. The helper calls a skill backend with the current `RetrievalRequest`, aggregates required tools from the returned `SkillBundle`, asks `ToolRuntime.prepare()` for available tool specs, and raises a clear error when required tools are not prepared.

`PromptBuilder.build()` accepts an optional `skill_context` dict. When present, it appends a JSON `Skill Context` section containing selected skills, retrieval tree paths, graph context summary, and required tools.

## Runtime Flow

1. `TaskRuntime.run()` creates the existing `RetrievalRequest`.
2. Memory retrieval still runs through `memory.search(request)`.
3. Skill retrieval runs through `prepare_skill_runtime_context(...)`.
4. The prepared `ToolBundle` is converted to JSON-compatible tool specs and passed to `llm.generate(...)`.
5. The generated `skill_context` is included in the prompt and persisted in run metadata with the `ToolBundle`.

## Error Handling

Missing required tools are reported by `MissingRequiredToolError` with the missing tool names. A tool is missing if it is absent from the registry, filtered out by `allowed_tools`, or there is no `ToolRuntime` while skills require tools.

## Testing

Tests cover real `GraphSkillBackend` retrieval, required tool aggregation, `ToolRuntime.prepare()` output, missing tool errors, prompt context injection, and `TaskRuntime` passing prepared tool specs to the LLM.
