# Memory Methods

This package contains native memory algorithms used by `MethodMemoryBackend`.
Methods are lower-level than runtime backends: they receive method-specific
search/ingest requests and return normalized memory results.

## Module Map

- `base.py`: method-level request, result, and protocol definitions.
- `mem0.py`: native mem0-style extraction, dedupe, embedding, and search.
- `everos.py`: native EverOS/EverMemOS-style scene and cell memory.
- `store.py`: SQLite persistence for memories, messages, entities, links, and
  scope versions.
- `retrieval.py`: scoring and ranking helpers.

## Boundaries

Methods should not import `TaskRuntime`. They may bind LLM and embedding
runtimes through `bind_runtimes()`, but task orchestration stays in
`runtime/`. All persistence paths should be provided by the Lab layout or
backend config, normally under `.evolab/memory`.

