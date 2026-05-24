# Params And Runtime V1 README

This document is the V1 sign-off reference for EvoLab runtime, API LLM
integration, and local-trainable rollout state handoff.

## Runtime Boundary

`TaskRuntime` orchestrates lifecycle calls only:

- MetaAgent route decisions over configured subagents
- memory search before prompt construction
- skill retrieval and tool preparation
- bounded LLM/tool loop
- trajectory and tool trace persistence
- memory update after the run
- skill observation after the run

Runtime does not implement memory algorithms, skill graph algorithms, reward
design, SFT/RL, or PEFT training. Those remain backend responsibilities.

## Short Experiment Configs

New experiment configs should be short and human-authored. The expected shape is:

```yaml
task: |
  Natural-language task description, success criteria, input files, schemas,
  policies, and output requirements.
meta_agent:
  system_prompt: |
    Route to one configured subagent or END. Return JSON only.
subagents:
  SurveyAgent:
    system_prompt: Survey assigned context.
  ExecAgent:
    system_prompt: Execute assigned operations.
lab_root: /tmp/evolab-run
backends:
  llm: ...
  memory: ...
  skill: ...
```

`clean-run` compiles this into internal `TaskRequest` and `TaskConfig` models.
The internal shape is still supported for deterministic demos, but humans should
not need to hand-write `task_config.roles` for new experiments.

MetaAgent route output:

```json
{"route":"SurveyAgent","instruction":"Survey current lab inputs.","metadata":{}}
```

Completion output:

```json
{"route":"END","instruction":"Done.","metadata":{"final_answer":"..."}}
```

The runtime validates routes against configured subagent names and includes a
compact Lab state summary in the MetaAgent input.

## API LLM Backend

`ApiLLMBackend` instantiates an OpenAI Responses-compatible client from config:

```json
{
  "backends": {
    "llm": {
      "aigocode-gpt": {
        "type": "api",
        "env_ref": "aigocode-gpt",
        "model": "gpt-5.1-codex"
      }
    }
  }
}
```

The referenced env entry is loaded from `.env`. `env_ref: "aigocode-gpt"` maps
to:

```dotenv
AIGOCODE_GPT_API=openai-responses
AIGOCODE_GPT_BASE_URL=https://api.aigocode.com
AIGOCODE_GPT_API_KEY=...
```

The backend forwards generation config fields such as model override,
temperature, max output tokens, previous response id, explicit response input
items, and strict structured output schema.

For tool calling, EvoLab `ToolSpec` records are converted to Responses function
tools. V1 handles:

- EvoLab `parameters_schema`
- plain Responses `parameters`
- object schemas that omit `properties`
- single-tool-step runtime by disabling parallel tool calls
- multi-turn function call replay for HTTP Responses requests

`ApiLLMBackend.evolve(...)` does not mutate a remote API model. It returns a
skipped `LLMEvolutionResult` with explanatory metadata.

## Tool Observations

Tool execution stays in `ToolRuntime`; `TaskRuntime` only records and relays
results. The next LLM turn receives a `role="tool"` message whose content
contains the result summary and, for successful tools with metadata/artifacts,
a JSON payload. This is required for tools such as `read_text` because the
retrieved text lives in `ToolResult.metadata`.

The complete `ToolResult` is still stored in:

- `SubagentRunRecord.tool_calls`
- `SubagentRunRecord.metadata.tool_trace`
- `LLMCallRecord.input_messages`

## Local Trainable Backend

`LocalTrainableLLMBackend` is the V1 rollout backend for local-trainable
states. It implements only the `LLMBackend` interface; EvolveWorker-managed
trainers such as SFT and OPSD own training/evolution.

Promoted dry-run SFT/OPSD results:

- write adapter and dataset artifacts
- write `state/local_trainable_state.json`
- returns `status="promoted_candidate"`
- returns a valid `local-trainable://...` state ref
- returns `StandardEvolutionMetrics`
- returns a `local_trainable_state` artifact ref

`LocalTrainableLLMBackend.instantiate(None)` returns a deterministic base local
runtime. `instantiate(state_ref)` resolves the registered state manifest and
uses that state's default content and metadata.

The runtime promotion executor applies V1 guards before registering backend
state: non-empty candidate state ref, artifact under artifact root, compatible
role, and valid metrics for non-cold-start promotion.

## Demo Configs

Real V1 path:

```bash
python3 -m evolab.cli clean-run configs/demo_v1.yaml --lab-root /tmp/evolab-demo-v1
```

This uses real API LLM config from `.env`, `GraphSkillBackend`, native mem0
method memory with lab-local SQLite stores, scientific IE tools, and SFT
dry-run evolution that produces a local-trainable rollout state.

CI/offline deterministic path:

```bash
python3 -m evolab.cli clean-run configs/demo_v1_ci.yaml --lab-root /tmp/evolab-demo-v1-ci
```

This preserves fake LLM and fake skill behavior for tests while exercising the
native mem0 method path with fake memory extraction and fake embeddings.

Biology component extraction uses the current short config path:

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_generic_subagents.yaml \
  --lab-root /tmp/evolab-biology-component-run
```

## V2 Boundary

V1 intentionally does not include:

- real LoRA/SFT training in the clean-run path
- async external `EvolveWorker`
- federated consolidation
- public skill governance
- full resource mining
- external HITL integrations
