# AgentSkillOS V1 Parity

EvoLab v1 maps the AgentSkillOS layers onto scientific task execution without copying AgentSkillOS code.

## Manage Skills

`GraphSkillBackend` is the Manage Skills layer. It stores reusable skills as independent packages, indexes a graph-backed capability library, instantiates a task-time tree/subtree retrieval view, hydrates package-backed skills, aggregates required tools, and returns retrieval trace metadata.

EvoLab-specific extensions:

- The canonical skill library is graph-backed rather than a pure tree.
- Task-time retrieval instantiates tree/subtree paths over that graph.
- Typed relationships support retrieval-time dependency completion and planning-time DAG edges.
- Domain packages keep task-specific schemas, ontologies, evidence policies, and negative patterns separate from reusable stable skills.

## Solve Tasks

`SkillWorkflowPlanner` is the minimal v1 skill orchestration layer. It converts a retrieved `SkillBundle` into a skill-level `WorkflowPlan` DAG.

`TaskRuntime` remains the executor. When workflow planning is enabled, it executes workflow nodes in topological order using the same LLM runtime and the same `ToolRuntime` used by the flat path.

`ToolRuntime` remains the only tool execution system. Human participation is represented as optional tools, not mandatory workflow nodes.

Reusable scientific IE skills declare generic `required_tools`. EvoLab v1 provides registered ToolSpec entries and executable handlers for every required tool in `skills/scientific_ie`. Domain-specific schema, ontology, evidence, and negative-pattern logic remains in `domain_packages`.

## V1 Scope

Implemented:

- `RetrievalRequest -> SkillBundle -> WorkflowPlan`
- plan-aware node execution
- `NodeExecutionRecord`
- run-level `ToolTrace`
- artifact collection
- `PlanExecutionTrace`
- `SkillObservationRequest -> skill.look_at(...)`
- optional human tools gated by `RuntimePolicy`
- complete generic scientific IE required-tool coverage

Not implemented in v1:

- production workflow engine
- production sandboxed tool infrastructure
- task-level decomposition planner
- GUI
- full self-evolution
- embedding or LLM retrieval
- mandatory human review nodes
