# Demo V0 Meta-Agent

Inspect the active role pool in `agents.md` before dynamic workflow planning.

The current pool should contain:

- `SolverAgent` for the initial catalyst recommendation.
- `ReviewerAgent` for evidence review.

Return exactly one `DispatchDecision` JSON object with route `END`. Include
`metadata.role_pool_update` when the reusable role pool should change, or
`metadata.no_role_pool_update_reason` when no reusable role-pool update is
needed. Do not route executable work; DynamicWorkflowPlanner creates the
runtime workflow after this step.
