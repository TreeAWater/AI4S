# EvoLab Demo V0 Agents

This file is the active role pool for the V0 catalyst demo. MetaAgent may update
it automatically during dynamic role-pool evolution.

```json
{
  "schema_version": "v1",
  "metadata": {
    "role_pool_seed": true,
    "role_pool_generation": 0,
    "domain": "demo_v0_catalyst_comparison"
  },
  "agents": [
    {
      "schema_version": "v1",
      "name": "SolverAgent",
      "system_prompt": "You solve scientific comparison tasks with concise evidence-backed recommendations. Read the assigned evidence before making a recommendation.",
      "llm_backend": {
        "backend_id": "fake-llm",
        "state_ref": "fake-llm-state-v0"
      },
      "allowed_tools": [
        "read_file"
      ],
      "required_skills": [],
      "metadata": {
        "role_pool_seed": true,
        "role_pool_generation": 0,
        "specialization": "initial catalyst recommendation"
      }
    },
    {
      "schema_version": "v1",
      "name": "ReviewerAgent",
      "system_prompt": "You review scientific recommendations against the provided evidence and flag missing support or uncertainty.",
      "llm_backend": {
        "backend_id": "fake-llm",
        "state_ref": "fake-llm-state-v0"
      },
      "allowed_tools": [
        "read_file"
      ],
      "required_skills": [],
      "metadata": {
        "role_pool_seed": true,
        "role_pool_generation": 0,
        "specialization": "recommendation review"
      }
    }
  ]
}
```
