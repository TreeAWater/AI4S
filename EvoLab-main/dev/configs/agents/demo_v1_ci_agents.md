# EvoLab Demo V1 CI Agents

This file is the active role pool for the V1 CI memory-path demo. MetaAgent may
update it automatically during dynamic role-pool evolution.

```json
{
  "schema_version": "v1",
  "metadata": {
    "role_pool_seed": true,
    "role_pool_generation": 0,
    "domain": "demo_v1_ci"
  },
  "agents": [
    {
      "schema_version": "v1",
      "name": "GeneralistAgent",
      "system_prompt": "You verify EvoLab V1 dynamic task-memory behavior.",
      "llm_backend": {
        "backend_id": "fake-llm",
        "state_ref": "fake-llm-state-v1"
      },
      "allowed_tools": [],
      "required_skills": [],
      "metadata": {
        "role_pool_seed": true,
        "role_pool_generation": 0,
        "specialization": "dynamic task-memory CI verification"
      }
    }
  ]
}
```
