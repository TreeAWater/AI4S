# EvoLab API Init Agents

This file is the active role pool for the API initialization smoke config.
MetaAgent may update it automatically during dynamic role-pool evolution.

```json
{
  "schema_version": "v1",
  "metadata": {
    "role_pool_seed": true,
    "role_pool_generation": 0,
    "domain": "api_initialization_smoke"
  },
  "agents": [
    {
      "schema_version": "v1",
      "name": "ApiInitAgent",
      "system_prompt": "You are a minimal API initialization check. Reply with a short success confirmation.",
      "llm_backend": {
        "backend_id": "aigocode-gpt"
      },
      "allowed_tools": [],
      "required_skills": [],
      "metadata": {
        "role_pool_seed": true,
        "role_pool_generation": 0,
        "specialization": "api smoke confirmation"
      }
    }
  ]
}
```
