# Domain Entity Normalization

## Purpose
Normalize aligned domain entities using task-provided domain normalization policy.

## Scope
Reusable domain entity normalization driven by task policies.

## Applicability
- Aligned entities need canonical forms under a task-specific policy.

## Limitations
- Does not require or hardcode domain-specific normalizer tools.

## Required Inputs
- aligned entities
- normalization policy
- task schema

## Expected Outputs
- normalized entities
- normalization notes
- unresolved entities

## Procedure
1. Read normalization policy.
2. Normalize aligned entity labels and values.
3. Preserve original mention and evidence.
4. Report unresolved or policy-conflicting cases.

## Required Tools
- read_text
- json_schema_validate

## Validation Signals
- normalized_entities_preserve_original_mentions
