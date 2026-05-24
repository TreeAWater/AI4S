# Domain Negative Pattern Filtering

## Purpose
Filter false positives using task-provided negative pattern policy.

## Scope
Reusable negative pattern filtering for candidate entities and records.

## Applicability
- A task or domain package provides negative patterns.

## Limitations
- Negative patterns are task resources, not stable skill IDs.

## Required Inputs
- candidate records or mentions
- negative pattern policy
- evidence context

## Expected Outputs
- filtered false positives
- kept records
- filter rationale

## Procedure
1. Read negative pattern policy.
2. Search evidence context for exclusion patterns.
3. Mark false positives with rationale.
4. Return kept and filtered records.

## Required Tools
- search_text
- json_schema_validate

## Validation Signals
- false_positive_filters_are_explainable
