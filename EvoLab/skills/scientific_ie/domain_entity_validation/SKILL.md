# Domain Entity Validation

## Purpose
Validate normalized entities against task-provided domain validation policy.

## Scope
Reusable validation of normalized domain entities using task-provided policies.

## Applicability
- Normalized entities need validation before final record acceptance.

## Limitations
- Does not hardcode biological, chemical, or material validation tools.

## Required Inputs
- normalized entities
- validation policy
- candidate records

## Expected Outputs
- entity validation results
- invalid entity report
- validation warnings

## Procedure
1. Validate policy schema.
2. Apply entity validation rules.
3. Preserve evidence and original values.
4. Return invalid entities and warnings.

## Required Tools
- json_schema_validate

## Validation Signals
- entity_validation_results_attached_to_records
