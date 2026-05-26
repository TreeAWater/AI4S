# Extraction Schema Interpretation

## Purpose
Interpret task-specific extraction schema into field definitions, required/nullable fields, evidence requirements, validation requirements, and extraction constraints.

## Scope
Reusable interpretation of task-provided extraction schemas.

## Applicability
- The task provides a schema or field specification.

## Limitations
- Does not invent missing domain ontology terms.

## Required Inputs
- extraction schema
- task goal
- domain package references

## Expected Outputs
- field definitions
- required and nullable fields
- evidence requirements
- validation constraints

## Procedure
1. Validate schema syntax.
2. List fields and nullability.
3. Extract evidence and validation requirements.
4. Identify constraints and domain package references.
5. Return interpreted schema contract.

## Required Tools
- json_schema_validate

## Validation Signals
- schema_fields_and_constraints_listed
