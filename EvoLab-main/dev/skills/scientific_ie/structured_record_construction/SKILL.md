# Structured Record Construction

## Purpose
Construct candidate structured records from schema interpretation, field mapping, source values, and provenance.

## Scope
Reusable construction of candidate scientific IE records.

## Applicability
- Field mappings and source values are available.

## Limitations
- Constructed records remain candidates until validation passes.

## Required Inputs
- interpreted schema
- field mappings
- source values
- evidence metadata

## Expected Outputs
- candidate records
- record provenance
- missing field report

## Procedure
1. Validate schema.
2. Read mapped source values.
3. Construct records with provenance.
4. Mark missing or nullable fields.
5. Write candidate JSONL when requested.

## Required Tools
- json_schema_validate
- read_table_slice
- write_jsonl

## Validation Signals
- candidate_records_schema_valid
