# Extraction Result Validation

## Purpose
Validate candidate records against schema, evidence requirements, domain validation results, and negative filtering results.

## Scope
Reusable validation of candidate scientific IE records.

## Applicability
- Candidate records and validation policies are available.

## Limitations
- Validation can flag issues but does not invent missing evidence.

## Required Inputs
- candidate records
- schema
- evidence requirements
- domain validation results
- negative filtering results

## Expected Outputs
- validated records
- record-level errors
- validation report

## Procedure
1. Validate candidate records against schema.
2. Check evidence requirements.
3. Incorporate domain validation and negative filters.
4. Write validation report.
5. Return accepted and rejected records.

## Required Tools
- json_schema_validate
- write_report

## Validation Signals
- validation_report_lists_accept_reject_reasons
