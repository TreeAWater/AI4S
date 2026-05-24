# Record Deduplication and Conflict Resolution

## Purpose
Deduplicate records and resolve conflicting fields while preserving evidence.

## Scope
Reusable record deduplication and conflict resolution for scientific IE outputs.

## Applicability
- Multiple candidate records may refer to the same scientific entity or event.

## Limitations
- Requires evidence preservation to resolve conflicts transparently.

## Required Inputs
- validated records
- deduplication policy
- evidence metadata

## Expected Outputs
- deduplicated records
- conflict report
- merged evidence references

## Procedure
1. Validate deduplication policy.
2. Group likely duplicate records.
3. Resolve field conflicts using evidence.
4. Preserve alternative evidence.
5. Write conflict report.

## Required Tools
- json_schema_validate
- write_report

## Validation Signals
- deduplicated_records_preserve_evidence
