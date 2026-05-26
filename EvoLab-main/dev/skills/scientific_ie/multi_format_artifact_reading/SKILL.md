# Multi-format Artifact Reading

## Purpose
Read scientific artifacts into normalized provenance-preserving artifact views.

## Scope
Reusable reading of text, spreadsheet, and table-like scientific artifacts.

## Applicability
- A task-relevant artifact was discovered.

## Limitations
- Does not interpret domain semantics without task schema and policy.

## Required Inputs
- artifact reference
- artifact metadata
- task goal

## Expected Outputs
- normalized artifact view
- provenance metadata
- reading warnings

## Procedure
1. Inspect artifact metadata.
2. Select an appropriate reader.
3. Read text, sheets, or table slices.
4. Normalize into provenance-preserving views.
5. Report reading warnings.

## Required Tools
- inspect_file_metadata
- read_text
- inspect_excel_workbook
- read_excel_sheet
- inspect_table
- read_table_slice

## Validation Signals
- artifact_views_have_source_metadata
