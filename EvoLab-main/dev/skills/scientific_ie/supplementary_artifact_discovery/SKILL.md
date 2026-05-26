# Supplementary Artifact Discovery

## Purpose
Discover and prioritize supplementary artifacts likely to contain task-relevant evidence or structured data.

## Scope
Reusable discovery of supplementary artifacts for scientific IE.

## Applicability
- Supplementary files or references are present.

## Limitations
- Cannot inspect encrypted or unsupported artifacts beyond metadata.

## Required Inputs
- document package manifest
- task goal
- supplementary references

## Expected Outputs
- ranked supplementary artifacts
- artifact type guesses
- priority rationale

## Procedure
1. List supplementary files.
2. Inspect metadata and filenames.
3. Read lightweight text where possible.
4. Prioritize likely evidence artifacts.
5. Return artifact candidates.

## Required Tools
- list_files
- inspect_file_metadata
- read_text
- inspect_excel_workbook
- inspect_table

## Validation Signals
- task_relevant_artifacts_prioritized
