# Final Artifact Writing

## Purpose
Write final structured JSONL records and report artifacts from validated upstream content without inventing records.

## Scope
Reusable final artifact writing for scientific information extraction tasks.

## Applicability
- Validated records or an explicit empty-output audit are available.
- The final artifact names and formats are known.

## Limitations
- Does not extract new evidence.
- Does not validate unsupported records by invention.
- Does not silently convert narrative summaries into records.

## Required Inputs
- validated records or empty-output audit
- coverage summary
- failure and skipped-item summaries
- target artifact names

## Expected Outputs
- final JSONL records artifact
- final report artifact
- coverage and failure audit

## Procedure
1. Read validated upstream content and artifact references.
2. If accepted records exist, pass a list of record objects to `write_jsonl`.
3. If no accepted records exist, pass an empty list to `write_jsonl` and write an explicit empty-output audit report.
4. Write a report with coverage, skipped items, failures, rejected records, and final artifact paths.
5. Do not call table-reading, extraction, validation, or ontology tools during final writing.

## Required Tools
- write_jsonl
- write_report

## Validation Signals
- final_jsonl_contains_only_json_objects
- report_links_final_artifacts
- empty_output_has_audit
