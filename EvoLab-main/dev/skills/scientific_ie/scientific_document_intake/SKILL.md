# Scientific Document Intake

## Purpose
Map a scientific document package into main text, supplementary files, table-like artifacts, unreadable files, and recommended next actions.

## Scope
Reusable intake for scientific document packages before task-specific extraction.

## Applicability
- A task provides a paper, document bundle, or supplementary package.

## Limitations
- Does not infer task-specific records or domain entities.

## Required Inputs
- document package path or manifest
- task goal

## Expected Outputs
- main text reference
- supplementary file list
- table-like artifacts
- unreadable file report
- recommended next actions

## Procedure
1. List package files.
2. Identify likely main text and supplementary artifacts.
3. Inspect file metadata.
4. Report unreadable or unsupported files.
5. Recommend next reading actions.

## Required Tools
- list_files
- read_text
- inspect_file_metadata

## Validation Signals
- main_text_found
- supplementary_manifest_complete
- unreadable_files_reported
