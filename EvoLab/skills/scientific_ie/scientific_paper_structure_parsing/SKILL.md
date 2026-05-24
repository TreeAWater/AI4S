# Scientific Paper Structure Parsing

## Purpose
Parse main text into normalized paper sections and supplementary references.

## Scope
Reusable section parsing for scientific papers.

## Applicability
- Main scientific text is available as text.

## Limitations
- Section labels can be ambiguous in poorly formatted documents.

## Required Inputs
- main text
- document metadata

## Expected Outputs
- normalized section map
- section spans
- supplementary references

## Procedure
1. Read main text.
2. Extract section headings.
3. Normalize common section names.
4. Locate supplementary references.
5. Return section spans and warnings.

## Required Tools
- read_text
- extract_sections
- search_text

## Validation Signals
- section_spans_non_overlapping
- supplementary_refs_found
