# Schema-Guided Field Mapping

## Purpose
Map source fields, table columns, text spans, or artifact metadata to schema fields.

## Scope
Reusable schema-guided mapping from sources to extraction fields.

## Applicability
- Schema fields and candidate text/table sources are available.

## Limitations
- Mappings require downstream validation before becoming final records.

## Required Inputs
- interpreted schema
- source spans or table structure
- artifact metadata

## Expected Outputs
- field-source mappings
- mapping confidence
- unmapped fields
- source warnings

## Procedure
1. Validate interpreted schema.
2. Inspect source columns or text spans.
3. Match source labels and context to schema fields.
4. Record confidence and unmapped fields.
5. Return mapping candidates.

## Required Tools
- json_schema_validate
- inspect_table
- read_table_slice
- search_text
- profile_table

## Validation Signals
- mapped_fields_have_sources
