# Evidence Source Attribution

## Purpose
Attach traceable evidence source metadata to claims, fields, or records.

## Scope
Reusable provenance attribution for extracted scientific claims and records.

## Applicability
- A field, claim, or record needs traceable source metadata.

## Limitations
- Cannot attribute evidence if source spans or table coordinates are unavailable.

## Required Inputs
- claim or field
- source span or table coordinates
- artifact metadata

## Expected Outputs
- evidence source reference
- source type
- location metadata
- confidence note

## Procedure
1. Identify source artifact.
2. Attach section, span, table, row, or column coordinates.
3. Record source type and evidence role.
4. Return provenance metadata.

## Required Tools
- read_text
- search_text
- read_table_slice
- inspect_table

## Validation Signals
- evidence_source_resolves_to_artifact
