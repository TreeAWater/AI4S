# Domain Ontology Alignment

## Purpose
Align extracted mentions or record fields to task-provided ontology concepts, entity types, or controlled vocabulary.

## Scope
Reusable alignment to task-provided domain ontologies.

## Applicability
- A domain package provides ontology concepts or controlled vocabulary.

## Limitations
- Does not hardcode domain-specific ontology content.

## Required Inputs
- candidate records or mentions
- domain ontology
- task schema

## Expected Outputs
- ontology alignments
- unmatched mentions
- alignment confidence

## Procedure
1. Read ontology concepts.
2. Compare record fields or mentions to ontology labels.
3. Assign entity types or concepts.
4. Report unmatched or ambiguous mentions.

## Required Tools
- read_text
- json_schema_validate

## Validation Signals
- aligned_entities_reference_task_ontology
