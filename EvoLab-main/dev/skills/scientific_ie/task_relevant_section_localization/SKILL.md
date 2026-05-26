# Task-Relevant Section Localization

## Purpose
Rank paper sections by task relevance using task goal, schema, section type, and supplementary references.

## Scope
Reusable localization of paper sections likely to contain task-relevant evidence.

## Applicability
- A task goal or extraction schema is available with parsed sections.

## Limitations
- May miss evidence only present in figures or unsupported artifacts.

## Required Inputs
- task goal
- optional extraction schema
- section map
- supplementary references

## Expected Outputs
- ranked relevant sections
- rationale
- supplementary follow-up targets

## Procedure
1. Read task goal and schema.
2. Score sections by goal terms, section type, and references.
3. Prioritize supplementary follow-up targets.
4. Return ranked sections with rationale.

## Required Tools
- read_text
- search_text

## Validation Signals
- ranked_sections_include_expected_evidence
