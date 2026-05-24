# Evidence-Aware Claim Classification

## Purpose
Classify scientific text spans as background, method, prediction, experimental observation, measured result, author conclusion, speculation, or unclear.

## Scope
Reusable evidence-aware classification for scientific text spans.

## Applicability
- Text spans may be used as evidence for extraction or validation.

## Limitations
- Does not determine domain correctness of extracted entities.

## Required Inputs
- text span
- section context
- task goal

## Expected Outputs
- claim class
- classification rationale
- uncertainty notes

## Procedure
1. Read candidate span and nearby context.
2. Classify the span by evidence role.
3. Flag unclear or speculative language.
4. Return classification with rationale.

## Required Tools
- read_text
- search_text

## Validation Signals
- claim_class_consistent_with_section_context
