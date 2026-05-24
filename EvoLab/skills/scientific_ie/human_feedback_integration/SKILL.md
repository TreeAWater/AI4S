# Human Feedback Integration

## Purpose
Normalize human feedback into corrections, constraints, review signals, and possible skill update signals.

## Scope
Reusable integration of human feedback into structured review and skill update signals.

## Applicability
- Human feedback is available for a task, trajectory, or extraction output.

## Limitations
- Does not decide promotion or graph mutation by itself.

## Required Inputs
- human feedback text
- related task or trajectory refs
- candidate outputs

## Expected Outputs
- normalized corrections
- constraints
- review signals
- candidate skill update signals

## Procedure
1. Read feedback text.
2. Classify corrections and constraints.
3. Link feedback to outputs or trajectories.
4. Summarize possible skill update signals.
5. Write feedback integration report.

## Required Tools
- read_text
- write_report

## Validation Signals
- feedback_report_links_to_source_outputs
