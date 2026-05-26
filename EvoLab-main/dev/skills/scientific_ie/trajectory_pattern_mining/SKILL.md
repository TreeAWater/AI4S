# Trajectory Pattern Mining

## Purpose
Mine reusable patterns, failure-repair patterns, tool-use patterns, and candidate skill suggestions from subagent trajectories and tool traces.

## Scope
Reusable mining of trajectory patterns for future skill improvement.

## Applicability
- Subagent trajectories and tool traces are available.

## Limitations
- Does not directly mutate the skill graph without review and update policy.

## Required Inputs
- trajectory records
- tool traces
- validation outcomes

## Expected Outputs
- reusable patterns
- failure-repair patterns
- candidate skill suggestions

## Procedure
1. Read trajectory and tool trace records.
2. Identify recurring successful patterns.
3. Identify recurring failures and repairs.
4. Summarize candidate skill updates.
5. Write pattern report.

## Required Tools
- read_text
- write_report

## Validation Signals
- pattern_report_links_to_source_runs
