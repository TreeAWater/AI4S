# Ground Truth Based Evaluation

## Purpose
Compare predicted records with ground truth using task-provided evaluation schema and matching policy.

## Scope
Reusable ground-truth evaluation for scientific IE records.

## Applicability
- Ground truth records and matching policy are available.

## Limitations
- Evaluation quality depends on task-provided matching policy.

## Required Inputs
- predicted records
- ground truth records
- evaluation schema
- matching policy

## Expected Outputs
- evaluation metrics
- match report
- error categories

## Procedure
1. Validate evaluation schema.
2. Match predictions to ground truth.
3. Compute task metrics.
4. Categorize errors.
5. Write evaluation report.

## Required Tools
- json_schema_validate
- write_report

## Validation Signals
- evaluation_report_contains_metrics_and_matches
