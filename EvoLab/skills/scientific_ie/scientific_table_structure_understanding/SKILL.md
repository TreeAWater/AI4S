# Scientific Table Structure Understanding

## Purpose
Understand table layout, headers, data region, annotation rows, column profiles, and structural warnings.

## Scope
Reusable table structure understanding for scientific IE.

## Applicability
- A table-like artifact may contain extraction evidence.

## Limitations
- Does not decide task-specific field semantics by itself.

## Required Inputs
- table artifact reference
- task goal
- optional schema fields

## Expected Outputs
- header rows
- data region
- column profiles
- annotation rows
- structural warnings

## Procedure
1. Inspect table dimensions.
2. Detect header and data regions.
3. Profile columns.
4. Identify annotation rows and merged header risks.
5. Return structural view and warnings.

## Required Tools
- inspect_table
- read_table_slice
- detect_table_header
- normalize_table
- profile_table

## Validation Signals
- table_structure_has_header_and_data_region
