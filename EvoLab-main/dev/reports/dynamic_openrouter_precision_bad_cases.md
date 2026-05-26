# Dynamic OpenRouter Precision Bad Case Report

- Source run: `/tmp/evolab-3-biology-dynamic-openrouter-accuracy-20260521T094024Z`
- Note: user-provided `/tmp/evolab-biology-dynamic-openrouter-accuracy-20260521T094247` was not present; this report uses the latest completed 3-article dynamic run available in `/tmp`.
- GT was used only for post-run FP/FN analysis, not extraction.

## Per-Article Metrics

| Article | TP | FP | FN | Precision | Recall | F1 | Main FP categories |
|---|---:|---:|---:|---:|---:|---:|---|
| `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | 150 | 1479 | 0 | 0.092081 | 1.0 | 0.168634 | barcode/replicate assay detail table (1419), over-broad component table acceptance (60) |
| `ai_knowledge_sigma70_design` | 5 | 0 | 31 | 1.0 | 0.138889 | 0.243902 | none observed |
| `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | 4350 | 1723 | 0 | 0.716285 | 1.0 | 0.834693 | off-target promoter sub-library / wrong context for benchmark target (1722), internal reference/control sequence (1) |

## Stage Attribution

- Document preparation: no dominant FP source found; inventories and source paths were valid.
- Candidate collection: broad source discovery correctly found relevant sources but did not distinguish primary component tables from secondary assay/design tables.
- Table selection / triage: dominant source of FPs. All sequence-bearing tables were treated as extraction candidates.
- Extraction boundary: rows were accepted when they had a DNA-like promoter sequence field, even when surrounding table semantics indicated barcode/replicate assay detail, yeast/PolII sub-library, or control sequences.
- Validation: validator mainly checked format/provenance/DNA-likeness; it did not enforce target-component semantics strongly enough.
- Final artifact selection: aggregation preserved non-empty records correctly, but it had no additional precision gate.

## Representative False Positives

| # | Article | Component | Source | Sequence length | FP type | Why wrong | Generic prevention | Recall risk |
|---:|---|---|---|---:|---|---|---|---|
| 1 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU61-35` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 62` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 2 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU6_VSPH-77` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 103` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 3 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU61-52` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 104` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 4 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU61-20` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 105` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 5 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU61-13` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 106` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 6 | `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | `SynU61-43` | `41587_2025_2896_MOESM3_ESM.xlsx::Table_S1 row 107` | 249 | over-broad component table acceptance | row has a DNA-like sequence but insufficient target-component evidence beyond the sequence field | require acceptance_reason tied to component role/table title/source coordinates before finalization | low-to-medium depending on source table quality |
| 7 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `J23100` | `workbook.md::workbook.md::table-1 row 4351` | 35 | internal reference/control sequence | record is an internal reference/control, not a target extracted component set member | validator flags control/reference rows unless task explicitly asks for controls | low for benchmark component extraction; medium if controls are in scope |
| 8 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `SLP2019-1-2` | `workbook.md::workbook.md::table-2 row 1` | 325 | off-target promoter sub-library / wrong context for benchmark target | table belongs to a distinct sub-library/context rather than the primary target component set | prefer primary component table(s) per source and require table title/section context to match task target semantics | medium for multi-target articles; stage non-primary target tables instead of deleting when task asks for all targets |
| 9 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `SLP2019-1-6` | `workbook.md::workbook.md::table-2 row 2` | 325 | off-target promoter sub-library / wrong context for benchmark target | table belongs to a distinct sub-library/context rather than the primary target component set | prefer primary component table(s) per source and require table title/section context to match task target semantics | medium for multi-target articles; stage non-primary target tables instead of deleting when task asks for all targets |
| 10 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `SLP2019-1-7` | `workbook.md::workbook.md::table-2 row 3` | 325 | off-target promoter sub-library / wrong context for benchmark target | table belongs to a distinct sub-library/context rather than the primary target component set | prefer primary component table(s) per source and require table title/section context to match task target semantics | medium for multi-target articles; stage non-primary target tables instead of deleting when task asks for all targets |
| 11 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `SLP2019-1-10` | `workbook.md::workbook.md::table-2 row 4` | 325 | off-target promoter sub-library / wrong context for benchmark target | table belongs to a distinct sub-library/context rather than the primary target component set | prefer primary component table(s) per source and require table title/section context to match task target semantics | medium for multi-target articles; stage non-primary target tables instead of deleting when task asks for all targets |
| 12 | `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | `SLP2019-1-12` | `workbook.md::workbook.md::table-2 row 5` | 325 | off-target promoter sub-library / wrong context for benchmark target | table belongs to a distinct sub-library/context rather than the primary target component set | prefer primary component table(s) per source and require table title/section context to match task target semantics | medium for multi-target articles; stage non-primary target tables instead of deleting when task asks for all targets |

## Proposed Generic Fixes

1. Add a precision-oriented primary component table filter that keeps the best primary component table(s) per source and stages secondary assay/design/detail tables.
2. Propagate table-level context into candidate rows/records so validation can inspect table title, section, headers, and source semantics.
3. Strengthen validation from format-only to evidence-grounded acceptance: records require target component evidence, not just a DNA-like sequence.
4. Reject or stage prediction-only/generated-only tables, primer/scaffold/plasmid/construct rows, and DNA-like columns without component semantics.
5. Keep exact sequence deduplication and add canonical reverse-complement/format deduplication where safe.

