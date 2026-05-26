# 28-Article Processed Scope Sanity Check

This report verifies the partial processed-output evaluation scope before changing extraction/runtime logic.

| Check | Status | Evidence |
|---|---|---|
| Corrected subset has exactly 28 work item ids | PASS | 28 ids in artifacts/eval_28_article_subset.txt |
| Processed-only subset has article-level validated artifacts only | PASS | 26 article-level validated files, 26 evaluated articles |
| Two unfinished articles are excluded | PASS | characterization_context_dependent_effects_synthetic_promoters, tn7_device_calibrated_heterologous_gene_expression_pseudomonas_putida |
| Evaluator uses accepted/validated records only | PASS | Records were loaded from article-level artifacts/tools/<work_item_id>/validated_records.json accepted_records. |
| Evaluator does not read candidate artifacts | PASS | No candidate_records/extraction_candidates paths are used for processed-output metrics. |
| Evaluator does not read top-level aggregate validated_records.json | PASS | Top-level aggregate exists at /tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z/artifacts/tools/validated_records.json, but was intentionally excluded. |
| Evaluator does not mix global final jsonl | PASS | Global final_records.jsonl / biology_component_records.jsonl may exist, but were intentionally excluded from this processed-output evaluation. |
| Predictions are deduplicated per article by normalized sequence | PASS | Deduped predicted sequence count is reported; duplicate row counts are included per article. |
| Article matching is article-local | PASS | Each work_item_id was aligned to its configured article title; predictions were matched only to GT with the same title. |
| Matching policy includes exact, substring, reverse-complement | PASS | Article-aligned evaluation over processed articles only. DNA sequences normalized by uppercasing/removing whitespace and deduplicated per article. Matching accepts exact, substring, reverse-complement exact, and reverse-complement substring matches. |

## Artifact Counts

- Planned article ids: 28
- Evaluated processed article ids: 26
- Article-level validated artifacts found: 26
- Top-level aggregate validated artifact exists: True
- Top-level final records exists: True
- Top-level biology component records exists: True

## Unfinished Articles Excluded

- `characterization_context_dependent_effects_synthetic_promoters`
- `tn7_device_calibrated_heterologous_gene_expression_pseudomonas_putida`

## Duplicate Sequence Rows In Validated Artifacts

| Work item | Raw accepted records | Deduped sequences | Duplicate sequence rows |
|---|---:|---:|---:|
| `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | 209 | 209 | 0 |
| `ai_knowledge_sigma70_design` | 5 | 5 | 0 |
| `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | 4351 | 4351 | 0 |
| `automated_model_predictive_design_synthetic_promoters_control_transcriptional` | 25389 | 25389 | 0 |
| `characterization_zymomonas_mobilis_promoters_that_are_functional_escherichia` | 0 | 0 | 0 |
| `characterization_divergent_promoters_pmaia_phyd_from_gordonia_co` | 0 | 0 | 0 |
| `characterization_endogenous_promoters_yarrowia_lipolytica_biomanufacturing_applications` | 0 | 0 | 0 |
| `composability_regulatory_sequences_controlling_transcription_translation_escherichia_coli` | 0 | 0 | 0 |
| `construction_characterization_mutant_library_p23_constitutive_promoter_lactic` | 0 | 0 | 0 |
| `construction_promoter_ribosome_binding_site_library_manipulating_gene` | 0 | 0 | 0 |
| `de_novo_promoter_design_method_deep_generative_dynamic` | 0 | 0 | 0 |
| `deep_learning_assisted_design_novel_promoters_escherichia_coli` | 50 | 50 | 0 |
| `deep_flanking_sequence_engineering_efficient_promoter_design_deepseed` | 0 | 0 | 0 |
| `deep_learning_guided_programmable_design_escherichia_coli_core` | 10264 | 10264 | 0 |
| `design_deep_learning_synthetic_b_cell_specific_promoters` | 9997 | 9997 | 0 |
| `design_synthetic_promoters_cyanobacteria_generative_deep_learning_model` | 0 | 0 | 0 |
| `metagenomic_mining_regulatory_elements_enables_programmable_species_selective` | 9968 | 9968 | 0 |
| `mix_match_promoters_terminators_tuning_gene_expression_methylotrophic` | 0 | 0 | 0 |
| `model_driven_generation_artificial_yeast_promoters` | 0 | 0 | 0 |
| `precise_prediction_promoter_strength_de_novo_synthetic_promoter` | 3664 | 3664 | 0 |
| `precise_strength_prediction_endogenous_promoters_from_escherichia_coli` | 10 | 10 | 0 |
| `quantitative_design_regulatory_elements_high_precision_strength_prediction` | 0 | 0 | 0 |
| `screening_broad_host_expression_promoters_shuttle_expression_vectors` | 0 | 0 | 0 |
| `synthetic_promoter_design_escherichia_coli_deep_generative_network` | 0 | 0 | 0 |
| `synthetic_promoter_multinomial_diffusion` | 33 | 33 | 0 |
| `systematic_representation_optimization_enable_inverse_design_cross_species` | 0 | 0 | 0 |
