# 28-Article Processed-Output Evaluation Report

## Evaluation Scope

- This is NOT a completed 28-article benchmark.
- This is a partial processed-output evaluation over articles that already had `validated_records.json` artifacts.
- Incomplete articles are excluded from the denominator because the run stopped on an OpenRouter HTTP 402 quota/balance failure.
- Lab root: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z`
- Config: `configs/tasks/biology_component_extraction_v1_28_article_work_items_dynamic_qwen30b.yaml`
- GT root: `/root/bio/Evaluation`

## Run Status

- Total planned articles: 28
- Processed / validated articles: 26
- Unfinished articles: 2
- Reason for interruption: OpenRouter HTTP 402 quota/balance failure.

### Unfinished Articles Excluded

- `characterization_context_dependent_effects_synthetic_promoters`
- `tn7_device_calibrated_heterologous_gene_expression_pseudomonas_putida`

## Matching Policy

Article-aligned evaluation over processed articles only. DNA sequences normalized by uppercasing/removing whitespace and deduplicated per article. Matching accepts exact, substring, reverse-complement exact, and reverse-complement substring matches.

## Processed-Output Metrics

| GT sequences | Predicted sequences | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 43026 | 63940 | 19111 | 44829 | 23915 | 0.298890 | 0.444173 | 0.357328 |

## Target Checks

- Precision >= 90%: no
- Recall >= 90%: no

## Per-Article Metrics

| Work item | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Validated records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | 150 | 209 | 150 | 59 | 0 | 0.717703 | 1.000000 | 0.835655 | 209 |
| `ai_knowledge_sigma70_design` | 36 | 5 | 5 | 0 | 31 | 1.000000 | 0.138889 | 0.243902 | 5 |
| `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | 4350 | 4351 | 4350 | 1 | 0 | 0.999770 | 1.000000 | 0.999885 | 4351 |
| `automated_model_predictive_design_synthetic_promoters_control_transcriptional` | 5391 | 25389 | 5389 | 20000 | 2 | 0.212257 | 0.999629 | 0.350162 | 25389 |
| `characterization_zymomonas_mobilis_promoters_that_are_functional_escherichia` | 3 | 0 | 0 | 0 | 3 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `characterization_divergent_promoters_pmaia_phyd_from_gordonia_co` | 2 | 0 | 0 | 0 | 2 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `characterization_endogenous_promoters_yarrowia_lipolytica_biomanufacturing_applications` | 74 | 0 | 0 | 0 | 74 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `composability_regulatory_sequences_controlling_transcription_translation_escherichia_coli` | 95 | 0 | 0 | 0 | 95 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `construction_characterization_mutant_library_p23_constitutive_promoter_lactic` | 246 | 0 | 0 | 0 | 246 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `construction_promoter_ribosome_binding_site_library_manipulating_gene` | 76 | 0 | 0 | 0 | 76 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `de_novo_promoter_design_method_deep_generative_dynamic` | 23 | 0 | 0 | 0 | 23 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `deep_learning_assisted_design_novel_promoters_escherichia_coli` | 298 | 50 | 49 | 1 | 249 | 0.980000 | 0.164430 | 0.281609 | 50 |
| `deep_flanking_sequence_engineering_efficient_promoter_design_deepseed` | 144 | 0 | 0 | 0 | 144 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `deep_learning_guided_programmable_design_escherichia_coli_core` | 80 | 10264 | 0 | 10264 | 80 | 0.000000 | 0.000000 | 0.000000 | 10264 |
| `design_deep_learning_synthetic_b_cell_specific_promoters` | 12095 | 9997 | 8 | 9989 | 12087 | 0.000800 | 0.000661 | 0.000724 | 9997 |
| `design_synthetic_promoters_cyanobacteria_generative_deep_learning_model` | 36 | 0 | 0 | 0 | 36 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `metagenomic_mining_regulatory_elements_enables_programmable_species_selective` | 15813 | 9968 | 5464 | 4504 | 10349 | 0.548154 | 0.345538 | 0.423878 | 9968 |
| `mix_match_promoters_terminators_tuning_gene_expression_methylotrophic` | 6 | 0 | 0 | 0 | 6 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `model_driven_generation_artificial_yeast_promoters` | 36 | 0 | 0 | 0 | 36 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `precise_prediction_promoter_strength_de_novo_synthetic_promoter` | 3663 | 3664 | 3663 | 1 | 0 | 0.999727 | 1.000000 | 0.999864 | 3664 |
| `precise_strength_prediction_endogenous_promoters_from_escherichia_coli` | 9 | 10 | 9 | 1 | 0 | 0.900000 | 1.000000 | 0.947368 | 10 |
| `quantitative_design_regulatory_elements_high_precision_strength_prediction` | 115 | 0 | 0 | 0 | 115 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `screening_broad_host_expression_promoters_shuttle_expression_vectors` | 10 | 0 | 0 | 0 | 10 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `synthetic_promoter_design_escherichia_coli_deep_generative_network` | 206 | 0 | 0 | 0 | 206 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `synthetic_promoter_multinomial_diffusion` | 24 | 33 | 24 | 9 | 0 | 0.727273 | 1.000000 | 0.842105 | 33 |
| `systematic_representation_optimization_enable_inverse_design_cross_species` | 45 | 0 | 0 | 0 | 45 | 0.000000 | 0.000000 | 0.000000 | 0 |

## Report Files

- JSON: `artifacts/eval_28_article_report_before.json`
- Bad cases markdown: `artifacts/eval_28_article_bad_cases_before.md`
- Bad cases JSON: `artifacts/eval_28_article_bad_cases_before.json`
