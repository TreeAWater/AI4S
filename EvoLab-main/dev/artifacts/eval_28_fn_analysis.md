# 28-Article False-Negative Analysis After Source-Gated Debug

## Low Recall Articles

| Article | GT | Pred | TP | FN | Recall | Review | Rejected | Likely generic cause |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `metagenomic_mining_regulatory_elements_enables_programmable_species_selective` | 15813 | 1408 | 837 | 14976 | 0.052931 | 7558 | 1012 | selected accepted source covers only a subset of GT or valid candidates were routed to review/rejected |
| `precise_prediction_promoter_strength_de_novo_synthetic_promoter` | 3663 | 0 | 0 | 3663 | 0.000000 | 7089 | 0 | source gate/recovery produced zero accepted records |
| `deep_learning_assisted_design_novel_promoters_escherichia_coli` | 298 | 0 | 0 | 298 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `construction_characterization_mutant_library_p23_constitutive_promoter_lactic` | 246 | 0 | 0 | 246 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `synthetic_promoter_design_escherichia_coli_deep_generative_network` | 206 | 0 | 0 | 206 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `deep_flanking_sequence_engineering_efficient_promoter_design_deepseed` | 144 | 0 | 0 | 144 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `quantitative_design_regulatory_elements_high_precision_strength_prediction` | 115 | 0 | 0 | 115 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `composability_regulatory_sequences_controlling_transcription_translation_escherichia_coli` | 95 | 0 | 0 | 95 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `construction_promoter_ribosome_binding_site_library_manipulating_gene` | 76 | 0 | 0 | 76 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `characterization_endogenous_promoters_yarrowia_lipolytica_biomanufacturing_applications` | 74 | 0 | 0 | 74 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `model_driven_generation_artificial_yeast_promoters` | 36 | 0 | 0 | 36 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `synthetic_promoter_multinomial_diffusion` | 24 | 0 | 0 | 24 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `de_novo_promoter_design_method_deep_generative_dynamic` | 23 | 0 | 0 | 23 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `screening_broad_host_expression_promoters_shuttle_expression_vectors` | 10 | 0 | 0 | 10 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `mix_match_promoters_terminators_tuning_gene_expression_methylotrophic` | 6 | 0 | 0 | 6 | 0.000000 | 6 | 0 | source gate/recovery produced zero accepted records |
| `characterization_zymomonas_mobilis_promoters_that_are_functional_escherichia` | 3 | 0 | 0 | 3 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `characterization_divergent_promoters_pmaia_phyd_from_gordonia_co` | 1 | 0 | 0 | 1 | 0.000000 | 0 | 0 | source gate/recovery produced zero accepted records |
| `ai_knowledge_sigma70_design` | 36 | 5 | 5 | 31 | 0.138889 | 0 | 0 | selected accepted source covers only a subset of GT or valid candidates were routed to review/rejected |
| `design_synthetic_promoters_cyanobacteria_generative_deep_learning_model` | 36 | 16 | 16 | 20 | 0.444444 | 0 | 0 | selected accepted source covers only a subset of GT or valid candidates were routed to review/rejected |

## FN Examples

- `ai_knowledge_sigma70_design` `ATTATGTTCAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGGGAGCA`
- `ai_knowledge_sigma70_design` `AATAAATTAAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `TATGTCATTAATTCTTGACAGCTTGCTTAGTCCTAGGTATAATCCTAGCA`
- `ai_knowledge_sigma70_design` `GTCAAAATAAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATACAAGCA`
- `ai_knowledge_sigma70_design` `GTAAAAAAAAATTCTTGACAGCTAGCTCAGTCCTTGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `CTAAAAAGTAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `CTTTTGTTTATTACTTGACAGCTAGCTTAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `ATAAAAATCAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `AATTACATAAATTCTTGACAGCTAGCTCAGTCCTAGCTATAATGCTACCA`
- `ai_knowledge_sigma70_design` `ATTGAGTTAAATTCTTGACAGCTTGCTTAGTCCTACGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `GTAGATTTTAATTCTTGACAGCTAGGTCACACCTAGGTATAATGCCAGCA`
- `ai_knowledge_sigma70_design` `CGTAAAAGAAACTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `CTGAAAAGAAATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `AATCAATTTAATTCTTGACAGCTAGATCAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `AAACTGATTAATTCTTGACAGCTAGCTCAGCCCTAGGTATAATGCTACCA`
- `ai_knowledge_sigma70_design` `AATCTTTTTAATGCTTGACAGCTAGCTCGGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `CATAAATGTAATCCTTGACAGCTAGCTCGGGCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `GGCGTATTTAATTCTTGACAGCTATATCAGTCCTAGCTATAATGCGAGCA`
- `ai_knowledge_sigma70_design` `CAAAAAATTAATTCATGACAGATGGCTTAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `ATTGTTTTAAATTCTTGACAGCTAGCTTAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `TTGAAAAGTAATTCTTGACAGCTAGCTCAATGCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `AGTAAAATAAATTCTTGACAGCTAGCTTAGTCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `AAACACTTTAATTCTTGACAGCTGGCTCACGCCTAGGTATAATGCTAGCA`
- `ai_knowledge_sigma70_design` `TTATATATATTTTTTTGAGCGTTATCTAGATCCTAGTTATAATGCTCTCC`
- `ai_knowledge_sigma70_design` `AAGAAAATTAATTCTTGACAGGTAGCTCAGTCCTACGTATAATGCTAGCA`
