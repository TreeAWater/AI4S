# 28-Article Processed-Output Before/After Debug Comparison

| Metric | Before source-gated debug | After source-gated debug | Delta |
|---|---:|---:|---:|
| gt_sequence_count | 43026 | 43025 | -1 |
| predicted_sequence_count | 61421 | 23352 | -38069 |
| true_positive | 24751 | 19906 | -4845 |
| false_positive | 36670 | 3446 | -33224 |
| false_negative | 18275 | 23119 | 4844 |
| precision | 0.402973 | 0.852432 | 0.44945899999999994 |
| recall | 0.575257 | 0.462661 | -0.11259600000000003 |
| f1 | 0.473944 | 0.599786 | 0.12584200000000006 |

The latest change is a generic source/table classifier and final accepted-record gate. It does not use article-specific titles, sequences, or GT during extraction.
