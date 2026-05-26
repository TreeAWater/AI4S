# LABBench2 Prop50 Small Evaluation Slice

Created: 2026-05-26

This is a fixed 50-question LABBench2 API smoke slice. It is intentionally close to balanced across top-level tags while giving extra room to `seqqa2`.

Selection rule:

- `seqqa2`: 8 questions, one from each selected validator type.
- All other top-level tags: 3 questions each.
- `cloning`: 3 questions selected to cover all three cloning subtypes.
- Source dataset: `EdisonScientific/labbench2`, config `all`, split `train`.
- IDs are globally unique in the dataset.
- Preferred run mode: `file`, because every file-backed item in this slice supports `file`; text-only items also run under `file`.

Use the companion IDs file:

```bash
eval_slices/labbench2_prop50_ids.txt
```

Example command template:

```bash
cd benchmark_repo/labbench2
conda run -n labbench2 python -m evals.run_evals \
  --agent native:openai-responses:<MODEL>@tools,low \
  --ids-file ../../eval_slices/labbench2_prop50_ids.txt \
  --mode file \
  --parallel 2 \
  --judge-model openai:gpt-5.4-mini@low \
  --report-path assets/reports/smoke/prop50_<MODEL>.json
```

## ID Table

| # | id | tag | type | supported_modes | has_files |
|---:|---|---|---|---|---|
| 1 | fb8fc27d-592a-40e8-a65f-9e1a60b7a708 | cloning | gibson | file, inject, retrieve | True |
| 2 | 21e4def0-1c9f-4628-8724-81db65cdb7f8 | cloning | golden-gate | file, inject, retrieve | True |
| 3 | ae62bcdb-197b-4815-991f-cb7a9c151ff6 | cloning | restriction-ligation | file, inject, retrieve | True |
| 4 | a1dae0d3-53c6-4d56-9e39-625cc1d32fa9 | dbqa2 |  | text-only | False |
| 5 | 3badfda1-1a86-4295-9ace-799aabdee469 | dbqa2 |  | text-only | False |
| 6 | 495aa4d1-576d-4553-9bf5-2ba84ca3e47b | dbqa2 |  | text-only | False |
| 7 | b9ba0817-f8c1-4817-8293-c71aa0d6efec | figqa2 |  | text-only | False |
| 8 | 4bbdabcd-920c-4e1a-b11b-eb948603a1d3 | figqa2 |  | text-only | False |
| 9 | b60fdf79-25b2-4bf2-a5bb-cb553d83770f | figqa2 |  | text-only | False |
| 10 | b60fdf79-25b2-4bf2-a5bb-cb553d83770f-img | figqa2-img |  | file | True |
| 11 | d9daf6c6-513a-4969-b5a6-86a1bd504c6c-img | figqa2-img |  | file | True |
| 12 | ec7abee3-61da-44d9-a4fc-1d7f540a60b0-img | figqa2-img |  | file | True |
| 13 | b60fdf79-25b2-4bf2-a5bb-cb553d83770f-pdf | figqa2-pdf |  | file | True |
| 14 | d9daf6c6-513a-4969-b5a6-86a1bd504c6c-pdf | figqa2-pdf |  | file | True |
| 15 | ec7abee3-61da-44d9-a4fc-1d7f540a60b0-pdf | figqa2-pdf |  | file | True |
| 16 | e3b5a4af-41d9-48db-becf-29a08d0ad28e | litqa3 |  | text-only | False |
| 17 | 76184ccf-4bf0-469e-a442-11d04b4ff8b0 | litqa3 |  | text-only | False |
| 18 | 39129e1c-096f-4414-bf4f-37fadbbe364c | litqa3 |  | text-only | False |
| 19 | dcbf1eb1-f1c1-4043-a40c-50f738c6c994 | patentqa |  | text-only | False |
| 20 | 5bf921b7-be55-4148-bbb8-b7d6181c9a16 | patentqa |  | text-only | False |
| 21 | 01c3e29d-81d9-488b-bf90-2c9f78e7de6b | patentqa |  | text-only | False |
| 22 | a68f494c-50de-4200-b12b-82108e9c1d8e | protocolqa2 |  | file | True |
| 23 | e0759c5d-f4eb-4bb5-850e-55a0adaede9d | protocolqa2 |  | file | True |
| 24 | 74c30601-5725-45d0-8472-6711f96b5c1a | protocolqa2 |  | file | True |
| 25 | a56306f0-676c-4e7f-94f8-5b0ce7f94448 | sourcequality |  | file | True |
| 26 | b79d5cad-ca69-49c9-b2a2-72d5077ef6f2 | sourcequality |  | file | True |
| 27 | 35e97565-b048-43bb-a4d3-6a8888ebad3d | sourcequality |  | file | True |
| 28 | 70be3149-5beb-443f-bfe8-cf14da0dd59c | suppqa2 |  | text-only | False |
| 29 | 3bfefaf7-f9c6-453d-8f2e-7fc9b5d2a6bc | suppqa2 |  | text-only | False |
| 30 | 2b166d76-eb57-448f-b031-06edee42fa13 | suppqa2 |  | text-only | False |
| 31 | 28cddb99-558a-41e5-9a83-46c8ed73c4f8 | tableqa2 |  | text-only | False |
| 32 | 867f2c1c-8849-43d1-a9d4-6905214031fc | tableqa2 |  | text-only | False |
| 33 | cf2a4612-2673-443b-9dae-e07c640450c0 | tableqa2 |  | text-only | False |
| 34 | 54f637f5-27cf-450b-8d9c-6069006d15d0-img | tableqa2-img |  | file | True |
| 35 | 37f51984-8119-4a55-bca4-ec11018dcd2f-img | tableqa2-img |  | file | True |
| 36 | 899534c8-e247-4ebf-86ad-c00bd8cd4fe6-img | tableqa2-img |  | file | True |
| 37 | 54f637f5-27cf-450b-8d9c-6069006d15d0-pdf | tableqa2-pdf |  | file | True |
| 38 | 37f51984-8119-4a55-bca4-ec11018dcd2f-pdf | tableqa2-pdf |  | file | True |
| 39 | 899534c8-e247-4ebf-86ad-c00bd8cd4fe6-pdf | tableqa2-pdf |  | file | True |
| 40 | 93f1229d-822e-4aa5-90d6-9c459c32dd0c | trialqa |  | text-only | False |
| 41 | d2e4fced-3f42-415e-be71-19ed67c56b59 | trialqa |  | text-only | False |
| 42 | 425e0a08-36cc-41dc-b44f-44da1840a729 | trialqa |  | text-only | False |
| 43 | 7b9689fb-35de-48a8-93b4-109172c3b870 | seqqa2 | gc_content | file, inject, retrieve | True |
| 44 | d7b3cad5-62aa-4aa6-a71a-b4f854ecb77e | seqqa2 | amplicon_length | file, inject, retrieve | True |
| 45 | f46ab5a5-5559-4829-a12c-0051fa54a967 | seqqa2 | tm_calculations | file, inject | True |
| 46 | 610f7342-ad5a-4b30-9b6d-3dbaf99ea4f0 | seqqa2 | molecular_weight | file, inject | True |
| 47 | 44f27ccc-0f2b-4f87-97b3-b3a9db1d069b | seqqa2 | restriction_digest | file, inject, retrieve | True |
| 48 | 792ebbe7-c49b-4447-b186-0ccf64e31188 | seqqa2 | primer_design | file, inject, retrieve | True |
| 49 | fbf7bc19-e357-4214-817b-bca44f88bf3a | seqqa2 | gibson_primers | file, inject, retrieve | True |
| 50 | 474c1805-3562-4de0-8471-30ee84a79ff9 | seqqa2 | msa_scoring | file, inject | True |
