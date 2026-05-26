Post-run evaluator rubric for promoter sequence extraction:

1. Scope:
   - Evaluate only the article_id and article_title in the ground_truth payload.
   - Do not penalize unrelated or unprocessed articles.
   - Ground truth is evaluator-only feedback, not extraction input.

2. Matching policy:
   - Normalize DNA sequences by uppercasing and removing whitespace/hyphens.
   - Match predictions only against ground truth for the same article.
   - Count exact matches, substring matches, exact reverse-complement matches,
     and reverse-complement substring matches as true positives.
   - Deduplicate normalized sequences per article before scoring.

3. Metrics:
   - precision = TP / (TP + FP), or 0 if no predictions.
   - recall = TP / (TP + FN), or 0 if no ground truth is matched.
   - f1 = 2PR / (P + R), or 0 if P + R is 0.
   - score should be the F1 value when metrics are computable.
   - If runtime-computed metrics are supplied, use them exactly. Do not
     replace them with model-estimated counts.

4. Passing threshold:
   - passed is true only if precision >= 0.90 and recall >= 0.90.

5. Feedback:
   - Attribute errors to generic stages: source discovery, table triage,
     candidate extraction, validation, final serialization, or aggregation.
   - Use runtime-provided FP/FN examples to make feedback concrete. Include
     example sequences only as post-run diagnostics, not as extraction rules.
   - If candidate or validated artifacts contain more useful records than the
     final accepted artifact, call out artifact handoff/final selection as the
     likely bottleneck.
   - Specific evolution instructions should be reusable and stage-targeted:
     state what skill/tool/runtime behavior should change, why, and which
     bounded diagnostic examples support it.
   - Recommend reusable improvements only. Do not recommend article-specific,
     sequence-specific, or GT-leaking extraction rules.
