TRACE2SKILL_LESSON_EXTRACTION_PROMPT = """Extract bounded, reusable skill-evolution lessons from compact execution traces.

Return JSON only. Do not include private reasoning. For each lesson include:
- evidence_summary
- reusable_principle
- proposed_delta
- target_skill_id if supported by the trace
- confidence between 0 and 1
- risk_level: low, medium, or high
- update_kind: skill_deepening or skill_creation

Keep lessons domain-generic and transferable across tasks.
"""


TRACE2SKILL_CONSOLIDATION_PROMPT = """Consolidate local skill patch proposals into conflict-free transferable updates.

Return JSON only. Prefer patches supported by multiple traces, reject unsupported tool additions,
and stage risky candidate-skill or relationship updates for policy review.
"""
