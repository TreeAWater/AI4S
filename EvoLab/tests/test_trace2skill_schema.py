from evolab.backends.skills.trace2skill.schema import (
    ConsolidatedSkillPatch,
    PatchConsolidationResult,
    SkillPatchProposal,
    Trace2SkillRunConfig,
    Trace2SkillRunResult,
    TracePool,
    TraceRecord,
    TrajectoryLesson,
)


def test_trace2skill_core_models_serialize():
    trace = TraceRecord(
        trace_id="trace-1",
        task_id="task-1",
        task_summary="generic task",
        selected_skill_ids=["skill.generic.v1"],
        tools_used=["read_text"],
        final_status="runtime_success",
        compact_execution_summary="done",
        created_at="2026-01-01T00:00:00+00:00",
    )
    pool = TracePool(
        pool_id="pool-1",
        traces=[trace],
        success_traces=[trace],
        stats={"trace_count": 1},
        created_at=trace.created_at,
    )
    lesson = TrajectoryLesson(
        lesson_id="lesson-1",
        source_trace_ids=[trace.trace_id],
        lesson_type="success_lesson",
        target_skill_id="skill.generic.v1",
        evidence_summary="worked",
        reusable_principle="reuse successful pattern",
        proposed_delta={"example_summary": "worked"},
        confidence=0.8,
    )
    patch = SkillPatchProposal(
        patch_id="patch-1",
        patch_type="example_memory_patch",
        target_skill_id="skill.generic.v1",
        source_lesson_ids=[lesson.lesson_id],
        source_trace_ids=[trace.trace_id],
        patch_content={"example_summary": "worked"},
        evidence_summary="worked",
        confidence=0.8,
        risk_level="low",
        created_at=trace.created_at,
    )
    consolidated = ConsolidatedSkillPatch(
        consolidated_patch_id="consolidated-1",
        patch_type="example_memory_patch",
        target_skill_id="skill.generic.v1",
        merged_content=patch.patch_content,
        source_patch_ids=[patch.patch_id],
        source_lesson_ids=[lesson.lesson_id],
        source_trace_ids=[trace.trace_id],
        confidence=0.8,
        risk_level="low",
        created_at=trace.created_at,
    )
    consolidation = PatchConsolidationResult(
        result_id="result-1",
        consolidated_patches=[consolidated],
        stats={"consolidated_patch_count": 1},
    )
    result = Trace2SkillRunResult(
        run_id="run-1",
        config=Trace2SkillRunConfig(dry_run=True),
        trace_pool_summary=pool.stats,
        lessons=[lesson],
        local_patches=[patch],
        consolidation_result=consolidation,
        status="dry_run",
    )

    assert trace.model_dump(mode="json")["tools_used"] == ["read_text"]
    assert pool.stats["trace_count"] == 1
    assert consolidated.model_dump(mode="json")["source_trace_ids"] == ["trace-1"]
    assert result.model_dump(mode="json")["status"] == "dry_run"
