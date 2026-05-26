from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_promoter_sequences import (
    _article_ids_from_file,
    evaluate_sequences,
    normalize_sequence,
    reverse_complement,
)


def test_normalize_sequence_keeps_only_acgtn_uppercase() -> None:
    assert normalize_sequence(" aa tcg\nn--x ") == "AATCGN"


def test_normalize_sequence_ignores_label_prefixes() -> None:
    assert normalize_sequence("Promoter:ttcTAG") == "TTCTAG"


def test_reverse_complement() -> None:
    assert reverse_complement("AATCGN") == "NCGATT"


def test_evaluate_sequences_matches_exact_substring_and_reverse_complement(tmp_path: Path) -> None:
    predictions = [
        {"sequence": "AAATTT", "article_id": "paper-a"},
        {"sequence": "GGCC", "article_id": "paper-a"},
        {"sequence": "TTGG", "article_id": "paper-b"},
    ]
    ground_truth = [
        {"sequence": "AAATTT", "article_id": "paper-a"},
        {"sequence": "TTGGCCAA", "article_id": "paper-a"},
        {"sequence": "CCAA", "article_id": "paper-b"},
        {"sequence": "NNNN", "article_id": "paper-c"},
    ]

    result = evaluate_sequences(predictions, ground_truth)

    assert result.summary["gt_sequence_count"] == 4
    assert result.summary["predicted_sequence_count"] == 3
    assert result.summary["true_positive"] == 3
    assert result.summary["false_positive"] == 0
    assert result.summary["false_negative"] == 1
    assert result.false_negatives == [{"article_id": "paper-c", "sequence": "NNNN"}]
    assert {row["article_id"] for row in result.per_article_metrics} == {"paper-a", "paper-b", "paper-c"}


def test_evaluator_writes_required_artifacts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"sequence": "AAATTT", "article_id": "paper-a"}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "evaluation"

    result = evaluate_sequences(
        predictions=[{"sequence": "AAATTT", "article_id": "paper-a"}],
        ground_truth=[{"sequence": "AAATTT", "article_id": "paper-a"}],
        output_dir=output_dir,
        prediction_artifact=predictions,
    )

    expected = {
        "promoter_eval_summary.json",
        "promoter_eval_summary.md",
        "promoter_false_positives.jsonl",
        "promoter_false_negatives.jsonl",
        "promoter_per_article_metrics.csv",
        "normalized_predictions.jsonl",
        "normalized_ground_truth.jsonl",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    assert result.summary["precision"] == 1.0


def test_evaluator_can_filter_to_requested_article_ids(tmp_path: Path) -> None:
    result = evaluate_sequences(
        predictions=[
            {"sequence": "AAATTT", "article_id": "paper-a"},
            {"sequence": "GGGCCC", "article_id": "paper-b"},
        ],
        ground_truth=[
            {"sequence": "AAATTT", "article_id": "paper-a"},
            {"sequence": "TTTAAA", "article_id": "paper-b"},
            {"sequence": "CCCCGG", "article_id": "paper-c"},
        ],
        article_ids=["paper-a"],
    )

    assert result.summary["gt_sequence_count"] == 1
    assert result.summary["predicted_sequence_count"] == 1
    assert result.summary["precision"] == 1.0
    assert result.summary["recall"] == 1.0
    assert result.summary["article_filter"] == ["paper-a"]


def test_evaluator_aligns_slug_filter_to_title_article_ids(tmp_path: Path) -> None:
    result = evaluate_sequences(
        predictions=[
            {
                "sequence": "AAATTT",
                "article_id": "automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic",
                "source_file": (
                    "/tmp/Automated design of thousands of nonrepetitive parts for engineering stable genetic systems/"
                    "supplementary/table.xlsx"
                ),
            }
        ],
        ground_truth=[
            {
                "sequence": "AAATTT",
                "Publication": "Automated design of thousands of nonrepetitive parts for engineering stable genetic systems",
            },
            {"sequence": "GGGCCC", "Publication": "Some other article"},
        ],
        article_ids=[
            "automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic",
            "Automated design of thousands of nonrepetitive parts for engineering stable genetic systems",
        ],
    )

    assert result.summary["gt_sequence_count"] == 1
    assert result.summary["predicted_sequence_count"] == 1
    assert result.summary["true_positive"] == 1
    assert result.summary["precision"] == 1.0
    assert result.summary["recall"] == 1.0


def test_evaluator_matches_sequences_within_article_only() -> None:
    result = evaluate_sequences(
        predictions=[{"sequence": "AAATTT", "article_id": "paper-a"}],
        ground_truth=[
            {"sequence": "AAATTT", "article_id": "paper-b"},
            {"sequence": "GGGCCC", "article_id": "paper-a"},
        ],
    )

    assert result.summary["true_positive"] == 0
    assert result.summary["false_positive"] == 1
    assert result.summary["false_negative"] == 2


def test_article_ids_file_reads_numbered_subset_file(tmp_path: Path) -> None:
    subset = tmp_path / "subset.txt"
    subset.write_text(
        "# comment\n"
        "1. paper_a\n"
        "   /tmp/article A\n"
        "2. paper_b # trailing note\n",
        encoding="utf-8",
    )

    assert _article_ids_from_file(str(subset)) == ["paper_a", "article A", "paper_b"]
