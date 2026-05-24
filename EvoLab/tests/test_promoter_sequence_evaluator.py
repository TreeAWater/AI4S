from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_promoter_sequences import evaluate_sequences, normalize_sequence, reverse_complement


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
