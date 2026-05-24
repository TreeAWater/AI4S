import json
from pathlib import Path

from evolab.tools.scientific_artifacts import (
    build_candidate_records,
    extract_candidate_rows,
    serialize_final_records,
    validate_candidate_records,
)


def test_prediction_only_table_does_not_enter_final_records(tmp_path: Path):
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"predicted_value": "14.7", "generated_sequences": "AAACCCGGGTTT"},
                sequence_field="generated_sequences",
                sequence="AAACCCGGGTTT",
            )
        ],
    )

    records = build_candidate_records(
        {
            "candidate_rows_path": str(candidate_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["record_count"] == 0
    assert payload["skipped_counts"]["no_profile_compatible_sequence_field"] == 1


def test_primer_scaffold_plasmid_construct_sequences_are_rejected(tmp_path: Path):
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"primer_name": "qPCR-F", "primer_sequence": "AAACCCGGGTTT"},
                sequence_field="primer_sequence",
                sequence="AAACCCGGGTTT",
            ),
            _candidate_row(
                tmp_path,
                row={"construct_id": "c1", "construct_sequence": "TTTCCCGGGAAA"},
                sequence_field="construct_sequence",
                sequence="TTTCCCGGGAAA",
            ),
        ],
    )

    records = build_candidate_records(
        {
            "candidate_rows_path": str(candidate_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["record_count"] == 0
    assert payload["skipped_counts"]["no_profile_compatible_sequence_field"] == 2


def test_dna_like_column_without_target_component_semantics_is_not_accepted(tmp_path: Path):
    source = tmp_path / "supplement.tsv"
    source.write_text("id,sequence\nrow1,AAACCCGGGTTT\n", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "row1",
                        "component_type": "sequence_component",
                        "sequence": "AAACCCGGGTTT",
                        "sequence_field": "sequence",
                        "source_file": str(source),
                        "evidence_source": {"path": str(source), "row_index": 2},
                        "evidence_text": '{"id":"row1","sequence":"AAACCCGGGTTT"}',
                        "acceptance_reason": "sequence-like value with provenance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validated = validate_candidate_records(
        {
            "candidate_records_path": str(records_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(validated.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 0
    assert payload["rejected_count"] == 1
    assert "missing target-component semantics" in payload["rejected_records"][0]["validation_issues"][0]


def test_explicit_target_component_evidence_is_retained(tmp_path: Path):
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"promoter_id": "P1", "promoter_sequence": "AAACCCGGGTTT", "activity": "measured"},
                sequence_field="promoter_sequence",
                sequence="AAACCCGGGTTT",
                label_field="promoter_id",
                label="P1",
            )
        ],
    )
    records = build_candidate_records(
        {
            "candidate_rows_path": str(candidate_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )
    validated = validate_candidate_records(
        {
            "candidate_records_path": records.artifact_refs[0].uri,
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )
    final = serialize_final_records(
        {"records_path": validated.artifact_refs[0].uri, "artifact_name": "final_records.jsonl"},
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(validated.artifact_refs[0].uri).read_text(encoding="utf-8"))
    final_lines = Path(final.artifact_refs[0].uri).read_text(encoding="utf-8").splitlines()
    assert payload["accepted_count"] == 1
    assert payload["accepted_records"][0]["component_name"] == "P1"
    assert len(final_lines) == 1


def test_duplicate_and_reverse_complement_sequences_are_deduplicated(tmp_path: Path):
    sequence = "AAACCCGGGTTA"
    reverse_complement = sequence[::-1].translate(str.maketrans("ACGT", "TGCA"))
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"promoter_id": "P1", "promoter_sequence": sequence},
                sequence_field="promoter_sequence",
                sequence=sequence,
                label="P1",
            ),
            _candidate_row(
                tmp_path,
                row={"promoter_id": "P1-rc", "promoter_sequence": reverse_complement},
                sequence_field="promoter_sequence",
                sequence=reverse_complement,
                label="P1-rc",
            ),
        ],
    )

    records = build_candidate_records(
        {
            "candidate_rows_path": str(candidate_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
            "deduplicate_sequences": True,
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    assert payload["skipped_counts"]["duplicate_sequence"] == 1


def test_primary_component_table_filter_stages_secondary_tables(tmp_path: Path):
    source = tmp_path / "workbook.md"
    source.write_text("synthetic workbook", encoding="utf-8")
    candidate_tables_path = tmp_path / "candidate_tables.json"
    candidate_tables_path.write_text(
        json.dumps(
            {
                "candidate_tables": [
                    {
                        "source_file": str(source),
                        "file_type": "markdown",
                        "table_id": "workbook.md::table-1",
                        "table_index": 1,
                        "headers": ["promoter_id", "promoter_sequence", "activity"],
                        "rows": [
                            ["promoter_id", "promoter_sequence", "activity"],
                            ["P-primary", "AAACCCGGGTTT", "measured"],
                        ],
                        "section_label": "Primary promoter library",
                    },
                    {
                        "source_file": str(source),
                        "file_type": "markdown",
                        "table_id": "workbook.md::table-2",
                        "table_index": 2,
                        "headers": ["promoter_id", "promoter_sequence", "prediction_score"],
                        "rows": [
                            ["promoter_id", "promoter_sequence", "prediction_score"],
                            ["P-secondary", "TTTCCCGGGAAA", "0.91"],
                        ],
                        "section_label": "Secondary design table",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = extract_candidate_rows(
        {
            "candidate_tables_path": str(candidate_tables_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
            "primary_component_tables_only": True,
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(rows.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["candidate_rows"][0]["row"]["promoter_id"] == "P-primary"
    assert payload["table_selection_report"]["selected_tables"] == ["workbook.md::table-1"]


def test_validator_rejects_insufficient_evidence(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("name,sequence\nP1,AAACCCGGGTTT\n", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "P1",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "evidence_source": {"path": str(source), "row_index": 2},
                        "evidence_text": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validated = validate_candidate_records(
        {
            "candidate_records_path": str(records_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(validated.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 0
    assert "missing acceptance_reason" in payload["rejected_records"][0]["validation_issues"]


def _candidate_rows_path(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "candidate_rows.json"
    path.write_text(json.dumps({"work_item_id": "article_a", "candidate_rows": rows}), encoding="utf-8")
    return path


def _candidate_row(
    tmp_path: Path,
    *,
    row: dict[str, str],
    sequence_field: str,
    sequence: str,
    label_field: str = "promoter_id",
    label: str = "P1",
) -> dict:
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    return {
        "source_file": str(source),
        "sheet_name": "Sheet1",
        "row_index": 2,
        "row": row,
        "candidate_sequence_fields": [{"field": sequence_field, "value": sequence}],
        "candidate_label_fields": [{"field": label_field, "value": label}],
        "table_context": {"table_id": "fixture", "section_label": "promoter table"},
    }
