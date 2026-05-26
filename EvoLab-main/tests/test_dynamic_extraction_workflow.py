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


def test_generic_seq_suffix_without_target_context_is_not_promoter_field(tmp_path: Path):
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"name": "row1", "p_seq": "AAACCCGGGTTT", "predict_strength": "0.9"},
                sequence_field="p_seq",
                sequence="AAACCCGGGTTT",
                label_field="name",
                label="row1",
                table_context={"table_id": "variant table", "section_label": "variant predictions"},
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


def test_predict_strength_only_variant_table_is_rejected_by_validation(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "variant-1",
                        "component_type": "sequence_component",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "evidence_source": {"path": str(source), "row_index": 2},
                        "evidence_text": json.dumps(
                            {
                                "name": "variant-1",
                                "promoter_sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                                "predict_strength_variant": "0.91",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert "prediction-only" in payload["rejected_records"][0]["validation_issues"][0]


def test_generated_design_only_table_is_rejected_by_source_gate(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "design-1",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "Generated promoter candidates"},
                        "evidence_source": {"path": str(source), "table_id": "generated promoters"},
                        "evidence_text": json.dumps(
                            {
                                "promoter_id": "design-1",
                                "promoter_sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                                "generation_score": "0.91",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert payload["rejected_records"][0]["source_classification"]["category"] == "design_or_generated_library_source"


def test_background_database_table_is_rejected_by_source_gate(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "Weak",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "RegulonDB"},
                        "evidence_source": {"path": str(source), "table_id": "RegulonDB"},
                        "evidence_text": json.dumps(
                            {
                                "promoter_sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                                "class": "Weak",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert payload["rejected_records"][0]["source_classification"]["category"] == "background_database_source"


def test_ambiguous_target_sequence_source_routes_to_human_review(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
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
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "promoter identifiers"},
                        "evidence_source": {"path": str(source), "table_id": "promoter identifiers"},
                        "evidence_text": json.dumps(
                            {
                                "promoter_id": "P1",
                                "promoter_sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert payload["human_review_count"] == 1
    assert payload["human_review_records"][0]["status"] == "human_review"


def test_data_like_spreadsheet_headers_are_redetected_before_candidate_record_build(tmp_path: Path):
    source = tmp_path / "supplement.md"
    source.write_text("fixture", encoding="utf-8")
    candidate_tables_path = tmp_path / "candidate_tables.json"
    candidate_tables_path.write_text(
        json.dumps(
            {
                "work_item_id": "article_a",
                "candidate_tables": [
                    {
                        "source_file": str(source),
                        "file_type": "markdown",
                        "table_id": "supplement::table-1",
                        "table_index": 1,
                        "header_row_index": 2,
                        "headers": [
                            "wgan_gp_round1_1",
                            "agtcaacattaaaagttaaaatagttgctatgatgtgctaaggtaaataa",
                            "55_1965893901138",
                        ],
                        "rows": [
                            [
                                "Sequence_name",
                                "Artificial sequence oligos",
                                "Promoter activity(Fluo/OD600)",
                            ],
                            ["", "", "biological_replicate1"],
                            ["design-1", "AGTCAACATTAAAAGTTAAAATAGTTGCTATGATGTGCTAAGGTAAATAA", "55.1"],
                        ],
                        "section_label": "experimentally tested promoter sequences",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = extract_candidate_rows(
        {
            "candidate_tables_path": str(candidate_tables_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
            "primary_component_tables_only": False,
        },
        artifact_root=tmp_path / "artifacts",
    )
    records = build_candidate_records(
        {
            "candidate_rows_path": rows.artifact_refs[0].uri,
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=tmp_path / "artifacts",
    )

    row_payload = json.loads(Path(rows.artifact_refs[0].uri).read_text(encoding="utf-8"))
    record_payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert row_payload["row_count"] == 1
    assert row_payload["candidate_rows"][0]["headers"][0] == "sequence_name"
    assert record_payload["record_count"] == 1
    assert record_payload["records"][0]["sequence_field"] == "artificial_sequence_oligos"


def test_extract_candidate_rows_reads_full_plain_text_markdown_table_block(tmp_path: Path):
    source = tmp_path / "supplement.md"
    lines = [
        "Supplementary material",
        "Table S1 Experimentally measured promoter sequences",
        "names",
        "promoter_sequences",
        "fi_od600",
    ]
    for index in range(1, 9):
        sequence = "AAACCCGGGTTTAAACCCGGG" + ("A" * index)
        lines.extend(
            [
                f"S{index}",
                sequence,
                str(1000 + index),
            ]
        )
    lines.append("Figure S1 next section")
    source.write_text("\n".join(lines), encoding="utf-8")
    candidate_tables_path = tmp_path / "candidate_tables.json"
    candidate_tables_path.write_text(
        json.dumps(
            {
                "work_item_id": "article_a",
                "candidate_tables": [
                    {
                        "source_file": str(source),
                        "file_type": "markdown",
                        "table_id": "supplement.md::table-0",
                        "headers": ["names", "promoter_sequences", "fi_od600"],
                        "row_count": 8,
                        "sample_rows": [
                            ["names", "promoter_sequences", "fi_od600"],
                            ["S1", "AAACCCGGGTTTAAACCCGGGA", "1001"],
                            ["S2", "AAACCCGGGTTTAAACCCGGGAA", "1002"],
                        ],
                        "plain_text_table_block": {
                            "caption": "Table S1 Experimentally measured promoter sequences",
                            "start_line": 2,
                            "end_line": len(lines) - 1,
                            "source_format": "plain_text_markdown",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = extract_candidate_rows(
        {
            "candidate_tables_path": str(candidate_tables_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
            "primary_component_tables_only": False,
            "max_rows_per_table": 100,
        },
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(Path(rows.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert payload["row_count"] == 8
    assert [row["row"]["names"] for row in payload["candidate_rows"]] == [f"S{index}" for index in range(1, 9)]


def test_validator_rejects_ambiguous_seq_alias_when_only_article_path_mentions_target(tmp_path: Path):
    article_root = tmp_path / "Promoter article"
    article_root.mkdir()
    source = article_root / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "variant-1",
                        "component_type": "sequence_component",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "p_seq",
                        "source_file": str(source),
                        "evidence_source": {"path": str(source), "row_index": 2},
                        "evidence_text": '{"name":"variant-1","p_seq":"AAACCCGGGTTTAAACCCGGGTTT","predict_strength":"0.9"}',
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
                        "notes": "profile-specific generated note mentioning promoter/regulatory extraction",
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


def test_high_confidence_bare_promoter_field_is_retained(tmp_path: Path):
    article_root = tmp_path / "LaFleur article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "LaFleur et al - model-predictive promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                source=source,
                row={
                    "promoter": "TTTTCTATCTACGTACTTGACACTATTTCCTATTTCTCTTATAATCCCCGCGGCTCTACCT",
                    "measured_transcription_rates_tx_txref": "27.1",
                },
                sequence_field="promoter",
                sequence="TTTTCTATCTACGTACTTGACACTATTTCCTATTTCTCTTATAATCCCCGCGGCTCTACCT",
                label_field="measured_transcription_rates_tx_txref",
                label="27.1",
                table_context={"sheet_name": "La Fleur et al. (Fig S12a)", "section_label": "promoter measurements"},
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

    records_payload = json.loads(Path(records.artifact_refs[0].uri).read_text(encoding="utf-8"))
    validated_payload = json.loads(Path(validated.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert records_payload["record_count"] == 1
    assert validated_payload["accepted_count"] == 1
    assert validated_payload["accepted_records"][0]["sequence_field"] == "promoter"


def test_external_reference_bare_promoter_field_is_staged_not_final(tmp_path: Path):
    article_root = tmp_path / "LaFleur article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "LaFleur et al - model-predictive promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                source=source,
                row={"promoter": "CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG"},
                sequence_field="promoter",
                sequence="CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG",
                table_context={"sheet_name": "Urtecho et al. (Fig S9b)", "section_label": "promoter benchmark"},
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


def test_bare_promoter_field_requires_table_level_target_or_article_provenance(tmp_path: Path):
    article_root = tmp_path / "Promoter design article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "LaFleur et al - model-predictive promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
    candidate_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                source=source,
                row={"promoter": "CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG"},
                sequence_field="promoter",
                sequence="CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG",
                table_context={"sheet_name": "Lagator 36N", "section_label": "benchmark measurements"},
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


def test_bare_promoter_gate_ignores_derived_table_selection_reasons(tmp_path: Path):
    article_root = tmp_path / "Promoter design article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "LaFleur et al - model-predictive promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
    row = _candidate_row(
        tmp_path,
        source=source,
        row={"promoter": "CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG"},
        sequence_field="promoter",
        sequence="CCTGCGTTTCCTCGCTTTTGGTGAAACGCATCTGGCTGATGTGCTGGTGAGCAAGCAGTTCCACG",
        table_context={"sheet_name": "Lagator 36N", "section_label": "benchmark measurements"},
    )
    row["table_selection"] = {"reasons": ["target-compatible sequence headers: promoter"]}
    candidate_rows_path = _candidate_rows_path(tmp_path, [row])

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


def test_validator_rejects_external_reference_dataset_records(tmp_path: Path):
    article_root = tmp_path / "LaFleur article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "LaFleur et al - model-predictive promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
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
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "Urtecho et al. (Fig S9b)"},
                        "evidence_source": {"path": str(source), "table_id": "Urtecho et al. (Fig S9b)"},
                        "evidence_text": '{"promoter_sequence":"AAACCCGGGTTTAAACCCGGGTTT"}',
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert "external/reference dataset" in payload["rejected_records"][0]["validation_issues"][0]


def test_validator_rejects_model_evaluation_detail_records(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
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
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "Model performance", "headers": ["abs_residual", "mae"]},
                        "evidence_source": {"path": str(source), "table_id": "model performance"},
                        "evidence_text": '{"observed_log_tx_txref":"1.0","predicted_log_tx_txref":"1.1","mae":"0.1"}',
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert "model-evaluation" in payload["rejected_records"][0]["validation_issues"][0]


def test_validator_rejects_external_model_prediction_observation_table(tmp_path: Path):
    article_root = tmp_path / "Current study article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "Current Study et al - promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "0",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "Benchmark 36N"},
                        "evidence_source": {"path": str(source), "table_id": "Benchmark 36N"},
                        "evidence_text": json.dumps(
                            {
                                "promoter": "AAACCCGGGTTTAAACCCGGGTTT",
                                "observed_log_tx_txref": "-1.0",
                                "predicted_log_tx_txref": "-0.9",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert "model-evaluation" in payload["rejected_records"][0]["validation_issues"][0]


def test_validator_keeps_current_article_model_measurement_table(tmp_path: Path):
    article_root = tmp_path / "Current study article"
    article_root.mkdir()
    (article_root / "manifest.json").write_text(
        json.dumps({"main_pdf": "Current Study et al - promoter design.pdf"}),
        encoding="utf-8",
    )
    source = article_root / "supplement.xlsx"
    source.write_text("fixture", encoding="utf-8")
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
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "Current Study et al. measured promoters"},
                        "evidence_source": {"path": str(source), "table_id": "Current Study et al."},
                        "evidence_text": json.dumps(
                            {
                                "promoter": "AAACCCGGGTTTAAACCCGGGTTT",
                                "measured_transcription_rates_tx_txref": "1.2",
                                "predicted_log_tx_txref": "-0.9",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert payload["accepted_count"] == 1


def test_validator_rejects_raw_replicate_library_without_component_identity(tmp_path: Path):
    source = tmp_path / "supplement.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "0.4",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTTAAACCCGGGTTT",
                        "sequence_field": "promoter_sequences",
                        "source_file": str(source),
                        "source_table_context": {"sheet_name": "promoter library"},
                        "evidence_source": {"path": str(source), "table_id": "promoter library"},
                        "evidence_text": json.dumps(
                            {
                                "promoter_sequences": "AAACCCGGGTTTAAACCCGGGTTT",
                                "biological_replicate_1": "0.4",
                                "biological_replicate_2": "0.5",
                            }
                        ),
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
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
    assert "raw replicate library" in payload["rejected_records"][0]["validation_issues"][0]


def test_validated_records_policy_overwrites_higher_count_existing_artifact(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    work_item_root = artifact_root / "article_a"
    work_item_root.mkdir(parents=True)
    validated_path = work_item_root / "validated_records.json"
    validated_path.write_text(
        json.dumps(
            {
                "work_item_id": "article_a",
                "accepted_count": 2,
                "accepted_records": [
                    {"component_name": "old-1", "sequence": "AAACCCGGGTTT"},
                    {"component_name": "old-2", "sequence": "TTTCCCGGGAAA"},
                ],
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    records_path = tmp_path / "candidate_records.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "article_id": "article_a",
                        "work_item_id": "article_a",
                        "component_name": "new-1",
                        "component_type": "promoter",
                        "sequence": "AAACCCGGGTTT",
                        "sequence_field": "promoter_sequence",
                        "source_file": str(source),
                        "evidence_source": {"path": str(source), "row_index": 2},
                        "evidence_text": '{"promoter_sequence":"AAACCCGGGTTT","activity":"measured"}',
                        "acceptance_reason": "target-compatible promoter sequence field with source provenance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validate_candidate_records(
        {
            "candidate_records_path": str(records_path),
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=artifact_root,
    )

    payload = json.loads(validated_path.read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 1
    assert payload["accepted_records"][0]["component_name"] == "new-1"


def test_smaller_candidate_record_artifact_does_not_overwrite_higher_coverage_handoff(tmp_path: Path):
    first_rows_path = _candidate_rows_path(
        tmp_path,
        [
            _candidate_row(
                tmp_path,
                row={"promoter_id": "P1", "promoter_sequence": "AAACCCGGGTTT"},
                sequence_field="promoter_sequence",
                sequence="AAACCCGGGTTT",
                label="P1",
            ),
            _candidate_row(
                tmp_path,
                row={"promoter_id": "P2", "promoter_sequence": "TTTCCCGGGAAA"},
                sequence_field="promoter_sequence",
                sequence="TTTCCCGGGAAA",
                label="P2",
            ),
        ],
    )
    artifact_root = tmp_path / "artifacts"
    first = build_candidate_records(
        {
            "candidate_rows_path": str(first_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=artifact_root,
    )
    second_rows_path = tmp_path / "candidate_rows_smaller.json"
    second_rows_path.write_text(
        json.dumps(
            {
                "work_item_id": "article_a",
                "candidate_rows": [
                    _candidate_row(
                        tmp_path,
                        row={"promoter_id": "P1", "promoter_sequence": "AAACCCGGGTTT"},
                        sequence_field="promoter_sequence",
                        sequence="AAACCCGGGTTT",
                        label="P1",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    second = build_candidate_records(
        {
            "candidate_rows_path": str(second_rows_path),
            "article_id": "article_a",
            "work_item_id": "article_a",
            "sequence_extraction_profile": "promoter",
        },
        artifact_root=artifact_root,
    )

    payload = json.loads(Path(first.artifact_refs[0].uri).read_text(encoding="utf-8"))
    assert first.artifact_refs[0].uri == second.artifact_refs[0].uri
    assert payload["record_count"] == 2
    assert second.metadata["record_count"] == 2
    assert "preserved existing higher-coverage handoff artifact" in second.metadata["warnings"][0]


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
    source: Path | None = None,
    table_context: dict | None = None,
) -> dict:
    if source is None:
        source = tmp_path / "supplement.csv"
        source.write_text("fixture", encoding="utf-8")
    return {
        "source_file": str(source),
        "sheet_name": "Sheet1",
        "row_index": 2,
        "row": row,
        "candidate_sequence_fields": [{"field": sequence_field, "value": sequence}],
        "candidate_label_fields": [{"field": label_field, "value": label}],
        "table_context": table_context or {"table_id": "fixture", "section_label": "promoter table"},
    }
