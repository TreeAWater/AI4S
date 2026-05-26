from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MIN_SEQUENCE_LENGTH = 1
MIN_SEQUENCE_RUN_LENGTH = 6
SEQUENCE_COLUMN_RE = re.compile(r"(^sequence\d*$|^sequence$|promoter.*sequence|sequence.*promoter)", re.I)
ARTICLE_ID_FIELDS = ("article_id", "paper_id", "publication", "Publication", "title", "Title")
DNA_RE = re.compile(r"[ACGTN]", re.I)
DNA_RUN_RE = re.compile(r"[ACGTN]+", re.I)
LETTER_RE = re.compile(r"[A-Z]", re.I)
GENERIC_ARTICLE_PATH_NAMES = {
    "supplemental",
    "supplementary",
    "supplementary data",
    "supplementary files",
    "supplementary material",
    "supplementary materials",
}
ARTICLE_ID_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    false_positives: list[dict[str, Any]]
    false_negatives: list[dict[str, Any]]
    per_article_metrics: list[dict[str, Any]]
    normalized_predictions: list[dict[str, Any]]
    normalized_ground_truth: list[dict[str, Any]]


def normalize_sequence(value: Any) -> str:
    if value is None:
        return ""
    compact = re.sub(r"[\s\-]+", "", str(value).upper())
    if not compact:
        return ""
    if re.fullmatch(r"[ACGTN]+", compact):
        return compact
    runs = DNA_RUN_RE.findall(compact)
    long_runs = [run.upper() for run in runs if len(run) >= MIN_SEQUENCE_RUN_LENGTH]
    if long_runs:
        return max(long_runs, key=len)
    return ""


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def sequences_match(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    if prediction == ground_truth or prediction in ground_truth or ground_truth in prediction:
        return True
    rc_prediction = reverse_complement(prediction)
    return rc_prediction == ground_truth or rc_prediction in ground_truth or ground_truth in rc_prediction


def evaluate_sequences(
    predictions: Iterable[dict[str, Any]],
    ground_truth: Iterable[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    prediction_artifact: Path | None = None,
    warnings: list[str] | None = None,
    article_ids: Iterable[str] | None = None,
) -> EvaluationResult:
    normalized_predictions = _dedupe_by_sequence(_normalize_records(predictions, source="prediction"))
    normalized_ground_truth = _dedupe_by_sequence(_normalize_records(ground_truth, source="ground_truth"))
    article_filter = _normalized_article_filter(article_ids)
    article_filter_display = _normalized_article_filter_display(article_ids)
    if article_filter:
        normalized_predictions = [
            record for record in normalized_predictions if _record_matches_article_filter(record, article_filter)
        ]
        normalized_ground_truth = [
            record
            for record in normalized_ground_truth
            if _record_matches_article_filter(record, article_filter)
        ]

    matches = _match_records(normalized_predictions, normalized_ground_truth)
    matched_prediction_indexes = {pred_i for pred_i, _ in matches}
    matched_gt_indexes = {gt_i for _, gt_i in matches}
    false_positives = [
        _example_record(record)
        for index, record in enumerate(normalized_predictions)
        if index not in matched_prediction_indexes
    ]
    false_negatives = [
        _example_record(record)
        for index, record in enumerate(normalized_ground_truth)
        if index not in matched_gt_indexes
    ]

    gt_count = len(normalized_ground_truth)
    pred_count = len(normalized_predictions)
    true_positive = len(matches)
    false_positive = len(false_positives)
    false_negative = len(false_negatives)
    precision = true_positive / pred_count if pred_count else 0.0
    recall = true_positive / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    validation_errors = []
    if prediction_artifact is not None and not prediction_artifact.exists():
        validation_errors.append(f"prediction artifact not found: {prediction_artifact}")

    summary = {
        "prediction_artifact": str(prediction_artifact) if prediction_artifact else None,
        "gt_sequence_count": gt_count,
        "predicted_sequence_count": pred_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "validation_status": {"valid": not validation_errors, "errors": validation_errors},
        "warnings": warnings or [],
        "article_filter": article_filter_display,
    }
    result = EvaluationResult(
        summary=summary,
        false_positives=false_positives,
        false_negatives=false_negatives,
        per_article_metrics=_per_article_metrics(normalized_predictions, normalized_ground_truth),
        normalized_predictions=normalized_predictions,
        normalized_ground_truth=normalized_ground_truth,
    )
    if output_dir is not None:
        write_evaluation_artifacts(result, output_dir)
    return result


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    records: list[dict[str, Any]] = []
    for candidate in paths:
        if _is_evaluation_output(candidate):
            continue
        for row in _read_jsonl(candidate):
            for record in _record_payloads(row):
                record = dict(record)
                record.setdefault("_source_file", str(candidate))
                records.append(record)
    return records


def load_ground_truth_records(gt_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(gt_root.rglob("*")):
        if path.suffix.lower() == ".xlsx":
            records.extend(_load_xlsx_ground_truth(path))
        elif path.suffix.lower() == ".jsonl":
            for row in _read_jsonl(path):
                for record in _record_payloads(row):
                    record = dict(record)
                    record.setdefault("_source_file", str(path))
                    records.append(record)
        elif path.suffix.lower() == ".csv":
            records.extend(_load_csv_ground_truth(path))
    return records


def write_evaluation_artifacts(result: EvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "promoter_eval_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "promoter_eval_summary.md").write_text(_summary_markdown(result.summary), encoding="utf-8")
    _write_jsonl(output_dir / "promoter_false_positives.jsonl", result.false_positives)
    _write_jsonl(output_dir / "promoter_false_negatives.jsonl", result.false_negatives)
    _write_jsonl(output_dir / "normalized_predictions.jsonl", result.normalized_predictions)
    _write_jsonl(output_dir / "normalized_ground_truth.jsonl", result.normalized_ground_truth)
    with (output_dir / "promoter_per_article_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "article_id",
            "gt_sequence_count",
            "predicted_sequence_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.per_article_metrics:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate EvoLab promoter sequence predictions against GT.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file or directory.")
    parser.add_argument("--gt-root", required=True, help="Ground-truth directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Write an evaluation report with zero predictions when the prediction path is missing.",
    )
    parser.add_argument(
        "--article-id",
        action="append",
        default=[],
        help="Restrict evaluation to one article id. May be provided multiple times.",
    )
    parser.add_argument(
        "--article-ids-file",
        help="Restrict evaluation to article ids listed in a text file. Lines may contain numbered ids or comments.",
    )
    args = parser.parse_args(argv)

    prediction_path = Path(args.predictions)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    warnings: list[str] = []
    if not prediction_path.exists():
        if not args.allow_missing_predictions:
            raise FileNotFoundError(f"prediction path not found: {prediction_path}")
        warnings.append(f"prediction path not found; evaluated as zero predictions: {prediction_path}")
    if not gt_root.exists():
        raise FileNotFoundError(f"ground-truth root not found: {gt_root}")

    result = evaluate_sequences(
        load_prediction_records(prediction_path),
        load_ground_truth_records(gt_root),
        output_dir=output_dir,
        prediction_artifact=prediction_path,
        warnings=warnings,
        article_ids=[*args.article_id, *_article_ids_from_file(args.article_ids_file)],
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def _normalize_records(records: Iterable[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        article_id = _article_id(record)
        article_keys = sorted(
            _record_article_match_keys(record, article_id, include_path_keys=source == "prediction" or not article_id)
        )
        article_key = _canonical_article_key_from_keys(article_keys)
        for raw_sequence in _sequence_values(record):
            if not _looks_like_sequence(raw_sequence):
                continue
            sequence = normalize_sequence(raw_sequence)
            if len(sequence) < MIN_SEQUENCE_LENGTH:
                continue
            normalized.append(
                {
                    "article_id": article_id,
                    "article_key": article_key,
                    "article_keys": article_keys,
                    "sequence": sequence,
                    "source": source,
                    "source_file": record.get("_source_file"),
                    "source_sheet": record.get("_source_sheet"),
                    "source_row": record.get("_source_row"),
                    "component_name": record.get("component_name") or record.get("Name") or record.get("Code"),
                }
            )
    return normalized


def _sequence_values(record: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key, value in record.items():
        if key == "sequence" or SEQUENCE_COLUMN_RE.search(str(key)):
            if value not in (None, ""):
                values.append(value)
    return values


def _article_id(record: dict[str, Any]) -> str | None:
    for field in ARTICLE_ID_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    evidence_source = record.get("evidence_source")
    if isinstance(evidence_source, dict):
        for field in ("article_id", "publication", "path"):
            value = evidence_source.get(field)
            if value not in (None, ""):
                return str(value)
    return None


def _normalized_article_filter(article_ids: Iterable[str] | None) -> set[str]:
    if article_ids is None:
        return set()
    keys: set[str] = set()
    for article_id in article_ids:
        keys.update(_article_match_keys(article_id))
    return keys


def _normalized_article_filter_display(article_ids: Iterable[str] | None) -> list[str]:
    if article_ids is None:
        return []
    return sorted(
        normalized
        for normalized in (_normalize_article_id(article_id) for article_id in article_ids)
        if normalized
    )


def _normalize_article_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _article_match_keys(value: Any) -> set[str]:
    normalized = _normalize_article_id(value)
    if not normalized:
        return set()
    keys = {normalized}
    stem = Path(normalized).stem
    if stem:
        keys.add(stem)
    for candidate in list(keys):
        slug = re.sub(r"[^a-z0-9]+", "_", candidate.casefold()).strip("_")
        if slug:
            keys.add(slug)
        compact = re.sub(r"[^a-z0-9]+", "", candidate.casefold())
        if compact:
            keys.add(compact)
        tokens = [token for token in re.split(r"[^a-z0-9]+", candidate.casefold()) if token]
        stopword_light = "_".join(token for token in tokens if token not in ARTICLE_ID_STOPWORDS)
        if stopword_light:
            keys.add(stopword_light)
            keys.add(stopword_light.replace("_", ""))
    return keys


def _canonical_article_key(value: Any) -> str:
    return _canonical_article_key_from_keys(_article_match_keys(value))


def _canonical_article_key_from_keys(keys_value: Iterable[str]) -> str:
    keys = set(keys_value)
    for key in sorted(keys, key=lambda item: (0 if "_" in item else 1, len(item))):
        if "_" in key:
            return key
    return sorted(keys)[0] if keys else ""


def _record_article_match_keys(record: dict[str, Any], article_id: Any, *, include_path_keys: bool) -> set[str]:
    keys = _article_match_keys(article_id)
    if not include_path_keys:
        return keys
    for value in _record_article_path_values(record):
        path = Path(str(value))
        keys.update(_article_match_keys(path.stem))
        for parent in list(path.parents)[:2]:
            if _normalize_article_id(parent.name) not in GENERIC_ARTICLE_PATH_NAMES:
                keys.update(_article_match_keys(parent.name))
    return keys


def _record_article_path_values(record: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for field in ("source_file",):
        value = record.get(field)
        if value not in (None, ""):
            values.append(value)
    evidence_source = record.get("evidence_source")
    if isinstance(evidence_source, dict):
        value = evidence_source.get("path")
        if value not in (None, ""):
            values.append(value)
    if not values and not _article_id(record):
        value = record.get("_source_file")
        if value not in (None, ""):
            values.append(value)
    return values


def _record_matches_article_filter(record: dict[str, Any], article_filter: set[str]) -> bool:
    keys = set(record.get("article_keys") or [])
    if not keys:
        keys = _article_match_keys(record.get("article_id"))
    return _article_keys_overlap(keys, article_filter)


def _article_keys_overlap(left: set[str], right: set[str]) -> bool:
    return bool(left & right)


def _article_ids_from_file(path_value: str | None) -> list[str]:
    if not path_value:
        return []
    ids: list[str] = []
    for line in Path(path_value).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^\d+\.\s+([^\s#]+)", stripped)
        if match:
            ids.append(match.group(1))
        elif stripped.startswith("/"):
            ids.append(Path(stripped).name)
        else:
            ids.append(stripped.split()[0])
    return ids


def _looks_like_sequence(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    sequence = normalize_sequence(text)
    if len(sequence) < MIN_SEQUENCE_LENGTH:
        return False
    letters = LETTER_RE.findall(text)
    if not letters:
        return False
    dna_letters = DNA_RE.findall(text)
    return len(dna_letters) / len(letters) >= 0.85 or len(sequence) >= MIN_SEQUENCE_RUN_LENGTH


def _dedupe_by_sequence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        sequence = record["sequence"]
        key = (str(record.get("article_key") or ""), sequence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _match_records(
    predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    unmatched_gt = set(range(len(ground_truth)))
    gt_keys_by_index = [set(record.get("article_keys") or []) for record in ground_truth]
    gt_indexes_by_key: dict[str, set[int]] = {}
    for gt_i, keys in enumerate(gt_keys_by_index):
        for key in keys:
            gt_indexes_by_key.setdefault(key, set()).add(gt_i)
    candidate_pool_cache: dict[tuple[str, ...], set[int]] = {}
    prediction_order = sorted(range(len(predictions)), key=lambda index: len(predictions[index]["sequence"]), reverse=True)
    for pred_i in prediction_order:
        pred_sequence = predictions[pred_i]["sequence"]
        pred_rc_sequence = reverse_complement(pred_sequence)
        candidate_pool = _candidate_gt_pool(
            set(predictions[pred_i].get("article_keys") or []),
            gt_indexes_by_key,
            candidate_pool_cache,
            unmatched_gt,
        )
        exact_candidates = [
            gt_i
            for gt_i in candidate_pool
            if pred_sequence == ground_truth[gt_i]["sequence"] or pred_rc_sequence == ground_truth[gt_i]["sequence"]
        ]
        substring_candidates = [
            gt_i
            for gt_i in candidate_pool
            if gt_i not in exact_candidates
            and _sequence_or_reverse_complement_substring_match(
                pred_sequence,
                pred_rc_sequence,
                ground_truth[gt_i]["sequence"],
            )
        ]
        candidates = exact_candidates or substring_candidates
        if candidates:
            gt_i = sorted(candidates, key=lambda index: len(ground_truth[index]["sequence"]), reverse=True)[0]
            matches.append((pred_i, gt_i))
            unmatched_gt.remove(gt_i)
    return matches


def _candidate_gt_pool(
    prediction_keys: set[str],
    gt_indexes_by_key: dict[str, set[int]],
    candidate_pool_cache: dict[tuple[str, ...], set[int]],
    unmatched_gt: set[int],
) -> set[int]:
    if not prediction_keys:
        return set(unmatched_gt)
    cache_key = tuple(sorted(prediction_keys))
    if cache_key not in candidate_pool_cache:
        pool: set[int] = set()
        for gt_key, indexes in gt_indexes_by_key.items():
            if _article_keys_overlap(prediction_keys, {gt_key}):
                pool.update(indexes)
        candidate_pool_cache[cache_key] = pool
    return candidate_pool_cache[cache_key] & unmatched_gt


def _sequence_or_reverse_complement_substring_match(
    prediction: str,
    reverse_complement_prediction: str,
    ground_truth: str,
) -> bool:
    return (
        prediction in ground_truth
        or ground_truth in prediction
        or reverse_complement_prediction in ground_truth
        or ground_truth in reverse_complement_prediction
    )


def _per_article_metrics(
    predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    article_key_groups = _article_key_groups(predictions + ground_truth)
    rows: list[dict[str, Any]] = []
    for article_keys in article_key_groups:
        pred_subset = [r for r in predictions if _article_keys_overlap(set(r.get("article_keys") or []), article_keys)]
        gt_subset = [r for r in ground_truth if _article_keys_overlap(set(r.get("article_keys") or []), article_keys)]
        if not pred_subset and not gt_subset:
            continue
        display_article_id = _display_article_id(sorted(article_keys)[0], pred_subset, gt_subset)
        matches = _match_records(pred_subset, gt_subset)
        tp = len(matches)
        fp = len(pred_subset) - tp
        fn = len(gt_subset) - tp
        precision = tp / len(pred_subset) if pred_subset else 0.0
        recall = tp / len(gt_subset) if gt_subset else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "article_id": display_article_id,
                "gt_sequence_count": len(gt_subset),
                "predicted_sequence_count": len(pred_subset),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return rows


def _article_key_groups(records: list[dict[str, Any]]) -> list[set[str]]:
    groups: list[set[str]] = []
    for record in records:
        keys = set(record.get("article_keys") or [])
        if not keys:
            continue
        matching_indexes = [
            index for index, group in enumerate(groups) if _article_keys_overlap(keys, group)
        ]
        if not matching_indexes:
            groups.append(set(keys))
            continue
        first = matching_indexes[0]
        groups[first].update(keys)
        for index in reversed(matching_indexes[1:]):
            groups[first].update(groups[index])
            del groups[index]
    return sorted(groups, key=lambda group: sorted(group)[0])


def _display_article_id(
    article_key: str,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> str:
    for record in [*predictions, *ground_truth]:
        article_id = record.get("article_id")
        if article_id:
            return str(article_id)
    return article_key


def _example_record(record: dict[str, Any]) -> dict[str, Any]:
    example = {
        "article_id": record.get("article_id"),
        "sequence": record.get("sequence"),
        "source_file": record.get("source_file"),
        "source_sheet": record.get("source_sheet"),
        "source_row": record.get("source_row"),
        "component_name": record.get("component_name"),
    }
    return {key: value for key, value in example.items() if value is not None}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload.setdefault("_source_row", line_number)
                rows.append(payload)
    return rows


def _record_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("records", "accepted_records", "predictions", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _is_evaluation_output(path: Path) -> bool:
    return path.name in {
        "promoter_false_positives.jsonl",
        "promoter_false_negatives.jsonl",
        "normalized_predictions.jsonl",
        "normalized_ground_truth.jsonl",
    }


def _load_csv_ground_truth(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for index, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["_source_file"] = str(path)
            row["_source_row"] = index
            rows.append(row)
    return rows


def _load_xlsx_ground_truth(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        for sheet_name, sheet_path in _workbook_sheets(archive):
            raw_rows = _sheet_rows(archive, sheet_path, shared_strings)
            if not raw_rows:
                continue
            header = [str(value).strip() for value in raw_rows[0]]
            if not any(SEQUENCE_COLUMN_RE.search(column) for column in header):
                continue
            for row_index, raw_row in enumerate(raw_rows[1:], start=2):
                if not any(value not in (None, "") for value in raw_row):
                    continue
                record = {header[i]: raw_row[i] if i < len(raw_row) else "" for i in range(len(header)) if header[i]}
                record["_source_file"] = str(path)
                record["_source_sheet"] = sheet_name
                record["_source_row"] = row_index
                rows.append(record)
    return rows


def _shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text_node.text or "" for text_node in item.iter(_xlsx_tag("t")))
        for item in root.findall(_xlsx_tag("si"))
    ]


def _workbook_sheets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets: list[tuple[str, str]] = []
    relationship_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    for sheet in workbook.find(_xlsx_tag("sheets")) or []:
        name = sheet.attrib.get("name", "")
        target = relmap[sheet.attrib[relationship_key]]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((name, target))
    return sheets


def _sheet_rows(archive: ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    sheet_data = root.find(_xlsx_tag("sheetData"))
    if sheet_data is None:
        return rows
    for row in sheet_data.findall(_xlsx_tag("row")):
        cells: list[str] = []
        for cell in row.findall(_xlsx_tag("c")):
            index = _cell_column_index(cell.attrib.get("r", ""))
            while len(cells) <= index:
                cells.append("")
            cells[index] = _cell_value(cell, shared_strings)
        rows.append(cells)
    return rows


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(_xlsx_tag("is"))
        if inline is None:
            return ""
        return "".join(text_node.text or "" for text_node in inline.iter(_xlsx_tag("t")))
    value = cell.find(_xlsx_tag("v"))
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def _cell_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    column = 0
    for letter in letters:
        column = column * 26 + ord(letter.upper()) - 64
    return max(column - 1, 0)


def _xlsx_tag(local_name: str) -> str:
    return f"{{http://schemas.openxmlformats.org/spreadsheetml/2006/main}}{local_name}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Promoter Sequence Evaluation",
        "",
        f"- GT sequences: {summary['gt_sequence_count']}",
        f"- Predicted sequences: {summary['predicted_sequence_count']}",
        f"- True positives: {summary['true_positive']}",
        f"- False positives: {summary['false_positive']}",
        f"- False negatives: {summary['false_negative']}",
        f"- Precision: {summary['precision']}",
        f"- Recall: {summary['recall']}",
        f"- F1: {summary['f1']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
