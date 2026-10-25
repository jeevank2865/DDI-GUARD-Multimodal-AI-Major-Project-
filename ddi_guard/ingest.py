"""Lossless, auditable ingestion for the eight DDInter 2.0 ATC exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import LABELS

ATC_CODES = ("A", "B", "D", "H", "L", "P", "R", "V")
REQUIRED_COLUMNS = ("DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level")
VALID_LABELS = set(LABELS)


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def stable_pair_id(drug_id_a: str, drug_id_b: str) -> str:
    low, high = sorted((drug_id_a.strip(), drug_id_b.strip()))
    digest = hashlib.sha256(f"{low}\0{high}".encode("utf-8")).hexdigest()[:20]
    return f"ddi_{digest}"


def discover_sources(input_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for code in ATC_CODES:
        path = input_dir / f"ddinter_downloads_code_{code}.csv"
        if path.is_file():
            found[code] = path
    missing = sorted(set(ATC_CODES) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing required DDInter ATC files: {', '.join(missing)}")
    return found


def _read_records(sources: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for code in ATC_CODES:
        path = sources[code]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_cols = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
            if missing_cols:
                raise ValueError(f"{path.name} missing required columns: {missing_cols}")
            for row_number, row in enumerate(reader, start=2):
                raw = {key: (value or "").strip() for key, value in row.items() if key is not None}
                source = {"file": path.name, "atc_code": code, "row_number": row_number}
                required_values = [raw.get(name, "") for name in REQUIRED_COLUMNS]
                if any(not value for value in required_values):
                    malformed.append({**source, **raw, "reason": "missing_required_value"})
                    continue
                a = raw["DDInterID_A"]
                b = raw["DDInterID_B"]
                if a == b:
                    malformed.append({**source, **raw, "reason": "self_interaction"})
                    continue
                records.append({**raw, "source": source})
    return records, malformed


def build_outputs(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sources = discover_sources(input_dir)
    records, malformed = _read_records(sources)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    drugs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        a, b = record["DDInterID_A"], record["DDInterID_B"]
        grouped[tuple(sorted((a, b)))].append(record)
        drugs[a].add(record["Drug_A"])
        drugs[b].add(record["Drug_B"])

    pairs: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = list(malformed)
    pair_label_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter(record["Level"] for record in records)
    raw_label_counts.update(row.get("Level", "<MISSING>") or "<MISSING>" for row in malformed)
    duplicate_records = 0
    conflict_pairs = 0
    unknown_records = 0
    unknown_pairs = 0

    for (id_low, id_high), group in sorted(grouped.items()):
        valid_labels = {record["Level"] for record in group if record["Level"] in VALID_LABELS}
        unknown = [record for record in group if record["Level"] not in VALID_LABELS]
        if len(valid_labels) > 1:
            conflict_pairs += 1
            for record in group:
                ledger.append({**record["source"], **{k: record[k] for k in REQUIRED_COLUMNS},
                               "reason": "conflicting_severity_labels", "pair_id": stable_pair_id(id_low, id_high)})
            continue
        if not valid_labels:
            unknown_pairs += 1
            unknown_records += len(group)
            for record in group:
                ledger.append({**record["source"], **{k: record[k] for k in REQUIRED_COLUMNS},
                               "reason": "unknown_or_invalid_severity", "pair_id": stable_pair_id(id_low, id_high)})
            continue
        label = next(iter(valid_labels))
        selected = [record for record in group if record["Level"] == label]
        canonical_record = selected[0]
        # Names are attached to stable DDInter IDs independent of which source orientation appeared first.
        name_by_id: dict[str, set[str]] = defaultdict(set)
        for record in selected:
            name_by_id[record["DDInterID_A"]].add(record["Drug_A"])
            name_by_id[record["DDInterID_B"]].add(record["Drug_B"])
        pair_id = stable_pair_id(id_low, id_high)
        pair = {
            "pair_id": pair_id,
            "drug_a_id": id_low,
            "drug_a_name": sorted(name_by_id[id_low])[0],
            "drug_b_id": id_high,
            "drug_b_name": sorted(name_by_id[id_high])[0],
            "severity": label,
            "source_records": [record["source"] for record in selected],
            "atc_codes": sorted({record["source"]["atc_code"] for record in selected}),
        }
        pairs.append(pair)
        pair_label_counts[label] += 1
        duplicate_records += max(0, len(selected) - 1)
        for record in unknown:
            unknown_records += 1
            ledger.append({**record["source"], **{k: record[k] for k in REQUIRED_COLUMNS},
                           "reason": "unknown_or_invalid_severity", "pair_id": pair_id})
        for record in selected[1:]:
            ledger.append({**record["source"], **{k: record[k] for k in REQUIRED_COLUMNS},
                           "reason": "duplicate_or_reversed_pair_collapsed", "pair_id": pair_id})

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    processed_dir = output_dir / "data" / "processed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pair_json = processed_dir / "ddi_pairs_full.json"
    pair_json.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = processed_dir / "ddi_pair_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("pair_id", "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name", "severity", "atc_codes", "source_records")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({**pair, "atc_codes": json.dumps(pair["atc_codes"]), "source_records": json.dumps(pair["source_records"])})

    ledger_path = processed_dir / "data_exclusion_ledger.csv"
    ledger_fields = ("file", "atc_code", "row_number", "DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level", "reason", "pair_id")
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger)

    mapping_path = processed_dir / "drug_identity_mapping.csv"
    name_to_ids: dict[str, set[str]] = defaultdict(set)
    for drug_id, names in drugs.items():
        for name in names:
            name_to_ids[normalize_name(name)].add(drug_id)
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("ddinter_id", "original_name", "normalized_name", "name_ambiguity", "pubchem_cid", "mapping_status")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for drug_id in sorted(drugs):
            for name in sorted(drugs[drug_id], key=str.casefold):
                normalized = normalize_name(name)
                writer.writerow({"ddinter_id": drug_id, "original_name": name, "normalized_name": normalized,
                                 "name_ambiguity": len(name_to_ids[normalized]) > 1, "pubchem_cid": "",
                                 "mapping_status": "pending_pubchem_enrichment"})

    class_counts = {label: pair_label_counts[label] for label in LABELS}
    class_path = processed_dir / "ddi_class_distribution.json"
    class_path.write_text(json.dumps(class_counts, indent=2) + "\n", encoding="utf-8")
    raw_rows = len(records) + len(malformed)
    categorized = len(pairs) + len(ledger)
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(input_dir),
        "source_files": {code: {"path": str(path), "raw_rows": sum(1 for row in records if row["source"]["atc_code"] == code) + sum(1 for row in malformed if row.get("atc_code") == code)} for code, path in sources.items()},
        "raw_records": raw_rows,
        "unique_canonical_pairs_including_unknown": len(grouped),
        "unique_labeled_pairs": len(pairs),
        "raw_label_counts": dict(raw_label_counts),
        "class_distribution_unique_labeled": class_counts,
        "unique_drug_ids": len(drugs),
        "duplicate_labeled_records_collapsed": duplicate_records,
        "duplicate_records_total_including_unknown": len(records) - len(grouped),
        "unknown_unique_canonical_pairs": unknown_pairs,
        "unknown_or_invalid_records": unknown_records,
        "conflicting_pair_count": conflict_pairs,
        "malformed_or_self_interaction_records": len(malformed),
        "exclusion_ledger_rows": len(ledger),
        "record_accounting_check": {"raw_records": raw_rows, "retained_unique_labeled_pairs": len(pairs), "ledger_rows": len(ledger), "accounted": raw_rows == categorized},
        "dataset_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {code: hashlib.sha256(path.read_bytes()).hexdigest() for code, path in sources.items()},
        "outputs": {
            "pairs_json": str(pair_json), "pair_manifest_csv": str(manifest_path),
            "exclusion_ledger_csv": str(ledger_path), "drug_identity_mapping_csv": str(mapping_path),
            "class_distribution_json": str(class_path),
        },
        "notes": [
            "Pairs are canonicalized by stable DDInter IDs, not display names.",
            "All records used by each retained pair remain referenced in source_records.",
            "Unknown and malformed labels are excluded from severity supervision and recorded in the ledger.",
            "No external molecule, target, or text features are synthesized by this ingestion step.",
        ],
    }
    audit_path = artifacts_dir / "ddinter_ingestion_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    audit = build_outputs(args.input_dir, args.output_dir)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
