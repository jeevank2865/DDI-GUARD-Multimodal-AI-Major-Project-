"""Re-evaluate PubChem query matches against DDInter's within-source name ambiguity."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def review(compounds_path: Path, mapping_csv: Path, audit_path: Path) -> dict[str, Any]:
    compounds = json.loads(compounds_path.read_text(encoding="utf-8"))
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        map_rows = list(csv.DictReader(handle))
        fields = list(map_rows[0]) if map_rows else []
    ambiguous_names = {norm_key(row["original_name"]) for row in map_rows
                       if row.get("name_ambiguity", "").casefold() in ("true", "1", "yes")}
    ambiguous_drug_ids = {row["ddinter_id"] for row in map_rows
                          if row.get("name_ambiguity", "").casefold() in ("true", "1", "yes")}
    per_id: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for item in compounds:
        query = item.get("query_name", "")
        is_valid = bool(item.get("canonical_smiles_rdkit_validated"))
        if is_valid and norm_key(query) in ambiguous_names:
            item["identity_match_status"] = "ambiguous_ddinter_name_requires_review"
            item["usable_for_model"] = False
            counts["ambiguous_match_excluded"] += 1
        elif is_valid:
            item["identity_match_status"] = "unique_name_match"
            item["usable_for_model"] = True
            counts["accepted_unique_name_match"] += 1
        else:
            item["identity_match_status"] = item.get("status", "unresolved")
            item["usable_for_model"] = False
            counts["unresolved_or_invalid"] += 1
        per_id[item["ddinter_id"]] = item
    compounds_path.write_text(json.dumps(compounds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resolved_ids = {key for key, item in per_id.items() if item.get("usable_for_model")}
    for row in map_rows:
        item = per_id.get(row["ddinter_id"])
        if row["ddinter_id"] in resolved_ids and item:
            row["pubchem_cid"] = str(item.get("pubchem_cid", ""))
            row["mapping_status"] = "resolved_rdkit_validated_unique_name"
        elif item and item.get("identity_match_status") == "ambiguous_ddinter_name_requires_review":
            row["pubchem_cid"] = ""
            row["mapping_status"] = "ambiguous_name_requires_review"
        elif item:
            row["pubchem_cid"] = ""
            row["mapping_status"] = item.get("status", "unresolved")
        else:
            row["pubchem_cid"] = ""
            row["mapping_status"] = "not_in_enrichment_checkpoint"
    with mapping_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(map_rows)
    total_ids = len({row["ddinter_id"] for row in map_rows})
    available_3d = sum(bool(item.get("structure_3d_available")) and item.get("usable_for_model") for item in compounds)
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PubChem PUG REST",
        "unique_ddinter_drug_ids": total_ids,
        "checkpointed_ids": len(per_id),
        "resolved_rdkit_validated": counts["accepted_unique_name_match"],
        "validated_structures_ambiguous_name_excluded": counts["ambiguous_match_excluded"],
        "unresolved_or_invalid": counts["unresolved_or_invalid"],
        "validated_3d_conformers": available_3d,
        "complete": len(per_id) == total_ids,
        "coverage_fraction_of_total": (counts["accepted_unique_name_match"] / total_ids) if total_ids else 0.0,
        "status_counts": dict(counts),
        "outputs": {"compounds_json": str(compounds_path), "mapping_csv": str(mapping_csv)},
        "notes": ["Only exact PubChem synonym matches with unambiguous DDInter display names are approved for model use.",
                  "Ambiguous candidate structures remain preserved for manual review but are excluded from coverage and model inputs.",
                  "3D structures are counted only when the source record and atom mapping passed RDKit validation."],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, default=Path("data/processed/pubchem_compounds.json"))
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/processed/drug_identity_mapping.csv"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/pubchem_coverage_audit.json"))
    args = parser.parse_args()
    print(json.dumps(review(args.compounds, args.mapping_csv, args.audit), indent=2))


if __name__ == "__main__":
    main()
