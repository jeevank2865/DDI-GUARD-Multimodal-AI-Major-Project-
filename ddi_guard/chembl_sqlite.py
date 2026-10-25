"""Version-pinned ChEMBL SQLite activity ingestion; never treats all activities as equal."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from contextlib import closing
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_RELEASE = "34"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in db.execute(f'PRAGMA table_info("{table}")')}


def find_release(db: sqlite3.Connection) -> str | None:
    tables = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "version" not in tables:
        return None
    cols = table_columns(db, "version")
    # ChEMBL release SQLite uses `version.name` with a value like ChEMBL_34;
    # small fixtures and some compatible exports use a `version` column.
    column = "version" if "version" in cols else "name" if "name" in cols else None
    if column is None:
        return None
    values = [str(row[0]) for row in db.execute(f'SELECT "{column}" FROM "version"') if row[0] is not None]
    for value in values:
        match = re.fullmatch(r"ChEMBL[_ -]?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        if column == "version" and value.isdigit():
            return value
    return None


def ingest_chembl(db_path: Path, mapping_csv: Path, output_root: Path,
                  expected_release: str = EXPECTED_RELEASE) -> dict[str, Any]:
    processed = output_root / "data" / "processed"
    artifacts = output_root / "artifacts"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    audit_path = artifacts / "chembl_coverage_audit.json"
    audit: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official ChEMBL relational database",
        "source_path": str(db_path), "required_release": expected_release,
        "mapped_drugs": 0, "unique_targets": 0, "assays": 0,
        "retained_bioactivity_records": 0, "usable_drug_target_edges": 0,
        "status": "blocked", "reason": None,
    }
    try:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except Exception as exc:
        audit["reason"] = f"Could not open SQLite database read-only: {type(exc).__name__}: {exc}"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return audit
    with closing(connection) as db:
        release = find_release(db)
        audit["observed_release"] = release
        if release != expected_release:
            audit["reason"] = f"Release mismatch or missing version metadata: expected {expected_release}, observed {release}"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            return audit
        tables = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required_tables = {"activities", "assays", "target_dictionary", "molecule_dictionary", "molecule_synonyms"}
        if not required_tables <= tables:
            audit["reason"] = f"Missing expected ChEMBL tables: {sorted(required_tables - tables)}"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            return audit
        activity_cols = table_columns(db, "activities")
        assay_cols = table_columns(db, "assays")
        target_cols = table_columns(db, "target_dictionary")
        molecule_cols = table_columns(db, "molecule_dictionary")
        synonym_cols = table_columns(db, "molecule_synonyms")
        required_columns = {
            "activities": {"activity_id", "assay_id", "molregno", "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value", "data_validity_comment"},
            "assays": {"assay_id", "confidence_score", "description"},
            "target_dictionary": {"pref_name", "target_type", "organism"},
            "molecule_dictionary": {"molregno", "chembl_id", "pref_name"},
            "molecule_synonyms": {"molregno", "synonyms"},
        }
        observed = {"activities": activity_cols, "assays": assay_cols,
                    "target_dictionary": target_cols, "molecule_dictionary": molecule_cols,
                    "molecule_synonyms": synonym_cols}
        missing = {table: sorted(cols - observed[table]) for table, cols in required_columns.items() if not cols <= observed[table]}
        if not ({"assay_chembl_id", "chembl_id"} & assay_cols):
            missing["assays"] = sorted(set(missing.get("assays", [])) | {"assay_chembl_id|chembl_id"})
        if not ({"target_chembl_id", "tid"} & assay_cols):
            missing["assays"] = sorted(set(missing.get("assays", [])) | {"target_chembl_id|tid"})
        if not ({"target_chembl_id", "chembl_id"} & target_cols):
            missing["target_dictionary"] = sorted(set(missing.get("target_dictionary", [])) | {"target_chembl_id|chembl_id"})
        if missing:
            audit["reason"] = f"ChEMBL schema differs from supported schema: {missing}"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            return audit
        assay_id_column = "assay_chembl_id" if "assay_chembl_id" in assay_cols else "chembl_id"
        assay_target_column = "target_chembl_id" if "target_chembl_id" in assay_cols else "tid"
        target_id_column = "target_chembl_id" if "target_chembl_id" in target_cols else "chembl_id"
        target_join_column = "target_chembl_id" if "target_chembl_id" in target_cols else "tid"
        mapped: dict[str, dict[str, Any]] = {}
        with mapping_csv.open(encoding="utf-8", newline="") as handle:
            dd_rows = list(csv.DictReader(handle))
        names_by_id: dict[str, set[str]] = defaultdict(set)
        for row in dd_rows:
            # Refuse identifiers whose display name is known to identify multiple DDInter IDs.
            if str(row.get("name_ambiguity", "")).casefold() not in ("true", "1", "yes"):
                names_by_id[row["ddinter_id"]].add(row["original_name"])
        wanted_names = sorted({norm(name) for names in names_by_id.values() for name in names if norm(name)})
        # Keep memory bounded: scan ChEMBL name columns in SQLite and retain only exact requested-name matches.
        db.execute("CREATE TEMP TABLE ddi_wanted_names(name_key TEXT PRIMARY KEY)")
        db.executemany("INSERT OR IGNORE INTO ddi_wanted_names(name_key) VALUES (?)", ((name,) for name in wanted_names))
        db.create_function("ddi_norm", 1, lambda value: norm(str(value)) if value is not None else "")
        by_name: dict[str, set[str]] = defaultdict(set)
        for row in db.execute("""
            SELECT DISTINCT m.chembl_id, m.pref_name
            FROM molecule_dictionary m JOIN ddi_wanted_names w ON w.name_key = ddi_norm(m.pref_name)
            WHERE m.pref_name IS NOT NULL
        """):
            by_name[norm(str(row[1]))].add(str(row[0]))
        for row in db.execute("""
            SELECT DISTINCT m.chembl_id, s.synonyms
            FROM molecule_synonyms s JOIN molecule_dictionary m ON m.molregno = s.molregno
            JOIN ddi_wanted_names w ON w.name_key = ddi_norm(s.synonyms)
            WHERE s.synonyms IS NOT NULL
        """):
            by_name[norm(str(row[1]))].add(str(row[0]))
        ambiguous_ids: set[str] = set()
        for drug_id, names in names_by_id.items():
            matches = {chembl_id for name in names for chembl_id in by_name.get(norm(name), set())}
            if len(matches) == 1:
                chembl_id = next(iter(matches))
                mapped[drug_id] = {"ddinter_id": drug_id, "chembl_id": chembl_id,
                                   "matched_names": sorted(names), "match_method": "exact_normalized_pref_name_or_synonym"}
            elif len(matches) > 1:
                ambiguous_ids.add(drug_id)
        chembl_ids = sorted({row["chembl_id"] for row in mapped.values()})
        if not chembl_ids:
            audit.update({"status": "blocked", "reason": "No unambiguous exact-name/synonym molecule mappings found."})
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            return audit
        molecule_to_ddi: dict[str, list[str]] = defaultdict(list)
        for drug_id, item in mapped.items():
            molecule_to_ddi[item["chembl_id"]].append(drug_id)
        doc_join = ""
        doc_select = "NULL AS document_id, NULL AS document_chembl_id, NULL AS pubmed_id, NULL AS doi, NULL AS document_year, NULL AS document_title"
        if "docs" in tables and "doc_id" in assay_cols:
            docs_cols = table_columns(db, "docs")
            select_doc_fields = {
                "document_chembl_id": "chembl_id", "pubmed_id": "pubmed_id", "doi": "doi",
                "document_year": "year", "document_title": "title",
            }
            doc_select = "s.doc_id AS document_id, " + ", ".join(
                f"d.{column} AS {alias}" if column in docs_cols else f"NULL AS {alias}"
                for alias, column in select_doc_fields.items())
            doc_join = "LEFT JOIN docs d ON d.doc_id = s.doc_id"
        manifest_path = processed / "chembl_bioactivity_manifest.csv"
        edges_path = processed / "chembl_drug_target_edges.csv"
        fields = ["activity_id", "ddinter_id", "molecule_chembl_id", "assay_chembl_id", "document_id",
                  "document_chembl_id", "pubmed_id", "doi", "document_year", "document_title",
                  "target_chembl_id", "target_name", "target_type", "organism", "activity_type",
                  "relation_operator", "activity_value", "activity_units", "pchembl_value",
                  "confidence_score", "assay_description", "data_validity_comment", "potential_duplicate",
                  "activity_comment", "bao_endpoint", "quality_status", "source_release"]
        retained, usable, target_ids, assays = 0, 0, set(), set()
        relation_keys: Counter[tuple[str, str, str]] = Counter()
        combined: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        with manifest_path.open("w", encoding="utf-8", newline="") as mf:
            writer = csv.DictWriter(mf, fieldnames=fields)
            writer.writeheader()
            for offset in range(0, len(chembl_ids), 500):
                id_chunk = chembl_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in id_chunk)
                query = """
                  SELECT a.activity_id, a.standard_type, a.standard_relation, a.standard_value,
                         a.standard_units, a.pchembl_value, a.data_validity_comment,
                         a.potential_duplicate, a.activity_comment, a.bao_endpoint,
                         m.chembl_id AS molecule_chembl_id, s.{assay_id_column} AS assay_chembl_id,
                         t.{target_id_column} AS target_chembl_id, s.confidence_score, s.description AS assay_description,
                         t.pref_name AS target_name, t.target_type, t.organism, {doc_select}
                  FROM activities a
                  JOIN molecule_dictionary m ON m.molregno = a.molregno
                  JOIN assays s ON s.assay_id = a.assay_id
                  JOIN target_dictionary t ON t.{target_join_column} = s.{assay_target_column}
                  {doc_join}
                  WHERE m.chembl_id IN ({placeholders})
                  ORDER BY a.activity_id
                """.format(assay_id_column=assay_id_column, target_id_column=target_id_column,
                           target_join_column=target_join_column,
                           assay_target_column=assay_target_column, doc_select=doc_select,
                           doc_join=doc_join, placeholders=placeholders)
                for activity in db.execute(query, id_chunk):
                    confidence = activity["confidence_score"]
                    valid_comment = activity["data_validity_comment"] in (None, "Manually validated")
                    quality_ok = (confidence is not None and float(confidence) >= 8 and
                                  activity["standard_value"] is not None and activity["standard_units"] not in (None, "") and valid_comment)
                    quality = "retained_usable" if quality_ok else "preserved_filtered"
                    for drug_id in molecule_to_ddi[activity["molecule_chembl_id"]]:
                        row = {
                            "activity_id": activity["activity_id"], "ddinter_id": drug_id,
                            "molecule_chembl_id": activity["molecule_chembl_id"],
                            "assay_chembl_id": activity["assay_chembl_id"], "document_id": activity["document_id"],
                            "document_chembl_id": activity["document_chembl_id"], "pubmed_id": activity["pubmed_id"],
                            "doi": activity["doi"], "document_year": activity["document_year"],
                            "document_title": activity["document_title"], "target_chembl_id": activity["target_chembl_id"],
                            "target_name": activity["target_name"], "target_type": activity["target_type"],
                            "organism": activity["organism"], "activity_type": activity["standard_type"],
                            "relation_operator": activity["standard_relation"], "activity_value": activity["standard_value"],
                            "activity_units": activity["standard_units"], "pchembl_value": activity["pchembl_value"],
                            "confidence_score": confidence, "assay_description": activity["assay_description"],
                            "data_validity_comment": activity["data_validity_comment"],
                            "potential_duplicate": activity["potential_duplicate"], "activity_comment": activity["activity_comment"],
                            "bao_endpoint": activity["bao_endpoint"], "quality_status": quality, "source_release": release,
                        }
                        writer.writerow(row)
                        retained += 1
                        target_ids.add(activity["target_chembl_id"])
                        assays.add(activity["assay_chembl_id"])
                        if quality_ok:
                            usable += 1
                            relation = (str(activity["standard_type"]), str(activity["standard_units"]),
                                        str(activity["standard_relation"]))
                            relation_keys[relation] += 1
                            key = (drug_id, str(activity["target_chembl_id"]), relation[0], relation[1], relation[2])
                            if key not in combined:
                                combined[key] = {"ddinter_id": drug_id,
                                                 "molecule_chembl_id": activity["molecule_chembl_id"],
                                                 "target_chembl_id": activity["target_chembl_id"],
                                                 "target_name": activity["target_name"], "organism": activity["organism"],
                                                 "activity_type": relation[0], "activity_units": relation[1],
                                                 "relation_operator": relation[2],
                                                 "confidence_score": float(confidence),
                                                 "edge_weight": min(float(confidence), 9.0) / 9.0,
                                                 "supporting_activity_ids": set(), "assay_chembl_ids": set(),
                                                 "source_release": release}
                            combined[key]["supporting_activity_ids"].add(str(activity["activity_id"]))
                            combined[key]["assay_chembl_ids"].add(str(activity["assay_chembl_id"]))
        relation_id = {key: index for index, key in enumerate(sorted(relation_keys))}
        edge_fields = ["ddinter_id", "molecule_chembl_id", "target_chembl_id", "target_name", "organism",
                       "activity_type", "activity_units", "relation_operator", "relation_id", "confidence_score",
                       "edge_weight", "supporting_activity_ids", "assay_chembl_ids", "source_release"]
        with edges_path.open("w", encoding="utf-8", newline="") as ef:
            writer = csv.DictWriter(ef, fieldnames=edge_fields)
            writer.writeheader()
            for edge in combined.values():
                rel_key = (edge["activity_type"], edge["activity_units"], edge["relation_operator"])
                edge["relation_id"] = relation_id[rel_key]
                edge["supporting_activity_ids"] = json.dumps(sorted(edge["supporting_activity_ids"]))
                edge["assay_chembl_ids"] = json.dumps(sorted(edge["assay_chembl_ids"]))
                writer.writerow(edge)
        audit.update({
            "status": "completed", "mapped_drugs": len(mapped), "ambiguous_drugs": len(ambiguous_ids),
            "unique_targets": len(target_ids), "assays": len(assays),
            "retained_bioactivity_records": retained, "usable_drug_target_edges": len(combined),
            "usable_activity_records": usable,
            "quality_filter": {"confidence_score_minimum": 8,
                                "standard_value_and_units_required": True,
                                "data_validity_comment_allowed": [None, "Manually validated"],
                                "relation_preservation": "Distinct relation IDs are maintained for each activity type and unit; no values or units are merged."},
            "relation_vocabulary": [{"relation_id": index, "activity_type": key[0], "activity_units": key[1],
                                      "relation_operator": key[2], "supporting_records": relation_keys[key]}
                                     for key, index in relation_id.items()],
            "outputs": {"bioactivity_manifest": str(manifest_path), "drug_target_edges": str(edges_path)},
        })
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="read-only official ChEMBL SQLite database")
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/processed/drug_identity_mapping.csv"))
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-release", default=EXPECTED_RELEASE)
    args = parser.parse_args()
    print(json.dumps(ingest_chembl(args.db, args.mapping_csv, args.output_root, args.expected_release), indent=2))


if __name__ == "__main__":
    main()
