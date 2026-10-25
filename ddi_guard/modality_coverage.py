"""Audit authentic feature availability by drug, pair, class, and fixed split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from . import LABELS

MODALITIES = ("molecular", "target", "text")


def _ids_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def load_verified_target_drugs(edges_path: Path, audit_path: Path) -> set[str]:
    """Accept only completed ChEMBL 34 import output with auditable source fields."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("observed_release") != "34":
        return set()
    required = {"ddinter_id", "target_chembl_id", "organism", "activity_type", "activity_units",
                "relation_operator", "supporting_activity_ids", "source_release"}
    result: set[str] = set()
    with edges_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"ChEMBL edge file is missing required provenance columns: {sorted(required - set(reader.fieldnames or ())) }")
        for row in reader:
            if row["source_release"] != "34":
                raise ValueError("ChEMBL edge row does not match the verified release")
            if all(row.get(field, "").strip() for field in
                   ("ddinter_id", "target_chembl_id", "organism", "activity_type", "activity_units",
                    "relation_operator", "supporting_activity_ids")):
                result.add(row["ddinter_id"])
    return result


def load_verified_text_drugs(manifest_path: Path, project_root: Path) -> set[str]:
    """Accept drug text vectors only when an evidence/source/hash manifest verifies them.

    Expected CSV columns: ddinter_id, source_id, evidence_id, source_url,
    source_version or source_date, evidence_scope, label_independent,
    embedding_path, embedding_sha256. This drug-level feature manifest accepts
    general drug evidence only; direct DDI passages belong in pair-keyed
    retrieval records, not in a drug embedding.
    """
    required = {"ddinter_id", "source_id", "evidence_id", "source_url", "label_independent",
                "embedding_path", "embedding_sha256"}
    result: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"Text manifest is missing required provenance columns: {sorted(required - set(reader.fieldnames or ())) }")
        headers = set(reader.fieldnames or ())
        if not ({"source_version", "source_date"} & headers) or "evidence_scope" not in headers:
            raise ValueError("Text manifest must include source_version or source_date and evidence_scope")
        for row in reader:
            parsed = urlparse(row["source_url"])
            if not (row["ddinter_id"].strip() and row["source_id"].strip() and row["evidence_id"].strip()
                    and parsed.scheme in {"https", "http"} and parsed.netloc
                    and (row.get("source_version", "").strip() or row.get("source_date", "").strip())
                    and row["evidence_scope"].strip().casefold() == "drug_general"
                    and _truthy(row["label_independent"])):
                continue
            embedding = (project_root / row["embedding_path"]).resolve()
            if not embedding.is_file() or not embedding.is_relative_to(project_root.resolve()):
                continue
            digest = hashlib.sha256(embedding.read_bytes()).hexdigest()
            if digest.casefold() == row["embedding_sha256"].strip().casefold():
                result.add(row["ddinter_id"])
    return result


def _summary_for_rows(rows: list[dict[str, Any]], feature_sets: dict[str, set[str]]) -> dict[str, Any]:
    classes = Counter(row["severity"] for row in rows)
    pair_sets: dict[str, set[str]] = {}
    for modality, drug_ids in feature_sets.items():
        pair_sets[modality] = {row["pair_id"] for row in rows
                               if row["drug_a_id"] in drug_ids and row["drug_b_id"] in drug_ids}

    combinations: dict[str, list[dict[str, Any]]] = {}
    for bits in product((False, True), repeat=len(MODALITIES)):
        key = "+".join(name for name, present in zip(MODALITIES, bits) if present) or "none"
        selected = [row for row in rows if all((row["pair_id"] in pair_sets[name]) == present
                                               for name, present in zip(MODALITIES, bits))]
        combinations[key] = {"pair_count": len(selected),
                             "class_counts": {label: sum(row["severity"] == label for row in selected)
                                              for label in LABELS},
                             "pair_ids_sha256": _ids_hash(row["pair_id"] for row in selected)}
    by_modality = {}
    for modality in MODALITIES:
        eligible = [row for row in rows if row["pair_id"] in pair_sets[modality]]
        excluded = [row for row in rows if row["pair_id"] not in pair_sets[modality]]
        by_modality[modality] = {
            "available_pair_count": len(eligible),
            "available_class_counts": {label: sum(row["severity"] == label for row in eligible) for label in LABELS},
            "excluded_pair_count": len(excluded),
            "excluded_class_counts": {label: sum(row["severity"] == label for row in excluded) for label in LABELS},
            "available_pair_ids_sha256": _ids_hash(row["pair_id"] for row in eligible),
        }
    return {"pair_count": len(rows), "class_counts": {label: classes[label] for label in LABELS},
            "by_modality": by_modality, "by_modality_intersection": combinations}


def build_coverage_report(pairs: list[dict[str, Any]], split_manifest: dict[str, Any], *,
                          molecular_drugs: set[str], target_drugs: set[str], text_drugs: set[str],
                          modality_sources: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    if len(pairs_by_id) != len(pairs):
        raise ValueError("pair manifest contains duplicate pair IDs")
    feature_sets = {"molecular": molecular_drugs, "target": target_drugs, "text": text_drugs}
    all_drugs = {row[key] for row in pairs for key in ("drug_a_id", "drug_b_id")}
    coverage_by_drug = {name: {"available_drugs": len(ids & all_drugs),
                                "unavailable_drugs": len(all_drugs - ids),
                                "drug_ids_sha256": _ids_hash(ids & all_drugs)}
                        for name, ids in feature_sets.items()}
    protocols: dict[str, Any] = {}
    for protocol in ("pair_disjoint", "cold_drug"):
        if protocol not in split_manifest:
            raise ValueError(f"fixed split manifest is missing protocol {protocol}")
        split = split_manifest[protocol]
        assigned: set[str] = set()
        protocol_rows: dict[str, Any] = {}
        split_drug_sets = {name: set(split.get(f"{name}_drug_ids", []))
                           for name in ("train", "validation", "test")}
        if protocol == "cold_drug":
            pairwise_drug_overlap = ((split_drug_sets["train"] & split_drug_sets["validation"])
                                     | (split_drug_sets["train"] & split_drug_sets["test"])
                                     | (split_drug_sets["validation"] & split_drug_sets["test"]))
            if pairwise_drug_overlap:
                raise ValueError(f"cold_drug protocol has {len(pairwise_drug_overlap)} drugs shared across splits")
        for split_name in ("train", "validation", "test"):
            ids = set(split[f"{split_name}_pair_ids"])
            overlap = assigned & ids
            unknown = ids - pairs_by_id.keys()
            if overlap or unknown:
                raise ValueError(f"invalid {protocol} {split_name} IDs: overlap={len(overlap)}, unknown={len(unknown)}")
            if protocol == "cold_drug":
                bad_pairs = [pair_id for pair_id in ids
                             if not {pairs_by_id[pair_id]["drug_a_id"], pairs_by_id[pair_id]["drug_b_id"]}
                             <= split_drug_sets[split_name]]
                if bad_pairs:
                    raise ValueError(f"cold_drug {split_name} includes {len(bad_pairs)} pairs outside its drug partition")
            assigned.update(ids)
            rows = [pairs_by_id[pair_id] for pair_id in ids]
            protocol_rows[split_name] = _summary_for_rows(rows, feature_sets)
        protocols[protocol] = {
            "split_pair_counts": {name: protocol_rows[name]["pair_count"] for name in protocol_rows},
            "splits": protocol_rows,
            "cold_drug_sets_disjoint": (not bool((split_drug_sets["train"] & split_drug_sets["validation"])
                                                  | (split_drug_sets["train"] & split_drug_sets["test"])
                                                  | (split_drug_sets["validation"] & split_drug_sets["test"])))
                                      if protocol == "cold_drug" else None,
        }
    return {"status": "coverage_audited", "source_dataset_pair_count": len(pairs),
            "split_seed": split_manifest.get("seed"), "modalities": list(MODALITIES),
            "feature_availability_definition": "A pair is covered by a modality when both drugs have verified source-backed features for that modality.",
            "modalities_by_drug": coverage_by_drug,
            "modality_sources": modality_sources or {}, "protocols": protocols}


def audit_project(project_root: Path) -> dict[str, Any]:
    pairs_path = project_root / "data/processed/ddi_pairs_full.json"
    split_path = project_root / "artifacts/split_manifest.json"
    graph_path = project_root / "data/processed/molecular_graph_manifest.json"
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    graph_rows = json.loads(graph_path.read_text(encoding="utf-8"))
    molecular = {row["ddinter_id"] for row in graph_rows if row.get("graph_status") == "validated"}
    target_edges = project_root / "data/processed/chembl_drug_target_edges.csv"
    target_audit = project_root / "artifacts/chembl_coverage_audit.json"
    if target_edges.is_file() and target_audit.is_file():
        targets = load_verified_target_drugs(target_edges, target_audit)
        target_source = {"status": "verified", "release": "34", "path": str(target_edges)}
    else:
        targets = set()
        target_source = {"status": "blocked_source_missing", "path": str(target_edges)}
    text_manifest = project_root / "data/processed/text_evidence_embeddings_manifest.csv"
    if text_manifest.is_file():
        texts = load_verified_text_drugs(text_manifest, project_root)
        text_source = {"status": "verified" if texts else "present_but_no_verified_vectors", "path": str(text_manifest)}
    else:
        texts = set()
        text_source = {"status": "blocked_source_missing", "path": str(text_manifest)}
    report = build_coverage_report(
        pairs, split_manifest, molecular_drugs=molecular, target_drugs=targets, text_drugs=texts,
        modality_sources={
            "molecular": {"status": "verified", "source": "RDKit validated graph manifest", "path": str(graph_path)},
            "target": target_source,
            "text": text_source,
        })
    report["dataset_manifest_sha256"] = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    report["split_manifest_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    report["molecular_graph_manifest_sha256"] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/multimodal_coverage_audit.json"))
    args = parser.parse_args()
    result = audit_project(args.project_root)
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "source_dataset_pair_count": result["source_dataset_pair_count"],
                      "modalities_by_drug": result["modalities_by_drug"]}, indent=2))


if __name__ == "__main__":
    main()
