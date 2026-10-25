"""Build a relation-aware, bidirectional graph from verified ChEMBL 34 edges."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_target_graph(edges_csv: Path, mapping_csv: Path, audit_path: Path,
                       output_root: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("observed_release") != "34":
        raise ValueError("target graph requires a completed, release-verified ChEMBL 34 import")
    required = {"ddinter_id", "target_chembl_id", "target_name", "organism", "activity_type",
                "activity_units", "relation_operator", "relation_id", "confidence_score",
                "edge_weight", "supporting_activity_ids", "assay_chembl_ids", "source_release"}
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        drug_ids = sorted({row["ddinter_id"] for row in csv.DictReader(handle) if row.get("ddinter_id")})
    if not drug_ids:
        raise ValueError("reviewed DDInter-to-drug mapping is empty")
    drug_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    relationships: list[dict[str, Any]] = []
    target_descriptors: dict[str, tuple[str, str]] = {}
    relation_descriptors: dict[int, tuple[str, str, str]] = {}
    with edges_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"edge manifest lacks required columns: {sorted(required - set(reader.fieldnames or ())) }")
        for row in reader:
            if row["source_release"] != "34":
                raise ValueError(f"edge source release mismatch for {row.get('target_chembl_id')}")
            if row["ddinter_id"] not in drug_index:
                raise ValueError(f"edge references unknown DDInter drug ID {row['ddinter_id']}")
            required_values = ("target_chembl_id", "organism", "activity_type", "activity_units",
                               "relation_operator", "supporting_activity_ids", "assay_chembl_ids")
            if any(not row.get(key, "").strip() for key in required_values):
                raise ValueError("edge has missing target, organism, activity, relation, or source provenance")
            target_id = row["target_chembl_id"]
            target_descriptor = (row["target_name"], row["organism"])
            if target_id in target_descriptors and target_descriptors[target_id] != target_descriptor:
                raise ValueError(f"inconsistent name or organism for target {target_id}")
            target_descriptors[target_id] = target_descriptor
            relation_id = int(row["relation_id"])
            relation_descriptor = (row["activity_type"], row["activity_units"], row["relation_operator"])
            if relation_id in relation_descriptors and relation_descriptors[relation_id] != relation_descriptor:
                raise ValueError(f"relation ID {relation_id} maps to inconsistent activity descriptors")
            relation_descriptors[relation_id] = relation_descriptor
            confidence = float(row["confidence_score"])
            edge_weight = float(row["edge_weight"])
            if confidence < 8 or not 0.0 < edge_weight <= 1.0:
                raise ValueError("edge does not meet the importer confidence/weight threshold")
            activities = json.loads(row["supporting_activity_ids"])
            assays = json.loads(row["assay_chembl_ids"])
            if not isinstance(activities, list) or not activities or not isinstance(assays, list) or not assays:
                raise ValueError("edge must retain source activity and assay identifiers")
            relationships.append({"drug_id": row["ddinter_id"], "target_id": target_id,
                                  "relation_id": relation_id, "edge_weight": edge_weight,
                                  "activity_ids": activities, "assay_ids": assays})
    if not relationships:
        raise ValueError("verified ChEMBL import contains no usable drug-target edges")
    target_ids = sorted(target_descriptors)
    target_index = {target_id: len(drug_ids) + index for index, target_id in enumerate(target_ids)}
    relation_ids = sorted(relation_descriptors)
    relation_index = {old_id: index for index, old_id in enumerate(relation_ids)}
    edge_pairs: list[tuple[int, int]] = []
    relation_values: list[int] = []
    edge_weights: list[float] = []
    for record in relationships:
        drug_node = drug_index[record["drug_id"]]
        target_node = target_index[record["target_id"]]
        relation = relation_index[record["relation_id"]]
        weight = record["edge_weight"]
        # Each verified interaction is represented in both directions so target
        # messages can update the corresponding drug embedding.
        edge_pairs.extend(((drug_node, target_node), (target_node, drug_node)))
        relation_values.extend((relation, relation))
        edge_weights.extend((weight, weight))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    relation_tensor = torch.tensor(relation_values, dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
    output_root.mkdir(parents=True, exist_ok=True)
    graph_path = output_root / "target_graph.pt"
    temp_graph = graph_path.with_name(graph_path.name + ".tmp")
    torch.save({"edge_index": edge_index, "relation": relation_tensor,
                "edge_weight": edge_weight_tensor}, temp_graph)
    os.replace(temp_graph, graph_path)
    manifest = {
        "format_version": 1, "status": "verified_chEMBL_34_target_graph",
        "source_release": "34", "source_edge_manifest": str(edges_csv),
        "source_edge_manifest_sha256": _sha256(edges_csv),
        "mapping_csv": str(mapping_csv), "mapping_csv_sha256": _sha256(mapping_csv),
        "chembl_import_audit": str(audit_path), "chembl_import_audit_sha256": _sha256(audit_path),
        "graph_tensor_path": str(graph_path), "graph_tensor_sha256": _sha256(graph_path),
        "drug_index_to_ddinter_id": drug_ids,
        "target_index_to_record": [{"node_index": target_index[target_id], "target_chembl_id": target_id,
                                     "target_name": target_descriptors[target_id][0],
                                     "organism": target_descriptors[target_id][1]}
                                    for target_id in target_ids],
        "relation_index_to_descriptor": [{"relation_index": relation_index[relation_id],
                                          "source_relation_id": relation_id,
                                          "activity_type": relation_descriptors[relation_id][0],
                                          "activity_units": relation_descriptors[relation_id][1],
                                          "relation_operator": relation_descriptors[relation_id][2]}
                                         for relation_id in relation_ids],
        "usable_source_edge_rows": len(relationships),
        "directed_message_edges": len(edge_pairs),
        "drug_count": len(drug_ids), "drug_count_with_edges": len({r["drug_id"] for r in relationships}),
        "target_count": len(target_ids), "relation_count": len(relation_ids),
        "edge_weight_definition": "verified importer confidence_score scaled to [0,1]",
        "label_leakage_check": "DDInter severity labels are not used to build nodes, edges, or relations.",
        "interpretation_limit": "ChEMBL bioactivity is drug-target evidence, not automatically a clinical DDI mechanism.",
    }
    manifest_path = output_root / "target_graph_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges-csv", type=Path, default=Path("data/processed/chembl_drug_target_edges.csv"))
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/processed/drug_identity_mapping.csv"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/chembl_coverage_audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/chembl_34_target_graph"))
    args = parser.parse_args()
    result = build_target_graph(args.edges_csv, args.mapping_csv, args.audit, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "usable_source_edge_rows", "directed_message_edges",
                                                   "drug_count_with_edges", "target_count", "relation_count",
                                                   "graph_tensor_path", "graph_tensor_sha256")}, indent=2))


if __name__ == "__main__":
    main()
