"""Validate PubChem SMILES and emit a molecule graph coverage manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem
from .features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, molecule_to_graph


def build(compounds_path: Path, manifest_path: Path, audit_path: Path) -> dict[str, Any]:
    compounds = json.loads(compounds_path.read_text(encoding="utf-8"))
    rows = []
    status_counts: Counter[str] = Counter()
    for compound in compounds:
        drug_id = compound.get("ddinter_id")
        smiles = compound.get("isomeric_smiles") or compound.get("canonical_smiles")
        if not compound.get("canonical_smiles_rdkit_validated") or not compound.get("usable_for_model", True) or not smiles:
            status = compound.get("status", "unresolved")
            status_counts[status] += 1
            rows.append({"ddinter_id": drug_id, "pubchem_cid": compound.get("pubchem_cid"),
                         "graph_status": status, "atom_count": None, "bond_count": None,
                         "smiles": None, "3d_available": False})
            continue
        try:
            graph = molecule_to_graph(smiles)
            mol = Chem.MolFromSmiles(smiles)
            graph_status = "validated"
            status_counts[graph_status] += 1
            rows.append({"ddinter_id": drug_id, "pubchem_cid": compound.get("pubchem_cid"),
                         "graph_status": graph_status, "atom_count": int(graph["x"].shape[0]),
                         "bond_count": int(mol.GetNumBonds()), "atom_feature_dim": ATOM_FEATURE_DIM,
                         "bond_feature_dim": BOND_FEATURE_DIM, "smiles": smiles,
                         "3d_available": bool(compound.get("structure_3d_available")),
                         "3d_status": compound.get("conformer_3d_status"),
                         "source_url": compound.get("source_url")})
        except Exception as exc:
            status_counts["invalid_graph"] += 1
            rows.append({"ddinter_id": drug_id, "pubchem_cid": compound.get("pubchem_cid"),
                         "graph_status": "invalid_graph", "error": f"{type(exc).__name__}: {exc}",
                         "atom_count": None, "bond_count": None, "smiles": smiles,
                         "3d_available": False})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_compounds_json": str(compounds_path), "drug_ids_in_compound_file": len(compounds),
        "validated_molecular_graphs": status_counts["validated"],
        "invalid_graphs": status_counts["invalid_graph"],
        "unresolved_or_unvalidated": len(compounds) - status_counts["validated"] - status_counts["invalid_graph"],
        "graph_status_counts": dict(status_counts),
        "atom_feature_dim": ATOM_FEATURE_DIM, "bond_feature_dim": BOND_FEATURE_DIM,
        "manifest_path": str(manifest_path),
        "note": "The graph manifest records validated 2D topology. The current model does not consume 3D coordinates."
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, default=Path("data/processed/pubchem_compounds.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/molecular_graph_manifest.json"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/molecular_graph_audit.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.compounds, args.manifest, args.audit), indent=2))


if __name__ == "__main__":
    main()
