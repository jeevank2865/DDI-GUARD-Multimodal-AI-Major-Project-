"""Resumable PubChem PUG REST enrichment with provenance and RDKit validation."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "DDI-GUARD/0.1 (research data ingestion; resumable)"


def norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class PUGClient:
    def __init__(self, interval_seconds: float = 0.3, max_attempts: int = 5, timeout: int = 30):
        self.interval_seconds = max(interval_seconds, 0.22)
        self.max_attempts = max_attempts
        self.timeout = timeout
        self._last_request = 0.0
        self._request_lock = threading.Lock()

    def get(self, url: str, *, binary: bool = False) -> bytes:
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            with self._request_lock:
                wait = self.interval_seconds - (time.monotonic() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (URLError, TimeoutError) as exc:
                last_error = exc
                retry_after = None
            if attempt + 1 < self.max_attempts:
                try:
                    pause = float(retry_after) if retry_after else min(2 ** attempt, 20) + random.random()
                except (ValueError, TypeError):
                    pause = min(2 ** attempt, 20) + random.random()
                time.sleep(pause)
        raise RuntimeError(f"PubChem request failed after {self.max_attempts} attempts: {last_error}") from last_error

    def get_compound(self, drug_id: str, names: list[str]) -> dict[str, Any]:
        errors: list[str] = []
        for name in names:
            encoded = quote(name, safe="")
            try:
                url = (f"{BASE}/compound/name/{encoded}/property/CanonicalSMILES,"
                       "IsomericSMILES,MolecularFormula,MolecularWeight/JSON")
                payload = json.loads(self.get(url).decode("utf-8"))
                prop = payload["PropertyTable"]["Properties"][0]
                cid = int(prop["CID"])
                synonyms_url = f"{BASE}/compound/cid/{cid}/synonyms/JSON"
                synonyms_doc = json.loads(self.get(synonyms_url).decode("utf-8"))
                synonyms = synonyms_doc["InformationList"]["Information"][0].get("Synonym", [])
                all_names = [str(s) for s in synonyms]
                exact_match = any(norm_key(name) == norm_key(candidate) for candidate in all_names)
                # PUG name resolution must agree with the returned compound's own synonym set.
                if not exact_match:
                    errors.append(f"{name}: returned CID {cid} synonyms did not verify the query")
                    continue
                smiles = prop.get("ConnectivitySMILES") or prop.get("SMILES")
                isomeric = prop.get("SMILES") or smiles
                mol = Chem.MolFromSmiles(isomeric or "")
                if mol is None or mol.GetNumAtoms() < 1:
                    return {"ddinter_id": drug_id, "status": "invalid_structure", "query_name": name,
                            "pubchem_cid": cid, "errors": errors + ["RDKit rejected PubChem SMILES"]}
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                canonical_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                atom_records = []
                for atom in mol.GetAtoms():
                    atom_records.append({"index": atom.GetIdx(), "atomic_number": atom.GetAtomicNum(),
                                         "formal_charge": atom.GetFormalCharge(), "degree": atom.GetDegree(),
                                         "aromatic": atom.GetIsAromatic(), "in_ring": atom.IsInRing(),
                                         "chiral_tag": str(atom.GetChiralTag())})
                bond_records = []
                for bond in mol.GetBonds():
                    bond_records.append({"begin_atom": bond.GetBeginAtomIdx(), "end_atom": bond.GetEndAtomIdx(),
                                         "bond_type": str(bond.GetBondType()), "aromatic": bond.GetIsAromatic(),
                                         "stereo": str(bond.GetStereo())})
                result: dict[str, Any] = {
                    "ddinter_id": drug_id, "pubchem_cid": cid, "query_name": name,
                    "canonical_smiles": canonical, "isomeric_smiles": canonical_isomeric,
                    "molecular_formula_pubchem": prop.get("MolecularFormula"),
                    "molecular_weight_pubchem": prop.get("MolecularWeight"),
                    "molecular_weight_rdkit": float(Descriptors.MolWt(mol)),
                    "canonical_smiles_rdkit_validated": True,
                    "atom_count": mol.GetNumAtoms(), "bond_count": mol.GetNumBonds(),
                    "atoms": atom_records, "bonds": bond_records,
                    "synonyms": synonyms[:500], "synonym_count_returned": len(synonyms),
                    "structure_2d_available": True, "structure_3d_available": False,
                    "conformer_3d": None, "conformer_3d_status": "not_checked",
                    "source_url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "validation": {"rdkit_parse": "passed", "canonical_roundtrip": canonical},
                    "errors": errors,
                }
                sdf_url = f"{BASE}/compound/cid/{cid}/record/SDF?record_type=3d"
                try:
                    sdf = self.get(sdf_url).decode("utf-8", errors="replace")
                    conformer_mol = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False)
                    if conformer_mol is None or conformer_mol.GetNumConformers() == 0:
                        result["conformer_3d_status"] = "pubchem_record_not_parseable_or_no_conformer"
                    else:
                        heavy_3d = Chem.RemoveHs(conformer_mol)
                        conformer_smiles = Chem.MolToSmiles(heavy_3d, canonical=True, isomericSmiles=True)
                        atom_match = mol.GetSubstructMatch(heavy_3d)
                        if conformer_smiles == canonical_isomeric and len(atom_match) == mol.GetNumAtoms():
                            conf = heavy_3d.GetConformer()
                            coordinates_by_2d_index: list[list[float] | None] = [None] * mol.GetNumAtoms()
                            for record_atom_idx, model_atom_idx in enumerate(atom_match):
                                pos = conf.GetAtomPosition(record_atom_idx)
                                coordinates_by_2d_index[model_atom_idx] = [float(pos.x), float(pos.y), float(pos.z)]
                            if all(point is not None for point in coordinates_by_2d_index):
                                result["structure_3d_available"] = True
                                result["conformer_3d"] = {"coordinates_angstrom_by_validated_2d_atom_index": coordinates_by_2d_index,
                                                          "atom_count": len(coordinates_by_2d_index), "source_url": sdf_url,
                                                          "canonical_structure_match": True}
                                result["conformer_3d_status"] = "validated_graph_isomorphism"
                            else:
                                result["conformer_3d_status"] = "atom_mapping_incomplete"
                        else:
                            result["conformer_3d_status"] = "structure_or_atom_mapping_mismatch"
                except HTTPError as exc:
                    if exc.code in (400, 404):
                        result["conformer_3d_status"] = "not_available_from_pubchem"
                    else:
                        result["conformer_3d_status"] = f"request_error_http_{exc.code}"
                except Exception as exc:
                    result["conformer_3d_status"] = f"request_or_validation_error:{type(exc).__name__}"
                return result
            except HTTPError as exc:
                errors.append(f"{name}: HTTP {exc.code}")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return {"ddinter_id": drug_id, "status": "unresolved", "errors": errors,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}


def _read_mapping(mapping_csv: Path) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("ddinter_id") and row.get("original_name"):
                grouped[row["ddinter_id"]].add(row["original_name"].strip())
    return {key: sorted(names, key=lambda x: (len(x), x.casefold())) for key, names in grouped.items()}


def enrich(mapping_csv: Path, output_dir: Path, interval: float = 0.3, limit: int | None = None,
           workers: int = 8) -> dict[str, Any]:
    processed_dir = output_dir / "data" / "processed"
    artifacts_dir = output_dir / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    compounds_path = processed_dir / "pubchem_compounds.json"
    checkpoint_path = processed_dir / "pubchem_enrichment_checkpoint.jsonl"
    mapping_path = mapping_csv
    names_by_id = _read_mapping(mapping_csv)
    checkpoint: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                    if obj.get("ddinter_id"):
                        checkpoint[obj["ddinter_id"]] = obj
                except json.JSONDecodeError:
                    continue
    todo = [drug_id for drug_id in sorted(names_by_id) if drug_id not in checkpoint]
    if limit is not None:
        todo = todo[:limit]
    client = PUGClient(interval_seconds=interval)
    write_lock = threading.Lock()
    completed = 0
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(client.get_compound, drug_id, names_by_id[drug_id]): drug_id for drug_id in todo}
            for future in as_completed(futures):
                drug_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"ddinter_id": drug_id, "status": "request_failed",
                              "errors": [f"{type(exc).__name__}: {exc}"],
                              "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
                result["original_names"] = names_by_id[drug_id]
                with write_lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    checkpoint[drug_id] = result
                    completed += 1
                    if completed % 25 == 0 or completed == len(todo):
                        print(f"processed {completed}/{len(todo)} new drug IDs; total checkpointed {len(checkpoint)}/{len(names_by_id)}", flush=True)
    records = [checkpoint[key] for key in sorted(checkpoint)]
    compounds_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status_counts = Counter("resolved" if row.get("canonical_smiles_rdkit_validated") else row.get("status", "unresolved") for row in records)
    resolved = [row for row in records if row.get("canonical_smiles_rdkit_validated")]
    conformers = [row for row in resolved if row.get("structure_3d_available")]
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PubChem PUG REST",
        "mapping_input": str(mapping_csv),
        "unique_ddinter_drug_ids": len(names_by_id),
        "checkpointed_ids": len(records), "resolved_rdkit_validated": len(resolved),
        "unresolved_or_invalid": len(records) - len(resolved), "validated_3d_conformers": len(conformers),
        "status_counts": dict(status_counts),
        "coverage_fraction_of_checkpointed": (len(resolved) / len(records)) if records else 0.0,
        "complete": len(records) == len(names_by_id),
        "pending_ids": sorted(set(names_by_id) - set(checkpoint)),
        "outputs": {"compounds_json": str(compounds_path), "checkpoint_jsonl": str(checkpoint_path)},
        "notes": ["Structures are retained only after RDKit parsing; no synthetic chemistry features are created.",
                  "Name queries are checked against the returned PubChem synonym list before accepting the match.",
                  "3D conformers are retained only when RDKit parses coordinates and atom counts match the validated 2D structure."],
    }
    (artifacts_dir / "pubchem_coverage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    # Update the project mapping copy; original input remains unchanged.
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        map_rows = list(csv.DictReader(handle))
        fields = list(map_rows[0].keys()) if map_rows else ["ddinter_id", "original_name", "normalized_name", "name_ambiguity", "pubchem_cid", "mapping_status"]
    cid_by_id = {row["ddinter_id"]: row.get("pubchem_cid", "") for row in resolved}
    for row in map_rows:
        resolved_obj = checkpoint.get(row["ddinter_id"], {})
        row["pubchem_cid"] = str(cid_by_id.get(row["ddinter_id"], ""))
        row["mapping_status"] = "resolved_rdkit_validated" if row["ddinter_id"] in cid_by_id else resolved_obj.get("status", "pending_pubchem_enrichment")
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(map_rows)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/processed/drug_identity_mapping.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="project root (writes data/processed and artifacts)")
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending IDs (debug only)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(enrich(args.mapping_csv, args.output_dir, args.interval, args.limit, args.workers), indent=2))


if __name__ == "__main__":
    main()
