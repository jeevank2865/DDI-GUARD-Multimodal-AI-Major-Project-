import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from ddi_guard.train_molecular import _protocol_rows, train_protocol


class MolecularTrainingCoverageTests(unittest.TestCase):
    def test_missing_structure_pairs_are_recorded_and_covered_pairs_retained(self):
        split = {"train_pair_ids": ["covered", "missing"]}
        pairs = {
            "covered": {"pair_id": "covered", "drug_a_id": "d1", "drug_b_id": "d2", "severity": "Minor"},
            "missing": {"pair_id": "missing", "drug_a_id": "d1", "drug_b_id": "d3", "severity": "Major"},
        }
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["protocol", "split", "pair_id", "reason", "missing_drug_ids"])
        writer.writeheader()

        result = _protocol_rows(split, pairs, {"d1", "d2"}, "pair_disjoint", ("train",), writer)

        self.assertEqual([row["pair_id"] for row in result["train"]], ["covered"])
        ledger = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(ledger[0]["protocol"], "pair_disjoint")
        self.assertEqual(ledger[0]["pair_id"], "missing")
        self.assertEqual(ledger[0]["missing_drug_ids"], "d3")

    def test_tiny_train_validation_test_run_writes_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("data/processed", "artifacts"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            drug_ids = [f"d{i}" for i in range(1, 19)]
            pairs = [
                {"pair_id": "tr1", "drug_a_id": "d1", "drug_b_id": "d2", "severity": "Minor"},
                {"pair_id": "tr2", "drug_a_id": "d3", "drug_b_id": "d4", "severity": "Moderate"},
                {"pair_id": "tr3", "drug_a_id": "d5", "drug_b_id": "d6", "severity": "Major"},
                {"pair_id": "va1", "drug_a_id": "d7", "drug_b_id": "d8", "severity": "Minor"},
                {"pair_id": "va2", "drug_a_id": "d9", "drug_b_id": "d10", "severity": "Moderate"},
                {"pair_id": "va3", "drug_a_id": "d11", "drug_b_id": "d12", "severity": "Major"},
                {"pair_id": "te1", "drug_a_id": "d13", "drug_b_id": "d14", "severity": "Minor"},
                {"pair_id": "te2", "drug_a_id": "d15", "drug_b_id": "d16", "severity": "Moderate"},
                {"pair_id": "te3", "drug_a_id": "d17", "drug_b_id": "d18", "severity": "Major"},
            ]
            smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1", "CCOC", "CCCl", "CCCC", "CCOC(=O)C", "CN(C)C", "CCS", "C1CC1", "CC(=O)N", "CO", "CCOCC", "C#N", "C=CC", "CC(C)O"]
            (root / "data/processed/ddi_pairs_full.json").write_text(json.dumps(pairs), encoding="utf-8")
            (root / "data/processed/pubchem_compounds.json").write_text(json.dumps([
                {"ddinter_id": drug_id, "usable_for_model": True,
                 "canonical_smiles_rdkit_validated": True, "isomeric_smiles": value}
                for drug_id, value in zip(drug_ids, smiles)
            ]), encoding="utf-8")
            (root / "data/processed/drug_identity_mapping.csv").write_text(
                "ddinter_id,original_name\n" + "".join(f"{drug_id},{drug_id}\n" for drug_id in drug_ids),
                encoding="utf-8")
            (root / "artifacts/pubchem_coverage_audit.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8")
            (root / "artifacts/split_manifest.json").write_text(json.dumps({"pair_disjoint": {
                "train_pair_ids": ["tr1", "tr2", "tr3"],
                "validation_pair_ids": ["va1", "va2", "va3"],
                "test_pair_ids": ["te1", "te2", "te3"],
            }}), encoding="utf-8")

            result = train_protocol(root, "pair_disjoint", seed=7, epochs=1,
                                    batch_size=2, patience=1, hidden_dim=8)

            self.assertEqual(result["status"], "completed_molecular_only_ablation")
            self.assertEqual(result["train_pair_count"], 3)
            self.assertEqual(result["validation_pair_count"], 3)
            self.assertEqual(result["test_pair_count"], 3)
            self.assertTrue((root / "artifacts/molecular_pair_disjoint/best_validation.pt").is_file())
            self.assertTrue((root / "artifacts/molecular_pair_disjoint/evaluation.json").is_file())
            resumed = train_protocol(root, "pair_disjoint", seed=7, epochs=2,
                                     batch_size=2, patience=2, hidden_dim=8, resume=True)
            self.assertEqual(resumed["resumed_from_epoch"], 1)
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertTrue((root / "artifacts/molecular_pair_disjoint/last_training_state.pt").is_file())


if __name__ == "__main__":
    unittest.main()
