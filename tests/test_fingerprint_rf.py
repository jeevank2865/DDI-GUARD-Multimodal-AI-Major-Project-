import unittest
import numpy as np
import json
import tempfile
from pathlib import Path

from ddi_guard.train_fingerprint_rf import run, symmetric_pair_fingerprint


class FingerprintBaselineTests(unittest.TestCase):
    def test_pair_features_are_symmetric(self):
        a = np.array([1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 1, 0], dtype=np.uint8)
        self.assertTrue(np.array_equal(symmetric_pair_fingerprint(a, b),
                                       symmetric_pair_fingerprint(b, a)))

    def test_tiny_random_forest_run_reports_real_split_metrics(self):
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
            (root / "artifacts/pubchem_coverage_audit.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8")
            (root / "artifacts/split_manifest.json").write_text(json.dumps({"pair_disjoint": {
                "train_pair_ids": ["tr1", "tr2", "tr3"],
                "validation_pair_ids": ["va1", "va2", "va3"],
                "test_pair_ids": ["te1", "te2", "te3"],
            }}), encoding="utf-8")

            result = run(root, "pair_disjoint", seed=7, fingerprint_size=64,
                         n_estimators=2, n_jobs=1)

            self.assertEqual(result["status"], "completed_random_forest_fingerprint_baseline")
            self.assertEqual(result["covered_split_pair_counts"],
                             {"train": 3, "validation": 3, "test": 3})
            self.assertEqual(result["test_metrics_raw"]["n"], 3)
            self.assertTrue(result["checkpoint_reload_verified"])
            self.assertTrue((root / "artifacts/fingerprint_rf_pair_disjoint/evaluation.json").is_file())
            self.assertTrue((root / "artifacts/fingerprint_rf_pair_disjoint/model.joblib").is_file())
            with np.load(root / "artifacts/fingerprint_rf_pair_disjoint/test_predictions.npz",
                         allow_pickle=False) as predictions:
                self.assertEqual(predictions["pair_ids"].tolist(), ["te1", "te2", "te3"])


if __name__ == "__main__":
    unittest.main()
