import csv
import json
import tempfile
import unittest
from pathlib import Path

from ddi_guard.review_pubchem_mapping import review


class PubchemReviewTests(unittest.TestCase):
    def test_ambiguous_exact_name_is_preserved_but_excluded_from_model_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compounds_path, mapping_path, audit_path = root / "compounds.json", root / "mapping.csv", root / "audit.json"
            compounds_path.write_text(json.dumps([{
                "ddinter_id": "DDInter1", "query_name": "Iron", "pubchem_cid": 123,
                "canonical_smiles_rdkit_validated": True, "canonical_smiles": "[Fe]",
                "structure_3d_available": False, "status": None
            }]))
            with mapping_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ddinter_id", "original_name", "name_ambiguity", "pubchem_cid", "mapping_status"])
                writer.writeheader()
                writer.writerows([
                    {"ddinter_id": "DDInter1", "original_name": "Iron", "name_ambiguity": "True", "pubchem_cid": "123", "mapping_status": "resolved"},
                    {"ddinter_id": "DDInter2", "original_name": "Iron", "name_ambiguity": "True", "pubchem_cid": "123", "mapping_status": "resolved"},
                ])
            audit = review(compounds_path, mapping_path, audit_path)
            self.assertEqual(audit["resolved_rdkit_validated"], 0)
            self.assertEqual(audit["validated_structures_ambiguous_name_excluded"], 1)
            result = json.loads(compounds_path.read_text())[0]
            self.assertFalse(result["usable_for_model"])
            with mapping_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(not row["pubchem_cid"] for row in rows))


if __name__ == "__main__":
    unittest.main()
