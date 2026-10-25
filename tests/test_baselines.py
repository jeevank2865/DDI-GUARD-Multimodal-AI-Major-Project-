import json
import tempfile
import unittest
from pathlib import Path

from ddi_guard.baselines import majority_baseline


class BaselineTests(unittest.TestCase):
    def test_majority_class_is_selected_from_train_and_scored_on_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs_path = root / "pairs.json"
            split_path = root / "splits.json"
            output_path = root / "baseline.json"
            rows = [
                {"pair_id": "p1", "severity": "Moderate"},
                {"pair_id": "p2", "severity": "Moderate"},
                {"pair_id": "p3", "severity": "Major"},
                {"pair_id": "p4", "severity": "Minor"},
                {"pair_id": "p5", "severity": "Major"},
                {"pair_id": "p6", "severity": "Moderate"},
            ]
            pairs_path.write_text(json.dumps(rows), encoding="utf-8")
            split_path.write_text(json.dumps({"pair_disjoint": {
                "train_pair_ids": ["p1", "p2", "p3"],
                "test_pair_ids": ["p4", "p5", "p6"],
            }}), encoding="utf-8")

            result = majority_baseline(pairs_path, split_path, output_path)

            self.assertEqual(result["selected_class_using_train_only"], "Moderate")
            self.assertEqual(result["test_pair_count"], 3)
            self.assertAlmostEqual(result["metrics"]["accuracy"], 1 / 3)
            self.assertEqual(result["metrics"]["n"], 3)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
