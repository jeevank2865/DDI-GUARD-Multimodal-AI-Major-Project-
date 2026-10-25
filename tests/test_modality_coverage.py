import unittest

from ddi_guard.modality_coverage import build_coverage_report


class ModalityCoverageTests(unittest.TestCase):
    def test_coverage_intersections_account_for_every_pair_by_class_and_split(self):
        pairs = [
            {"pair_id": "p1", "drug_a_id": "a", "drug_b_id": "b", "severity": "Minor"},
            {"pair_id": "p2", "drug_a_id": "a", "drug_b_id": "c", "severity": "Moderate"},
            {"pair_id": "p3", "drug_a_id": "b", "drug_b_id": "c", "severity": "Major"},
            {"pair_id": "p4", "drug_a_id": "c", "drug_b_id": "d", "severity": "Moderate"},
            {"pair_id": "p5", "drug_a_id": "e", "drug_b_id": "f", "severity": "Major"},
        ]
        split = {"seed": 42,
                 "pair_disjoint": {"train_pair_ids": ["p1"], "validation_pair_ids": ["p2"], "test_pair_ids": ["p3"]},
                 "cold_drug": {"train_pair_ids": ["p1"], "validation_pair_ids": ["p4"], "test_pair_ids": ["p5"],
                               "train_drug_ids": ["a", "b"], "validation_drug_ids": ["c", "d"],
                               "test_drug_ids": ["e", "f"]}}
        report = build_coverage_report(pairs, split, molecular_drugs={"a", "b"},
                                       target_drugs={"a", "c"}, text_drugs=set())
        self.assertEqual(report["source_dataset_pair_count"], 5)
        for protocol in ("pair_disjoint", "cold_drug"):
            for split_name in ("train", "validation", "test"):
                summary = report["protocols"][protocol]["splits"][split_name]
                intersection_count = sum(row["pair_count"] for row in summary["by_modality_intersection"].values())
                self.assertEqual(intersection_count, summary["pair_count"])
        pair_test = report["protocols"]["pair_disjoint"]["splits"]["test"]
        self.assertEqual(pair_test["by_modality"]["molecular"]["available_pair_count"], 0)
        self.assertEqual(pair_test["by_modality"]["text"]["excluded_class_counts"]["Major"], 1)
        self.assertTrue(report["protocols"]["cold_drug"]["cold_drug_sets_disjoint"])

    def test_rejects_unknown_or_overlapping_split_ids(self):
        pairs = [{"pair_id": "p1", "drug_a_id": "a", "drug_b_id": "b", "severity": "Minor"}]
        split = {"pair_disjoint": {"train_pair_ids": ["p1"], "validation_pair_ids": ["p1"], "test_pair_ids": []},
                 "cold_drug": {"train_pair_ids": ["p1"], "validation_pair_ids": [], "test_pair_ids": []}}
        with self.assertRaisesRegex(ValueError, "invalid pair_disjoint"):
            build_coverage_report(pairs, split, molecular_drugs=set(), target_drugs=set(), text_drugs=set())


if __name__ == "__main__":
    unittest.main()
