import unittest

from ddi_guard.splits import cold_drug_split, pair_disjoint_split


class SplitTests(unittest.TestCase):
    def setUp(self):
        labels = ["Minor", "Moderate", "Major"]
        self.pairs = []
        for i in range(90):
            self.pairs.append({"pair_id": f"p{i}", "drug_a_id": f"d{i % 12}",
                               "drug_b_id": f"d{(i * 5 + 1) % 12}", "severity": labels[i % 3]})

    def test_pair_disjoint_ids_do_not_cross_splits(self):
        split = pair_disjoint_split(self.pairs, seed=42)
        train, validation, test = (set(split[k]) for k in ("train_pair_ids", "validation_pair_ids", "test_pair_ids"))
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, {row["pair_id"] for row in self.pairs})

    def test_cold_drug_has_disjoint_drug_sets_and_excludes_cross_pairs(self):
        split = cold_drug_split(self.pairs, seed=11, drug_holdout_fraction=0.25)
        train_drugs = set(split["train_drug_ids"])
        validation_drugs = set(split["validation_drug_ids"])
        test_drugs = set(split["test_drug_ids"])
        heldout = validation_drugs | test_drugs
        self.assertFalse(train_drugs & heldout)
        self.assertFalse(validation_drugs & test_drugs)
        by_id = {row["pair_id"]: row for row in self.pairs}
        for pair_id in split["train_pair_ids"]:
            pair = by_id[pair_id]
            self.assertTrue({pair["drug_a_id"], pair["drug_b_id"]} <= train_drugs)
        for pair_id in split["validation_pair_ids"]:
            pair = by_id[pair_id]
            self.assertTrue({pair["drug_a_id"], pair["drug_b_id"]} <= validation_drugs)
        for pair_id in split["test_pair_ids"]:
            pair = by_id[pair_id]
            self.assertTrue({pair["drug_a_id"], pair["drug_b_id"]} <= test_drugs)
        partitions = [set(split[f"{name}_pair_ids"]) for name in ("train", "validation", "test")]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual(sum(len(part) for part in partitions) + len(split["excluded_cross_partition_pair_ids"]), len(self.pairs))


if __name__ == "__main__":
    unittest.main()
