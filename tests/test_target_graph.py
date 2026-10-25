import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ddi_guard.target_graph import build_target_graph


class TargetGraphTests(unittest.TestCase):
    def test_verified_target_edges_become_bidirectional_relation_aware_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "mapping.csv"
            mapping.write_text("ddinter_id,original_name\nDDInter1,Aspirin\nDDInter2,Warfarin\n", encoding="utf-8")
            edges = root / "edges.csv"
            fields = ["ddinter_id", "target_chembl_id", "target_name", "organism", "activity_type",
                      "activity_units", "relation_operator", "relation_id", "confidence_score", "edge_weight",
                      "supporting_activity_ids", "assay_chembl_ids", "source_release"]
            with edges.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"ddinter_id": "DDInter1", "target_chembl_id": "CHEMBL_T1",
                                 "target_name": "Target 1", "organism": "Homo sapiens",
                                 "activity_type": "IC50", "activity_units": "nM", "relation_operator": "=",
                                 "relation_id": 0, "confidence_score": 9, "edge_weight": 1,
                                 "supporting_activity_ids": '["ACT1"]', "assay_chembl_ids": '["ASSAY1"]',
                                 "source_release": "34"})
            audit = root / "audit.json"
            audit.write_text(json.dumps({"status": "completed", "observed_release": "34"}), encoding="utf-8")
            result = build_target_graph(edges, mapping, audit, root / "out")
            graph = torch.load(result["graph_tensor_path"], weights_only=True)
            self.assertEqual(result["directed_message_edges"], 2)
            self.assertEqual(result["drug_count_with_edges"], 1)
            self.assertEqual(graph["edge_index"].tolist(), [[0, 2], [2, 0]])
            self.assertEqual(graph["relation"].tolist(), [0, 0])
            self.assertEqual(result["target_index_to_record"][0]["organism"], "Homo sapiens")
            self.assertIn("not automatically a clinical DDI mechanism", result["interpretation_limit"])

    def test_rejects_unverified_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "mapping.csv"
            mapping.write_text("ddinter_id,original_name\nDDInter1,Aspirin\n", encoding="utf-8")
            edges = root / "edges.csv"
            edges.write_text("\n", encoding="utf-8")
            audit = root / "audit.json"
            audit.write_text(json.dumps({"status": "completed", "observed_release": "37"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified ChEMBL 34"):
                build_target_graph(edges, mapping, audit, root / "out")


if __name__ == "__main__":
    unittest.main()
