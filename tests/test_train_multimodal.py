from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ddi_guard.modality_coverage import load_verified_text_drugs
from ddi_guard.train_multimodal import _load_sources, _load_text_source, _pair_mask, _select_rows, _validate_fixed_split


class MultimodalTrainingInputTests(unittest.TestCase):
    def test_target_training_fails_closed_without_verified_chembl_34_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "verified ChEMBL 34 edges and target graph are required"):
                _load_sources(Path(temp), ("target",))

    def test_text_vector_requires_source_fields_label_independence_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data/processed").mkdir(parents=True)
            vector_path = root / "data/processed/vector.npy"
            np.save(vector_path, np.asarray([0.2, 0.7], dtype=np.float32))
            digest = hashlib.sha256(vector_path.read_bytes()).hexdigest()
            manifest = root / "data/processed/text_evidence_embeddings_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("ddinter_id", "source_id", "evidence_id", "source_url",
                    "source_version", "evidence_scope", "label_independent", "embedding_path", "embedding_sha256"))
                writer.writeheader()
                writer.writerow({"ddinter_id": "DDI1", "source_id": "SRC1", "evidence_id": "EV1",
                    "source_url": "https://example.org/evidence/1", "source_version": "v1",
                    "evidence_scope": "drug_general", "label_independent": "true",
                    "embedding_path": "data/processed/vector.npy", "embedding_sha256": digest})
                writer.writerow({"ddinter_id": "DDI2", "source_id": "SRC1", "evidence_id": "EV2",
                    "source_url": "https://example.org/evidence/2", "source_version": "v1",
                    "evidence_scope": "direct_ddi", "label_independent": "true",
                    "embedding_path": "data/processed/vector.npy", "embedding_sha256": digest})
            vectors, metadata = _load_text_source(root)
            self.assertEqual(vectors["DDI1"].shape, (2,))
            self.assertEqual(metadata["verified_drug_count"], 1)
            self.assertEqual(load_verified_text_drugs(manifest, root), {"DDI1"})

            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("ddinter_id", "source_id", "evidence_id", "source_url",
                    "source_version", "evidence_scope", "label_independent", "embedding_path", "embedding_sha256"))
                writer.writeheader()
                writer.writerow({"ddinter_id": "DDI1", "source_id": "SRC1", "evidence_id": "EV1",
                    "source_url": "https://example.org/evidence/1", "source_version": "v1",
                    "evidence_scope": "drug_general", "label_independent": "true",
                    "embedding_path": "data/processed/vector.npy", "embedding_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "no hash-verified"):
                _load_text_source(root)

    def test_pair_masks_preserve_missing_modalities_explicitly(self):
        sources = {
            "molecular_graphs": {"DDI1": object(), "DDI2": object()},
            "target_drugs": {"DDI1"},
            "text_vectors": {"DDI1": object(), "DDI2": object()},
        }
        row = {"drug_a_id": "DDI1", "drug_b_id": "DDI2"}
        self.assertEqual(_pair_mask(row, sources, ("molecular", "target", "text")),
                         [True, False, True])
        self.assertEqual(_pair_mask(row, sources, ("target",)), [False, False, False])

    def test_required_support_filter_keeps_common_ids_and_selected_modality_mask(self):
        sources = {"molecular_graphs": {drug: object() for drug in ("A", "B", "C")},
                   "target_drugs": {"A", "B"}, "text_vectors": {drug: object() for drug in ("A", "B", "C")}}
        pairs = {"P1": {"pair_id": "P1", "drug_a_id": "A", "drug_b_id": "B", "severity": "Minor"},
                 "P2": {"pair_id": "P2", "drug_a_id": "A", "drug_b_id": "C", "severity": "Moderate"}}
        split = {name + "_pair_ids": ["P1", "P2"] for name in ("train", "validation", "test")}
        ledger = csv.DictWriter(io.StringIO(), fieldnames=("protocol", "split", "pair_id", "reason", "available_modalities"))
        selected = _select_rows(split, pairs, sources, ("molecular", "text"), "pair_disjoint", ledger,
                                support_modalities=("molecular", "target", "text"))
        self.assertEqual([row["pair_id"] for row in selected["test"]], ["P1"])
        self.assertEqual(selected["test"][0]["modality_mask"], [True, False, True])

    def test_pair_disjoint_split_rejects_pair_overlap(self):
        pairs = {"P1": {"drug_a_id": "A", "drug_b_id": "B"},
                 "P2": {"drug_a_id": "A", "drug_b_id": "C"},
                 "P3": {"drug_a_id": "B", "drug_b_id": "C"}}
        split = {"train_pair_ids": ["P1"], "validation_pair_ids": ["P1"], "test_pair_ids": ["P3"]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_fixed_split("pair_disjoint", pairs, split)

    def test_cold_drug_split_rejects_drug_overlap(self):
        pairs = {"P1": {"drug_a_id": "A", "drug_b_id": "B"},
                 "P2": {"drug_a_id": "B", "drug_b_id": "C"},
                 "P3": {"drug_a_id": "D", "drug_b_id": "E"}}
        split = {"train_pair_ids": ["P1"], "validation_pair_ids": ["P2"], "test_pair_ids": ["P3"],
                 "train_drug_ids": ["A", "B"], "validation_drug_ids": ["B", "C"],
                 "test_drug_ids": ["D", "E"]}
        with self.assertRaisesRegex(ValueError, "drug-ID overlap"):
            _validate_fixed_split("cold_drug", pairs, split)


if __name__ == "__main__":
    unittest.main()
