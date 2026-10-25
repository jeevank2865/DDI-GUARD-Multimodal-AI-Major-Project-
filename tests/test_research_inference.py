import json
import unittest

import numpy as np
import torch
from fastapi.testclient import TestClient
from unittest.mock import patch

from ddi_guard.api import _drug_catalog, app
from ddi_guard.evaluation import apply_temperature
from ddi_guard.features import batch_graphs
from ddi_guard.research_inference import (
    MODEL_SPECS, PAIR_DATA, ROOT, ResearchInferenceError, _bundle,
    _feature_presence, _select_model, _sources, infer_research_pair,
)
from ddi_guard.train_multimodal import MODALITY_ORDER, _make_inputs, _pair_mask


class ResearchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = _sources()
        cls.bundles = {name: _bundle(name) for name in MODEL_SPECS}
        cls.catalog_ids = set(_drug_catalog())
        cls.available = {
            "molecular": set(cls.sources["molecular_graphs"]),
            "target": set(cls.sources["target_drugs"]),
            "text": set(cls.sources["text_vectors"]),
        }
        cls.pairs = {row["pair_id"]: row for row in json.loads(PAIR_DATA.read_text())}

        def find_pair(predicate):
            for a in sorted(cls.catalog_ids):
                for b in sorted(cls.catalog_ids):
                    if a >= b:
                        continue
                    pa = _feature_presence(a, cls.sources)
                    pb = _feature_presence(b, cls.sources)
                    model = _select_model(pa, pb)
                    if predicate(pa, pb, model):
                        return a, b, model
            return None

        cls.examples = {
            "molecular_target_text": find_pair(lambda a, b, m: m == "molecular_target_text"),
            "molecular_target": find_pair(lambda a, b, m: m == "molecular_target"),
            "molecular": find_pair(lambda a, b, m: m == "molecular"),
            "target": find_pair(lambda a, b, m: m == "target"),
            "unsupported": find_pair(lambda a, b, m: m is None),
        }
        if any(cls.examples[name] is None for name in ("molecular_target_text", "molecular_target", "molecular", "target")):
            raise AssertionError("one or more claimed modality routing strata have no real example")

    def test_all_registered_checkpoints_load_with_pinned_hashes(self):
        for name, bundle in self.bundles.items():
            with self.subTest(model=name):
                spec = MODEL_SPECS[name]
                self.assertEqual(bundle["checkpoint_sha256"], spec["checkpoint_sha256"])
                self.assertEqual(bundle["config_sha256"], spec["config_sha256"])
                self.assertEqual(bundle["evaluation"]["checkpoint_reload_verified"], True)
                self.assertEqual(bundle["modalities"], tuple(MODEL_SPECS[name]["modalities"]))

    def test_saved_pair_disjoint_predictions_match_training_preprocessing(self):
        for name, spec in MODEL_SPECS.items():
            bundle = self.bundles[name]
            run = ROOT / spec["run"]
            with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
                selected = None
                for index, saved_pair_id in enumerate(saved["pair_ids"]):
                    candidate = self.pairs[str(saved_pair_id)].copy()
                    mask = _pair_mask(candidate, self.sources, bundle["modalities"])
                    if all(mask[MODALITY_ORDER.index(modality)] for modality in bundle["modalities"]):
                        selected = (str(saved_pair_id), mask, saved["probabilities_calibrated"][index])
                        break
                if selected is None:
                    self.fail(f"{name} test artifact lacks a pair with all trained modalities present")
                pair_id, input_mask, expected = selected
            row = self.pairs[pair_id].copy()
            row["modality_mask"] = input_mask
            if bundle["modalities"] == ("molecular",):
                inputs = {
                    "molecule_a": batch_graphs([self.sources["molecular_graphs"][row["drug_a_id"]]]),
                    "molecule_b": batch_graphs([self.sources["molecular_graphs"][row["drug_b_id"]]]),
                }
            else:
                inputs = _make_inputs([row], self.sources, bundle["modalities"],
                                      self.sources["target_drug_index"],
                                      int(bundle["model"].config.text_dim))
            with torch.inference_mode():
                actual = apply_temperature(bundle["model"](**inputs)["logits"].cpu(),
                                           bundle["temperature"])[0].numpy()
            with self.subTest(model=name, pair_id=pair_id):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_each_real_modality_stratum_runs_repeatable_inference(self):
        for expected_model in ("molecular_target_text", "molecular_target", "molecular", "target"):
            a, b, selected_model = self.examples[expected_model]
            first = infer_research_pair(a, b)
            second = infer_research_pair(a, b)
            with self.subTest(model=expected_model):
                self.assertEqual(selected_model, expected_model)
                self.assertEqual(first["model_version"], f"{expected_model}_pair_disjoint_v1")
                self.assertEqual(first["class_probabilities"], second["class_probabilities"])
                self.assertEqual(first["class_order"], ["Minor", "Moderate", "Major"])
                values = list(first["class_probabilities"].values())
                self.assertTrue(all(0 <= value <= 1 for value in values))
                self.assertAlmostEqual(sum(values), 1, places=6)
                self.assertEqual(first["calibration"]["fit_partition"], "validation_only")
                self.assertEqual(first["preprocessing_verification"]["status"], "verified")

    def test_unknown_invalid_same_and_unsupported_pairs_are_rejected(self):
        a = self.examples["molecular_target_text"][0]
        with self.assertRaises(ResearchInferenceError):
            infer_research_pair("", a)
        with self.assertRaisesRegex(ResearchInferenceError, "Unknown drug"):
            infer_research_pair("not-a-reviewed-drug", a)
        with self.assertRaisesRegex(ResearchInferenceError, "different drugs"):
            infer_research_pair(a, a)
        if self.examples["unsupported"]:
            x, y, _ = self.examples["unsupported"]
            with self.assertRaisesRegex(ResearchInferenceError, "No verified checkpoint"):
                infer_research_pair(x, y)

    def test_production_prediction_remains_disabled_by_policy(self):
        a, b = "DDInter1", "DDInter1033"
        response = TestClient(app).post("/api/predict", json={"drug_a": a, "drug_b": b})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json().get("approved", False))

    def test_research_endpoint_is_closed_until_policy_gate_is_enabled(self):
        a, b = "DDInter1", "DDInter1033"
        with patch("ddi_guard.api._research_demo_authorized", return_value=False):
            response = TestClient(app).post("/api/research/predict", json={"drug_a": a, "drug_b": b})
        self.assertEqual(response.status_code, 503)
        self.assertIn("verification gate", response.json()["detail"])

    def test_authorized_research_endpoint_returns_actual_inference(self):
        # Use drugs that also meet the search selector's strict source-backed eligibility rule.
        a, b = "DDInter1", "DDInter1033"
        response = TestClient(app).post("/api/research/predict", json={"drug_a": a, "drug_b": b})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "research_prediction")
        self.assertEqual(body["predicted_severity"], "Major")
        self.assertEqual(body["checkpoint_sha256"], MODEL_SPECS["molecular_target_text"]["checkpoint_sha256"])
        self.assertAlmostEqual(sum(body["class_probabilities"].values()), 1.0, places=6)
        self.assertEqual(body["calibration"]["fit_partition"], "validation_only")

    def test_research_endpoint_surfaces_unknown_and_unsupported_drugs(self):
        a, b, _ = self.examples["molecular_target_text"]
        client = TestClient(app)
        unknown = client.post("/api/research/predict", json={"drug_a": "unknown", "drug_b": b})
        self.assertEqual(unknown.status_code, 404)
        ineligible = client.post("/api/research/predict", json={"drug_a": "DDInter2", "drug_b": "DDInter1"})
        self.assertEqual(ineligible.status_code, 422)
        self.assertEqual(ineligible.json()["detail"]["code"], "drug_not_prediction_eligible")
        if self.examples["unsupported"]:
            x, y, _ = self.examples["unsupported"]
            unsupported = client.post("/api/research/predict", json={"drug_a": x, "drug_b": y})
            self.assertEqual(unsupported.status_code, 422)

    def test_ui_keeps_source_annotations_distinct_from_model_predictions(self):
        client = TestClient(app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("DDInter source annotation", page.text)
        self.assertIn('apiFetch("/api/research/predict"', page.text)
        self.assertIn("MODEL RESULT · RESEARCH USE", page.text)
        pair_summary = page.text[
            page.text.index("function renderPairSummary("):
            page.text.index("async function inspectPair()")
        ]
        self.assertNotIn("dataset_annotation", pair_summary)
        self.assertNotIn("renderDrugEvidence", pair_summary)
        a, b, _ = self.examples["molecular_target_text"]
        lookup = client.get("/api/interactions/lookup", params={"drug_a_id": a, "drug_b_id": b})
        self.assertEqual(lookup.status_code, 200)
        self.assertEqual(lookup.json()["model_prediction"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
