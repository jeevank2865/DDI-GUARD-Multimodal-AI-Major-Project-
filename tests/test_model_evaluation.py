import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ddi_guard.api import app
from ddi_guard.model_evaluation import EvaluationArtifactError, get_model_evaluation


class ModelEvaluationTests(unittest.TestCase):
    def test_pair_disjoint_metrics_are_recomputed_from_verified_artifacts(self):
        result = get_model_evaluation("pair_disjoint")
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["sample_count"], 1802)
        self.assertAlmostEqual(result["metrics"]["macro_f1"], 0.7473041966619568)
        self.assertEqual(result["metrics"]["confusion_matrix"], [[85, 25, 12], [61, 1210, 105], [6, 63, 235]])
        self.assertEqual(sum(result["test_class_support"].values()), result["sample_count"])
        self.assertEqual(result["calibration"]["fit_on"], "validation split only")
        self.assertTrue(result["calibration"]["verified"])
        self.assertIn("text_feature_manifest_sha256", result["artifacts"])

    def test_cold_drug_evaluation_highlights_sparse_minor_support(self):
        result = get_model_evaluation("cold_drug")
        self.assertEqual(result["sample_count"], 253)
        self.assertEqual(result["test_class_support"], {"Minor": 14, "Moderate": 206, "Major": 33})
        self.assertAlmostEqual(result["metrics"]["macro_f1"], 0.36635632287806197)
        self.assertEqual(result["metrics"]["confusion_matrix"], [[2, 12, 0], [7, 194, 5], [0, 32, 1]])

    def test_evaluation_is_repeatable_and_protocols_are_separate(self):
        first = get_model_evaluation("cold_drug")
        second = get_model_evaluation("cold_drug")
        pair_disjoint = get_model_evaluation("pair_disjoint")
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["protocol"], "cold_drug")
        self.assertEqual(pair_disjoint["protocol"], "pair_disjoint")
        self.assertNotEqual(first["artifacts"]["prediction_sha256"], pair_disjoint["artifacts"]["prediction_sha256"])

    def test_unsupported_protocol_is_rejected(self):
        with self.assertRaises(EvaluationArtifactError):
            get_model_evaluation("test_tuned")

    def test_api_serves_verified_protocols_and_rejects_invalid_protocol(self):
        client = TestClient(app)
        response = client.get("/api/model-evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["protocol"], "pair_disjoint")
        cold = client.get("/api/model-evaluation?protocol=cold_drug")
        self.assertEqual(cold.status_code, 200)
        self.assertEqual(cold.json()["test_class_support"]["Minor"], 14)
        self.assertEqual(client.get("/api/model-evaluation?protocol=invalid").status_code, 422)

    def test_api_fails_closed_if_an_artifact_hash_does_not_match(self):
        with patch("ddi_guard.model_evaluation._require_hash", side_effect=EvaluationArtifactError("fixture mismatch")):
            response = TestClient(app).get("/api/model-evaluation")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Verified model evaluation unavailable", response.json()["detail"])

    def test_frontend_has_real_evaluation_controls_and_research_disclaimer(self):
        html = TestClient(app).get("/").text
        for marker in ("data-page=\"evaluation\"", "id=\"evaluation-protocol\"",
                       "id=\"evaluation-matrix\"", "id=\"evaluation-per-class\"",
                       "/api/model-evaluation?protocol=", "not clinically validated"):
            self.assertIn(marker, html)


if __name__ == "__main__":
    unittest.main()
