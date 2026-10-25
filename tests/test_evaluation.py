import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

from ddi_guard.evaluation import (apply_temperature, compute_metrics,
                                  fit_temperature_on_validation, save_prediction_artifact)


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_probability_matrix_and_actual_multiclass_auc(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.7,.2,.1],[.2,.7,.1],[.2,.1,.7]])
        metrics = compute_metrics(y, p)
        self.assertEqual(metrics["n"], 6)
        self.assertAlmostEqual(metrics["roc_auc_ovr_macro"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0)
        self.assertLessEqual(metrics["expected_calibration_error"], 1)

    def test_temperature_is_fit_from_validation_logits(self):
        logits = torch.tensor([[3., 0., -1.], [0., 3., -1.], [-1., 0., 3.], [2., 0., -1.]])
        labels = torch.tensor([0, 1, 2, 0])
        temperature = fit_temperature_on_validation(logits, labels)
        self.assertGreater(temperature, 0)
        probabilities = apply_temperature(logits, temperature)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(4), atol=1e-6))

    def test_prediction_artifact_keeps_pair_ids_labels_and_both_probability_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.npz"
            labels = np.array([0, 1, 2])
            probabilities = np.eye(3, dtype=float)
            info = save_prediction_artifact(path, ["p1", "p2", "p3"], labels,
                                            probabilities, probabilities,
                                            {"protocol": "pair_disjoint"})
            self.assertEqual(info["pair_count"], 3)
            with np.load(path, allow_pickle=False) as artifact:
                self.assertEqual(artifact["pair_ids"].tolist(), ["p1", "p2", "p3"])
                self.assertEqual(artifact["labels"].tolist(), [0, 1, 2])
                np.testing.assert_array_equal(artifact["probabilities_raw"], probabilities)
                np.testing.assert_array_equal(artifact["probabilities_calibrated"], probabilities)
                self.assertEqual(json.loads(str(artifact["metadata"]))["protocol"], "pair_disjoint")


if __name__ == "__main__":
    unittest.main()
