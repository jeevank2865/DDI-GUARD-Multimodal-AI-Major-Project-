import unittest

import numpy as np

from ddi_guard.compare_experiments import paired_bootstrap_macro_f1


class MatchedComparisonTests(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible_and_reports_model_delta(self):
        labels = np.array([0, 1, 2, 0, 1, 2] * 10)
        first = labels.copy()
        second = np.full_like(labels, 1)
        one = paired_bootstrap_macro_f1(labels, first, second, replicates=100, seed=19)
        two = paired_bootstrap_macro_f1(labels, first, second, replicates=100, seed=19)
        self.assertEqual(one, two)
        self.assertGreater(one["observed_first_minus_second_macro_f1"], 0)
        self.assertEqual(one["test_rows"], len(labels))


if __name__ == "__main__":
    unittest.main()
