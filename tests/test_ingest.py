import csv
import json
import tempfile
import unittest
from pathlib import Path

from ddi_guard.ingest import ATC_CODES, build_outputs


class IngestTests(unittest.TestCase):
    def test_raw_record_accounting_dedup_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source, output = tmp_path / "source", tmp_path / "out"
            source.mkdir()
            fields = ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]
            rows = {
                "A": [
                    ["DDInter1", "Drug A", "DDInter2", "Drug B", "Major"],
                    ["DDInter2", "Drug B", "DDInter1", "Drug A", "Major"],
                    ["DDInter3", "Drug C", "DDInter4", "Drug D", "Unknown"],
                    ["DDInter5", "Drug E", "DDInter5", "Drug E", "Minor"],
                    ["DDInter6", "Drug F", "DDInter7", "Drug G", "Minor"],
                    ["DDInter7", "Drug G", "DDInter6", "Drug F", "Major"],
                ]
            }
            for code in ATC_CODES:
                with (source / f"ddinter_downloads_code_{code}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows.get(code, []))
            audit = build_outputs(source, output)
            self.assertEqual(audit["raw_records"], 6)
            self.assertEqual(audit["unique_labeled_pairs"], 1)
            self.assertEqual(audit["conflicting_pair_count"], 1)
            self.assertEqual(audit["malformed_or_self_interaction_records"], 1)
            self.assertTrue(audit["record_accounting_check"]["accounted"])
            with (output / "data/processed/ddi_pair_manifest.csv").open() as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(len(json.loads(manifest[0]["source_records"])), 2)
            with (output / "data/processed/data_exclusion_ledger.csv").open() as handle:
                ledger = list(csv.DictReader(handle))
            reasons = {row["reason"] for row in ledger}
            self.assertTrue({"duplicate_or_reversed_pair_collapsed", "unknown_or_invalid_severity",
                             "self_interaction", "conflicting_severity_labels"} <= reasons)
            self.assertEqual(len(manifest) + len(ledger), 6)


if __name__ == "__main__":
    unittest.main()
