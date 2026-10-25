import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ddi_guard.dailymed_evidence import _label_sections, _list_labels, ingest, vectorize, _sha


class DailyMedEvidenceTests(unittest.TestCase):
    def test_extracts_label_sections_with_documented_codes(self):
        xml = b'''<document xmlns="urn:hl7-org:v3"><section><code code="34073-7"/><title>Drug Interactions</title><text>May increase exposure.</text></section><section><code code="99999-9"/><title>Other</title><text>Ignored.</text></section></document>'''
        rows = _label_sections(xml)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["section_name"], "DRUG INTERACTIONS")
        self.assertIn("May increase exposure", rows[0]["text"])

    @patch("ddi_guard.dailymed_evidence._request")
    def test_search_requires_exact_normalized_name_in_title(self, request):
        request.return_value = json.dumps({"data": [
            {"setid": "1", "title": "ASPIRIN TABLET"},
            {"setid": "2", "title": "ASPIRIN AND CAFFEINE TABLET"},
        ]}).encode()
        _, rows = _list_labels("aspirin", timeout=2)
        self.assertEqual([row["setid"] for row in rows], ["1"])

    def test_vectorization_writes_hashed_provenance_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "evidence.jsonl"
            row = {"ddinter_id": "DDI1", "drug_name": "drug", "evidence_id": "dailymed:x:34073-7",
                   "source_id": "DailyMed-SPL", "source_url": "https://dailymed.nlm.nih.gov/x",
                   "source_version": "2", "source_date": "2026-01-01", "evidence_scope": "drug_general",
                   "label_independent": True, "source_document_sha256": "abc123",
                   "text": "May increase exposure through hepatic metabolism."}
            corpus.write_text(json.dumps(row) + "\n")
            manifest = root / "manifest.csv"
            self.assertEqual(vectorize(corpus, root / "vectors", manifest, max_features=32), 1)
            with manifest.open() as f:
                saved = next(csv.DictReader(f))
            self.assertEqual(saved["ddinter_id"], "DDI1")
            self.assertEqual(saved["evidence_scope"], "drug_general")
            self.assertEqual(json.loads(saved["source_urls_json"]), ["https://dailymed.nlm.nih.gov/x"])
            self.assertEqual(json.loads(saved["source_document_sha256s_json"]), ["abc123"])
            self.assertEqual(saved["fit_scope"], "all_corpus")
            self.assertTrue((root / saved["embedding_path"]).is_file())

    def test_vectorizer_fits_only_the_declared_training_drugs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for drug_id, text in (("TRAIN", "hepatic metabolism interaction evidence"),
                                  ("HELDOUT", "unseen quasar lexicon")):
                rows.append({"ddinter_id": drug_id, "drug_name": drug_id, "evidence_id": f"dailymed:{drug_id}",
                             "source_id": "DailyMed-SPL", "source_url": f"https://dailymed.nlm.nih.gov/{drug_id}",
                             "source_version": "1", "source_date": "2026-01-01", "evidence_scope": "drug_general",
                             "label_independent": True, "source_document_sha256": "abc", "text": text})
            corpus = root / "evidence.jsonl"
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "manifest.csv"
            self.assertEqual(vectorize(corpus, root / "vectors", manifest,
                                      fit_drug_ids={"TRAIN"}, fit_scope="pair_disjoint_train_drug_text_only"), 2)
            with manifest.open() as handle:
                saved = {row["ddinter_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(saved["TRAIN"]["fit_scope"], "pair_disjoint_train_drug_text_only")
            self.assertEqual(saved["HELDOUT"]["fit_drug_count"], "1")
            self.assertEqual(saved["HELDOUT"]["fit_drug_ids_sha256"], _sha(b"TRAIN"))

    @patch("ddi_guard.dailymed_evidence._list_labels", return_value=("https://dailymed.nlm.nih.gov/list", []))
    def test_empty_fetch_still_writes_coverage_audit(self, _list):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "mapping.csv"
            mapping.write_text("ddinter_id,normalized_name\nDDI1,missingdrug\n")
            audit = ingest(mapping, root / "out.jsonl", root / "audit.json", pause=0)
            self.assertEqual(audit["covered_drugs"], 0)
            self.assertEqual(audit["status"], "completed_no_coverage")
            self.assertEqual(audit["scope_note"].split(".")[0], "These passages are general drug-label evidence")

    @patch("ddi_guard.dailymed_evidence._list_labels", side_effect=[TimeoutError("temporary timeout"), ("https://dailymed.nlm.nih.gov/list", [])])
    def test_partial_source_failures_are_not_reported_as_complete(self, _list):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "mapping.csv"
            mapping.write_text("ddinter_id,normalized_name\nDDI1,drugone\nDDI2,drugtwo\n")
            audit = ingest(mapping, root / "out.jsonl", root / "audit.json", pause=0)
            self.assertEqual(audit["status"], "completed_with_partial_errors")
            self.assertEqual(audit["counts"]["request_errors"], 1)

    @patch("ddi_guard.dailymed_evidence._list_labels")
    def test_per_drug_checkpoint_resumes_without_repeating_completed_ids(self, list_labels):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "mapping.csv"
            mapping.write_text("ddinter_id,normalized_name\nDDI1,drugone\nDDI2,drugtwo\n")
            output, audit = root / "out.jsonl", root / "audit.json"
            list_labels.side_effect = [("https://dailymed.nlm.nih.gov/list", []), KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):
                ingest(mapping, output, audit, pause=0)
            checkpoint = root / "audit.checkpoint.json"
            self.assertTrue(checkpoint.is_file())
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(saved["processed_ddinter_ids"], ["DDI1"])
            list_labels.reset_mock()
            list_labels.side_effect = [("https://dailymed.nlm.nih.gov/list", [])]
            result = ingest(mapping, output, audit, pause=0)
            self.assertEqual(result["input_drugs"], 2)
            self.assertEqual(list_labels.call_count, 1)
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
