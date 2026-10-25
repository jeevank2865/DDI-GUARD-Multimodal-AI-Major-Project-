import csv
import hashlib
import io
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path

from ddi_guard.chembl_sqlite import find_release, ingest_chembl
from ddi_guard.prepare_chembl_archive import prepare


def make_chembl_fixture(path: Path, version: str = "34") -> None:
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE version(version TEXT);
      CREATE TABLE molecule_dictionary(molregno INTEGER, chembl_id TEXT, pref_name TEXT);
      CREATE TABLE molecule_synonyms(molregno INTEGER, synonyms TEXT);
      CREATE TABLE activities(activity_id INTEGER, assay_id INTEGER, molregno INTEGER,
        standard_type TEXT, standard_relation TEXT, standard_value REAL, standard_units TEXT,
        pchembl_value REAL, data_validity_comment TEXT, potential_duplicate INTEGER,
        activity_comment TEXT, bao_endpoint TEXT);
      CREATE TABLE assays(assay_id INTEGER, assay_chembl_id TEXT, target_chembl_id TEXT,
        confidence_score INTEGER, description TEXT);
      CREATE TABLE target_dictionary(target_chembl_id TEXT, pref_name TEXT, target_type TEXT, organism TEXT);
    """)
    db.execute("INSERT INTO version VALUES (?)", (version,))
    db.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1', 'Aspirin')")
    db.execute("INSERT INTO molecule_synonyms VALUES (1, 'Acetylsalicylic acid')")
    db.execute("INSERT INTO target_dictionary VALUES ('CHEMBL_T1', 'Cyclooxygenase-1', 'SINGLE PROTEIN', 'Homo sapiens')")
    db.execute("INSERT INTO assays VALUES (1, 'CHEMBL_A1', 'CHEMBL_T1', 9, 'binding assay')")
    db.executemany("INSERT INTO activities VALUES (?, 1, 1, ?, ?, ?, ?, ?, ?, 0, NULL, 'BAO_0000190')", [
        (1, "IC50", "=", 10.0, "nM", 7.0, None),
        (2, "Kd", "=", 0.01, "uM", 8.0, None),
        (3, "IC50", "<", 5.0, "nM", 7.3, None),
        (4, "Ki", "=", 20.0, "nM", 6.7, "Outside typical range"),
    ])
    db.commit()
    db.close()


class ChemblTests(unittest.TestCase):
    def test_release_reader_handles_official_name_column(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE version(name TEXT, creation_date TEXT, comments TEXT)")
        connection.execute("INSERT INTO version VALUES ('ChEMBL_34', '2024-03-28', 'release')")
        self.assertEqual(find_release(connection), "34")
        connection.close()

    def test_archive_checksum_and_safe_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fixture.tar.gz"
            db_fixture = root / "fixture.db"
            connection = sqlite3.connect(db_fixture)
            connection.execute("CREATE TABLE version(version TEXT)")
            connection.execute("INSERT INTO version VALUES ('34')")
            connection.commit()
            connection.close()
            payload = db_fixture.read_bytes()
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("chembl_34/chembl_34.db")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            result = prepare(archive, root / "extracted", expected_sha256=digest)
            self.assertEqual(result["status"], "extracted")
            self.assertEqual((root / "extracted/chembl_34/chembl_34.db").read_bytes(), payload)
            self.assertEqual(result["database_readability_checks"][0]["observed_release"], "34")
            resumed = prepare(archive, root / "extracted", expected_sha256=digest)
            self.assertEqual(resumed["database_readability_checks"][0]["observed_release"], "34")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare(archive, root / "bad", expected_sha256="0" * 64)

    def test_official_chembl_version_table_name_schema_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "official-schema.tar.gz"
            db_path = root / "fixture.db"
            db = sqlite3.connect(db_path)
            db.execute("CREATE TABLE version(name TEXT, creation_date TEXT, comments TEXT)")
            db.execute("INSERT INTO version VALUES ('ChEMBL_34', '2024-03-28', 'ChEMBL Release 34')")
            db.commit()
            db.close()
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("chembl_34/chembl_34.db")
                payload = db_path.read_bytes()
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            result = prepare(archive, root / "out", expected_sha256=digest)
            self.assertEqual(result["database_readability_checks"][0]["observed_release"], "34")

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.tar.gz"
            payload = b"must not escape"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../escape.db")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                prepare(archive, root / "extracted", expected_sha256=digest)
            self.assertFalse((root / "escape.db").exists())

    def test_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, mapping, out = root / "chembl.sqlite", root / "mapping.csv", root / "out"
            make_chembl_fixture(db_path, "37")
            mapping.write_text("ddinter_id,original_name,name_ambiguity\nDDInter1,Aspirin,false\n")
            audit = ingest_chembl(db_path, mapping, out)
            self.assertEqual(audit["status"], "blocked")
            self.assertIn("expected 34", audit["reason"])
            self.assertFalse((out / "data/processed/chembl_bioactivity_manifest.csv").exists())

    def test_activity_units_and_types_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, mapping, out = root / "chembl.sqlite", root / "mapping.csv", root / "out"
            make_chembl_fixture(db_path)
            mapping.write_text("ddinter_id,original_name,name_ambiguity\nDDInter1,Aspirin,false\n")
            audit = ingest_chembl(db_path, mapping, out)
            self.assertEqual(audit["status"], "completed")
            self.assertEqual(audit["mapped_drugs"], 1)
            self.assertEqual(audit["retained_bioactivity_records"], 4)
            self.assertEqual(audit["usable_activity_records"], 3)
            with (out / "data/processed/chembl_drug_target_edges.csv").open() as handle:
                edges = list(csv.DictReader(handle))
            self.assertEqual(len(edges), 3)
            self.assertEqual({edge["activity_type"] for edge in edges}, {"IC50", "Kd"})
            self.assertEqual({edge["activity_units"] for edge in edges}, {"nM", "uM"})
            self.assertEqual({edge["relation_operator"] for edge in edges}, {"=", "<"})
            self.assertEqual(len({edge["relation_id"] for edge in edges}), 3)


if __name__ == "__main__":
    unittest.main()
