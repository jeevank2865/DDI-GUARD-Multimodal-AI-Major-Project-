"""Train a symmetric Morgan-fingerprint Random Forest baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
import joblib

from . import LABELS, LABEL_TO_ID
from .evaluation import (apply_temperature, compute_metrics, fit_temperature_on_validation,
                         save_prediction_artifact)
from .train_molecular import _load_approved_compounds, _majority_metrics, _protocol_rows


def _morgan_bits(smiles: str, generator: Any, size: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("approved PubChem SMILES no longer parses in RDKit")
    fingerprint = generator.GetFingerprint(mol)
    bits = np.zeros((size,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    return bits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0], "numpy": np.__version__}
    for distribution in ("scikit-learn", "rdkit", "joblib"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def symmetric_pair_fingerprint(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """Order-invariant union, exclusive-or, and intersection of two fingerprints."""
    if bits_a.shape != bits_b.shape or bits_a.ndim != 1:
        raise ValueError("fingerprints must be equal-length vectors")
    return np.concatenate((np.bitwise_or(bits_a, bits_b),
                           np.bitwise_xor(bits_a, bits_b),
                           np.bitwise_and(bits_a, bits_b)))


def _build_matrix(rows: list[dict[str, Any]], fingerprints: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(rows), next(iter(fingerprints.values())).size * 3), dtype=np.uint8)
    y = np.empty((len(rows),), dtype=np.int64)
    for index, row in enumerate(rows):
        x[index] = symmetric_pair_fingerprint(fingerprints[row["drug_a_id"]],
                                              fingerprints[row["drug_b_id"]])
        y[index] = LABEL_TO_ID[row["severity"]]
    return x, y


def run(project_root: Path, protocol: str = "pair_disjoint", *, seed: int = 42,
        fingerprint_size: int = 1024, n_estimators: int = 150,
        n_jobs: int = 4, run_tag: str | None = None) -> dict[str, Any]:
    if protocol not in ("pair_disjoint", "cold_drug"):
        raise ValueError("protocol must be pair_disjoint or cold_drug")
    if fingerprint_size < 64 or n_estimators < 1 or n_jobs < 1:
        raise ValueError("fingerprint_size, n_estimators, and n_jobs are invalid")
    started = time.monotonic()
    pairs_path = project_root / "data/processed/ddi_pairs_full.json"
    split_path = project_root / "artifacts/split_manifest.json"
    compounds_path = project_root / "data/processed/pubchem_compounds.json"
    pubchem_audit = project_root / "artifacts/pubchem_coverage_audit.json"
    mapping_path = project_root / "data/processed/drug_identity_mapping.csv"
    records = _load_approved_compounds(compounds_path, pubchem_audit)
    compounds = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs_by_id = {row["pair_id"]: row for row in compounds}
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split = split_manifest[protocol]
    ids_to_fingerprint = sorted({row[f"drug_{side}_id"] for pair_id in
                                 split["train_pair_ids"] + split["validation_pair_ids"] + split["test_pair_ids"]
                                 for row in (pairs_by_id[pair_id],)
                                 for side in ("a", "b")
                                 if row["drug_a_id"] in records and row["drug_b_id"] in records})
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size)
    fingerprints = {drug_id: _morgan_bits(records[drug_id].get("isomeric_smiles") or
                                          records[drug_id]["canonical_smiles"], generator, fingerprint_size)
                    for drug_id in ids_to_fingerprint}

    if run_tag is not None and (not run_tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_tag)):
        raise ValueError("run_tag may contain only letters, digits, hyphen, or underscore")
    output_name = f"fingerprint_rf_{protocol}" + (f"_{run_tag}" if run_tag else "")
    output_root = project_root / "artifacts" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "missing_structure_pairs.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "split", "pair_id", "reason", "missing_drug_ids"])
        writer.writeheader()
        selected = _protocol_rows(split, pairs_by_id, set(fingerprints), protocol,
                                  ("train", "validation", "test"), writer)
    train_rows, validation_rows, test_rows = (selected[name] for name in ("train", "validation", "test"))
    counts = Counter(row["severity"] for row in train_rows)
    if not train_rows or not validation_rows or not test_rows or any(counts[label] == 0 for label in LABELS):
        raise ValueError("the covered split is empty or lacks one or more training classes")
    print(f"{protocol}: covered train/validation/test={len(train_rows)}/{len(validation_rows)}/{len(test_rows)}; "
          f"approved molecules={len(fingerprints)}", flush=True)

    x_train, y_train = _build_matrix(train_rows, fingerprints)
    x_validation, y_validation = _build_matrix(validation_rows, fingerprints)
    x_test, y_test = _build_matrix(test_rows, fingerprints)
    print(f"{protocol}: fingerprint matrices ready; train shape={x_train.shape}; "
          f"train matrix bytes={x_train.nbytes}", flush=True)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt",
                                        min_samples_leaf=2, class_weight="balanced_subsample",
                                        n_jobs=n_jobs, random_state=seed)
    classifier.fit(x_train, y_train)
    print(f"{protocol}: Random Forest fitting complete ({n_estimators} trees)", flush=True)
    validation_probabilities = classifier.predict_proba(x_validation)
    if list(classifier.classes_) != list(range(len(LABELS))):
        raise ValueError("RF training did not produce the complete documented class order")
    validation_metrics = compute_metrics(y_validation, validation_probabilities)
    temperature = fit_temperature_on_validation(
        torch.from_numpy(np.log(np.clip(validation_probabilities, 1e-12, 1.0))).float(),
        torch.from_numpy(y_validation).long())
    test_probabilities = classifier.predict_proba(x_test)
    calibrated = apply_temperature(
        torch.from_numpy(np.log(np.clip(test_probabilities, 1e-12, 1.0))).float(), temperature).numpy()
    model_path = output_root / "model.joblib"
    temporary_model = output_root / "model.joblib.tmp"
    joblib.dump(classifier, temporary_model, compress=3)
    os.replace(temporary_model, model_path)
    reloaded_classifier = joblib.load(model_path)
    validation_reloaded = reloaded_classifier.predict_proba(x_validation)
    test_reloaded = reloaded_classifier.predict_proba(x_test)
    checkpoint_reload_verified = (np.allclose(validation_probabilities, validation_reloaded, rtol=1e-12, atol=1e-14)
                                  and np.allclose(test_probabilities, test_reloaded, rtol=1e-12, atol=1e-14))
    if not checkpoint_reload_verified:
        raise AssertionError("reloaded RF checkpoint does not reproduce saved probabilities")
    validation_probabilities, test_probabilities = validation_reloaded, test_reloaded
    calibrated = apply_temperature(
        torch.from_numpy(np.log(np.clip(test_probabilities, 1e-12, 1.0))).float(), temperature).numpy()
    config_path = output_root / "model_config.json"
    config = {"model_type": "RandomForestClassifier", "seed": seed,
              "fingerprint": {"source": "PubChem accepted isomeric SMILES", "algorithm": "Morgan",
                              "radius": 2, "size_bits": fingerprint_size,
                              "pair_combination": ["union", "xor", "intersection"]},
              "classifier": {"n_estimators": n_estimators, "max_features": "sqrt",
                             "min_samples_leaf": 2, "class_weight": "balanced_subsample",
                             "n_jobs": n_jobs}}
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_root / "test_predictions.npz"
    prediction_artifact = save_prediction_artifact(
        prediction_path, [row["pair_id"] for row in test_rows], y_test,
        test_probabilities, calibrated,
        {"protocol": protocol, "seed": seed, "model_mode": "fingerprint_random_forest_baseline",
         "temperature_fitted_on_validation_only": temperature})
    coverage_manifest_path = output_root / "pair_coverage_manifest.json"
    coverage_manifest_path.write_text(json.dumps({
        "protocol": protocol, "seed": seed,
        "covered_pair_ids": {key: sorted(row["pair_id"] for row in selected[key])
                             for key in ("train", "validation", "test")},
        "missing_structure_ledger": str(ledger_path),
        "dataset_manifest_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    evaluation = {
        "status": "completed_random_forest_fingerprint_baseline",
        "protocol": protocol, "seed": seed,
        "feature_source": "PubChem PUG REST isomeric SMILES; RDKit Morgan radius 2",
        "dataset_manifest_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "dependency_versions": _dependency_versions(),
        "fingerprint_size_bits": fingerprint_size,
        "symmetric_pair_features": ["bitwise_union", "bitwise_xor", "bitwise_intersection"],
        "classifier": {"type": "RandomForestClassifier", "n_estimators": n_estimators,
                       "max_features": "sqrt", "min_samples_leaf": 2,
                       "class_weight": "balanced_subsample", "n_jobs": n_jobs},
        "all_split_pair_counts": {key: len(split[f"{key}_pair_ids"]) for key in ("train", "validation", "test")},
        "covered_split_pair_counts": {key: len(selected[key]) for key in ("train", "validation", "test")},
        "excluded_missing_structure_pair_counts": {key: len(split[f"{key}_pair_ids"]) - len(selected[key])
                                                    for key in ("train", "validation", "test")},
        "train_class_counts": dict(counts),
        "validation_class_counts": dict(Counter(row["severity"] for row in validation_rows)),
        "test_class_counts": dict(Counter(row["severity"] for row in test_rows)),
        "validation_metrics": validation_metrics,
        "temperature_fitted_on_validation_only": temperature,
        "test_metrics_raw": compute_metrics(y_test, test_probabilities),
        "test_metrics_validation_temperature": compute_metrics(y_test, calibrated),
        "coverage_matched_majority_baseline": _majority_metrics(train_rows, test_rows),
        "checkpoint": str(model_path),
        "checkpoint_sha256": _sha256(model_path),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "tree_node_count": int(sum(tree.tree_.node_count for tree in reloaded_classifier.estimators_)),
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": _sha256(config_path),
        "test_prediction_artifact": prediction_artifact,
        "pair_coverage_manifest": str(coverage_manifest_path),
        "missing_structure_ledger": str(ledger_path),
        "training_duration_seconds": round(time.monotonic() - started, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limits": ["This is a fingerprint baseline, not the multimodal graph model.",
                   "Model is not wired into the prediction API."],
    }
    path = output_root / "evaluation.json"
    path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", choices=("pair_disjoint", "cold_drug"), default="pair_disjoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-size", type=int, default=1024)
    parser.add_argument("--estimators", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--run-tag", help="write a versioned artifact folder instead of replacing the default run")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.protocol, seed=args.seed,
                         fingerprint_size=args.fingerprint_size, n_estimators=args.estimators,
                         n_jobs=args.jobs, run_tag=args.run_tag), indent=2))


if __name__ == "__main__":
    main()
