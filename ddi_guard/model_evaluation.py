"""Verified, artifact-only reporting for the saved DDI-GUARD test evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ddi_guard import LABELS
from ddi_guard.evaluation import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
PAIRED_REPORT = ROOT / "artifacts/paired_multimodal_validation_v1/paired_multimodal_validation.json"
_SUPPORTED_PROTOCOLS = {"pair_disjoint", "cold_drug"}


class EvaluationArtifactError(ValueError):
    """Raised when saved evaluation results fail provenance or consistency checks."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise EvaluationArtifactError("Evaluation metadata points outside the project artifact directory.")
    if not resolved.is_file():
        raise EvaluationArtifactError(f"Required evaluation artifact is missing: {resolved.relative_to(ROOT)}")
    return resolved


def _require_hash(path: Path, expected: str | None, label: str) -> str:
    actual = _sha256(path)
    if not expected or actual != expected:
        raise EvaluationArtifactError(f"{label} SHA-256 verification failed.")
    return actual


def _same_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key in ("accuracy", "macro_f1", "weighted_f1", "multiclass_brier_score",
                "negative_log_likelihood", "expected_calibration_error"):
        if key not in expected or not np.isclose(float(actual[key]), float(expected[key]), rtol=1e-9, atol=1e-10):
            return False
    return (actual["per_class"] == expected["per_class"]
            and actual["confusion_matrix"] == expected["confusion_matrix"]
            and actual["n"] == expected["n"])


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    class_metrics = metrics["per_class"]
    best = max(LABELS, key=lambda name: (class_metrics[name]["f1"], -LABELS.index(name)))
    weakest = min(LABELS, key=lambda name: (class_metrics[name]["f1"], LABELS.index(name)))
    matrix = metrics["confusion_matrix"]
    errors = [(matrix[row][col], row, col) for row in range(3) for col in range(3)
              if row != col and matrix[row][col] > 0]
    if errors:
        count, actual, predicted = max(errors, key=lambda item: (item[0], -item[1], -item[2]))
        frequent = {"actual": LABELS[actual], "predicted": LABELS[predicted], "count": int(count)}
    else:
        frequent = None
    rare_support = class_metrics[weakest]["support"]
    recommendation = (
        "Compare class-weighted loss or balanced sampling using training data, choose settings on validation data only, and keep this held-out test set unchanged. Collect more independent cold-drug examples before treating rare-class estimates as stable."
        if rare_support < 50 else
        "If improving the weaker class, compare class-weighted loss or balanced sampling using training data and choose settings on validation data only; keep this held-out test set unchanged."
    )
    return {"best_per_class_f1": best, "weakest_per_class_f1": weakest,
            "most_frequent_misclassification": frequent,
            "recommended_follow_up": recommendation,
            "interpretation": (f"The model's highest per-class F1 is {best}; its lowest is {weakest}. "
                               f"Macro-F1 is {metrics['macro_f1']:.1%} across the three labels."),
            "limitations": [
                "This is a research evaluation of DDInter dataset labels, not clinical outcomes or patient-specific interaction risks.",
                "The full-model scores apply only to pairs with the required molecular, target, and text feature support; they do not describe every DDInter pair.",
                "Cold-drug test support is small for Minor and Major labels; per-class estimates for rare classes are uncertain.",
                "A held-out split reduces leakage risk but cannot establish clinical validity, external generalization, or label correctness.",
            ]}


def get_model_evaluation(protocol: str = "pair_disjoint") -> dict[str, Any]:
    """Load a full three-modality test report after verifying its saved provenance."""
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise EvaluationArtifactError("Unsupported evaluation protocol.")
    if not PAIRED_REPORT.is_file():
        raise EvaluationArtifactError("Verified paired evaluation report is unavailable.")
    report = json.loads(PAIRED_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("labels") != list(LABELS):
        raise EvaluationArtifactError("Saved evaluation report status or label order is invalid.")
    protocol_data = report.get("protocols", {}).get(protocol)
    run = (protocol_data or {}).get("models", {}).get("Full Molecular + Target + Text")
    if not run:
        raise EvaluationArtifactError(f"No verified full-model evaluation exists for {protocol}.")

    evaluation_path = _artifact_path(run.get("evaluation_path", ""))
    prediction_path = _artifact_path(run.get("prediction_path", ""))
    checkpoint_path = _artifact_path(run.get("checkpoint_path", ""))
    split_path = _artifact_path(run.get("split_manifest_path", ""))
    dataset_path = _artifact_path(run.get("dataset_manifest_path", ""))
    coverage_path = _artifact_path(run.get("pair_coverage_manifest_path", ""))
    evaluation_hash = _require_hash(evaluation_path, run.get("evaluation_sha256"), "Evaluation artifact")
    prediction_hash = _require_hash(prediction_path, run.get("prediction_sha256"), "Prediction artifact")
    checkpoint_hash = _require_hash(checkpoint_path, run.get("checkpoint_sha256"), "Checkpoint")
    split_hash = _require_hash(split_path, run.get("split_manifest_sha256"), "Split manifest")
    dataset_hash = _require_hash(dataset_path, run.get("dataset_manifest_sha256"), "Dataset manifest")
    coverage_hash = _require_hash(coverage_path, run.get("pair_coverage_manifest_sha256"), "Pair coverage manifest")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    dataset_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_pair_ids = [row.get("pair_id") for row in dataset_rows]
    if any(not isinstance(row.get(key), str) or not row.get(key).strip()
           for row in dataset_rows for key in ("pair_id", "drug_a_id", "drug_b_id", "severity")):
        raise EvaluationArtifactError("Source dataset has rows with missing pair identity or label fields.")
    source_pairs = [tuple(sorted((row.get("drug_a_id"), row.get("drug_b_id")))) for row in dataset_rows]
    if (not dataset_rows
            or len(set(source_pair_ids)) != len(source_pair_ids)
            or len(set(source_pairs)) != len(source_pairs)):
        raise EvaluationArtifactError("Source dataset contains empty or duplicate canonical drug pairs.")
    if evaluation.get("protocol") != protocol or evaluation.get("modalities") != ["molecular", "target", "text"]:
        raise EvaluationArtifactError("Evaluation protocol or modalities do not match the selected report.")
    if evaluation.get("checkpoint_sha256") != checkpoint_hash or not evaluation.get("checkpoint_reload_verified"):
        raise EvaluationArtifactError("Checkpoint identity or reload verification does not match the evaluation.")
    if evaluation.get("split_manifest_sha256") != split_hash or evaluation.get("dataset_manifest_sha256") != dataset_hash:
        raise EvaluationArtifactError("Evaluation does not reference the verified split and dataset manifests.")
    molecular_manifest = _artifact_path("data/processed/molecular_graph_manifest.json")
    pubchem_compounds = _artifact_path("data/processed/pubchem_compounds.json")
    target_sources = evaluation.get("sources", {}).get("target", {})
    target_graph = _artifact_path(target_sources.get("graph_path", ""))
    target_manifest = _artifact_path(target_sources.get("graph_manifest_path", ""))
    text_manifest = _artifact_path(run.get("text_feature_manifest_verified_path", ""))
    molecular_manifest_hash = _require_hash(
        molecular_manifest, evaluation.get("sources", {}).get("molecular", {}).get("graph_manifest_sha256"),
        "Molecular graph manifest")
    pubchem_hash = _require_hash(
        pubchem_compounds, evaluation.get("sources", {}).get("molecular", {}).get("pubchem_compounds_sha256"),
        "PubChem compound source")
    target_manifest_hash = _require_hash(
        target_manifest, run.get("target_graph_manifest_sha256"), "Target graph manifest")
    _require_hash(target_graph, target_sources.get("graph_sha256"), "Target graph")
    text_manifest_hash = _require_hash(
        text_manifest, run.get("text_feature_manifest_sha256"), "DailyMed feature manifest")
    if target_sources.get("manifest_sha256") != target_manifest_hash:
        raise EvaluationArtifactError("Target graph evaluation provenance does not match its verified manifest.")
    if evaluation.get("sources", {}).get("text", {}).get("manifest_sha256") != text_manifest_hash:
        raise EvaluationArtifactError("Text feature evaluation provenance does not match its verified manifest.")
    if evaluation.get("pair_coverage_manifest") != str(coverage_path.relative_to(ROOT)):
        raise EvaluationArtifactError("Evaluation does not reference the verified pair coverage manifest.")
    if not run.get("temperature_fitted_on_validation_only") or not run.get("checkpoint_reload_verified"):
        raise EvaluationArtifactError("Validation-only calibration or checkpoint reload is not verified.")
    temperature = float(evaluation.get("temperature_fitted_on_validation_only", 0))
    if temperature <= 0 or not np.isclose(temperature, float(run.get("temperature", 0)), rtol=1e-10, atol=1e-12):
        raise EvaluationArtifactError("Validation-only calibration temperature is inconsistent.")

    with np.load(prediction_path, allow_pickle=False) as saved:
        ids = saved["pair_ids"].astype(str)
        labels = saved["labels"].astype(np.int64)
        raw_probabilities = saved["probabilities_raw"].astype(np.float64)
        probabilities = saved["probabilities_calibrated"].astype(np.float64)
        metadata = json.loads(str(saved["metadata"]))
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids) or labels.shape != ids.shape:
        raise EvaluationArtifactError("Saved test pair IDs or labels are empty or duplicated.")
    id_hash = hashlib.sha256("\n".join(sorted(ids.tolist())).encode()).hexdigest()
    prediction_meta = evaluation.get("test_prediction_artifact", {})
    if (id_hash != run.get("test_pair_ids_sha256") or id_hash != prediction_meta.get("pair_ids_sha256")
            or len(ids) != int(run.get("test_pair_count", -1))
            or len(ids) != int(prediction_meta.get("pair_count", -1))):
        raise EvaluationArtifactError("Saved prediction support differs from the validated evaluation support.")
    if metadata.get("protocol") != protocol or metadata.get("class_order") != list(LABELS):
        raise EvaluationArtifactError("Prediction artifact metadata has an invalid protocol or class order.")
    if not np.isclose(float(metadata.get("temperature_fitted_on_validation_only", 0)), temperature):
        raise EvaluationArtifactError("Prediction artifact calibration metadata does not match the validation temperature.")
    if probabilities.shape != (len(ids), len(LABELS)) or raw_probabilities.shape != probabilities.shape:
        raise EvaluationArtifactError("Saved test prediction probability dimensions are invalid.")

    split_test_ids = set(split.get(protocol, {}).get("test_pair_ids", []))
    split_record = split.get(protocol, {})
    split_ids = {name: set(split_record.get(f"{name}_pair_ids", []))
                 for name in ("train", "validation", "test")}
    if any(split_ids[left] & split_ids[right]
           for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise EvaluationArtifactError("Pair IDs overlap across the selected protocol's splits.")
    cold_drug_disjoint = None
    if protocol == "cold_drug":
        drug_sets = {name: set(split_record.get(f"{name}_drug_ids", []))
                     for name in ("train", "validation", "test")}
        if any(drug_sets[left] & drug_sets[right]
               for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise EvaluationArtifactError("Drug IDs overlap across the cold-drug protocol splits.")
        cold_drug_disjoint = True
    covered_test_ids = set(coverage.get("selected_pair_ids", {}).get("test", []))
    if not set(ids.tolist()).issubset(split_test_ids) or set(ids.tolist()) != covered_test_ids:
        raise EvaluationArtifactError("Test predictions do not match the declared protocol and feature-coverage support.")
    pair_labels = {row["pair_id"]: LABELS.index(row["severity"])
                   for row in dataset_rows if row.get("severity") in LABELS}
    if any(pair_id not in pair_labels for pair_id in ids) or not np.array_equal(labels, [pair_labels[pair_id] for pair_id in ids]):
        raise EvaluationArtifactError("Saved ground-truth labels do not match the source dataset records.")
    reconstructed = np.power(np.clip(raw_probabilities, 1e-15, 1.0), 1.0 / temperature)
    reconstructed /= reconstructed.sum(axis=1, keepdims=True)
    if not np.allclose(reconstructed, probabilities, rtol=2e-7, atol=2e-7):
        raise EvaluationArtifactError("Saved calibrated probabilities do not match validation-only temperature scaling.")

    computed = compute_metrics(labels, probabilities)
    expected = evaluation.get("test_metrics_validation_temperature", {})
    if not _same_metrics(computed, expected):
        raise EvaluationArtifactError("Recomputed test metrics do not match the saved evaluation artifact.")
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        labels, probabilities.argmax(axis=1), labels=list(range(len(LABELS))), average="macro", zero_division=0)
    metrics = {**computed, "macro_precision": float(macro_precision), "macro_recall": float(macro_recall)}
    per_class_support = sum(item["support"] for item in metrics["per_class"].values())
    if sum(map(sum, metrics["confusion_matrix"])) != len(ids) or per_class_support != len(ids):
        raise EvaluationArtifactError("Confusion matrix or per-class support does not sum to the evaluated sample count.")

    protocol_title = "Pair-disjoint · unseen pairs among known drugs" if protocol == "pair_disjoint" else "Cold-drug · held-out drugs"
    return {"status": "verified", "protocol": protocol, "protocol_title": protocol_title,
            "model": run.get("label"), "modalities": run.get("modalities"), "labels": list(LABELS),
            "sample_count": len(ids), "test_class_support": {name: int(np.sum(labels == i)) for i, name in enumerate(LABELS)},
            "split_counts": {key: len(split_record.get(f"{key}_pair_ids", [])) for key in ("train", "validation", "test")},
            "metrics": metrics, "summary": _summary(metrics),
            "integrity_checks": {"source_pair_ids_unique": True, "canonical_unordered_pairs_unique": True,
                                 "pair_splits_disjoint": True, "cold_drug_sets_disjoint": cold_drug_disjoint,
                                 "test_labels_match_source_records": True,
                                 "probabilities_match_validation_temperature": True,
                                 "checkpoint_and_feature_provenance_verified": True},
            "artifacts": {"evaluation": str(evaluation_path.relative_to(ROOT)),
                          "prediction": str(prediction_path.relative_to(ROOT)),
                          "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                          "split_manifest": str(split_path.relative_to(ROOT)),
                          "dataset": str(dataset_path.relative_to(ROOT)),
                          "pair_coverage_manifest": str(coverage_path.relative_to(ROOT)),
                          "evaluation_sha256": evaluation_hash, "prediction_sha256": prediction_hash,
                          "checkpoint_sha256": checkpoint_hash, "split_manifest_sha256": split_hash,
                          "dataset_sha256": dataset_hash, "pair_coverage_sha256": coverage_hash,
                          "molecular_graph_manifest": str(molecular_manifest.relative_to(ROOT)),
                          "molecular_graph_manifest_sha256": molecular_manifest_hash,
                          "pubchem_compounds": str(pubchem_compounds.relative_to(ROOT)),
                          "pubchem_compounds_sha256": pubchem_hash,
                          "target_graph": str(target_graph.relative_to(ROOT)),
                          "target_graph_sha256": target_sources.get("graph_sha256"),
                          "target_graph_manifest": str(target_manifest.relative_to(ROOT)),
                          "target_graph_manifest_sha256": target_manifest_hash,
                          "text_feature_manifest_path": str(text_manifest.relative_to(ROOT)),
                          "text_feature_manifest_sha256": text_manifest_hash},
            "calibration": {"method": "scalar temperature scaling", "temperature": temperature,
                            "fit_on": "validation split only", "verified": True},
            "limitations": _summary(metrics)["limitations"],
            "clinical_status": "Research prototype; not clinically validated. Predictions are not confirmed clinical interaction risks."}
