"""Matched-coverage evaluation and paired bootstrap uncertainty for saved runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from . import LABELS
from .evaluation import compute_metrics


def paired_bootstrap_macro_f1(labels: np.ndarray, first: np.ndarray, second: np.ndarray,
                              *, replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(first, dtype=np.int64)
    b = np.asarray(second, dtype=np.int64)
    if y.ndim != 1 or a.shape != y.shape or b.shape != y.shape or y.size < 2:
        raise ValueError("paired bootstrap inputs must be aligned, one-dimensional arrays with at least two rows")
    if replicates < 100:
        raise ValueError("use at least 100 paired bootstrap replicates")
    rng = np.random.default_rng(seed)
    samples_a = np.empty(replicates, dtype=np.float64)
    samples_b = np.empty(replicates, dtype=np.float64)
    samples_delta = np.empty(replicates, dtype=np.float64)
    class_ids = list(range(len(LABELS)))
    for index in range(replicates):
        sample = rng.integers(0, y.size, size=y.size)
        samples_a[index] = f1_score(y[sample], a[sample], labels=class_ids, average="macro", zero_division=0)
        samples_b[index] = f1_score(y[sample], b[sample], labels=class_ids, average="macro", zero_division=0)
        samples_delta[index] = samples_a[index] - samples_b[index]
    return {
        "method": "paired non-stratified percentile bootstrap over held-out pair IDs",
        "replicates": replicates, "seed": seed, "confidence_level": 0.95,
        "first_macro_f1_ci95": np.quantile(samples_a, [0.025, 0.975]).tolist(),
        "second_macro_f1_ci95": np.quantile(samples_b, [0.025, 0.975]).tolist(),
        "first_minus_second_macro_f1_ci95": np.quantile(samples_delta, [0.025, 0.975]).tolist(),
        "observed_first_minus_second_macro_f1": float(
            f1_score(y, a, labels=class_ids, average="macro", zero_division=0)
            - f1_score(y, b, labels=class_ids, average="macro", zero_division=0)),
        "test_rows": int(y.size),
        "note": "Uncertainty summaries are descriptive and are not used for model selection or tuning.",
    }


def _read_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as artifact:
        ids = artifact["pair_ids"].astype(str)
        labels = artifact["labels"].astype(np.int64)
        probabilities = artifact["probabilities_raw"].astype(np.float64)
        metadata = json.loads(str(artifact["metadata"]))
    if metadata.get("class_order") != list(LABELS):
        raise ValueError(f"class order mismatch in {path}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"duplicate pair IDs in {path}")
    return {pair_id: np.asarray([label, *probability], dtype=np.float64)
            for pair_id, label, probability in zip(ids, labels, probabilities)}, {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pair_count": int(len(ids))}


def compare_protocol(project_root: Path, protocol: str, *, rf_run_tag: str = "prediction_export_v1",
                     molecular_run_tag: str = "no_placeholder_v1",
                     replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
    molecular_folder = f"molecular_{protocol}" + (f"_{molecular_run_tag}" if molecular_run_tag else "")
    molecular_path = project_root / "artifacts" / molecular_folder / "test_predictions.npz"
    rf_path = project_root / "artifacts" / f"fingerprint_rf_{protocol}_{rf_run_tag}" / "test_predictions.npz"
    molecular, molecular_meta = _read_predictions(molecular_path)
    rf, rf_meta = _read_predictions(rf_path)
    molecular_coverage_path = molecular_path.parent / "pair_coverage_manifest.json"
    rf_coverage_path = rf_path.parent / "pair_coverage_manifest.json"
    molecular_coverage = json.loads(molecular_coverage_path.read_text(encoding="utf-8"))
    rf_coverage = json.loads(rf_coverage_path.read_text(encoding="utf-8"))
    for split_name in ("train", "validation", "test"):
        if set(molecular_coverage["covered_pair_ids"][split_name]) != set(rf_coverage["covered_pair_ids"][split_name]):
            raise ValueError(f"models do not share identical covered {split_name} IDs")
    if set(molecular) != set(rf):
        raise ValueError(f"models do not share an exact held-out support: GNN={len(molecular)}, RF={len(rf)}")
    pair_ids = sorted(molecular)
    labels = np.asarray([molecular[pair_id][0] for pair_id in pair_ids], dtype=np.int64)
    rf_labels = np.asarray([rf[pair_id][0] for pair_id in pair_ids], dtype=np.int64)
    if not np.array_equal(labels, rf_labels):
        raise ValueError("model prediction artifacts disagree on held-out labels")
    gnn_prob = np.stack([molecular[pair_id][1:] for pair_id in pair_ids])
    rf_prob = np.stack([rf[pair_id][1:] for pair_id in pair_ids])
    if gnn_prob.shape[1:] != (len(LABELS),) or rf_prob.shape != gnn_prob.shape:
        raise ValueError("prediction arrays do not match the documented three-class order")
    gnn_metrics, rf_metrics = compute_metrics(labels, gnn_prob), compute_metrics(labels, rf_prob)
    gnn_pred, rf_pred = gnn_prob.argmax(axis=1), rf_prob.argmax(axis=1)
    split_manifest = json.loads((project_root / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    pair_rows = json.loads((project_root / "data/processed/ddi_pairs_full.json").read_text(encoding="utf-8"))
    pairs_by_id = {row["pair_id"]: row for row in pair_rows}
    train_counts = Counter(pairs_by_id[pair_id]["severity"]
                           for pair_id in molecular_coverage["covered_pair_ids"]["train"])
    majority_class = max(LABELS, key=lambda label: (train_counts[label], -LABELS.index(label)))
    majority_label = LABELS.index(majority_class)
    majority = np.full_like(labels, majority_label)
    majority_prob = np.zeros((labels.size, len(LABELS)), dtype=np.float64)
    majority_prob[:, majority_label] = 1.0
    return {
        "status": "completed_matched_coverage_comparison",
        "protocol": protocol,
        "test_pair_count": int(len(pair_ids)),
        "test_pair_ids_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
        "test_class_counts": {label: int(np.count_nonzero(labels == index)) for index, label in enumerate(LABELS)},
        "models": {
            "molecular_gnn": {"prediction_artifact": molecular_meta, "metrics_raw": gnn_metrics},
            "fingerprint_random_forest": {"prediction_artifact": rf_meta, "metrics_raw": rf_metrics},
            "training_majority_baseline": {"selected_class_from_training_only": majority_class,
                                           "metrics": compute_metrics(labels, majority_prob)},
        },
        "paired_bootstrap": paired_bootstrap_macro_f1(labels, gnn_pred, rf_pred,
                                                      replicates=replicates, seed=seed),
        "model_comparison_note": "Pair-disjoint and cold-drug protocols are analyzed separately. Both models must have exactly the same held-out pair IDs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", choices=("pair_disjoint", "cold_drug", "both"), default="both")
    parser.add_argument("--rf-run-tag", default="prediction_export_v1")
    parser.add_argument("--molecular-run-tag", default="no_placeholder_v1")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    protocols = ("pair_disjoint", "cold_drug") if args.protocol == "both" else (args.protocol,)
    results = {}
    comparison_tag = f"{args.molecular_run_tag or 'default'}_{args.rf_run_tag}"
    for protocol in protocols:
        report = compare_protocol(args.project_root, protocol, rf_run_tag=args.rf_run_tag,
                                  molecular_run_tag=args.molecular_run_tag,
                                  replicates=args.replicates, seed=args.seed)
        path = args.project_root / "artifacts" / f"matched_comparison_{protocol}_{comparison_tag}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results[protocol] = {"path": str(path), "test_pair_count": report["test_pair_count"],
                             "gnn_macro_f1": report["models"]["molecular_gnn"]["metrics_raw"]["macro_f1"],
                             "rf_macro_f1": report["models"]["fingerprint_random_forest"]["metrics_raw"]["macro_f1"],
                             "paired_delta_ci95": report["paired_bootstrap"]["first_minus_second_macro_f1_ci95"]}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
