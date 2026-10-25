#!/usr/bin/env python3
"""Audit saved matched-support DDI-GUARD runs without fitting or changing models."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paired_multimodal_validation_v1"
LABELS = ["Minor", "Moderate", "Major"]
MODALITIES = {"Text": ("text",), "Molecular + Text": ("molecular", "text"),
              "Target + Text": ("target", "text"), "Molecular + Target": ("molecular", "target"),
              "Full Molecular + Target + Text": ("molecular", "target", "text")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, *, reps: int = 10000,
              seed: int = 20261020) -> dict:
    """Paired class-stratified bootstrap; preserves each observed class support."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == i) for i in range(len(LABELS))]
    deltas = np.empty(reps)
    def macro_f1(pred: np.ndarray, ix: np.ndarray) -> np.ndarray:
        encoded = y[ix] * 3 + pred[ix]
        cm = np.stack([np.bincount(row, minlength=9).reshape(3, 3) for row in encoded])
        tp = np.diagonal(cm, axis1=1, axis2=2)
        denom = 2 * tp + cm.sum(axis=1) - tp + cm.sum(axis=2) - tp
        scores = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
        return scores.mean(axis=1)
    for start in range(0, reps, 200):
        size = min(200, reps - start)
        ix = np.concatenate([rng.choice(group, size=(size, len(group)), replace=True) for group in by_class], axis=1)
        deltas[start:start + size] = macro_f1(a, ix) - macro_f1(b, ix)
    return {"method": "paired class-stratified percentile bootstrap over fixed test pair IDs",
            "replicates": reps, "seed": seed, "confidence_level": 0.95,
            "difference_definition": "full molecular+target+text minus comparator macro-F1",
            "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "bootstrap_delta_median": float(np.median(deltas)),
            "note": "Preserves observed test class counts; descriptive uncertainty only; no model selection."}


def load_run(protocol: str, label: str, modalities: tuple[str, ...]) -> dict:
    mtag = "_".join(modalities)
    folder = ROOT / "artifacts" / f"multimodal_{protocol}_{mtag}_dailymed_common_support_trainfit_v1_support_molecular_target_text"
    ep, pp = folder / "evaluation.json", folder / "test_predictions.npz"
    ev = json.loads(ep.read_text())
    split_path = ROOT / "artifacts/split_manifest.json"
    dataset_path = ROOT / "data/processed/ddi_pairs_full.json"
    if ev["split_manifest_sha256"] != sha(split_path):
        raise ValueError(f"split manifest hash mismatch: {folder}")
    if ev["dataset_manifest_sha256"] != sha(dataset_path):
        raise ValueError(f"dataset manifest hash mismatch: {folder}")
    if tuple(ev["modalities"]) != modalities:
        raise ValueError(f"evaluation modality declaration mismatch: {folder}")
    cp = Path(ev["checkpoint"])
    if not cp.is_absolute():
        cp = ROOT / cp
    with np.load(pp, allow_pickle=False) as z:
        ids = z["pair_ids"].astype(str).tolist()
        y = z["labels"].astype(np.int64)
        p_raw = z["probabilities_raw"].astype(np.float64)
        p = z["probabilities_calibrated"].astype(np.float64)
        metadata = json.loads(str(z["metadata"]))
    if not ev.get("temperature_fitted_on_validation_only") or not np.isclose(p.sum(axis=1), 1).all():
        raise ValueError(f"calibration/probability check failed: {folder}")
    temperature = float(ev["temperature_fitted_on_validation_only"])
    reconstructed = np.power(np.clip(p_raw, 1e-15, 1.0), 1.0 / temperature)
    reconstructed /= reconstructed.sum(axis=1, keepdims=True)
    if not np.allclose(reconstructed, p, rtol=2e-7, atol=2e-7):
        raise ValueError(f"saved calibrated probabilities do not match recorded validation temperature: {folder}")
    if ev["checkpoint_sha256"] != sha(cp) or ev["test_prediction_artifact"]["sha256"] != sha(pp):
        raise ValueError(f"checkpoint or prediction hash mismatch: {folder}")
    if ev["test_prediction_artifact"]["pair_ids_sha256"] != id_hash(ids):
        raise ValueError(f"pair ID hash mismatch: {folder}")
    if ev["pair_ids_sha256"]["test"] != id_hash(ids):
        raise ValueError(f"evaluation split IDs mismatch: {folder}")
    if metadata.get("class_order") != LABELS:
        raise ValueError(f"class order mismatch: {folder}")
    pred = p.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    class_metrics = {LABELS[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                 "f1": float(f1[i]), "support": int(sup[i])} for i in range(3)}
    conf = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    metrics = ev["test_metrics_validation_temperature"]
    if not np.isclose(metrics["macro_f1"], np.mean(f1), atol=1e-10):
        raise ValueError(f"recomputed calibrated macro-F1 mismatch: {folder}")
    target_manifest = ev["sources"].get("target", {}).get("manifest_sha256")
    text_manifest = ev["sources"].get("text", {}).get("manifest_sha256")
    verified_text_manifest_path = None
    declared_text_path = ev["sources"].get("text", {}).get("manifest_path")
    declared_text_sha = None
    if text_manifest:
        text_path = ROOT / "data/processed" / ("text_evidence_embeddings_manifest_cold_drug_trainfit_v1.csv"
                      if protocol == "cold_drug" else "text_evidence_embeddings_manifest_pair_disjoint_trainfit_v1.csv")
        if sha(text_path) != text_manifest:
            raise ValueError(f"text feature manifest hash mismatch: {folder}")
        verified_text_manifest_path = str(text_path)
        declared_path = Path(declared_text_path or "")
        if not declared_path.is_absolute():
            declared_path = ROOT / declared_path
        if declared_path.is_file():
            declared_text_sha = sha(declared_path)
    cov_path = folder / "pair_coverage_manifest.json"
    cov = json.loads(cov_path.read_text())
    if set(cov["selected_pair_ids"]["test"]) != set(ids):
        raise ValueError(f"coverage manifest test IDs mismatch: {folder}")
    return {"label": label, "modalities": list(modalities), "folder": str(folder),
            "evaluation_path": str(ep), "evaluation_sha256": sha(ep), "prediction_path": str(pp),
            "prediction_sha256": sha(pp), "checkpoint_path": str(cp),
            "checkpoint_sha256": sha(cp), "checkpoint_reload_verified": ev["checkpoint_reload_verified"],
            "split_manifest_sha256": ev["split_manifest_sha256"],
            "split_manifest_path": str(split_path), "dataset_manifest_path": str(dataset_path),
            "dataset_manifest_sha256": ev["dataset_manifest_sha256"],
            "pair_coverage_manifest_path": str(cov_path), "pair_coverage_manifest_sha256": sha(cov_path),
            "text_feature_manifest_declared_path": declared_text_path,
            "text_feature_manifest_verified_path": verified_text_manifest_path,
            "text_feature_manifest_declared_path_sha256": declared_text_sha,
            "text_feature_manifest_path_mismatch": bool(text_manifest and declared_text_sha != text_manifest),
            "text_feature_manifest_sha256": text_manifest, "target_graph_manifest_sha256": target_manifest,
            "temperature_fitted_on_validation_only": True, "validation_fit_temperature": temperature,
            "temperature": temperature, "best_epoch": ev["best_epoch"],
            "validation_macro_f1": ev["best_validation_macro_f1"],
            "test_pair_count": len(ids), "test_pair_ids_sha256": id_hash(ids),
            "test_class_support": dict(zip(LABELS, map(int, np.bincount(y, minlength=3)))),
            "calibrated_test_metrics": metrics, "per_class": class_metrics,
            "confusion_matrix_rows_true_columns_predicted": conf,
            "_ids": ids, "_y": y, "_p": p}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "completed", "analysis_only": True, "labels": LABELS,
              "bootstrap_replicates": 10000, "bootstrap_seed": 20261020,
              "protocols": {}, "dailymed_429_audit": {}}
    for protocol in ("pair_disjoint", "cold_drug"):
        runs = {label: load_run(protocol, label, mods) for label, mods in MODALITIES.items()}
        reference = runs["Full Molecular + Target + Text"]
        for run in runs.values():
            if run["_ids"] != reference["_ids"] or not np.array_equal(run["_y"], reference["_y"]):
                raise ValueError(f"non-matched support/labels in {protocol}: {run['label']}")
        comparisons = {}
        for label in ("Molecular + Text", "Target + Text", "Text", "Molecular + Target"):
            comp = runs[label]
            comparisons[label] = bootstrap(reference["_y"], reference["_p"].argmax(axis=1),
                                           comp["_p"].argmax(axis=1))
            comparisons[label]["observed_macro_f1_difference"] = (
                reference["calibrated_test_metrics"]["macro_f1"] - comp["calibrated_test_metrics"]["macro_f1"])
        for r in runs.values():
            del r["_ids"], r["_y"], r["_p"]
        report["protocols"][protocol] = {"matched_test_support": reference["test_pair_count"],
                                          "test_pair_ids_sha256": reference["test_pair_ids_sha256"],
                                          "test_class_support": reference["test_class_support"],
                                          "models": runs, "paired_macro_f1_differences": comparisons}
    audit_path = ROOT / "artifacts/dailymed_evidence_coverage_audit.json"
    audit = json.loads(audit_path.read_text())
    errors = audit.get("errors", [])
    report["dailymed_429_audit"] = {"audit_path": str(audit_path), "audit_sha256": sha(audit_path),
        "status": audit.get("status"), "processed_drugs": audit.get("processed_drugs", audit.get("input_drugs")),
        "covered_drugs": audit.get("covered_drugs"), "evidence_records": audit.get("evidence_records"),
        "error_counts": dict(Counter(e.get("error", "unknown").split(":", 1)[-1].strip() for e in errors)),
        "total_errors": len(errors), "retry_executed": False,
        "retry_recommendation": {"scope": "only IDs recorded as failed; preserve successful corpus rows and original audit",
          "policy": "honor Retry-After when present; otherwise exponential backoff with full jitter, base 30 seconds, cap 10 minutes, at most 5 retries; global concurrency 1; target no more than one request/second; open circuit after 3 consecutive 429 responses and defer remaining work",
          "provenance": "write retries as an append-only attempt shard with request time, URL, response status/headers, attempt number and source hash; verify and stage a merged corpus/manifest separately; never overwrite source corpus or vectors in place",
          "note": "These are conservative client-side recommendations, not a claimed DailyMed-published quota."}}
    approval_path = ROOT / "artifacts/model_approval.json"
    approval = json.loads(approval_path.read_text())
    report["prediction_gate"] = {"approval_path": str(approval_path), "approval_sha256": sha(approval_path),
        "approved": approval.get("approved"), "approval_status": approval.get("approval_status"),
        "api_should_remain_disabled": not bool(approval.get("approved")),
        "stale_blocked_reasons": approval.get("blocked_reasons", []),
        "note": "The gate remains closed. Several blocked reasons are stale relative to this verified research phase; the approval file was deliberately left unchanged.",
        "clinical_validation": approval.get("clinical_validation")}
    path = OUT / "paired_multimodal_validation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Paired Multimodal Research Validation", "",
      "Generated from saved artifacts only; no model was retrained and no test data were used for selection, calibration, or fitting.", "",
      "Class order and confusion matrix orientation: Minor, Moderate, Major; rows are true labels and columns are predicted labels.", "",
      "All probabilities below are calibrated with a scalar temperature fitted on validation only. ECE uses 15 equal-width bins.", ""]
    for protocol, pr in report["protocols"].items():
        lines += [f"## {protocol.replace('_', ' ').title()}", "",
          f"Matched test support: {pr['matched_test_support']}; pair-ID SHA-256 `{pr['test_pair_ids_sha256']}`; class supports {pr['test_class_support']}.", "",
          "| Model | Validation macro-F1 | Test macro-F1 | AUROC OVR macro | Brier | NLL | ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
        for label, run in pr["models"].items():
            m = run["calibrated_test_metrics"]
            lines.append(f"| {label} | {run['validation_macro_f1']:.4f} | {m['macro_f1']:.4f} | {m['roc_auc_ovr_macro']:.4f} | {m['multiclass_brier_score']:.4f} | {m['negative_log_likelihood']:.4f} | {m['expected_calibration_error']:.4f} |")
        lines += ["", "Paired bootstrap for full model minus comparator (10,000 resamples; stratified within observed class supports):", "",
          "| Comparator | Δ macro-F1 | 95% CI |", "|---|---:|---:|"]
        for label, delta in pr["paired_macro_f1_differences"].items():
            ci = delta["ci95"]
            lines.append(f"| {label} | {delta['observed_macro_f1_difference']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines += ["", "Per-class calibrated test metrics (precision / recall / F1 / support):", "",
          "| Model | Minor | Moderate | Major |", "|---|---|---|---|"]
        for label, run in pr["models"].items():
            cells = []
            for cls in LABELS:
                cm = run["per_class"][cls]
                cells.append(f"{cm['precision']:.3f} / {cm['recall']:.3f} / {cm['f1']:.3f} / {cm['support']}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "Confusion matrices (true rows × predicted columns; [Minor, Moderate, Major]):", ""]
        for label, run in pr["models"].items():
            lines.append(f"- **{label}:** `{run['confusion_matrix_rows_true_columns_predicted']}`")
        lines += ["", "Artifact-integrity checks passed for evaluation, prediction, checkpoint, split IDs, common-support manifest, checkpoint reload flag, and protocol-specific text feature manifest hash. The cold-drug evaluation records declare the generic text-manifest path, whose current bytes are pair-disjoint scoped, while recording the cold-drug manifest hash. The cold-drug feature hash was independently verified against the cold-drug train-fit manifest; original evaluation artifacts were preserved. Pair-disjoint generic and specific manifest bytes match.", ""]
    dd = report["dailymed_429_audit"]
    lines += ["## DailyMed rate-limit audit", "",
      f"The preserved ingestion audit reports {dd['processed_drugs']} attempted IDs, {dd['covered_drugs']} covered drugs, {dd['evidence_records']} passages, and {dd['total_errors']} errors: {dd['error_counts']}.", "",
      "No retries were sent during this analysis. For a future, separately staged retry, the recommended conservative client policy is to retry failed IDs only; honor Retry-After; otherwise use full-jitter exponential backoff starting at 30 s, capped at 10 min, no more than five retries, global concurrency one and at most one request/s; stop after three consecutive 429s and defer the remainder. Record each attempt and response provenance in an append-only shard, verify hashes, and build a staged merged corpus/manifest without overwriting the existing corpus. These are operational recommendations, not a claimed NLM quota.", "",
      "## Interpretation and limitations", "",
      "The cold-drug matched test contains only 253 pairs: 14 Minor, 206 Moderate, and 33 Major. The 14-pair Minor support makes its precision/recall/F1 highly unstable; a stratified bootstrap preserves n=14 but does not create additional independent Minor examples. Cold-drug performance is materially below pair-disjoint performance, especially on Major and Minor classes. Results are descriptive of the fixed split and limited feature coverage. This analysis did not inspect or alter held-out pairs to tune models. It does not establish clinical validity.", "",
      "The prediction approval gate remains closed (`approved=false`, `approval_status=blocked`). The existing approval manifest contains stale blocked reasons that say ChEMBL, text evidence, and full multimodal training are absent. That approval file was left unchanged; the API must remain disabled until the project's full approval criteria are reviewed and satisfied.", "",
      "Reproduce: `python scripts/paired_multimodal_validation.py`", ""]
    md_path = OUT / "paired_multimodal_validation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    for protocol, pr in report["protocols"].items():
        print(protocol, pr["matched_test_support"], pr["test_class_support"])
        for name, diff in pr["paired_macro_f1_differences"].items():
            print(f"  full - {name}: {diff['observed_macro_f1_difference']:+.4f} CI95 {diff['ci95']}")


if __name__ == "__main__":
    main()
