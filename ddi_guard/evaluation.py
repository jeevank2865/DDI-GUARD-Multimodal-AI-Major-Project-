"""Evaluation helpers that compute metrics from real labels and probabilities."""
from __future__ import annotations

from typing import Any
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
import torch
from torch import nn

from . import LABELS


def compute_metrics(y_true: list[int] | np.ndarray, probabilities: np.ndarray, ece_bins: int = 15) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape != (y.size, len(LABELS)):
        raise ValueError(f"probabilities must have shape [N,{len(LABELS)}]")
    if not y.size or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("labels/probabilities are empty, non-finite, or outside [0,1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    if (y < 0).any() or (y >= len(LABELS)).any():
        raise ValueError("labels are outside the documented class mapping")
    predictions = p.argmax(axis=1)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y, predictions, labels=list(range(len(LABELS))), zero_division=0)
    one_hot = np.eye(len(LABELS), dtype=np.float64)[y]
    clipped = np.clip(p, 1e-12, 1.0)
    confidence = p.max(axis=1)
    correct = predictions == y
    ece = 0.0
    for low in np.linspace(0.0, 1.0, ece_bins + 1)[:-1]:
        high = low + 1.0 / ece_bins
        selected = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    try:
        auc = float(roc_auc_score(y, p, labels=list(range(len(LABELS))), multi_class="ovr", average="macro"))
        auc_note = None
    except ValueError as exc:
        auc = None
        auc_note = str(exc)
    return {
        "n": int(y.size),
        "class_order": list(LABELS),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", labels=list(range(len(LABELS))), zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", labels=list(range(len(LABELS))), zero_division=0)),
        "per_class": {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(per_class_f1[i]), "support": int(support[i])}
                      for i in range(len(LABELS))},
        "confusion_matrix": confusion_matrix(y, predictions, labels=list(range(len(LABELS)))).tolist(),
        "roc_auc_ovr_macro": auc,
        "roc_auc_note": auc_note,
        "multiclass_brier_score": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(-np.log(clipped[np.arange(y.size), y]).mean()),
        "expected_calibration_error": float(ece),
        "ece_bins_equal_width": int(ece_bins),
    }


def fit_temperature_on_validation(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit scalar temperature on validation logits only by minimizing validation NLL."""
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or labels.shape != (logits.shape[0],):
        raise ValueError("validation logits/labels have incompatible shapes")
    if not torch.isfinite(logits).all() or labels.numel() == 0:
        raise ValueError("validation data is empty or non-finite")
    # Optimize log-temperature, guaranteeing T > 0.
    log_t = nn.Parameter(torch.zeros((), dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_t.exp().clamp(1e-3, 1e3)
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(1e-3, 1e3).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / temperature, dim=-1)


def save_prediction_artifact(path: str | Path, pair_ids: list[str], labels: np.ndarray,
                             probabilities_raw: np.ndarray, probabilities_calibrated: np.ndarray,
                             metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist exact held-out predictions and provenance in a compressed NPZ."""
    path = Path(path)
    ids = np.asarray(pair_ids, dtype=np.str_)
    y = np.asarray(labels, dtype=np.int64)
    raw = np.asarray(probabilities_raw, dtype=np.float64)
    calibrated = np.asarray(probabilities_calibrated, dtype=np.float64)
    if ids.ndim != 1 or len(set(ids.tolist())) != ids.size:
        raise ValueError("prediction artifact pair IDs must be a unique vector")
    if y.shape != ids.shape or raw.shape != (ids.size, len(LABELS)) or calibrated.shape != raw.shape:
        raise ValueError("prediction artifact arrays have inconsistent shapes")
    compute_metrics(y, raw)
    compute_metrics(y, calibrated)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded_metadata = json.dumps({"class_order": list(LABELS), **metadata}, sort_keys=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, pair_ids=ids, labels=y,
                            probabilities_raw=raw, probabilities_calibrated=calibrated,
                            metadata=np.asarray(encoded_metadata))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "pair_count": int(ids.size),
            "pair_ids_sha256": hashlib.sha256("\n".join(sorted(pair_ids)).encode()).hexdigest()}
