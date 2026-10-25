"""Auditable non-learned baseline calculations on preserved pair-disjoint splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import LABELS, LABEL_TO_ID
from .evaluation import compute_metrics


def majority_baseline(pairs_path: Path, split_path: Path, output_path: Path) -> dict[str, Any]:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))["pair_disjoint"]
    by_id = {row["pair_id"]: row for row in pairs}
    train_ids = split["train_pair_ids"]
    test_ids = split["test_pair_ids"]
    if set(train_ids) & set(test_ids):
        raise ValueError("train/test pair IDs overlap")
    train_counts = Counter(by_id[pair_id]["severity"] for pair_id in train_ids)
    majority_class = max(LABELS, key=lambda label: (train_counts[label], -LABEL_TO_ID[label]))
    test_rows = [by_id[pair_id] for pair_id in test_ids]
    y_true = np.array([LABEL_TO_ID[row["severity"]] for row in test_rows], dtype=np.int64)
    probabilities = np.zeros((len(test_rows), len(LABELS)), dtype=np.float64)
    probabilities[:, LABEL_TO_ID[majority_class]] = 1.0
    metrics = compute_metrics(y_true, probabilities)
    result = {
        "status": "completed",
        "baseline": "majority_class_from_training_split",
        "selected_class_using_train_only": majority_class,
        "train_pair_count": len(train_ids), "train_class_counts": dict(train_counts),
        "test_pair_count": len(test_rows),
        "test_class_counts": dict(Counter(row["severity"] for row in test_rows)),
        "test_pair_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        "source_pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": ["The predicted class is selected only from training label frequencies.",
                  "All test probabilities are the one-hot majority-class output; AUROC is computed from these probabilities, not a heuristic.",
                  "This is a baseline calculation, not a trained model or clinical performance estimate."],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/ddi_pairs_full.json"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/majority_baseline_pair_disjoint.json"))
    args = parser.parse_args()
    print(json.dumps(majority_baseline(args.pairs, args.splits, args.output), indent=2))


if __name__ == "__main__":
    main()
