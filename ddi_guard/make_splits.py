"""Create exact, reproducible pair-disjoint and cold-drug split manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .splits import cold_drug_split, pair_disjoint_split


def write_splits(pairs_path: Path, output_path: Path, seed: int = 42) -> dict[str, Any]:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    pair_split = pair_disjoint_split(pairs, seed=seed)
    cold_split = cold_drug_split(pairs, seed=seed)
    label_by_id = {row["pair_id"]: row["severity"] for row in pairs}
    pair_split["class_counts"] = {name: dict(Counter(label_by_id[pair_id] for pair_id in pair_ids))
                                  for name, pair_ids in (("train", pair_split["train_pair_ids"]),
                                                         ("validation", pair_split["validation_pair_ids"]),
                                                         ("test", pair_split["test_pair_ids"]))}
    cold_split["class_counts"] = {name: dict(Counter(label_by_id[pair_id] for pair_id in pair_ids))
                                   for name, pair_ids in (("train", cold_split["train_pair_ids"]),
                                                          ("validation", cold_split["validation_pair_ids"]),
                                                          ("test", cold_split["test_pair_ids"]))}
    result = {
        "format_version": 1,
        "seed": seed,
        "source_pairs_path": str(pairs_path),
        "source_pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "pair_disjoint": pair_split,
        "cold_drug": cold_split,
        "protocol_notes": {
            "pair_disjoint": "Canonical pair IDs are unique and stratified by severity; test pairs do not occur in train or validation.",
            "cold_drug": "Only pairs whose two drugs both belong to the held-out drug partition are test examples. Cross-partition pairs are excluded from this protocol."
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/ddi_pairs_full.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = write_splits(args.pairs, args.output, args.seed)
    print(json.dumps({
        "source_pairs_sha256": result["source_pairs_sha256"],
        "pair_disjoint_sizes": {k: len(result["pair_disjoint"][f"{k}_pair_ids"]) for k in ("train", "validation", "test")},
        "cold_drug_sizes": {"train": len(result["cold_drug"]["train_pair_ids"]),
                            "validation": len(result["cold_drug"]["validation_pair_ids"]),
                            "test": len(result["cold_drug"]["test_pair_ids"]),
                            "excluded_cross_partition": len(result["cold_drug"]["excluded_cross_partition_pair_ids"])},
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
