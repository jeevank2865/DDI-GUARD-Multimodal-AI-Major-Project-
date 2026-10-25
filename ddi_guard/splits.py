"""Deterministic leakage-aware pair-disjoint and cold-drug split assignment."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split


def pair_disjoint_split(pairs: list[dict[str, Any]], seed: int = 42,
                        fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three values summing to one")
    ids = np.array([row["pair_id"] for row in pairs])
    labels = np.array([row["severity"] for row in pairs])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate pair ids cannot be split safely")
    train_ids, rem_ids, train_y, rem_y = train_test_split(ids, labels, test_size=fractions[1] + fractions[2],
                                                         random_state=seed, stratify=labels)
    val_fraction_of_rem = fractions[1] / (fractions[1] + fractions[2])
    val_ids, test_ids = train_test_split(rem_ids, test_size=1 - val_fraction_of_rem,
                                         random_state=seed, stratify=rem_y)
    return {"train_pair_ids": sorted(train_ids.tolist()), "validation_pair_ids": sorted(val_ids.tolist()),
            "test_pair_ids": sorted(test_ids.tolist()), "seed": seed}


def cold_drug_split(pairs: list[dict[str, Any]], seed: int = 42,
                    drug_holdout_fraction: float = 0.30) -> dict[str, Any]:
    """Three-way cold-drug split; held-out drugs never occur in training.

    Each retained pair has both endpoints inside one drug partition. Cross-partition
    pairs are explicitly excluded from this protocol.
    """
    drugs = sorted({row[k] for row in pairs for k in ("drug_a_id", "drug_b_id")})
    if len(drugs) < 6:
        raise ValueError("cold-drug split requires at least six unique drugs")
    train_drugs, heldout_drugs = train_test_split(drugs, test_size=drug_holdout_fraction, random_state=seed)
    validation_drugs, test_drugs = train_test_split(heldout_drugs, test_size=0.5, random_state=seed)
    train_set, validation_set, test_set = set(train_drugs), set(validation_drugs), set(test_drugs)
    train, validation, test, excluded = [], [], [], []
    for row in pairs:
        endpoints = {row["drug_a_id"], row["drug_b_id"]}
        if endpoints <= train_set:
            train.append(row["pair_id"])
        elif endpoints <= validation_set:
            validation.append(row["pair_id"])
        elif endpoints <= test_set:
            test.append(row["pair_id"])
        else:
            excluded.append(row["pair_id"])
    return {"train_pair_ids": sorted(train), "validation_pair_ids": sorted(validation),
            "test_pair_ids": sorted(test), "excluded_cross_partition_pair_ids": sorted(excluded),
            "train_drug_ids": sorted(train_set), "validation_drug_ids": sorted(validation_set),
            "test_drug_ids": sorted(test_set), "heldout_drug_ids": sorted(validation_set | test_set), "seed": seed}
