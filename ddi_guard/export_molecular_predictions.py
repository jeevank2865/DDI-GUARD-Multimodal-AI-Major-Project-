"""Recompute and persist held-out probabilities from an existing selected GNN checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation import apply_temperature, compute_metrics, save_prediction_artifact
from .features import molecule_to_graph
from .model import load_checkpoint
from .train_molecular import _load_approved_compounds, _predict


def export(project_root: Path, protocol: str) -> dict[str, Any]:
    output_root = project_root / "artifacts" / f"molecular_{protocol}"
    evaluation_path = output_root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "completed_molecular_only_ablation" or not evaluation.get("checkpoint_reload_verified"):
        raise ValueError("only a completed, reload-verified molecular ablation can be exported")
    split_manifest = json.loads((project_root / "artifacts/split_manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((output_root / "pair_coverage_manifest.json").read_text(encoding="utf-8"))
    pair_rows = json.loads((project_root / "data/processed/ddi_pairs_full.json").read_text(encoding="utf-8"))
    pairs_by_id = {row["pair_id"]: row for row in pair_rows}
    test_ids = coverage["covered_pair_ids"]["test"]
    if len(set(test_ids)) != len(test_ids) or not set(test_ids) <= set(split_manifest[protocol]["test_pair_ids"]):
        raise ValueError("covered test IDs are duplicate or outside the fixed held-out test split")
    test_rows = [pairs_by_id[pair_id] for pair_id in test_ids]
    records = _load_approved_compounds(project_root / "data/processed/pubchem_compounds.json",
                                       project_root / "artifacts/pubchem_coverage_audit.json")
    graphs = {drug_id: molecule_to_graph(item.get("isomeric_smiles") or item["canonical_smiles"])
              for drug_id, item in records.items()
              if item.get("usable_for_model") and item.get("canonical_smiles_rdkit_validated")}
    model = load_checkpoint(evaluation["checkpoint"], evaluation["checkpoint_config"]).eval()
    logits, labels = _predict(model, test_rows, graphs, batch_size=128)
    raw = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    temperature = float(evaluation["temperature_fitted_on_validation_only"])
    calibrated = apply_temperature(torch.from_numpy(logits).float(), temperature).numpy()
    raw_metrics, calibrated_metrics = compute_metrics(labels, raw), compute_metrics(labels, calibrated)
    for section, observed in (("test_metrics_raw", raw_metrics),
                              ("test_metrics_validation_temperature", calibrated_metrics)):
        expected = evaluation[section]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                       "roc_auc_ovr_macro", "multiclass_brier_score",
                       "negative_log_likelihood", "expected_calibration_error"):
            if expected[metric] is None:
                if observed[metric] is not None:
                    raise AssertionError(f"recomputed {section}.{metric} changed from None")
            elif observed[metric] is None or not np.isclose(expected[metric], observed[metric], rtol=1e-7, atol=1e-9):
                raise AssertionError(f"recomputed {section}.{metric} differs: {expected[metric]} != {observed[metric]}")
    artifact = save_prediction_artifact(
        output_root / "test_predictions.npz", test_ids, labels, raw, calibrated,
        {"protocol": protocol, "seed": evaluation["seed"],
         "model_mode": "molecular_only_ablation", "checkpoint": evaluation["checkpoint"],
         "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
         "temperature_fitted_on_validation_only": temperature,
         "metrics_recomputed_and_matched_existing_evaluation": True})
    evaluation["test_prediction_artifact"] = artifact
    evaluation["test_metrics_verified_against_saved_predictions"] = True
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", choices=("pair_disjoint", "cold_drug"), required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.project_root, args.protocol), indent=2))


if __name__ == "__main__":
    main()
