"""Train and evaluate the molecular-only ablation on verified PubChem coverage.

This is not the full multimodal production model. It requires completed PubChem
identity review, keeps missing-structure pairs in an exclusion ledger, selects a
checkpoint using validation macro-F1, and touches the test split only afterward.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import LABELS, LABEL_TO_ID
from .evaluation import (apply_temperature, compute_metrics, fit_temperature_on_validation,
                         save_prediction_artifact)
from .features import batch_graphs, molecule_to_graph
from .model import (DDIModelConfig, MultimodalDDIGuard, parameter_counts,
                    save_checkpoint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0], "torch": torch.__version__,
                                    "numpy": np.__version__}
    for distribution in ("rdkit", "scikit-learn", "torch-geometric"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def _save_training_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _load_approved_compounds(compounds_path: Path, audit_path: Path) -> dict[str, dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("complete"):
        raise ValueError("PubChem enrichment/review is incomplete; refusing to start a final ablation run")
    compounds = json.loads(compounds_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for item in compounds:
        if item.get("usable_for_model") and item.get("canonical_smiles_rdkit_validated"):
            drug_id = str(item["ddinter_id"])
            if drug_id in records:
                raise ValueError(f"duplicate approved PubChem record for {drug_id}")
            records[drug_id] = item
    return records


def _protocol_rows(split: dict[str, Any], pairs_by_id: dict[str, dict[str, Any]],
                   available_drugs: set[str], protocol_name: str, split_names: tuple[str, ...],
                   exclusion_writer: csv.DictWriter) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for split_name in split_names:
        rows: list[dict[str, Any]] = []
        for pair_id in split[f"{split_name}_pair_ids"]:
            row = pairs_by_id[pair_id]
            missing = [drug_id for drug_id in (row["drug_a_id"], row["drug_b_id"])
                       if drug_id not in available_drugs]
            if missing:
                exclusion_writer.writerow({"protocol": protocol_name, "split": split_name,
                                           "pair_id": pair_id, "reason": "missing_approved_pubchem_structure",
                                           "missing_drug_ids": ";".join(sorted(set(missing)))})
            else:
                rows.append(row)
        results[split_name] = rows
    return results


def _make_batch(rows: list[dict[str, Any]], graphs: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    graph_a = batch_graphs([graphs[row["drug_a_id"]] for row in rows])
    graph_b = batch_graphs([graphs[row["drug_b_id"]] for row in rows])
    labels = torch.tensor([LABEL_TO_ID[row["severity"]] for row in rows], dtype=torch.long)
    return graph_a, graph_b, labels


def _predict(model: MultimodalDDIGuard, rows: list[dict[str, Any]],
             graphs: dict[str, dict[str, torch.Tensor]], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            graph_a, graph_b, labels = _make_batch(batch, graphs)
            output = model(molecule_a=graph_a, molecule_b=graph_b)
            logits_out.extend(output["logits"].cpu().numpy())
            labels_out.extend(labels.numpy().tolist())
    return np.asarray(logits_out), np.asarray(labels_out, dtype=np.int64)


def _majority_metrics(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_counts = Counter(row["severity"] for row in train_rows)
    selected = max(LABELS, key=lambda label: (train_counts[label], -LABEL_TO_ID[label]))
    probabilities = np.zeros((len(test_rows), len(LABELS)), dtype=np.float64)
    probabilities[:, LABEL_TO_ID[selected]] = 1.0
    return {"selected_class_using_train_only": selected,
            "metrics": compute_metrics([LABEL_TO_ID[row["severity"]] for row in test_rows], probabilities)}


def train_protocol(project_root: Path, protocol_name: str, *, seed: int = 42, epochs: int = 20,
                   batch_size: int = 128, patience: int = 4, hidden_dim: int = 96,
                   learning_rate: float = 1e-3, resume: bool = False,
                   run_tag: str | None = None) -> dict[str, Any]:
    if protocol_name not in ("pair_disjoint", "cold_drug"):
        raise ValueError("protocol_name must be pair_disjoint or cold_drug")
    if min(epochs, batch_size, patience, hidden_dim) < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience, hidden_dim, and learning_rate must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pairs_path = project_root / "data/processed/ddi_pairs_full.json"
    split_path = project_root / "artifacts/split_manifest.json"
    compounds_path = project_root / "data/processed/pubchem_compounds.json"
    pubchem_audit_path = project_root / "artifacts/pubchem_coverage_audit.json"
    for path in (pairs_path, split_path, compounds_path, pubchem_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if run_tag is not None and (not run_tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_tag)):
        raise ValueError("run_tag may contain only letters, digits, hyphen, or underscore")
    output_name = f"molecular_{protocol_name}" + (f"_{run_tag}" if run_tag else "")
    output_root = project_root / "artifacts" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split = split_manifest[protocol_name]
    records = _load_approved_compounds(compounds_path, pubchem_audit_path)
    smiles = {drug_id: item.get("isomeric_smiles") or item.get("canonical_smiles")
              for drug_id, item in records.items()}
    graphs = {drug_id: molecule_to_graph(value) for drug_id, value in smiles.items() if value}
    available_drugs = set(graphs)

    exclusions_path = output_root / "missing_structure_pairs.csv"
    with exclusions_path.open("w", encoding="utf-8", newline="") as exclusion_handle:
        exclusion_writer = csv.DictWriter(exclusion_handle, fieldnames=[
            "protocol", "split", "pair_id", "reason", "missing_drug_ids"])
        exclusion_writer.writeheader()
        selected = _protocol_rows(split, pairs_by_id, available_drugs, protocol_name,
                                  ("train", "validation", "test"), exclusion_writer)
    train_rows = selected["train"]
    validation_rows = selected["validation"]
    test_rows = selected["test"]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError(f"insufficient covered pairs for {protocol_name}: "
                         f"train={len(train_rows)}, validation={len(validation_rows)}, test={len(test_rows)}")
    train_counts = Counter(row["severity"] for row in train_rows)
    if any(train_counts[label] == 0 for label in LABELS):
        raise ValueError(f"covered training split lacks a target class: {dict(train_counts)}")

    # A molecular ablation must not allocate nominal target/text branches or fake
    # target/relation vocabularies. Their data is absent, so those modules are absent.
    config = DDIModelConfig(num_drugs=0, num_targets=0, num_relations=0,
                            text_dim=0, hidden_dim=hidden_dim, attention_heads=4,
                            molecular_layers=3, use_molecular=True, use_target=False, use_text=False)
    model = MultimodalDDIGuard(config)
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    class_weights = torch.tensor([
        len(train_rows) / (len(LABELS) * train_counts[label]) for label in LABELS
    ], dtype=torch.float32)
    history_path = output_root / "training_history.csv"
    checkpoint_path = output_root / "best_validation.pt"
    config_path = output_root / "model_config.json"
    state_path = output_root / "last_training_state.pt"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_val_logits: np.ndarray | None = None
    stale_epochs = 0
    start_epoch = 1
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but no saved training state exists: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("model_config") != asdict(config):
            raise ValueError("saved training state uses a different model configuration")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        best_macro_f1 = float(state["best_macro_f1"])
        best_validation_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        best_val_logits = state["best_val_logits"].numpy() if state["best_val_logits"] is not None else None
        stale_epochs = int(state["stale_epochs"])
        initial_parameters = state["initial_parameters"]
        torch.set_rng_state(state["torch_rng_state"])
        start_epoch = int(state["last_epoch"]) + 1
        if start_epoch > epochs:
            raise ValueError(f"saved training already reached epoch {start_epoch - 1}; increase --epochs to resume")
    started = time.monotonic()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_rows = train_rows[:]
        random.Random(seed + epoch).shuffle(epoch_rows)
        losses: list[float] = []
        for start in range(0, len(epoch_rows), batch_size):
            batch = epoch_rows[start:start + batch_size]
            graph_a, graph_b, labels = _make_batch(batch, graphs)
            optimizer.zero_grad(set_to_none=True)
            output = model(molecule_a=graph_a, molecule_b=graph_b)
            loss = F.cross_entropy(output["logits"], labels, weight=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        val_logits, val_labels = _predict(model, validation_rows, graphs, batch_size)
        val_loss = float(F.cross_entropy(torch.from_numpy(val_logits).float(),
                                         torch.from_numpy(val_labels),
                                         weight=class_weights).item())
        val_metrics = compute_metrics(val_labels, torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy())
        scheduler.step(val_loss)
        current = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "validation_loss": val_loss, "validation_macro_f1": val_metrics["macro_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(current)
        improved = (val_metrics["macro_f1"] > best_macro_f1 or
                    (val_metrics["macro_f1"] == best_macro_f1 and val_loss < best_validation_loss))
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_validation_loss = val_loss
            best_epoch = epoch
            best_val_logits = val_logits.copy()
            stale_epochs = 0
            save_checkpoint(model, checkpoint_path, config_path)
        else:
            stale_epochs += 1
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss",
                                                        "validation_macro_f1", "learning_rate"])
            writer.writeheader()
            writer.writerows(history)
        _save_training_state(state_path, {
            "model_config": asdict(config), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "history": history, "best_macro_f1": best_macro_f1,
            "best_validation_loss": best_validation_loss, "best_epoch": best_epoch,
            "best_val_logits": (torch.as_tensor(best_val_logits) if best_val_logits is not None else None),
            "stale_epochs": stale_epochs, "last_epoch": epoch,
            "initial_parameters": initial_parameters, "torch_rng_state": torch.get_rng_state(),
        })
        print(f"{protocol_name} epoch {epoch}/{epochs}: train_loss={current['train_loss']:.5f} "
              f"validation_loss={val_loss:.5f} validation_macro_f1={val_metrics['macro_f1']:.5f} "
              f"best_epoch={best_epoch} stale_epochs={stale_epochs}", flush=True)
        if stale_epochs >= patience:
            break

    # Load the selected validation checkpoint before one final test evaluation.
    from .model import load_checkpoint
    model = load_checkpoint(checkpoint_path, config_path).eval()
    updated_elements_by_module: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        changed = int(torch.count_nonzero(parameter.detach() != initial_parameters[name]).item())
        updated_elements_by_module[name.split(".", 1)[0]] += changed
    modules_updated = sorted(name for name, count in updated_elements_by_module.items() if count > 0)
    val_logits, val_labels = _predict(model, validation_rows, graphs, batch_size)
    checkpoint_reload_verified = (best_val_logits is not None and
                                  np.allclose(best_val_logits, val_logits, rtol=1e-6, atol=1e-7))
    if not checkpoint_reload_verified:
        raise AssertionError("reloaded validation checkpoint changed logits")
    test_logits, test_labels = _predict(model, test_rows, graphs, batch_size)
    temperature = fit_temperature_on_validation(torch.from_numpy(val_logits).float(),
                                               torch.from_numpy(val_labels).long())
    raw_test_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    calibrated_test_probabilities = apply_temperature(torch.from_numpy(test_logits).float(), temperature).numpy()
    prediction_path = output_root / "test_predictions.npz"
    prediction_artifact = save_prediction_artifact(
        prediction_path, [row["pair_id"] for row in test_rows], test_labels,
        raw_test_probabilities, calibrated_test_probabilities,
        {"protocol": protocol_name, "seed": seed, "model_mode": "molecular_only_ablation",
         "temperature_fitted_on_validation_only": temperature})
    coverage_manifest_path = output_root / "pair_coverage_manifest.json"
    coverage_manifest = {
        "protocol": protocol_name,
        "run_tag": run_tag,
        "seed": seed,
        "dataset_manifest_sha256": _sha256(pairs_path),
        "split_manifest_sha256": _sha256(split_path),
        "covered_pair_ids": {key: sorted(row["pair_id"] for row in selected[key])
                             for key in ("train", "validation", "test")},
        "missing_structure_ledger": str(exclusions_path),
    }
    coverage_manifest_path.write_text(json.dumps(coverage_manifest, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    evaluation = {
        "status": "completed_molecular_only_ablation",
        "protocol": protocol_name,
        "seed": seed,
        "model_mode": "molecular_only; ChEMBL target and biomedical text modalities unavailable and masked",
        "training_config": {"optimizer": "AdamW", "initial_learning_rate": learning_rate,
                            "weight_decay": 1e-4, "scheduler": "ReduceLROnPlateau",
                            "scheduler_mode": "min_validation_loss", "scheduler_factor": 0.5,
                            "scheduler_patience": 2, "early_stopping_patience": patience,
                            "batch_size": batch_size, "gradient_clip_norm": 2.0,
                            "class_weighting": "inverse training class frequency; train split only"},
        "selection_metric": "validation_macro_f1_then_validation_loss",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "epochs_completed": len(history),
        "train_pair_count": len(train_rows),
        "validation_pair_count": len(validation_rows),
        "test_pair_count": len(test_rows),
        "all_split_pair_counts": {key: len(split[f"{key}_pair_ids"]) for key in ("train", "validation", "test")},
        "covered_split_pair_counts": {key: len(selected[key]) for key in ("train", "validation", "test")},
        "excluded_missing_structure_pair_counts": {
            key: len(split[f"{key}_pair_ids"]) - len(selected[key]) for key in ("train", "validation", "test")},
        "train_class_counts": dict(train_counts),
        "validation_class_counts": dict(Counter(row["severity"] for row in validation_rows)),
        "test_class_counts": dict(Counter(row["severity"] for row in test_rows)),
        "train_pair_ids_sha256": hashlib.sha256("\n".join(sorted(row["pair_id"] for row in train_rows)).encode()).hexdigest(),
        "validation_pair_ids_sha256": hashlib.sha256("\n".join(sorted(row["pair_id"] for row in validation_rows)).encode()).hexdigest(),
        "test_pair_ids_sha256": hashlib.sha256("\n".join(sorted(row["pair_id"] for row in test_rows)).encode()).hexdigest(),
        "dataset_manifest_sha256": _sha256(pairs_path),
        "split_manifest_sha256": _sha256(split_path),
        "pubchem_compounds_sha256": _sha256(compounds_path),
        "feature_schema_sha256": _sha256_if_file(project_root / "artifacts/feature_schema.json"),
        "dependency_versions": _dependency_versions(),
        "unique_approved_molecules": len(graphs),
        "class_weights_train_only": {label: float(class_weights[index]) for index, label in enumerate(LABELS)},
        "training_history": history,
        "validation_metrics_raw": compute_metrics(val_labels, torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()),
        "temperature_fitted_on_validation_only": temperature,
        "test_metrics_raw": compute_metrics(test_labels, raw_test_probabilities),
        "test_metrics_validation_temperature": compute_metrics(test_labels, calibrated_test_probabilities),
        "coverage_matched_majority_baseline": _majority_metrics(train_rows, test_rows),
        "parameter_counts": parameter_counts(model),
        "updated_parameter_elements_by_module": dict(updated_elements_by_module),
        "modules_updated": modules_updated,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": _sha256(config_path),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "test_prediction_artifact": prediction_artifact,
        "pair_coverage_manifest": str(coverage_manifest_path),
        "missing_structure_ledger": str(exclusions_path),
        "training_history_csv": str(history_path),
        "resumable_training_state": str(state_path),
        "resumable_training_state_sha256": _sha256(state_path),
        "resumed_from_epoch": start_epoch - 1,
        "training_duration_seconds": round(time.monotonic() - started, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limits": ["This is a molecular-only ablation, not the multimodal production model.",
                   "Pairs without two identity-approved molecular structures are listed in the exclusion ledger.",
                   "The checkpoint has not been integrated into the prediction API."],
    }
    evaluation_path = output_root / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", choices=("pair_disjoint", "cold_drug"), default="pair_disjoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="continue from last_training_state.pt")
    parser.add_argument("--run-tag", help="write a versioned artifact folder instead of replacing the default run")
    args = parser.parse_args()
    print(json.dumps(train_protocol(args.project_root, args.protocol, seed=args.seed,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    patience=args.patience, hidden_dim=args.hidden_dim,
                                    learning_rate=args.learning_rate, resume=args.resume,
                                    run_tag=args.run_tag), indent=2))


if __name__ == "__main__":
    main()
