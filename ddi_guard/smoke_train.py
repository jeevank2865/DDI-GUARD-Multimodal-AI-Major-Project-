"""Run a clearly labeled molecular-only smoke training pass on real covered pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np

from . import LABEL_TO_ID, LABELS
from .evaluation import compute_metrics
from .features import batch_graphs, molecule_to_graph
from .model import DDIModelConfig, MultimodalDDIGuard, load_checkpoint, parameter_counts, save_checkpoint


def _safe_pubchem_records(checkpoint_path: Path, mapping_csv: Path) -> dict[str, dict[str, Any]]:
    ambiguous = set()
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("name_ambiguity", "")).casefold() in ("true", "1", "yes"):
                ambiguous.add("".join(ch for ch in row["original_name"].casefold() if ch.isalnum()))
    records = {}
    with checkpoint_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate only a partial final line while the resumable job appends
            query_key = "".join(ch for ch in item.get("query_name", "").casefold() if ch.isalnum())
            if item.get("canonical_smiles_rdkit_validated") and query_key not in ambiguous:
                records[item["ddinter_id"]] = item
    return records


def _sample(split_ids: list[str], pairs_by_id: dict[str, dict[str, Any]], available_drugs: set[str],
            per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for pair_id in split_ids:
        row = pairs_by_id.get(pair_id)
        if (row is not None and row["severity"] in grouped and
                row["drug_a_id"] in available_drugs and row["drug_b_id"] in available_drugs):
            grouped[row["severity"]].append(row)
    picked = []
    for label in LABELS:
        candidates = sorted(grouped[label], key=lambda row: row["pair_id"])
        if len(candidates) < per_class:
            raise ValueError(f"not enough {label} pairs with two validated, unambiguous molecules: {len(candidates)} < {per_class}")
        picked.extend(rng.sample(candidates, per_class))
    rng.shuffle(picked)
    return picked


def run_smoke(project_root: Path, train_per_class: int = 24, validation_per_class: int = 6,
              epochs: int = 2, seed: int = 2026, batch_size: int = 12) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    pairs_path = project_root / "data/processed/ddi_pairs_full.json"
    split_path = project_root / "artifacts/split_manifest.json"
    checkpoint_path = project_root / "data/processed/pubchem_enrichment_checkpoint.jsonl"
    mapping_path = project_root / "data/processed/drug_identity_mapping.csv"
    if not all(path.is_file() for path in (pairs_path, split_path, checkpoint_path, mapping_path)):
        raise FileNotFoundError("DDInter pairs, split manifest, PubChem checkpoint, and drug mapping are required")
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))["pair_disjoint"]
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    records = _safe_pubchem_records(checkpoint_path, mapping_path)
    available_drugs = set(records)
    train_rows = _sample(splits["train_pair_ids"], pairs_by_id, available_drugs, train_per_class, seed)
    validation_rows = _sample(splits["validation_pair_ids"], pairs_by_id, available_drugs, validation_per_class, seed + 1)
    selected = train_rows + validation_rows
    selected_drugs = {row[k] for row in selected for k in ("drug_a_id", "drug_b_id")}
    missing = sorted(selected_drugs - records.keys())
    if missing:
        raise ValueError(f"selected pairs lost PubChem structures for {len(missing)} drugs")
    smiles_by_id = {drug_id: records[drug_id]["isomeric_smiles"] for drug_id in selected_drugs}
    graphs = {drug_id: molecule_to_graph(smiles) for drug_id, smiles in smiles_by_id.items()}
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        all_drug_count = len({row["ddinter_id"] for row in csv.DictReader(handle)})
    config = DDIModelConfig(num_drugs=max(1, all_drug_count), num_targets=1, num_relations=1,
                            text_dim=768, hidden_dim=32, attention_heads=4, molecular_layers=3)
    model = MultimodalDDIGuard(config)
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_losses = []
    train_ids = [row["pair_id"] for row in train_rows]
    validation_ids = [row["pair_id"] for row in validation_rows]
    if set(train_ids) & set(validation_ids):
        raise AssertionError("pair-disjoint train and validation IDs overlap")

    def train_batch(batch: list[dict[str, Any]]) -> float:
        model.train()
        graph_a = batch_graphs([graphs[row["drug_a_id"]] for row in batch])
        graph_b = batch_graphs([graphs[row["drug_b_id"]] for row in batch])
        labels = torch.tensor([LABEL_TO_ID[row["severity"]] for row in batch], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        output = model(molecule_a=graph_a, molecule_b=graph_b)
        loss = torch.nn.functional.cross_entropy(output["logits"], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite smoke training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("non-finite smoke training gradient")
        optimizer.step()
        return float(loss.detach())

    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train_rows)
        epoch_losses = []
        for offset in range(0, len(train_rows), batch_size):
            epoch_losses.append(train_batch(train_rows[offset:offset + batch_size]))
        train_losses.append(sum(epoch_losses) / len(epoch_losses))

    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for offset in range(0, len(validation_rows), batch_size):
            batch = validation_rows[offset:offset + batch_size]
            graph_a = batch_graphs([graphs[row["drug_a_id"]] for row in batch])
            graph_b = batch_graphs([graphs[row["drug_b_id"]] for row in batch])
            output = model(molecule_a=graph_a, molecule_b=graph_b)
            probabilities.extend(output["probabilities"].cpu().numpy().tolist())
            labels.extend(LABEL_TO_ID[row["severity"]] for row in batch)
    validation_metrics = compute_metrics(labels, np.asarray(probabilities))
    updated_elements_by_module: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        changed = int(torch.count_nonzero(parameter.detach() != initial_parameters[name]).item())
        updated_elements_by_module[name.split(".", 1)[0]] += changed
    data_digest = hashlib.sha256()
    for drug_id in sorted(smiles_by_id):
        data_digest.update(f"{drug_id}\0{smiles_by_id[drug_id]}\n".encode())
    source_digest = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    config_path = project_root / "artifacts/ddi_guard_smoke_model_config.json"
    weights_path = project_root / "artifacts/ddi_guard_smoke_molecular.pt"
    save_checkpoint(model, weights_path, config_path)
    reloaded = load_checkpoint(weights_path, config_path).eval()
    reloaded_probabilities = []
    with torch.no_grad():
        for offset in range(0, len(validation_rows), batch_size):
            batch = validation_rows[offset:offset + batch_size]
            graph_a = batch_graphs([graphs[row["drug_a_id"]] for row in batch])
            graph_b = batch_graphs([graphs[row["drug_b_id"]] for row in batch])
            reloaded_probabilities.extend(reloaded(molecule_a=graph_a, molecule_b=graph_b)["probabilities"].cpu().numpy().tolist())
    checkpoint_reload_verified = bool(np.allclose(np.asarray(probabilities), np.asarray(reloaded_probabilities), atol=1e-7))
    if not checkpoint_reload_verified:
        raise AssertionError("smoke checkpoint reload changed validation probabilities")
    report = {
        "status": "smoke_training_completed_only",
        "production_training_completed": False,
        "model_mode": "molecular_only_ablation; target and text modalities masked/unavailable",
        "seed": seed, "epochs_completed": epochs, "batch_size": batch_size,
        "train_pair_count": len(train_rows), "validation_pair_count": len(validation_rows),
        "test_pair_count": 0, "train_class_counts": dict(Counter(row["severity"] for row in train_rows)),
        "validation_class_counts": dict(Counter(row["severity"] for row in validation_rows)),
        "train_pair_ids": sorted(train_ids), "validation_pair_ids": sorted(validation_ids),
        "training_loss_by_epoch": train_losses, "validation_metrics_smoke_only": validation_metrics,
        "unique_molecules_used": len(smiles_by_id), "molecule_feature_source": "PubChem PUG REST; RDKit validated",
        "pair_labels_source": "local DDInter 2.0 full labeled manifest",
        "pairs_manifest_sha256": source_digest, "selected_molecular_data_sha256": data_digest.hexdigest(),
        "parameter_counts": parameter_counts(model),
        "updated_parameter_elements_by_module": dict(updated_elements_by_module),
        "modules_updated": sorted(name for name, count in updated_elements_by_module.items() if count > 0),
        "checkpoint": str(weights_path), "checkpoint_config": str(config_path),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Small smoke sample only; not a completed research training run.",
                        "No ChEMBL target graph or text encoder/features were supplied to this experiment.",
                        "Validation metrics are diagnostic only and are not independent final performance."],
    }
    (project_root / "artifacts/smoke_training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--validation-per-class", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.project_root, args.train_per_class, args.validation_per_class,
                               args.epochs, args.seed, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
