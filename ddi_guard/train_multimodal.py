"""Train source-backed modality ablations or the complete DDI-GUARD model.

The full command requires verified molecular, ChEMBL 34 target-graph, and
provenance/hash-checked text-vector inputs. Pairs with some but not all requested
modalities are retained with explicit masks; pairs with none are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F

from . import LABELS, LABEL_TO_ID
from .evaluation import apply_temperature, compute_metrics, fit_temperature_on_validation, save_prediction_artifact
from .features import batch_graphs, molecule_to_graph
from .model import DDIModelConfig, MultimodalDDIGuard, load_checkpoint, parameter_counts, save_checkpoint
from .train_molecular import _load_approved_compounds, _sha256, _sha256_if_file

MODALITY_ORDER = ("molecular", "target", "text")


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0], "torch": torch.__version__,
                                    "numpy": np.__version__}
    for distribution in ("rdkit", "scikit-learn", "torch-geometric"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def _load_target_source(project_root: Path) -> tuple[dict[str, torch.Tensor], dict[str, int], set[str], dict[str, Any]]:
    audit_path = project_root / "artifacts/chembl_coverage_audit.json"
    edges_path = project_root / "data/processed/chembl_drug_target_edges.csv"
    graph_root = project_root / "data/processed/chembl_34_target_graph"
    graph_path = graph_root / "target_graph.pt"
    manifest_path = graph_root / "target_graph_manifest.json"
    if not all(path.is_file() for path in (audit_path, edges_path, graph_path, manifest_path)):
        raise ValueError("target modality is unavailable: verified ChEMBL 34 edges and target graph are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("observed_release") != "34":
        raise ValueError("target modality is unavailable: ChEMBL audit must verify release 34")
    if manifest.get("status") != "verified_chEMBL_34_target_graph" or manifest.get("source_release") != "34":
        raise ValueError("target graph manifest is not verified as ChEMBL 34")
    if manifest.get("source_edge_manifest_sha256") != _sha256(edges_path):
        raise ValueError("target graph source-edge hash does not match the current ChEMBL edge file")
    if manifest.get("graph_tensor_sha256") != _sha256(graph_path):
        raise ValueError("target graph tensor hash does not match its manifest")
    mapping_path = project_root / "data/processed/drug_identity_mapping.csv"
    if not mapping_path.is_file() or manifest.get("mapping_csv_sha256") != _sha256(mapping_path):
        raise ValueError("target graph drug mapping hash does not match the reviewed identity mapping")
    if manifest.get("chembl_import_audit_sha256") != _sha256(audit_path):
        raise ValueError("target graph ChEMBL audit hash does not match the current importer audit")
    drug_ids = manifest.get("drug_index_to_ddinter_id")
    if not isinstance(drug_ids, list) or len(drug_ids) != len(set(drug_ids)) or not drug_ids:
        raise ValueError("target graph has an invalid drug-index mapping")
    drug_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    with edges_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ddinter_id", "target_chembl_id", "organism", "activity_type", "activity_units",
                    "relation_operator", "supporting_activity_ids", "assay_chembl_ids", "source_release"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("target edge manifest lacks source provenance fields")
        target_drugs: set[str] = set()
        for row in reader:
            if row["source_release"] != "34":
                raise ValueError("target edge row has an unexpected ChEMBL release")
            if not all(row.get(key, "").strip() for key in
                       ("ddinter_id", "target_chembl_id", "organism", "activity_type", "activity_units",
                        "relation_operator", "supporting_activity_ids", "assay_chembl_ids")):
                raise ValueError("target edge row is missing source provenance")
            target_drugs.add(row["ddinter_id"])
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("safe weights_only loading is required for the target graph") from exc
    if not isinstance(graph, dict) or not all(key in graph for key in ("edge_index", "relation", "edge_weight")):
        raise ValueError("target graph tensor file has an unsupported structure")
    tensors = {key: graph[key] for key in ("edge_index", "relation", "edge_weight")}
    if tensors["edge_index"].dtype != torch.long or tensors["relation"].dtype != torch.long:
        raise ValueError("target graph indices must be integer tensors")
    if tensors["edge_weight"].ndim != 1 or not torch.isfinite(tensors["edge_weight"]).all():
        raise ValueError("target graph weights are invalid")
    if tensors["edge_index"].shape != (2, tensors["relation"].numel()) or tensors["relation"].numel() != tensors["edge_weight"].numel():
        raise ValueError("target graph tensors have inconsistent edge counts")
    target_count = int(manifest["target_count"])
    relation_count = int(manifest["relation_count"])
    if target_count < 1 or relation_count < 1:
        raise ValueError("target graph has an empty target or relation vocabulary")
    if tensors["edge_index"].numel() and int(tensors["edge_index"].max()) >= len(drug_ids) + target_count:
        raise ValueError("target graph has an entity index outside its manifest")
    if tensors["relation"].numel() and (int(tensors["relation"].min()) < 0 or int(tensors["relation"].max()) >= relation_count):
        raise ValueError("target graph has a relation index outside its manifest")
    return tensors, drug_index, target_drugs, {
        "graph_path": str(graph_path), "graph_manifest_path": str(manifest_path),
        "graph_sha256": _sha256(graph_path), "manifest_sha256": _sha256(manifest_path),
        "audit_sha256": _sha256(audit_path), "num_drugs": len(drug_ids),
        "num_targets": target_count, "num_relations": relation_count,
    }


def _load_text_source(project_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = project_root / "data/processed/text_evidence_embeddings_manifest.csv"
    if not manifest_path.is_file():
        raise ValueError("text modality is unavailable: source-backed embedding manifest is missing")
    required = {"ddinter_id", "source_id", "evidence_id", "source_url", "label_independent",
                "evidence_scope", "embedding_path", "embedding_sha256"}
    vectors: dict[str, np.ndarray] = {}
    rows_used: list[dict[str, str]] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"text manifest is missing columns: {sorted(required - set(reader.fieldnames or ())) }")
        headers = set(reader.fieldnames or ())
        if not ({"source_version", "source_date"} & headers):
            raise ValueError("text manifest must include source_version or source_date")
        for row in reader:
            url = urlparse(row["source_url"])
            if not (row["ddinter_id"].strip() and row["source_id"].strip() and row["evidence_id"].strip()
                    and url.scheme in {"https", "http"} and url.netloc
                    and (row.get("source_version", "").strip() or row.get("source_date", "").strip())
                    and row["evidence_scope"].strip().casefold() == "drug_general"
                    and row["label_independent"].strip().casefold() in {"true", "1", "yes"}):
                continue
            path = (project_root / row["embedding_path"]).resolve()
            if not path.is_relative_to(project_root.resolve()) or not path.is_file():
                continue
            if _sha256(path).casefold() != row["embedding_sha256"].strip().casefold():
                continue
            if row["ddinter_id"] in vectors:
                raise ValueError(f"text manifest has multiple aggregate vectors for {row['ddinter_id']}; curate one vector per drug")
            try:
                vector = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot load verified text vector {path}: {exc}") from exc
            if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
                raise ValueError(f"text vector must be a finite one-dimensional array: {path}")
            vectors[row["ddinter_id"]] = vector.astype(np.float32, copy=False)
            rows_used.append({key: row[key] for key in sorted(required | ({"source_version"} if "source_version" in headers else {"source_date"}))})
    if not vectors:
        raise ValueError("text modality is unavailable: manifest has no hash-verified, label-independent vectors")
    dimensions = {vector.shape[0] for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"text vectors have inconsistent dimensions: {sorted(dimensions)}")
    return vectors, {"manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
                     "verified_drug_count": len(vectors), "vector_dimension": dimensions.pop(),
                     "verified_records": rows_used}


def _load_sources(project_root: Path, modalities: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"molecular_graphs": {}, "target_graph": None,
                              "target_drug_index": {}, "target_drugs": set(),
                              "text_vectors": {}, "sources": {}}
    if "molecular" in modalities:
        records = _load_approved_compounds(project_root / "data/processed/pubchem_compounds.json",
                                           project_root / "artifacts/pubchem_coverage_audit.json")
        smiles = {drug_id: item.get("isomeric_smiles") or item.get("canonical_smiles")
                  for drug_id, item in records.items()}
        result["molecular_graphs"] = {drug_id: molecule_to_graph(value) for drug_id, value in smiles.items() if value}
        if not result["molecular_graphs"]:
            raise ValueError("molecular modality is unavailable: no reviewed RDKit-valid structures")
        result["sources"]["molecular"] = {
            "graph_manifest_sha256": _sha256_if_file(project_root / "data/processed/molecular_graph_manifest.json"),
            "pubchem_compounds_sha256": _sha256(project_root / "data/processed/pubchem_compounds.json"),
            "drug_count": len(result["molecular_graphs"]),
        }
    if "target" in modalities:
        graph, index, drugs, meta = _load_target_source(project_root)
        result.update({"target_graph": graph, "target_drug_index": index, "target_drugs": drugs})
        result["sources"]["target"] = meta
    if "text" in modalities:
        vectors, meta = _load_text_source(project_root)
        result["text_vectors"] = vectors
        result["sources"]["text"] = meta
    return result


def _pair_mask(row: dict[str, Any], sources: dict[str, Any], modalities: tuple[str, ...]) -> list[bool]:
    available = {
        "molecular": set(sources["molecular_graphs"]),
        "target": sources["target_drugs"],
        "text": set(sources["text_vectors"]),
    }
    return [name in modalities and row["drug_a_id"] in available[name] and row["drug_b_id"] in available[name]
            for name in MODALITY_ORDER]


def _select_rows(split: dict[str, Any], pairs_by_id: dict[str, dict[str, Any]], sources: dict[str, Any],
                 modalities: tuple[str, ...], protocol: str, ledger: csv.DictWriter,
                 support_modalities: tuple[str, ...] = ()) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for name in ("train", "validation", "test"):
        rows = []
        for pair_id in split[f"{name}_pair_ids"]:
            row = pairs_by_id[pair_id]
            if support_modalities:
                support_mask = _pair_mask(row, sources, support_modalities)
                if not all(support_mask[MODALITY_ORDER.index(modality)] for modality in support_modalities):
                    ledger.writerow({"protocol": protocol, "split": name, "pair_id": pair_id,
                                     "reason": "outside_common_support_modalities",
                                     "available_modalities": ""})
                    continue
            mask = _pair_mask(row, sources, modalities)
            if not any(mask):
                ledger.writerow({"protocol": protocol, "split": name, "pair_id": pair_id,
                                 "reason": "no_verified_selected_modality",
                                 "available_modalities": ""})
                continue
            row = dict(row)
            row["modality_mask"] = mask
            rows.append(row)
        selected[name] = rows
    return selected


def _validate_fixed_split(protocol: str, pairs_by_id: dict[str, dict[str, Any]],
                          split: dict[str, Any]) -> None:
    assigned: set[str] = set()
    split_ids: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        ids = split.get(f"{name}_pair_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError(f"{protocol} {name} pair IDs are missing or duplicated")
        current = set(ids)
        if current & assigned:
            raise ValueError(f"{protocol} pair IDs overlap across train/validation/test")
        unknown = current - pairs_by_id.keys()
        if unknown:
            raise ValueError(f"{protocol} {name} includes unknown pair IDs")
        split_ids[name] = current
        assigned.update(current)
    if protocol != "cold_drug":
        return
    drug_sets = {name: set(split.get(f"{name}_drug_ids", [])) for name in split_ids}
    if ((drug_sets["train"] & drug_sets["validation"])
            or (drug_sets["train"] & drug_sets["test"])
            or (drug_sets["validation"] & drug_sets["test"])):
        raise ValueError("cold_drug split has drug-ID overlap across partitions")
    for name, pair_ids in split_ids.items():
        if any(not {pairs_by_id[pair_id]["drug_a_id"], pairs_by_id[pair_id]["drug_b_id"]} <= drug_sets[name]
               for pair_id in pair_ids):
            raise ValueError(f"cold_drug {name} pair references a drug outside its partition")


def _make_inputs(rows: list[dict[str, Any]], sources: dict[str, Any], modalities: tuple[str, ...],
                 target_drug_index: dict[str, int], text_dim: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"batch_size": len(rows),
                              "modality_mask": torch.tensor([row["modality_mask"] for row in rows], dtype=torch.bool)}
    if "molecular" in modalities:
        positions = [index for index, row in enumerate(rows) if row["modality_mask"][0]]
        if positions:
            kwargs["molecule_a"] = batch_graphs([sources["molecular_graphs"][rows[i]["drug_a_id"]] for i in positions])
            kwargs["molecule_b"] = batch_graphs([sources["molecular_graphs"][rows[i]["drug_b_id"]] for i in positions])
            kwargs["molecule_pair_indices"] = torch.tensor(positions, dtype=torch.long)
    if "target" in modalities and any(row["modality_mask"][1] for row in rows):
        kwargs["target_graph"] = sources["target_graph"]
        kwargs["drug_a_index"] = torch.tensor([target_drug_index[row["drug_a_id"]] for row in rows], dtype=torch.long)
        kwargs["drug_b_index"] = torch.tensor([target_drug_index[row["drug_b_id"]] for row in rows], dtype=torch.long)
    if "text" in modalities and any(row["modality_mask"][2] for row in rows):
        vectors = torch.zeros((len(rows), 2, text_dim), dtype=torch.float32)
        for i, row in enumerate(rows):
            if row["modality_mask"][2]:
                vectors[i, 0] = torch.from_numpy(sources["text_vectors"][row["drug_a_id"]])
                vectors[i, 1] = torch.from_numpy(sources["text_vectors"][row["drug_b_id"]])
        kwargs["text_features"] = vectors
    return kwargs


def _predict(model: MultimodalDDIGuard, rows: list[dict[str, Any]], sources: dict[str, Any],
             modalities: tuple[str, ...], target_drug_index: dict[str, int], text_dim: int,
             batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output = model(**_make_inputs(batch, sources, modalities, target_drug_index, text_dim))
            logits.extend(output["logits"].cpu().numpy())
            labels.extend(LABEL_TO_ID[row["severity"]] for row in batch)
    return np.asarray(logits), np.asarray(labels, dtype=np.int64)


def train_protocol(project_root: Path, protocol: str, *, modalities: tuple[str, ...] = MODALITY_ORDER,
                   support_modalities: tuple[str, ...] = (),
                   seed: int = 42, epochs: int = 20, batch_size: int = 128, patience: int = 4,
                   hidden_dim: int = 96, learning_rate: float = 1e-3, resume: bool = False,
                   run_tag: str | None = None) -> dict[str, Any]:
    requested_modalities = set(modalities)
    requested_support = set(support_modalities)
    if requested_modalities - set(MODALITY_ORDER):
        raise ValueError(f"unknown modalities: {sorted(requested_modalities - set(MODALITY_ORDER))}")
    if requested_support - set(MODALITY_ORDER):
        raise ValueError(f"unknown support modalities: {sorted(requested_support - set(MODALITY_ORDER))}")
    modalities = tuple(name for name in MODALITY_ORDER if name in requested_modalities)
    support_modalities = tuple(name for name in MODALITY_ORDER if name in requested_support)
    if protocol not in ("pair_disjoint", "cold_drug"):
        raise ValueError("protocol must be pair_disjoint or cold_drug")
    if not modalities or min(epochs, batch_size, patience, hidden_dim) < 1 or learning_rate <= 0:
        raise ValueError("select at least one modality and positive training settings")
    if run_tag is not None and (not run_tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_tag)):
        raise ValueError("run_tag may contain only letters, digits, hyphen, or underscore")
    project_root = project_root.resolve()
    pairs_path = project_root / "data/processed/ddi_pairs_full.json"
    split_path = project_root / "artifacts/split_manifest.json"
    for path in (pairs_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_modalities = tuple(name for name in MODALITY_ORDER if name in set(modalities) | set(support_modalities))
    sources = _load_sources(project_root, source_modalities)
    sources_fingerprint = hashlib.sha256(json.dumps(sources["sources"], sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest()
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split = split_manifest[protocol]
    _validate_fixed_split(protocol, pairs_by_id, split)
    output_name = f"multimodal_{protocol}_" + "_".join(modalities) + (f"_{run_tag}" if run_tag else "")
    if support_modalities:
        output_name += "_support_" + "_".join(support_modalities)
    output_root = project_root / "artifacts" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "uncovered_pairs.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        ledger = csv.DictWriter(handle, fieldnames=("protocol", "split", "pair_id", "reason", "available_modalities"))
        ledger.writeheader()
        selected = _select_rows(split, pairs_by_id, sources, modalities, protocol, ledger,
                                support_modalities=support_modalities)
    coverage_manifest_path = output_root / "pair_coverage_manifest.json"
    coverage_manifest = {
        "protocol": protocol, "modalities": list(modalities), "support_modalities": list(support_modalities), "seed": seed,
        "dataset_manifest_sha256": _sha256(pairs_path), "split_manifest_sha256": _sha256(split_path),
        "source_fingerprint_sha256": sources_fingerprint,
        "selected_pair_ids": {name: [row["pair_id"] for row in selected[name]] for name in selected},
        "per_pair_modality_mask": {
            name: [{"pair_id": row["pair_id"],
                    "availability": {modality: bool(row["modality_mask"][index])
                                     for index, modality in enumerate(MODALITY_ORDER)}}
                   for row in selected[name]] for name in selected},
        "excluded_no_selected_modality_ledger": str(ledger_path.relative_to(project_root)),
    }
    coverage_manifest_path.write_text(json.dumps(coverage_manifest, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    train_rows, validation_rows, test_rows = (selected[key] for key in ("train", "validation", "test"))
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError(f"no complete experiment coverage for {protocol}/{modalities}: "
                         f"train={len(train_rows)}, validation={len(validation_rows)}, test={len(test_rows)}; "
                         "see modality coverage audit and source manifests")
    train_counts = Counter(row["severity"] for row in train_rows)
    if any(train_counts[label] == 0 for label in LABELS):
        raise ValueError(f"covered training pairs lack a class: {dict(train_counts)}")
    target_meta = sources["sources"].get("target", {})
    text_meta = sources["sources"].get("text", {})
    text_dim = int(text_meta.get("vector_dimension", 0)) if "text" in modalities else 0
    config = DDIModelConfig(
        num_drugs=int(target_meta.get("num_drugs", 0)),
        num_targets=int(target_meta.get("num_targets", 0)),
        num_relations=int(target_meta.get("num_relations", 0)), text_dim=text_dim,
        hidden_dim=hidden_dim, attention_heads=4, molecular_layers=3,
        use_molecular="molecular" in modalities, use_target="target" in modalities,
        use_text="text" in modalities)
    model = MultimodalDDIGuard(config)
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    class_weights = torch.tensor([len(train_rows) / (len(LABELS) * train_counts[label]) for label in LABELS],
                                 dtype=torch.float32)
    config_path = output_root / "model_config.json"
    checkpoint_path = output_root / "best_validation.pt"
    state_path = output_root / "last_training_state.pt"
    history_path = output_root / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_val_logits: np.ndarray | None = None
    stale = 0
    start_epoch = 1
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but training state is missing: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if (state.get("model_config") != asdict(config) or state.get("modalities") != list(modalities)
                or state.get("sources_fingerprint_sha256") != sources_fingerprint):
            raise ValueError("saved training state has different modalities, configuration, or source inputs")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        best_macro_f1, best_loss = float(state["best_macro_f1"]), float(state["best_loss"])
        best_epoch, stale = int(state["best_epoch"]), int(state["stale_epochs"])
        best_val_logits = state["best_val_logits"].numpy() if state["best_val_logits"] is not None else None
        initial_parameters = state["initial_parameters"]
        torch.set_rng_state(state["torch_rng_state"])
        start_epoch = int(state["last_epoch"]) + 1
        if start_epoch > epochs:
            raise ValueError(f"training state already reached epoch {start_epoch - 1}; increase --epochs")
    started = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_rows = train_rows[:]
        random.Random(seed + epoch).shuffle(epoch_rows)
        batch_losses: list[float] = []
        for begin in range(0, len(epoch_rows), batch_size):
            batch = epoch_rows[begin:begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(**_make_inputs(batch, sources, modalities, sources["target_drug_index"], text_dim))
            loss = F.cross_entropy(output["logits"], torch.tensor([LABEL_TO_ID[row["severity"]] for row in batch]),
                                   weight=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        val_logits, val_labels = _predict(model, validation_rows, sources, modalities,
                                          sources["target_drug_index"], text_dim, batch_size)
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()
        val_metrics = compute_metrics(val_labels, val_probs)
        val_loss = float(F.cross_entropy(torch.from_numpy(val_logits).float(), torch.from_numpy(val_labels),
                                         weight=class_weights).item())
        scheduler.step(val_loss)
        improved = val_metrics["macro_f1"] > best_macro_f1 or (val_metrics["macro_f1"] == best_macro_f1 and val_loss < best_loss)
        if improved:
            best_macro_f1, best_loss, best_epoch, stale = val_metrics["macro_f1"], val_loss, epoch, 0
            best_val_logits = val_logits.copy()
            save_checkpoint(model, checkpoint_path, config_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)),
                        "validation_loss": val_loss, "validation_macro_f1": val_metrics["macro_f1"],
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        state = {"model_config": asdict(config), "modalities": list(modalities),
                 "sources_fingerprint_sha256": sources_fingerprint,
                 "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "history": history,
                 "best_macro_f1": best_macro_f1, "best_loss": best_loss, "best_epoch": best_epoch,
                 "best_val_logits": None if best_val_logits is None else torch.from_numpy(best_val_logits),
                 "stale_epochs": stale, "last_epoch": epoch, "initial_parameters": initial_parameters,
                 "torch_rng_state": torch.get_rng_state()}
        temporary = state_path.with_name(state_path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(state_path)
        print(f"{protocol} {modalities} epoch {epoch}/{epochs}: train_loss={history[-1]['train_loss']:.5f} "
              f"validation_loss={val_loss:.5f} validation_macro_f1={val_metrics['macro_f1']:.5f} "
              f"best_epoch={best_epoch} stale_epochs={stale}", flush=True)
        if stale >= patience:
            break

    model = load_checkpoint(checkpoint_path, config_path).eval()
    val_logits, val_labels = _predict(model, validation_rows, sources, modalities,
                                      sources["target_drug_index"], text_dim, batch_size)
    reload_verified = best_val_logits is not None and np.allclose(best_val_logits, val_logits, rtol=1e-6, atol=1e-7)
    if not reload_verified:
        raise AssertionError("reloaded best validation checkpoint changed validation logits")
    test_logits, test_labels = _predict(model, test_rows, sources, modalities,
                                        sources["target_drug_index"], text_dim, batch_size)
    temperature = fit_temperature_on_validation(torch.from_numpy(val_logits).float(), torch.from_numpy(val_labels).long())
    raw_probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    calibrated_probs = apply_temperature(torch.from_numpy(test_logits).float(), temperature).numpy()
    prediction = save_prediction_artifact(output_root / "test_predictions.npz",
                                          [row["pair_id"] for row in test_rows], test_labels,
                                          raw_probs, calibrated_probs,
                                          {"protocol": protocol, "modalities": list(modalities),
                                           "support_modalities": list(support_modalities), "seed": seed,
                                           "temperature_fitted_on_validation_only": temperature})
    counts = {split_name: {modality: sum(row["modality_mask"][index] for row in selected[split_name])
                           for index, modality in enumerate(MODALITY_ORDER)}
              for split_name in ("train", "validation", "test")}
    changed: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        changed[name.split(".", 1)[0]] += int(torch.count_nonzero(parameter.detach() != initial_parameters[name]).item())
    result = {
        "status": "completed_multimodal_or_source_backed_ablation", "protocol": protocol,
        "modalities": list(modalities), "support_modalities": list(support_modalities),
        "seed": seed, "selection_metric": "validation_macro_f1_then_validation_loss",
        "training_config": {"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": 1e-4,
                            "scheduler": "ReduceLROnPlateau", "class_weighting": "inverse training class frequency",
                            "gradient_clip_norm": 2.0, "batch_size": batch_size, "patience": patience},
        "best_epoch": best_epoch, "best_validation_macro_f1": best_macro_f1,
        "epochs_completed": len(history), "selected_pair_counts": {key: len(selected[key]) for key in selected},
        "modality_pair_coverage": counts,
        "excluded_no_selected_modality": {key: len(split[f"{key}_pair_ids"]) - len(selected[key]) for key in selected},
        "class_counts": {key: dict(Counter(row["severity"] for row in selected[key])) for key in selected},
        "pair_ids_sha256": {key: hashlib.sha256("\n".join(sorted(row["pair_id"] for row in selected[key])).encode()).hexdigest()
                             for key in selected},
        "dataset_manifest_sha256": _sha256(pairs_path), "split_manifest_sha256": _sha256(split_path),
        "sources": sources["sources"], "source_fingerprint_sha256": sources_fingerprint,
        "feature_schema_sha256": _sha256_if_file(project_root / "artifacts/feature_schema.json"),
        "dependency_versions": _dependency_versions(),
        "parameter_counts": parameter_counts(model), "updated_parameter_elements_by_module": dict(changed),
        "modules_updated": sorted(name for name, count in changed.items() if count),
        "validation_metrics_raw": compute_metrics(val_labels, torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()),
        "temperature_fitted_on_validation_only": temperature,
        "test_metrics_raw": compute_metrics(test_labels, raw_probs),
        "test_metrics_validation_temperature": compute_metrics(test_labels, calibrated_probs),
        "checkpoint": str(checkpoint_path.relative_to(project_root)), "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_config": str(config_path.relative_to(project_root)), "checkpoint_reload_verified": reload_verified,
        "test_prediction_artifact": prediction, "uncovered_pair_ledger": str(ledger_path.relative_to(project_root)),
        "pair_coverage_manifest": str(coverage_manifest_path.relative_to(project_root)),
        "training_history_csv": str(history_path.relative_to(project_root)),
        "resumable_training_state": str(state_path.relative_to(project_root)),
        "training_duration_seconds": round(time.monotonic() - started, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Only source-backed modalities in this run are enabled.",
                        "DDInter labels are dataset annotations, not patient outcomes or clinical validation."],
    }
    (output_root / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", choices=("pair_disjoint", "cold_drug"), default="pair_disjoint")
    parser.add_argument("--modalities", default="molecular,target,text",
                        help="comma-separated subset of molecular,target,text; defaults to all three")
    parser.add_argument("--support-modalities", default="",
                        help="optional comma-separated modalities required for matched support; IDs are filtered within the fixed split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag")
    args = parser.parse_args()
    modality_values = tuple(value.strip() for value in args.modalities.split(",") if value.strip())
    unknown = set(modality_values) - set(MODALITY_ORDER)
    if unknown:
        parser.error(f"unknown modalities: {sorted(unknown)}")
    support_values = tuple(value.strip() for value in args.support_modalities.split(",") if value.strip())
    unknown_support = set(support_values) - set(MODALITY_ORDER)
    if unknown_support:
        parser.error(f"unknown support modalities: {sorted(unknown_support)}")
    result = train_protocol(args.project_root, args.protocol, modalities=modality_values,
                            support_modalities=support_values, seed=args.seed,
                            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                            hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
                            resume=args.resume, run_tag=args.run_tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
