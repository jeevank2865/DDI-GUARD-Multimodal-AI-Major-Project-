"""Verified, research-only inference using the project's trained checkpoints.

Candidate adapters reuse the training-time feature loaders and batch builders.
No model in this module is exposed through the production /api/predict route.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import LABELS
from .api import _drug_catalog, _resolve_drug_identifier, _target_evidence_by_drug
from .evaluation import apply_temperature
from .features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, batch_graphs
from .model import load_checkpoint
from .train_multimodal import _load_sources, _make_inputs, _pair_mask, MODALITY_ORDER

ROOT = Path(__file__).resolve().parents[1]
TEXT_MANIFEST = ROOT / "data/processed/text_evidence_embeddings_manifest_pair_disjoint_trainfit_v1.csv"
PAIR_DATA = ROOT / "data/processed/ddi_pairs_full.json"
SPLIT_MANIFEST = ROOT / "artifacts/split_manifest.json"
FEATURE_SCHEMA = ROOT / "artifacts/feature_schema.json"

# SHA-256 values pin the exact evaluated model/config/evaluation/prediction files.
# The three fallback models were selected by pair-disjoint validation artifacts;
# no held-out test result is used for model selection at inference.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "molecular_target_text": {
        "run": "artifacts/multimodal_pair_disjoint_molecular_target_text_dailymed_common_support_trainfit_v1_support_molecular_target_text",
        "modalities": ("molecular", "target", "text"),
        "checkpoint_sha256": "9259c34a964c48d30c849cb5480c8ac0a1453abf3b4936608c1a2435054d965c",
        "config_sha256": "e151d9b71f3487b55980f584acd56cad1f6d283914933c2cce244db07121efee",
        "evaluation_sha256": "a58a7d3a3a71e54bdc582e7cbefbdf5540528b03defb807bdfc42a608da572aa",
        "test_predictions_sha256": "479e6734d191c7ef677c0a22aa6370b9f5a31a2e0b8b83f1a54238c2af1792b5",
    },
    "molecular_target": {
        "run": "artifacts/multimodal_pair_disjoint_molecular_target_chembl34_diagonal_v1",
        "modalities": ("molecular", "target"),
        "checkpoint_sha256": "f6d6f8b760358c5575e314b74b9127bb239cebe495ab0ae00599681cae2d90dc",
        "config_sha256": "458d0c722b3958733de21ba7ae8b891aa46c8a4249a9ad4562ffdf376fb3b0b9",
        "evaluation_sha256": "86ebb0642d856fa702b6f5ec3dcbfa3f4c12fc8da577fe4419c1438a73df9c44",
        "test_predictions_sha256": "ec422d7feae9e1079a6bd5f831182a509cba405c7a88cbe91a94b0bc7e3f1828",
    },
    "target": {
        "run": "artifacts/multimodal_pair_disjoint_target_diagonal_relation_v1",
        "modalities": ("target",),
        "checkpoint_sha256": "7c2da588f58185becb4de502c9efe41cc3ced174f4463716a2c8516d1e17ad8b",
        "config_sha256": "4e916241493b82ef5dd9d1ba358ba388ca424b870daf3449756061b74ff42b31",
        "evaluation_sha256": "384be663a92321a8649daf511b6d9b59bfd9d65b6b229abfdc869809886ca3d4",
        "test_predictions_sha256": "f38bc34d2730c431e9c7db9bfc051e24991569d7bbdc072db1318bf61e969c49",
    },
    "molecular": {
        "run": "artifacts/molecular_pair_disjoint_no_placeholder_v1",
        "modalities": ("molecular",),
        "checkpoint_sha256": "273c55b26107e13d1da3d370de2ba2367c3df515aeed36be65bf012b477fff2e",
        "config_sha256": "1300090197deb62909debb1b49ff1ecc0d85ec958cde29d930c4fac55436c0c9",
        "evaluation_sha256": "7a59c60986ecd182037b0d209b01eb08d87cdca87d191327221d5563921114b8",
        "test_predictions_sha256": "d8f5b5f48e68106f83d777d91c1dbc9d976b6eb7fc2dd5e956213afa9855daf3",
    },
}

LIMITATIONS = [
    "Academic research demonstration only; not for diagnosis, treatment, medication changes, or clinical decisions.",
    "DDInter severity labels are dataset annotations, not patient outcomes or clinical validation.",
    "DailyMed evidence is general drug-label text, not pair-specific proof of an interaction.",
    "ChEMBL target records are bioactivity evidence, not direct evidence of a drug-drug mechanism.",
    "Cold-drug performance is weak; the selected pair-disjoint model is not clinically validated.",
    "Model probabilities are research-model outputs, not clinical risk probabilities or validated confidence estimates.",
]


class ResearchInferenceError(ValueError):
    """Input or provenance could not safely support a research inference."""

    def __init__(self, message: str, *, code: str = "invalid_input", http_status: int = 422,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.details = details or {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise RuntimeError("candidate evaluation references a missing or out-of-project artifact")
    return path


def _load_text_provenance() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with TEXT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["ddinter_id"]] = {
                key: row.get(key, "") for key in
                ("source_id", "evidence_id", "source_url", "source_version", "source_date",
                 "embedding_path", "embedding_sha256", "evidence_scope", "label_independent")
                if row.get(key) is not None
            }
    return result


def _dependency_versions() -> dict[str, str | None]:
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "numpy", "rdkit", "scikit-learn", "torch-geometric"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@lru_cache(maxsize=1)
def _sources() -> dict[str, Any]:
    """Load all verified source modalities once through the shared training loaders."""
    return _load_sources(ROOT, MODALITY_ORDER)


def _feature_presence(drug_id: str, sources: dict[str, Any]) -> dict[str, bool]:
    return {
        "molecular": drug_id in sources["molecular_graphs"],
        "target": drug_id in sources["target_drugs"],
        "text": drug_id in sources["text_vectors"],
    }


@lru_cache(maxsize=4)
def _bundle(name: str = "molecular_target_text") -> dict[str, Any]:
    """Load a pinned candidate only if all artifact, source, config, and shape checks pass."""
    if name not in MODEL_SPECS:
        raise RuntimeError("research model candidate is not registered")
    spec = MODEL_SPECS[name]
    run = ROOT / spec["run"]
    evaluation_path, checkpoint_path, config_path = (run / "evaluation.json", run / "best_validation.pt",
                                                      run / "model_config.json")
    prediction_path = run / "test_predictions.npz"
    for path, key in ((evaluation_path, "evaluation_sha256"), (checkpoint_path, "checkpoint_sha256"),
                      (config_path, "config_sha256"), (prediction_path, "test_predictions_sha256")):
        if not path.is_file() or _sha(path).casefold() != spec[key].casefold():
            raise RuntimeError(f"verified {name} candidate artifact is missing or has changed: {path.relative_to(ROOT)}")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    modalities = spec["modalities"]
    if evaluation.get("status") not in {"completed_multimodal_or_source_backed_ablation", "completed_molecular_only_ablation"}:
        raise RuntimeError(f"{name} candidate evaluation is not complete")
    # The molecular-only evaluator predates the common `modalities` field; its
    # saved config is authoritative and is checked against the registry below.
    recorded_modalities = evaluation.get("modalities")
    if evaluation.get("protocol") != "pair_disjoint" or (recorded_modalities is not None and tuple(recorded_modalities) != modalities):
        raise RuntimeError(f"{name} candidate protocol/modalities do not match its registry")
    if evaluation.get("checkpoint_reload_verified") is not True:
        raise RuntimeError(f"{name} checkpoint reload was not verified during training")
    if float(evaluation.get("temperature_fitted_on_validation_only", 0)) <= 0:
        raise RuntimeError(f"{name} validation-only temperature is missing")
    if evaluation.get("dataset_manifest_sha256") != _sha(PAIR_DATA):
        raise RuntimeError(f"{name} DDInter dataset manifest hash mismatch")
    if evaluation.get("split_manifest_sha256") != _sha(SPLIT_MANIFEST):
        raise RuntimeError(f"{name} fixed split manifest hash mismatch")
    if metadata.get("format_version") != 1 or metadata.get("class_order") != list(LABELS):
        raise RuntimeError(f"{name} checkpoint format or class order is unsupported")
    if evaluation.get("checkpoint") != str(checkpoint_path.relative_to(ROOT)):
        raise RuntimeError(f"{name} evaluation points to a different checkpoint")
    if evaluation.get("checkpoint_config") != str(config_path.relative_to(ROOT)):
        raise RuntimeError(f"{name} evaluation points to a different model config")
    if evaluation.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
        raise RuntimeError(f"{name} evaluation checkpoint hash mismatch")
    if evaluation.get("test_prediction_artifact", {}).get("sha256") != spec["test_predictions_sha256"]:
        raise RuntimeError(f"{name} evaluation test-prediction hash mismatch")
    if float(evaluation["temperature_fitted_on_validation_only"]) <= 0:
        raise RuntimeError(f"{name} has an invalid validation calibration value")

    config = metadata["config"]
    enabled = tuple(modality for modality, flag in zip(MODALITY_ORDER,
                   (config.get("use_molecular"), config.get("use_target"), config.get("use_text"))) if flag)
    if enabled != modalities or config.get("num_classes") != len(LABELS):
        raise RuntimeError(f"{name} config modality flags/class count do not match evaluation")
    versions = _dependency_versions()
    expected_versions = evaluation.get("dependency_versions", {})
    for package, current in versions.items():
        recorded = expected_versions.get(package)
        if recorded is not None and recorded != current:
            raise RuntimeError(f"{name} dependency version mismatch for {package}: {recorded} != {current}")

    sources = _sources()
    if modalities == ("molecular",):
        if evaluation.get("pubchem_compounds_sha256") != _sha(ROOT / "data/processed/pubchem_compounds.json"):
            raise RuntimeError("molecular-only candidate PubChem source hash mismatch")
        if evaluation.get("feature_schema_sha256") != _sha(FEATURE_SCHEMA):
            raise RuntimeError("molecular-only candidate feature schema hash mismatch")
        if config.get("num_drugs") != 0 or config.get("num_targets") != 0 or config.get("text_dim") != 0:
            raise RuntimeError("molecular-only checkpoint contains unexpected non-molecular inputs")
        if not sources["molecular_graphs"]:
            raise RuntimeError("molecular-only candidate has no verified molecular graphs")
    else:
        selected_sources = {key: sources["sources"][key] for key in modalities}
        fingerprint = hashlib.sha256(json.dumps(selected_sources, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        if fingerprint != evaluation.get("source_fingerprint_sha256"):
            raise RuntimeError(f"{name} source feature fingerprint mismatch")
        for modality in modalities:
            recorded = evaluation.get("sources", {}).get(modality, {})
            loaded = sources["sources"].get(modality, {})
            for key, value in recorded.items():
                if key.endswith("_sha256") and loaded.get(key) != value:
                    raise RuntimeError(f"{name} {modality} source hash mismatch for {key}")
        if "text" in modalities:
            if evaluation["sources"]["text"].get("manifest_sha256") != _sha(TEXT_MANIFEST):
                raise RuntimeError("pair-disjoint text feature manifest hash mismatch")
            if config.get("text_dim") != sources["sources"]["text"].get("vector_dimension"):
                raise RuntimeError("text vector dimension differs from checkpoint config")
        if "target" in modalities:
            target_meta = sources["sources"]["target"]
            if any(config.get(key) != target_meta.get(source_key) for key, source_key in
                   (("num_drugs", "num_drugs"), ("num_targets", "num_targets"),
                    ("num_relations", "num_relations"))):
                raise RuntimeError("target graph entity/relation dimensions differ from checkpoint config")

    if config.get("use_molecular") and (ATOM_FEATURE_DIM < 1 or BOND_FEATURE_DIM < 1):
        raise RuntimeError("molecular feature schema has invalid dimensions")
    model = load_checkpoint(checkpoint_path, config_path, map_location="cpu").eval()
    with np.load(prediction_path, allow_pickle=False) as saved:
        if saved["probabilities_calibrated"].shape[1] != len(LABELS):
            raise RuntimeError(f"{name} saved predictions have the wrong class dimension")
        artifact_metadata = json.loads(str(saved["metadata"]))
        if artifact_metadata.get("class_order") != list(LABELS):
            raise RuntimeError(f"{name} saved predictions use a different class order")
    return {"name": name, "model": model, "evaluation": evaluation, "sources": sources,
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": _sha(checkpoint_path),
            "config_path": config_path, "config_sha256": _sha(config_path),
            "temperature": float(evaluation["temperature_fitted_on_validation_only"]),
            "modalities": modalities}


def _load_sources_for_pair() -> dict[str, Any]:
    return _sources()


def _select_model(present_a: dict[str, bool], present_b: dict[str, bool]) -> str | None:
    both = {name: present_a[name] and present_b[name] for name in MODALITY_ORDER}
    if all(both.values()):
        return "molecular_target_text"
    if both["molecular"] and both["target"]:
        return "molecular_target"
    # The molecular-only checkpoint has the higher validation macro-F1 of the
    # verified single-modality pair-disjoint fallbacks; this is preselected by
    # validation results, never by held-out test outcomes.
    if both["molecular"]:
        return "molecular"
    if both["target"]:
        return "target"
    return None


def infer_research_pair(drug_a: str, drug_b: str) -> dict[str, Any]:
    """Infer a research-only result from the best verified model supported by both drugs."""
    if not isinstance(drug_a, str) or not isinstance(drug_b, str) or not drug_a.strip() or not drug_b.strip():
        raise ResearchInferenceError("Both drug inputs must be non-empty strings.")
    try:
        id_a = _resolve_drug_identifier(drug_a)
        id_b = _resolve_drug_identifier(drug_b)
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            raise ResearchInferenceError("Unknown drug ID or exact reviewed name.", code="unknown_drug", http_status=404) from exc
        if getattr(exc, "status_code", None) == 422:
            raise ResearchInferenceError(str(getattr(exc, "detail", exc)), code="ambiguous_drug", http_status=422) from exc
        raise
    if id_a == id_b:
        raise ResearchInferenceError("Select two different drugs.", code="same_drug", http_status=422)

    sources = _load_sources_for_pair()
    present_a, present_b = _feature_presence(id_a, sources), _feature_presence(id_b, sources)
    model_name = _select_model(present_a, present_b)
    if model_name is None:
        raise ResearchInferenceError(
            "No verified checkpoint supports this pair. Both drugs need shared molecular features or shared target features; text-only fallback is not validated for this support.",
            code="insufficient_features", http_status=422,
            details={"drug_a_modalities": present_a, "drug_b_modalities": present_b})
    bundle = _bundle(model_name)
    modalities = bundle["modalities"]
    row = {"drug_a_id": id_a, "drug_b_id": id_b}
    row["modality_mask"] = _pair_mask(row, sources, modalities)
    model = bundle["model"]
    if modalities == ("molecular",):
        graph_a = batch_graphs([sources["molecular_graphs"][id_a]])
        graph_b = batch_graphs([sources["molecular_graphs"][id_b]])
        model_inputs = {"molecule_a": graph_a, "molecule_b": graph_b}
    else:
        model_inputs = _make_inputs([row], sources, modalities, sources["target_drug_index"],
                                    int(model.config.text_dim))
    with torch.inference_mode():
        output = model(**model_inputs)
        probabilities = apply_temperature(output["logits"].cpu(), bundle["temperature"])[0].numpy()
    if probabilities.shape != (len(LABELS),) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise RuntimeError("model returned invalid class probabilities")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise RuntimeError("model probabilities do not sum to one")

    catalog = _drug_catalog()
    targets = _target_evidence_by_drug()
    text_provenance = _load_text_provenance()

    def provenance(drug_id: str) -> dict[str, Any]:
        record = catalog[drug_id]
        text = text_provenance.get(drug_id)
        return {
            "drug_id": drug_id, "drug_name": record["name"],
            "available_modalities": _feature_presence(drug_id, sources),
            "model_input_modalities": {key: key in modalities for key in MODALITY_ORDER},
            "sources": {
                "molecular": {"source_id": "PubChem", "available": drug_id in sources["molecular_graphs"],
                              "source_url": record.get("structure_source_url"), "pubchem_cid": record.get("pubchem_cid")},
                "target": {"source_id": "ChEMBL 34", "available": drug_id in sources["target_drugs"],
                           "records": targets.get(drug_id, [])[:10], "record_count": len(targets.get(drug_id, []))},
                "text": {"source_id": (text or {}).get("source_id"), "available": drug_id in sources["text_vectors"],
                         "evidence_id": (text or {}).get("evidence_id"), "source_url": (text or {}).get("source_url"),
                         "source_version": (text or {}).get("source_version"), "scope": "drug_general" if text else None},
            },
        }

    evaluation = bundle["evaluation"]
    return {
        "status": "research_prediction",
        "drug_a": {"drug_id": id_a, "name": catalog[id_a]["name"]},
        "drug_b": {"drug_id": id_b, "name": catalog[id_b]["name"]},
        "predicted_severity": LABELS[int(np.argmax(probabilities))],
        "class_order": list(LABELS),
        "class_probabilities": {label: float(probabilities[i]) for i, label in enumerate(LABELS)},
        "probability_interpretation": "validation-temperature-scaled model probabilities; not clinical confidence or individual risk",
        "model_version": f"{model_name}_pair_disjoint_v1",
        "checkpoint_path": str(bundle["checkpoint_path"].relative_to(ROOT)),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "calibration": {"method": "validation-fitted scalar temperature", "fit_partition": "validation_only",
                        "temperature": bundle["temperature"]},
        "split_protocol": "pair_disjoint",
        "best_epoch": evaluation.get("best_epoch"),
        "validation_macro_f1": evaluation.get("best_validation_macro_f1"),
        "modality_coverage": {
            "drug_a": present_a, "drug_b": present_b,
            "model_inputs": {name: name in modalities for name in MODALITY_ORDER},
        },
        "preprocessing_verification": {"status": "verified", "training_preprocessing_reused": True,
                                        "checkpoint_reload": True},
        "supporting_evidence": {"drug_a": provenance(id_a), "drug_b": provenance(id_b)},
        "limitations": LIMITATIONS,
    }
