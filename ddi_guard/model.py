"""Trainable, order-invariant multimodal graph model for three-class DDI severity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from . import LABELS
from .features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, validate_graph


def symmetric_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Order-invariant pair descriptor, unchanged by exchanging a and b."""
    return torch.cat((a + b, torch.abs(a - b), a * b), dim=-1)


class MolecularEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 96, layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_projection = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.message_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + BOND_FEATURE_DIM, hidden_dim), nn.ReLU(),
                           nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                           nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_graph(graph)
        x = graph["x"]
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]
        batch = graph["batch"]
        h = F.relu(self.atom_projection(x))
        src, dst = edge_index
        for mlp, update, norm in zip(self.message_mlps, self.updates, self.norms):
            aggregate = torch.zeros_like(h)
            if src.numel():
                msg = mlp(torch.cat((h[src], edge_attr), dim=-1))
                aggregate.index_add_(0, dst, msg)
            h = norm(h + update(torch.cat((h, aggregate), dim=-1)))
            h = F.relu(h)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled = torch.zeros((n_graphs, self.hidden_dim), device=h.device, dtype=h.dtype)
        counts = torch.zeros((n_graphs, 1), device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones((batch.numel(), 1), device=h.device, dtype=h.dtype))
        return pooled / counts.clamp_min(1)


class RelationalDrugTargetEncoder(nn.Module):
    """Relation-aware message passing over a verified drug-target graph.

    Entity ids are assigned by the data pipeline. Each directed edge carries a relation
    id preserving activity/evidence type; edge values are confidence weights, never
    converted into invented binding affinities.
    """
    def __init__(self, num_drugs: int, num_targets: int, num_relations: int = 1, hidden_dim: int = 96):
        super().__init__()
        if min(num_drugs, num_targets, num_relations) < 1:
            raise ValueError("num_drugs, num_targets, and num_relations must be positive")
        self.num_drugs = num_drugs
        self.num_targets = num_targets
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.entities = nn.Embedding(num_drugs + num_targets, hidden_dim)
        # Relation-specific diagonal transforms keep full-graph message passing
        # bounded in memory. A dense matrix per edge would materialize
        # [num_edges, hidden_dim, hidden_dim] (multi-GB for the ChEMBL graph).
        self.relation_weights = nn.Parameter(torch.empty(num_relations, hidden_dim))
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.ones_(self.relation_weights)

    def forward(self, edge_index: torch.Tensor, relation: torch.Tensor, edge_weight: torch.Tensor,
                drug_a: torch.Tensor, drug_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.num_drugs + self.num_targets
        if edge_index.shape[0] != 2 or edge_index.shape[1] != relation.numel() or relation.numel() != edge_weight.numel():
            raise ValueError("target graph edge tensors have inconsistent shapes")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
            raise ValueError("target graph edge references an unknown entity")
        if relation.numel() and (int(relation.min()) < 0 or int(relation.max()) >= self.num_relations):
            raise ValueError("target graph edge has an unknown relation id")
        if (drug_a.numel() and (int(drug_a.min()) < 0 or int(drug_a.max()) >= self.num_drugs)) or (drug_b.numel() and (int(drug_b.min()) < 0 or int(drug_b.max()) >= self.num_drugs)):
            raise ValueError("query references an unknown drug entity")
        h = self.entities.weight
        aggregate = torch.zeros_like(h)
        degree = torch.zeros((n, 1), device=h.device, dtype=h.dtype)
        if edge_index.numel():
            src, dst = edge_index
            relation_scale = self.relation_weights[relation]
            msg = h[src] * relation_scale
            weights = edge_weight.to(h.dtype).reshape(-1, 1)
            aggregate.index_add_(0, dst, msg * weights)
            degree.index_add_(0, dst, weights.abs())
        aggregate = aggregate / degree.clamp_min(1.0)
        encoded = F.relu(self.norm(h + self.update(torch.cat((h, aggregate), dim=-1))))
        return encoded[drug_a], encoded[drug_b]


@dataclass
class DDIModelConfig:
    num_drugs: int
    num_targets: int
    num_relations: int = 1
    text_dim: int = 768
    hidden_dim: int = 96
    molecular_layers: int = 3
    attention_heads: int = 4
    num_classes: int = 3
    use_molecular: bool = True
    use_target: bool = True
    use_text: bool = True


class MultimodalDDIGuard(nn.Module):
    """Molecular GNN + relational drug-target GNN + trainable text projection.

    ``text_features`` must be real, provenance-backed vectors for the two drugs. This
    module does not create language embeddings or clinical evidence.
    """
    def __init__(self, config: DDIModelConfig):
        super().__init__()
        if config.num_classes != len(LABELS):
            raise ValueError(f"DDI-GUARD requires class order {LABELS}")
        if not any((config.use_molecular, config.use_target, config.use_text)):
            raise ValueError("at least one real modality must be enabled")
        if config.use_text and config.text_dim < 1:
            raise ValueError("text_dim must be positive when text features are enabled")
        self.config = config
        h = config.hidden_dim
        self.molecular_encoder = MolecularEncoder(h, config.molecular_layers) if config.use_molecular else None
        self.molecular_pair = (nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.LayerNorm(h))
                               if config.use_molecular else None)
        self.target_encoder = (RelationalDrugTargetEncoder(config.num_drugs, config.num_targets,
                                                           config.num_relations, h)
                               if config.use_target else None)
        self.target_pair = (nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.LayerNorm(h))
                            if config.use_target else None)
        self.text_pair = (nn.Sequential(nn.Linear(3 * config.text_dim, h), nn.ReLU(), nn.LayerNorm(h))
                          if config.use_text else None)
        self.cross_modal_attention = nn.MultiheadAttention(h, config.attention_heads, batch_first=True)
        self.gate = nn.Linear(h, 1)
        self.classifier = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.15), nn.Linear(h, config.num_classes))

    def forward(self, *, molecule_a: dict[str, torch.Tensor] | None = None,
                molecule_b: dict[str, torch.Tensor] | None = None,
                molecule_pair_indices: torch.Tensor | None = None,
                target_graph: dict[str, torch.Tensor] | None = None,
                drug_a_index: torch.Tensor | None = None, drug_b_index: torch.Tensor | None = None,
                text_features: torch.Tensor | None = None,
                modality_mask: torch.Tensor | None = None,
                batch_size: int | None = None) -> dict[str, torch.Tensor]:
        batch_sizes: list[int] = []
        if molecule_a is not None and molecule_pair_indices is None:
            batch_sizes.append(int(molecule_a["batch"].max().item()) + 1)
        if molecule_b is not None and molecule_pair_indices is None:
            batch_sizes.append(int(molecule_b["batch"].max().item()) + 1)
        if batch_size is not None:
            batch_sizes.append(int(batch_size))
        if drug_a_index is not None:
            batch_sizes.append(drug_a_index.numel())
        if drug_b_index is not None:
            batch_sizes.append(drug_b_index.numel())
        if text_features is not None:
            batch_sizes.append(text_features.shape[0])
        if not batch_sizes and molecule_pair_indices is not None and molecule_pair_indices.numel():
            batch_sizes.append(int(molecule_pair_indices.max().item()) + 1)
        if not batch_sizes or len(set(batch_sizes)) != 1:
            raise ValueError("inputs must define a consistent non-empty batch size")
        bsz = batch_sizes[0]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        tokens = torch.zeros((bsz, 3, self.config.hidden_dim), device=device, dtype=dtype)
        derived_mask = torch.zeros((bsz, 3), device=device, dtype=torch.bool)

        if molecule_a is not None or molecule_b is not None:
            if not self.config.use_molecular:
                raise ValueError("molecular features were supplied to a model with that modality disabled")
            if molecule_a is None or molecule_b is None:
                raise ValueError("both molecule graphs are required for the molecular modality")
            assert self.molecular_encoder is not None and self.molecular_pair is not None
            ea, eb = self.molecular_encoder(molecule_a), self.molecular_encoder(molecule_b)
            if ea.shape[0] != eb.shape[0]:
                raise ValueError("paired molecular graph batch sizes differ")
            indices = (torch.arange(ea.shape[0], device=device) if molecule_pair_indices is None
                       else molecule_pair_indices.to(device=device, dtype=torch.long))
            if indices.numel() != ea.shape[0] or (indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= bsz)):
                raise ValueError("molecule_pair_indices do not map every graph pair to the batch")
            if indices.unique().numel() != indices.numel():
                raise ValueError("molecule_pair_indices must be unique")
            tokens[indices, 0] = self.molecular_pair(symmetric_pair(ea, eb))
            derived_mask[indices, 0] = True

        if target_graph is not None or drug_a_index is not None or drug_b_index is not None:
            if not self.config.use_target:
                raise ValueError("target features were supplied to a model with that modality disabled")
            if target_graph is None or drug_a_index is None or drug_b_index is None:
                raise ValueError("target graph and both drug index tensors are required together")
            assert self.target_encoder is not None and self.target_pair is not None
            ta, tb = self.target_encoder(target_graph["edge_index"], target_graph["relation"],
                                         target_graph["edge_weight"], drug_a_index, drug_b_index)
            tokens[:, 1] = self.target_pair(symmetric_pair(ta, tb))
            derived_mask[:, 1] = True

        if text_features is not None:
            if not self.config.use_text:
                raise ValueError("text features were supplied to a model with that modality disabled")
            if text_features.shape != (bsz, 2, self.config.text_dim):
                raise ValueError(f"text_features must have shape [B,2,{self.config.text_dim}]")
            assert self.text_pair is not None
            ta, tb = text_features[:, 0], text_features[:, 1]
            tokens[:, 2] = self.text_pair(symmetric_pair(ta, tb))
            derived_mask[:, 2] = True

        if modality_mask is None:
            modality_mask = derived_mask
        else:
            modality_mask = modality_mask.to(device=device, dtype=torch.bool)
            if modality_mask.shape != (bsz, 3):
                raise ValueError("modality_mask must have shape [B,3] in molecular,target,text order")
            if bool((modality_mask & ~derived_mask).any()):
                raise ValueError("modality_mask claims a modality is available but its real features were not supplied")
        if bool((~modality_mask.any(dim=1)).any()):
            raise ValueError("each prediction requires at least one available, real modality")

        attended, _ = self.cross_modal_attention(tokens, tokens, tokens, key_padding_mask=~modality_mask, need_weights=False)
        scores = self.gate(attended).squeeze(-1).masked_fill(~modality_mask, torch.finfo(dtype).min)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities, "modality_mask": modality_mask,
                "fusion_weights": weights, "predicted_class": probabilities.argmax(dim=-1)}


def parameter_counts(model: nn.Module) -> dict[str, Any]:
    by_module = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                 for name, module in model.named_children()}
    return {"trainable_total": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_total": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "trainable_by_top_level_module": by_module}


def save_checkpoint(model: MultimodalDDIGuard, checkpoint_path: str | Path, config_path: str | Path) -> None:
    """Save tensor-only weights and JSON architecture metadata atomically."""
    checkpoint_path, config_path = Path(checkpoint_path), Path(config_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_weights = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_config = config_path.with_suffix(config_path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_weights)
    temporary_config.write_text(json.dumps({"format_version": 1, "config": asdict(model.config),
                                            "class_order": list(LABELS)}, indent=2) + "\n", encoding="utf-8")
    temporary_weights.replace(checkpoint_path)
    temporary_config.replace(config_path)


def load_checkpoint(checkpoint_path: str | Path, config_path: str | Path,
                    map_location: str | torch.device = "cpu") -> MultimodalDDIGuard:
    """Reconstruct only from JSON metadata and a weights-only state dictionary."""
    metadata = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("class_order") != list(LABELS):
        raise ValueError("unsupported checkpoint format or class order")
    model = MultimodalDDIGuard(DDIModelConfig(**metadata["config"]))
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("This PyTorch version lacks safe weights_only checkpoint loading") from exc
    if not isinstance(state, dict) or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                                               for key, value in state.items()):
        raise ValueError("checkpoint is not a tensor-only state dictionary")
    model.load_state_dict(state, strict=True)
    return model
