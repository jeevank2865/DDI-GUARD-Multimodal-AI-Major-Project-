"""Deterministic, chemically meaningful 2D atom/bond features from validated SMILES."""
from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem

ATOM_NUMBERS = tuple(range(1, 21))
DEGREES = tuple(range(0, 7))
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
)
BOND_TYPES = (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)
STEREO_TYPES = tuple(Chem.rdchem.BondStereo.values)
ATOM_FEATURE_DIM = len(ATOM_NUMBERS) + 1 + len(DEGREES) + 1 + 5 + 5 + 2 + 1 + 1 + 4
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1 + 1 + 1 + len(STEREO_TYPES)


def _one_hot(value: Any, choices: tuple[Any, ...]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def _atom_features(atom: Chem.Atom) -> list[float]:
    charge = max(-2, min(2, atom.GetFormalCharge()))
    chiral = int(atom.GetChiralTag())
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + [float(charge == item) for item in (-2, -1, 0, 1, 2)]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
        + [float(chiral == item) for item in range(4)]
        + [float(atom.GetTotalNumHs(includeNeighbors=True))]
        + [float(atom.GetMass() / 250.0)]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    stereo = bond.GetStereo()
    return (
        [float(bond.GetBondType() == kind) for kind in BOND_TYPES]
        + [float(bond.GetIsAromatic()), float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(stereo, STEREO_TYPES)
    )


def molecule_to_graph(smiles: str) -> dict[str, torch.Tensor]:
    """Parse a real SMILES and return graph tensors; invalid input raises ValueError."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("RDKit could not parse a non-empty molecule from SMILES")
    x = torch.tensor([_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_pairs.extend(((u, v), (v, u)))
        edge_features.extend((feat, feat))
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
            "batch": torch.zeros(x.shape[0], dtype=torch.long)}


def validate_graph(graph: dict[str, torch.Tensor]) -> None:
    required = {"x", "edge_index", "edge_attr", "batch"}
    if not required.issubset(graph):
        raise ValueError(f"Molecular graph missing tensors: {sorted(required - set(graph))}")
    x, edge_index, edge_attr, batch = graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"]
    if x.ndim != 2 or x.shape[1] != ATOM_FEATURE_DIM:
        raise ValueError(f"atom feature shape must be [N,{ATOM_FEATURE_DIM}]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must be a long tensor with shape [2,E]")
    if edge_attr.shape != (edge_index.shape[1], BOND_FEATURE_DIM):
        raise ValueError(f"edge_attr must have shape [E,{BOND_FEATURE_DIM}]")
    if batch.shape != (x.shape[0],) or batch.dtype != torch.long:
        raise ValueError("batch must be a long tensor with one graph id per atom")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]):
        raise ValueError("edge_index contains an atom index outside x")
    if not torch.isfinite(x).all() or not torch.isfinite(edge_attr).all():
        raise ValueError("molecular features contain non-finite values")


def batch_graphs(graphs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Batch only present, validated molecular graphs without inventing dummy atoms."""
    if not graphs:
        raise ValueError("cannot batch an empty graph list")
    xs, edges, attrs, assignments = [], [], [], []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        validate_graph(graph)
        x, edge_index, edge_attr = graph["x"], graph["edge_index"], graph["edge_attr"]
        xs.append(x)
        edges.append(edge_index + offset)
        attrs.append(edge_attr)
        assignments.append(torch.full((x.shape[0],), graph_id, dtype=torch.long, device=x.device))
        offset += x.shape[0]
    return {"x": torch.cat(xs, dim=0), "edge_index": torch.cat(edges, dim=1),
            "edge_attr": torch.cat(attrs, dim=0), "batch": torch.cat(assignments, dim=0)}
