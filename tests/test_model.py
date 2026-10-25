import copy
import unittest
import torch

from ddi_guard.features import batch_graphs, molecule_to_graph
from ddi_guard.model import DDIModelConfig, MultimodalDDIGuard, load_checkpoint, parameter_counts, save_checkpoint


def inputs():
    mol_a = molecule_to_graph("CCO")
    mol_b = molecule_to_graph("c1ccccc1")
    target_graph = {
        "edge_index": torch.tensor([[0, 1, 4, 5], [4, 5, 0, 1]], dtype=torch.long),
        "relation": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "edge_weight": torch.tensor([0.8, 1.0, 0.8, 1.0]),
    }
    text = torch.randn(1, 2, 12)
    return mol_a, mol_b, target_graph, text


class ModelTests(unittest.TestCase):
    def test_forward_backward_updates_all_intended_modules_and_symmetric(self):
        torch.manual_seed(3)
        config = DDIModelConfig(num_drugs=4, num_targets=3, num_relations=2,
                                text_dim=12, hidden_dim=32, attention_heads=4)
        model = MultimodalDDIGuard(config)
        a, b, tg, text = inputs()
        model.eval()
        kwargs = {"molecule_a": a, "molecule_b": b, "target_graph": tg,
                  "drug_a_index": torch.tensor([0]), "drug_b_index": torch.tensor([1]),
                  "text_features": text}
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = model(**kwargs)
        swapped = model(**{**kwargs, "molecule_a": b, "molecule_b": a,
                           "drug_a_index": torch.tensor([1]), "drug_b_index": torch.tensor([0]),
                           "text_features": text.flip(1)})
        self.assertEqual(result["logits"].shape, (1, 3))
        self.assertTrue(torch.allclose(result["probabilities"].sum(1), torch.ones(1), atol=1e-6))
        self.assertTrue(torch.allclose(result["logits"], swapped["logits"], atol=1e-6))
        self.assertGreater(parameter_counts(model)["trainable_total"], 99)
        loss = torch.nn.functional.cross_entropy(result["logits"], torch.tensor([2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        updated_modules = ("molecular_encoder", "target_encoder", "text_pair", "cross_modal_attention", "gate", "classifier")
        for module_name in updated_modules:
            parameters = list(getattr(model, module_name).parameters())
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), module_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        changed_names = {name for name, p in model.named_parameters() if not torch.equal(before[name], p.detach())}
        for module_name in updated_modules:
            self.assertTrue(any(name.startswith(module_name + ".") for name in changed_names), module_name)

    def test_missing_modalities_and_bad_masks(self):
        model = MultimodalDDIGuard(DDIModelConfig(num_drugs=2, num_targets=1, text_dim=12, hidden_dim=32, attention_heads=4))
        out = model(text_features=torch.randn(2, 2, 12), modality_mask=torch.tensor([[False, False, True], [False, False, True]]))
        self.assertEqual(out["probabilities"].shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "at least one available"):
            model(text_features=torch.randn(1, 2, 12), modality_mask=torch.zeros(1, 3, dtype=torch.bool))

    def test_molecular_ablation_has_no_placeholder_target_or_text_parameters(self):
        model = MultimodalDDIGuard(DDIModelConfig(
            num_drugs=0, num_targets=0, num_relations=0, text_dim=0, hidden_dim=32,
            attention_heads=4, use_molecular=True, use_target=False, use_text=False))
        names = {name for name, _ in model.named_parameters()}
        self.assertFalse(any(name.startswith(("target_encoder", "target_pair", "text_pair")) for name in names))
        a, b = molecule_to_graph("CC"), molecule_to_graph("CO")
        output = model(molecule_a=batch_graphs([a]), molecule_b=batch_graphs([b]))
        self.assertEqual(tuple(output["logits"].shape), (1, 3))
        with self.assertRaisesRegex(ValueError, "target features were supplied"):
            model(target_graph={"edge_index": torch.empty((2, 0), dtype=torch.long),
                                "relation": torch.empty(0, dtype=torch.long),
                                "edge_weight": torch.empty(0)},
                  drug_a_index=torch.tensor([0]), drug_b_index=torch.tensor([0]))

    def test_variable_size_graph_batch_and_per_pair_missing_molecule_mask(self):
        model = MultimodalDDIGuard(DDIModelConfig(num_drugs=3, num_targets=2, text_dim=12, hidden_dim=32, attention_heads=4))
        ethanol = molecule_to_graph("CCO")
        benzene = molecule_to_graph("c1ccccc1")
        one_batch = batch_graphs([ethanol, benzene])
        other_batch = batch_graphs([benzene, ethanol])
        result = model(molecule_a=one_batch, molecule_b=other_batch,
                       molecule_pair_indices=torch.tensor([0, 2]),
                       text_features=torch.randn(3, 2, 12),
                       modality_mask=torch.tensor([[True, False, True], [False, False, True], [True, False, True]]))
        self.assertEqual(result["logits"].shape, (3, 3))
        self.assertEqual(result["modality_mask"].tolist(), [[True, False, True], [False, False, True], [True, False, True]])

    def test_state_dict_reload_equivalence(self):
        torch.manual_seed(8)
        config = DDIModelConfig(num_drugs=4, num_targets=3, num_relations=2,
                                text_dim=12, hidden_dim=32, attention_heads=4)
        model = MultimodalDDIGuard(config).eval()
        a, b, tg, text = inputs()
        kwargs = {"molecule_a": a, "molecule_b": b, "target_graph": tg,
                  "drug_a_index": torch.tensor([0]), "drug_b_index": torch.tensor([1]),
                  "text_features": text}
        expected = model(**kwargs)["probabilities"].detach()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "model.pt"
            config_path = Path(tmp) / "model_config.json"
            save_checkpoint(model, weights_path, config_path)
            clone = load_checkpoint(weights_path, config_path).eval()
            self.assertTrue(torch.allclose(expected, clone(**kwargs)["probabilities"], atol=1e-7))


if __name__ == "__main__":
    unittest.main()
