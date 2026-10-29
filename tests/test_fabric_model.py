"""ML checks skip on the lightweight web-only environment."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'), 'Optional ML environment not installed')
class FabricModelTests(unittest.TestCase):
    def test_metrics_count_all_classes(self):
        from backend.fabric_model import classification_metrics
        metrics = classification_metrics([0, 1, 1], [0, 0, 1], ['a', 'b'])
        self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0], [1, 1]])
        self.assertAlmostEqual(metrics['macro_f1'], 2 / 3)

    def test_network_shape_and_feature_normalization(self):
        import torch
        from backend.fabric_model import FabricNetwork
        torch.set_num_threads(2)
        network = FabricNetwork(7, pretrained=False).eval()
        with torch.no_grad():
            images = torch.ones(1, 3, 224, 224)
            features = network.encode(images)
            self.assertEqual(tuple(features.shape), (1, 1280))
            raw = network.pool(network.features(images)).flatten(1)
            self.assertTrue(torch.allclose(features, torch.nn.functional.normalize(raw, dim=1)))
            # Randomly initialised backbones can yield near-zero activations.
            # F.normalize intentionally uses epsilon there; norm need not be 1.
            self.assertLessEqual(float(features.norm()), 1.00001)
            self.assertTrue(bool(torch.isfinite(features).all()))
            self.assertEqual(tuple(network.head(features).shape), (1, 7))

    def test_finetune_updates_only_allowed_stages(self):
        import torch
        from backend.fabric_model import FabricNetwork
        from scripts.finetune_fabric import configure_finetuning
        torch.set_num_threads(2)
        network = FabricNetwork(7)
        configure_finetuning(network)
        self.assertFalse(network.training)
        self.assertFalse(any(p.requires_grad for p in network.features[:-2].parameters()))
        self.assertTrue(all(p.requires_grad for p in network.features[-2:].parameters()))
        protected = {k: v.clone() for k, v in network.state_dict().items() if k.startswith('features.0.') or 'running_' in k or 'num_batches_tracked' in k}
        old_head = network.head.weight.clone()
        optimizer = torch.optim.SGD([p for p in network.parameters() if p.requires_grad], lr=0.1)
        loss = torch.nn.functional.cross_entropy(network(torch.rand(2, 3, 224, 224)), torch.tensor([0, 1]))
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(old_head, network.head.weight))
        for key, before in protected.items():
            self.assertTrue(torch.equal(before, network.state_dict()[key]), key)

    def test_development_loader_forbids_test_rows(self):
        from backend.training_data import DevelopmentImages
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            DevelopmentImages('.', [{'split': 'test'}], ['cotton'])

    def test_development_manifest_integrity_and_overlap(self):
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from backend.training_data import read_development
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            content = '\n'.join(json.dumps({'split': split, 'group_id': 'same', 'label': 'cotton'}) for split in ('train', 'validation', 'test')).encode()
            (folder / 'manifest.jsonl').write_bytes(content)
            (folder / 'preparation.json').write_text(json.dumps({'labels': ['cotton'], 'manifest_sha256': hashlib.sha256(content).hexdigest()}))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                read_development(folder)
            (folder / 'manifest.jsonl').write_bytes(content + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_development(folder)


if __name__ == '__main__': unittest.main()
