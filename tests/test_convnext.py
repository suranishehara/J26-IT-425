import unittest
import torch
from PIL import Image
from backend.convnext_model import ConvNextFabric, WEIGHTS


class ConvNextTests(unittest.TestCase):
    def test_finetune_boundary(self):
        from scripts.finetune_convnext import configure
        torch.set_num_threads(2)
        model = ConvNextFabric(7, pretrained=False)
        configure(model)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.features[:-2].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-2:].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.norm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_shape_normalization_and_no_download(self):
        torch.set_num_threads(2)
        model = ConvNextFabric(7, pretrained=False).eval()
        tensor = WEIGHTS.transforms()(Image.new('RGB', (256, 256), 'gray')).unsqueeze(0)
        with torch.no_grad():
            encoded = model.encode(tensor)
            logits = model(tensor)
        self.assertEqual(tuple(encoded.shape), (1, 768))
        self.assertTrue(torch.allclose(encoded.norm(dim=1), torch.ones(1), atol=1e-5))
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(torch.isfinite(logits).all())
