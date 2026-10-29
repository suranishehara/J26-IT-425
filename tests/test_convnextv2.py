"""Randomly initialized mechanics fixtures; no downloads or dataset claims."""
import importlib.util
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('timm'), 'Optional timm experiment dependency absent')
class ConvNextV2Tests(unittest.TestCase):
    def test_nine_category_output_is_a_new_head_not_relabelled_logits(self):
        import torch
        from backend.convnextv2_model import ConvNextV2Fabric
        torch.set_num_threads(2)
        model = ConvNextV2Fabric(9).eval()
        self.assertEqual(model.head.out_features, 9)
        with torch.no_grad(): result = model(torch.rand(1, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (1, 9))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_no_download_shapes_and_boundary(self):
        import torch
        from PIL import Image
        from backend.convnextv2_model import ConvNextV2Fabric, configure_finetuning
        torch.set_num_threads(2)
        with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('Inference must not download')):
            model = ConvNextV2Fabric(7).eval()
            image = model.transform(Image.new('RGB', (256, 256), 'gray')).unsqueeze(0)
            with torch.no_grad():
                features = model.encode(image); logits = model(image)
        self.assertEqual(tuple(features.shape), (1, 768))
        self.assertTrue(torch.allclose(features.norm(dim=1), torch.ones(1), atol=1e-5))
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        configure_finetuning(model)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.backbone.stem.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.stages[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.stages[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_no_implicit_pretrained_download(self):
        from backend.convnextv2_model import ConvNextV2Fabric
        with self.assertRaisesRegex(ValueError, 'explicit local'):
            ConvNextV2Fabric(7, pretrained=True)

    def test_download_requires_pinned_revision_and_checksum(self):
        from backend.convnextv2_model import download_pretrained, REVISION, REPOSITORY
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fake.safetensors'; path.write_bytes(b'not real weights')
            with patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    download_pretrained(Path(temp))
                self.assertEqual(download.call_args.kwargs['revision'], REVISION)
                self.assertEqual(download.call_args.kwargs['repo_id'], REPOSITORY)
                self.assertFalse(download.call_args.kwargs['token'])
