import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
from explanations import patch_gradcam

class GradCAMTests(unittest.TestCase):
    def test_class_token_is_excluded_and_positive_map_normalized(self):
        activations=torch.tensor([[[999.],[0.],[1.],[2.],[3.]]])
        gradient=torch.ones_like(activations)
        cam,informative=patch_gradcam(activations,gradient,2)
        self.assertTrue(informative)
        self.assertEqual(cam.shape,(2,2))
        self.assertAlmostEqual(cam[1,1].item(),1)
        self.assertEqual(cam[0,0].item(),0)
    def test_zero_gradient_is_not_a_fabricated_heatmap(self):
        a=torch.ones(1,5,2)
        cam,informative=patch_gradcam(a,torch.zeros_like(a),2)
        self.assertFalse(informative)
        self.assertEqual(cam.sum().item(),0)
    def test_rejects_malformed_or_nonfinite_values(self):
        with self.assertRaises(ValueError):patch_gradcam(torch.ones(1,4,2),torch.ones(1,4,2),2)
        with self.assertRaises(ValueError):patch_gradcam(torch.full((1,5,2),float('nan')),torch.ones(1,5,2),2)
