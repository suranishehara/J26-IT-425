import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
from detect_clothing import build_mask

class MaskRulesTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.zeros((80,60),dtype=np.uint8)
        self.labels[15:40,20:40]=4
        self.labels[5:15,23:37]=11
        self.labels[40:70,20:40]=6

    def test_upper_mask_excludes_face_and_trousers(self):
        mask=build_mask(self.labels,'upper-body')
        self.assertEqual(mask[25,30],255)
        self.assertFalse(mask[self.labels==11].any())
        self.assertFalse(mask[self.labels==6].any())

    def test_full_body_includes_lower_clothing(self):
        for category in ('dress', 'saree', 'kandyan-osariya'):
            with self.subTest(category=category):
                mask=build_mask(self.labels,category)
                self.assertEqual(mask[25,30],255)
                self.assertEqual(mask[55,30],255)
                self.assertFalse(mask[self.labels==11].any())

    def test_top_only_replaces_bodice_of_one_piece_dress(self):
        self.labels[15:70,20:40]=7
        mask=build_mask(self.labels,'upper-body')
        self.assertEqual(mask[25,30],255)
        self.assertEqual(mask[60,30],0)
        for category in ('dress', 'saree', 'kandyan-osariya'):
            with self.subTest(category=category):
                self.assertEqual(build_mask(self.labels,category)[60,30],255)

    def test_full_outfit_includes_space_between_trouser_legs(self):
        self.labels[40:70,20:40]=0
        self.labels[40:70,15:23]=6
        self.labels[40:70,37:45]=6
        for category in ('dress', 'saree', 'kandyan-osariya'):
            with self.subTest(category=category):
                self.assertEqual(build_mask(self.labels,category)[55,30],255)
        self.assertEqual(build_mask(self.labels,'upper-body')[55,30],0)

