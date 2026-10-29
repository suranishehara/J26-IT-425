import unittest
from PIL import Image
from backend.photo_review import quality, decision


class PhotoReviewTests(unittest.TestCase):
    def suggestions(self, first=.8, second=.1):
        return [{'category': 'cotton', 'relative_score': first}, {'category': 'silk', 'relative_score': second}]

    def test_ambiguous_example_abstains(self):
        result = decision(self.suggestions(.387, .339), {'warnings': []})
        self.assertFalse(result['allow_quick_apply'])
        self.assertEqual(result['suggested_category'], 'unknown')

    def test_high_score_still_requires_review(self):
        result = decision(self.suggestions(), {'warnings': []})
        self.assertTrue(result['allow_quick_apply'])
        self.assertEqual(result['status'], 'candidate_review_required')

    def test_small_margin_abstains(self):
        self.assertFalse(decision(self.suggestions(.6, .46), {'warnings': []})['allow_quick_apply'])

    def test_capture_warning_blocks_quick_apply(self):
        self.assertFalse(decision(self.suggestions(), {'warnings': ['glare']})['allow_quick_apply'])

    def test_no_prediction(self):
        self.assertEqual(decision([], {'warnings': []})['status'], 'no_prediction')

    def test_dark_bright_low_resolution(self):
        for value in (0, 255):
            result = quality(Image.new('RGB', (100, 100), (value,) * 3))
            self.assertEqual(result['status'], 'capture_review_recommended')
            self.assertGreaterEqual(len(result['warnings']), 3)

    def test_textured_fixture_no_warning(self):
        image = Image.new('RGB', (256, 256))
        image.putdata([(50, 50, 50) if (x // 8 + y // 8) % 2 else (180, 180, 180) for y in range(256) for x in range(256)])
        self.assertEqual(quality(image)['warnings'], [])
