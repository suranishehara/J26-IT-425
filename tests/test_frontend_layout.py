"""Structural guardrails for the four-panel frontend; geometry checked in browser."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayoutParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = []; self.sections = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag == 'section': self.sections.append(attrs.get('aria-labelledby'))


class FrontendLayoutTests(unittest.TestCase):
    def test_three_classifier_labels_and_default(self):
        html = (ROOT / 'frontend/index.html').read_text(encoding='utf-8')
        self.assertIn('<option value="convnext_v2" selected>Experimental ConvNeXt V2-Tiny</option>', html)
        self.assertIn('<option value="convnext_v1">Experimental ConvNeXt-Tiny</option>', html)
        self.assertIn('<option value="efficientnet_b0">Experimental EfficientNet-B0</option>', html)
        self.assertNotIn('value="review_only"', html)
        script = (ROOT / 'frontend/app.js').read_text(encoding='utf-8')
        self.assertIn("mode: 'experimental', model_id: $('analysis-mode').value", script)

    def test_long_analysis_section_removed_errors_remain_visible(self):
        script = (ROOT / 'frontend/app.js').read_text(encoding='utf-8')
        self.assertNotIn('About this analysis · limitations and licence', script)
        self.assertIn("'Analysis complete. Review the category suggestions below.' : result.notice", script)
        self.assertIn('result.capture_quality.warnings', script)
        self.assertIn('apply.disabled = !review || !review.allow_quick_apply', script)

    def test_four_accessibly_named_panels(self):
        parser = LayoutParser(); parser.feed((ROOT / 'frontend/index.html').read_text(encoding='utf-8'))
        self.assertEqual(parser.sections, ['photo-heading', 'properties-heading', 'profile-heading', 'design-heading'])
        self.assertEqual(len(parser.ids), len(set(parser.ids)), 'Duplicate control IDs')

    def test_controls_remain_available_after_reorganisation(self):
        parser = LayoutParser(); parser.feed((ROOT / 'frontend/index.html').read_text(encoding='utf-8'))
        for control in ('photo', 'permission', 'analysis-mode', 'analyse', 'preview', 'suggestions', 'fabric-category',
                        'material', 'technique', 'pattern', 'texture', 'drapability', 'weight', 'gsm', 'stretch',
                        'source', 'reference', 'unit', 'measurements', 'fit', 'style', 'silhouette', 'reviewed',
                        'recommend', 'status', 'warnings', 'results'):
            self.assertIn(control, parser.ids)


if __name__ == '__main__': unittest.main()
