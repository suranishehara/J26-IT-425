import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_nlu_review import make_pack, render


class ReviewPackTests(unittest.TestCase):
    def fixture(self, text='Compare silk and cotton'):
        return make_pack([{'id': 'q1', 'text': text, 'source_ids': ['m1'], 'provided_intents': ['compare']}],
                         [{'id': 'm1', 'conversation_id': 'c1', 'text': text}], 'sha')

    def test_suggestions_never_become_gold(self):
        pack = self.fixture()
        self.assertEqual(pack['items'][0]['suggestion']['intent'], 'compare')
        review = pack['items'][0]['review']
        for name in ('intent', 'entities', 'slots', 'remove_slots'):
            self.assertIsNone(review[name])
        self.assertEqual(review['status'], 'pending')
        self.assertFalse(review['assisted'])

    def test_untrusted_text_is_not_html(self):
        text = '</script><img src=x onerror=alert(1)>'
        html = render(self.fixture(text))
        payload = html.split('<script id="pack" type="application/json">')[1].split('</script>')[0]
        self.assertNotIn('<', payload)
        self.assertEqual(json.loads(payload)['items'][0]['text'], text)

    def test_missing_links_rejected_and_reproducible(self):
        with self.assertRaises(ValueError): make_pack([{'id': 'q', 'source_ids': ['missing']}], [], 'sha')
        self.assertEqual(self.fixture(), self.fixture())

    def test_review_javascript_validation(self):
        node = shutil.which('node')
        if not node: self.skipTest('Node unavailable')
        result = subprocess.run([node, str(ROOT / 'tests/review_validation.cjs')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
