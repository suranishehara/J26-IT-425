from pathlib import Path
import shutil
import subprocess
import unittest


class SpecimenAutofillUITests(unittest.TestCase):
    def test_real_ui_handlers(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node.js required for frontend handler regression test')
        result = subprocess.run([node, str(Path(__file__).with_name('specimen_autofill.cjs'))], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_optional_sections_and_visible_sample_selector(self):
        html = (Path(__file__).resolve().parents[1] / 'frontend/index.html').read_text(encoding='utf-8')
        self.assertLess(html.index('id="specimen"'), html.index('<details><summary>Save a sample'))
        for name in ('optional-labels', 'optional-measurements', 'saved-provenance', 'manual-provenance'):
            self.assertIn(f'id="{name}"', html)
