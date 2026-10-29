import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image
spec=importlib.util.spec_from_file_location('drape_inventory',Path(__file__).resolve().parents[1]/'scripts/audit_drape_candidates.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DrapeInventoryTests(unittest.TestCase):
    def test_jfif_is_supported_without_renaming(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for style in module.STYLES:(root/style).mkdir()
            Image.new('RGB',(20,30),'blue').save(root/'nivi/photo.jfif',format='JPEG')
            report=module.inventory(root)
            self.assertEqual(report['readable_counts']['nivi'],1)
            self.assertEqual(report['ignored_files'],[])
    def test_non_destructive_duplicate_conflict_and_bad_file_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for style in module.STYLES:(root/style).mkdir()
            Image.new('RGB',(20,30),'red').save(root/'nivi/a.png')
            Image.new('RGB',(20,30),'red').save(root/'kandyan_osariya/b.png')
            (root/'nivi/bad.jpg').write_bytes(b'not an image')
            (root/'nivi/readme.txt').write_text('source note')
            report=module.inventory(root)
            self.assertEqual(report['unique_decoded_images'],1)
            self.assertTrue(report['exact_duplicate_groups'][0]['cross_label_conflict'])
            self.assertEqual(len(report['failures']),1)
            self.assertEqual(len(report['ignored_files']),1)
            self.assertTrue(all(row['split']=='unassigned' for row in report['records']))
            self.assertTrue((root/'nivi/a.png').is_file())
