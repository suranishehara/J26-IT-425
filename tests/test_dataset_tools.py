import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dataset_tools import map_category,classification_metrics

spec=importlib.util.spec_from_file_location('indofashion_importer',ROOT/'scripts'/'prepare_indofashion.py')
importer=importlib.util.module_from_spec(spec);spec.loader.exec_module(importer)

class DatasetToolsTests(unittest.TestCase):
    def test_duplicate_cleanup_protects_evaluation_and_raw_splits(self):
        records=[dict(sample_id='train-a',source_group='same',split='train',garment_type='saree'),dict(sample_id='test-a',source_group='same',split='test',garment_type='saree')]
        kept,removed=importer.clean_duplicates(records)
        self.assertEqual([row['sample_id'] for row in kept],['test-a'])
        self.assertEqual(removed[0]['sample_id'],'train-a')
        self.assertEqual(len(records),2)
    def test_conflicting_duplicate_labels_are_not_guessed(self):
        records=[dict(sample_id='a',source_group='same',split='train',garment_type='saree'),dict(sample_id='b',source_group='same',split='test',garment_type='blouse')]
        kept,removed=importer.clean_duplicates(records)
        self.assertEqual(kept,[])
        self.assertEqual(len(removed),2)
    def test_mapping_does_not_invent_osariya_labels(self):
        self.assertEqual(map_category('kurta_men'),'kurta')
        self.assertEqual(map_category('Saree'),'saree')
        self.assertIsNone(map_category('sherwani'))
        self.assertIsNone(map_category(3))
    def test_metrics_include_out_of_scope_prediction_errors(self):
        report=classification_metrics([('saree','saree'),('saree','shirt'),('blouse','blouse')])
        self.assertAlmostEqual(report['accuracy'],2/3)
        self.assertAlmostEqual(report['per_class']['saree']['recall'],.5)
        self.assertEqual(report['confusion_matrix']['saree']['shirt'],1)
        self.assertAlmostEqual(report['macro_f1'],5/6)
    def test_manifest_preserves_split_and_leaves_unknown_attributes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'one.png').write_bytes(b'fixture-one')
            (root/'val.json').write_text(json.dumps([{'image_path':'one.png','class_label':'saree'}]))
            importer.prepare(root,root,root/'output')
            row=json.loads((root/'output'/'manifest.jsonl').read_text())
            self.assertEqual(row['split'],'validation')
            self.assertEqual(row['drape_style'],'unknown')
    def test_official_json_lines_and_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'one.png').write_bytes(b'fixture-one')
            (root/'two.png').write_bytes(b'fixture-two')
            records=[{'image_path':'one.png','class_label':'saree'},{'image_path':'two.png','class_label':'blouse'}]
            (root/'val_data.json').write_text('\n'.join(json.dumps(row) for row in records))
            report=importer.prepare(root,root,root/'output')
            self.assertEqual(report['counts'],{'validation:saree':1,'validation:blouse':1})
    def test_exact_cross_split_duplicate_blocks_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'one.png').write_bytes(b'fixture-one')
            records=[{'image_path':'one.png','class_label':'saree'}]
            for name in ('train.json','test.json'):
                (root/name).write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                importer.prepare(root,root,root/'output')
            self.assertFalse((root/'output'/'manifest.jsonl').exists())
