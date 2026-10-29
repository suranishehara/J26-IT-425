"""Generated fixtures test pipeline mechanics, not local garment accuracy."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from local_attributes import audit_manifest,head_plan

class LocalDataTests(unittest.TestCase):
    def test_unavailable_release_does_not_advertise_a_model(self):
        from unittest.mock import patch
        from local_attributes import available_heads
        with patch('local_attributes.configured_heads',side_effect=ValueError('checksum mismatch')):
            self.assertEqual(available_heads('fixture'),{})
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[]
        for i,(split,label) in enumerate([('train','solid'),('train','floral'),('validation','solid'),('validation','floral'),('test','unknown')]):
            Image.new('RGB',(10,12),(20+i*30,10,200)).save(self.root/(str(i)+'.png'))
            self.rows.append(dict(sample_id=str(i),image_path=str(i)+'.png',source_group='g'+str(i),split=split,
                garment_type='kurta',pattern_type=label,drape_style='unknown',fabric_texture='unknown',annotator='fixture',review_status='reviewed',
                source=dict(image_kind='synthetic',permission_confirmed=True,permission_reference='Generated unit-test pixels')))
    def audit(self):
        path=self.root/'manifest.jsonl';path.write_text('\n'.join(json.dumps(r) for r in self.rows),encoding='utf-8')
        return audit_manifest(path,self.root)
    def test_unknown_targets_masked_and_test_not_in_plan(self):
        rows,report=self.audit();plan=head_plan(rows,['pattern_type'],1,1)
        self.assertEqual(plan['pattern_type']['labels'],['solid','floral'])
        self.assertEqual(sum(plan['pattern_type']['train_counts'].values()),2)
        self.assertFalse(report['test_feature_extraction'])
    def test_rejects_group_leakage(self):
        self.rows[2]['source_group']='g0'
        with self.assertRaisesRegex(ValueError,'leakage'):self.audit()
    def test_rejects_duplicate_decoded_pixels(self):
        self.rows[2]['image_path']='0.png'
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.audit()
    def test_unreviewed_or_unpermitted_rejected(self):
        self.rows[0]['review_status']='unreviewed'
        with self.assertRaisesRegex(ValueError,'review'):self.audit()
        self.rows[0]['review_status']='reviewed';self.rows[0]['source']['permission_confirmed']=False
        with self.assertRaisesRegex(ValueError,'permission'):self.audit()
    def test_path_escape_and_invalid_drape_rejected(self):
        self.rows[0]['image_path']='../escape.png'
        with self.assertRaisesRegex(ValueError,'inside'):self.audit()
        self.rows[0]['image_path']='0.png';self.rows[0]['drape_style']='kandyan_osariya'
        with self.assertRaisesRegex(ValueError,'saree'):self.audit()
    def test_minimum_counts_and_missing_training_class_rejected(self):
        rows,_=self.audit()
        with self.assertRaisesRegex(ValueError,'needs at least'):head_plan(rows,['pattern_type'])
        rows[2]['pattern_type']='striped'
        with self.assertRaisesRegex(ValueError,'absent'):head_plan(rows,['pattern_type'],1,1)
    def test_tiny_head_training_is_deterministic(self):
        import torch
        spec=importlib.util.spec_from_file_location('local_training',Path(__file__).resolve().parents[1]/'scripts/train_local_attributes.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        x=torch.tensor([[1.,0.],[0.,1.],[.9,.1],[.1,.9]])
        y=torch.tensor([0,1,0,1])
        one,history,epoch=module.fit_head(x,y,x,y,2,epochs=10)
        two,_,_=module.fit_head(x,y,x,y,2,epochs=10)
        self.assertEqual(len(history),10)
        self.assertTrue(1<=epoch<=10)
        self.assertTrue(torch.equal(one.weight,two.weight))
