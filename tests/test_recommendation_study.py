"""Artificial fixtures test the evaluator, not research performance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('study_evaluator',Path(__file__).resolve().parents[1]/'scripts/evaluate_recommendations.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StudyTests(unittest.TestCase):
    def setUp(self):
        self.packet={'catalogue':[{'id':'a','family':'global'},{'id':'b','family':'traditional'}],
            'scenarios':[{'id':'fixture','candidate_order':['a','b'],'rankings':{'baseline':['b','a'],'sri_lanka_guidance':['a','b']}}]}
        self.packet['packet_id']=hashlib.sha256(json.dumps(self.packet,sort_keys=True).encode()).hexdigest()
        self.ratings={'packet_id':self.packet['packet_id'],'reviewer_id':'unit-test-only','judgments':{'fixture':{'a':3,'b':0}}}
    def test_evaluates_frozen_fixture(self):
        report=module.evaluate(self.packet,self.ratings,1)
        self.assertEqual(report['aggregate']['baseline']['precision_at_k']['mean'],0)
        self.assertEqual(report['aggregate']['sri_lanka_guidance']['ndcg_at_k']['mean'],1)
    def test_rejects_incomplete_ratings(self):
        del self.ratings['judgments']['fixture']['b']
        with self.assertRaises(ValueError):module.evaluate(self.packet,self.ratings,1)
    def test_rejects_changed_packet(self):
        self.packet['catalogue'][0]['family']='traditional'
        with self.assertRaises(ValueError):module.evaluate(self.packet,self.ratings,1)
