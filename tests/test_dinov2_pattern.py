import sys
from pathlib import Path
import unittest
import tempfile
import json
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from dinov2_pattern import development_rows,aligned_validation,validate_run,MODEL_ID,MODEL_REVISION

class DinoComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'sample_id':'fit','split':'train','pattern_type':'solid'},
            {'sample_id':'val','split':'validation','pattern_type':'striped'},
            {'sample_id':'test','split':'test','pattern_type':'floral'},
            {'sample_id':'unknown','split':'validation','pattern_type':'unknown'}]
        self.predictions=[{'sample_id':'val','attribute':'pattern_type','truth':'striped','baseline':'striped','adapted':'solid'}]
    def test_test_and_unknown_excluded(self):
        self.assertEqual([r['sample_id'] for r in development_rows(self.rows)],['fit','val'])
    def test_exact_validation_alignment(self):
        result=aligned_validation(self.rows,self.predictions,['solid','striped'])
        self.assertEqual(set(result),{'val'})
    def test_rejects_different_samples_or_truth(self):
        with self.assertRaisesRegex(ValueError,'sample IDs'):aligned_validation(self.rows,[],['solid','striped'])
        self.predictions[0]['truth']='solid'
        with self.assertRaisesRegex(ValueError,'labels differ'):aligned_validation(self.rows,self.predictions,['solid','striped'])
    def test_rejects_duplicate_or_unsupported_prediction(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):aligned_validation(self.rows,self.predictions*2,['solid','striped'])
        self.predictions[0]['adapted']='floral'
        with self.assertRaisesRegex(ValueError,'labels differ'):aligned_validation(self.rows,self.predictions,['solid','striped'])
    def test_checkpoint_checksum_and_revision_are_verified(self):
        from local_attributes import taxonomy_hash
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weights=b'unit fixture, not executable weights'
            (root/'pattern_head.safetensors').write_bytes(weights)
            report={'model':MODEL_ID,'revision':MODEL_REVISION,'taxonomy_sha256':taxonomy_hash(),
                'embedding_dim':384,'labels':['solid','striped'],'checkpoint_sha256':hashlib.sha256(weights).hexdigest()}
            (root/'report.json').write_text(json.dumps(report))
            self.assertEqual(validate_run(root)['labels'],['solid','striped'])
            (root/'pattern_head.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):validate_run(root)
            (root/'pattern_head.safetensors').write_bytes(weights);report['revision']='unverified'
            (root/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'mismatch'):validate_run(root)
