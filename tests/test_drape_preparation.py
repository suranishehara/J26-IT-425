import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_drape_dataset import prepare

class DrapePreparationTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for style in ('nivi','kandyan_osariya'):
            for i in range(10):
                self.rows.append(dict(candidate_id=style+str(i),image_path=style+'/'+str(i)+'.jpg',folder_label=style,
                    width=100,height=200,decoded_sha256=style+str(i)))
    def test_reproducible_stratified_proposal_not_false_review(self):
        one,report=prepare(self.rows,{})
        two,_=prepare(list(reversed(self.rows)),{})
        self.assertEqual(one,two)
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['split_counts']['train'],{'nivi':7,'kandyan_osariya':7})
        self.assertTrue(all(row['grouping_status']=='provisional' and row['review_status']=='unreviewed' for row in one))
    def test_duplicates_excluded_related_views_share_split(self):
        self.rows[1]['decoded_sha256']=self.rows[0]['decoded_sha256']
        rows,report=prepare(self.rows,{'related_garment_groups':[['nivi/2.jpg','nivi/3.jpg']]})
        self.assertEqual(len(rows),19)
        pair=[row for row in rows if row['image_path'] in ('nivi/2.jpg','nivi/3.jpg')]
        self.assertEqual(pair[0]['source_group'],pair[1]['source_group'])
        self.assertEqual(pair[0]['split'],pair[1]['split'])
    def test_cross_label_duplicates_fail(self):
        self.rows[-1]['decoded_sha256']=self.rows[0]['decoded_sha256']
        with self.assertRaisesRegex(ValueError,'crosses folder'):prepare(self.rows,{})
