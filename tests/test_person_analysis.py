import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from person_math import rgb_to_lab,estimate_skin,pose_proportions

class PersonMathTests(unittest.TestCase):
    def test_consent_is_required_before_models_or_image_processing(self):
        from person_analysis import analyse_person
        for payload in ({}, {'consent':False}, {'image':'data:image/png;base64,AAAA'}):
            with self.assertRaises(ValueError):
                analyse_person(payload)
    def test_lab_black_white(self):
        self.assertAlmostEqual(rgb_to_lab([0,0,0])[0],0)
        self.assertAlmostEqual(rgb_to_lab([255,255,255])[0],100,places=3)
    def test_no_face_does_not_invent_skin_tone(self):
        labels=np.zeros((50,50),dtype=int)
        result,_=estimate_skin(np.zeros((50,50,3),dtype='uint8'),labels,np.ones((50,50)))
        self.assertEqual(result['status'],'unavailable')
        self.assertNotIn('suggested_mst',result)
    def test_colour_is_unvalidated_and_requires_review(self):
        labels=np.full((60,40),11)
        rgb=np.full((60,40,3),[160,126,86],dtype='uint8')
        result,_=estimate_skin(rgb,labels,np.ones((60,40)))
        self.assertEqual(result['suggested_mst'],6)
        self.assertFalse(result['calibrated'])
        self.assertTrue(result['needs_review'])
    def test_occluded_hip_blocks_ratio(self):
        points=[dict(x=.5,y=.5,visibility=1) for _ in range(33)]
        points[23]['visibility']=.2
        self.assertEqual(pose_proportions(points,600,800)['status'],'unavailable')
    def test_ratio_is_image_space_not_centimetres(self):
        points=[dict(x=.5,y=.5,visibility=1) for _ in range(33)]
        for i,x,y in [(11,.3,.3),(12,.7,.3),(23,.4,.6),(24,.6,.6)]:
            points[i].update(x=x,y=y)
        result=pose_proportions(points,600,800)
        self.assertEqual(result['projected_shoulder_hip_ratio'],2)
        self.assertNotIn('measurements_cm',result)
