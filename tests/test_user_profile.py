import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'backend'))
from user_profile import validate_profile
from engine import recommend

class ProfileTests(unittest.TestCase):
    def test_inch_measurements_are_converted_once(self):
        profile=validate_profile(dict(unit='in',measurements={'waist':30},colours=[' Blue ','blue']))
        self.assertEqual(profile['measurements_cm']['waist'],76.2)
        self.assertEqual(profile['colours'],['blue'])
        result=recommend([],{'user_profile':profile})
        self.assertTrue(result['profile_context']['measurements_provided'])
        self.assertFalse(result['profile_context']['fit_assessed'])
    def test_optional_measurements(self):
        self.assertEqual(validate_profile({})['measurements_cm'],{})
    def test_photo_fields_require_review_and_are_not_measurements(self):
        with self.assertRaises(ValueError):
            validate_profile({'skin_tone_mst':6})
        result=validate_profile({'skin_tone_mst':6,'projected_shoulder_hip_ratio':1.4,'analysis_reviewed':True})
        self.assertEqual(result['skin_tone_mst'],6)
        self.assertEqual(result['measurements_cm'],{})
        self.assertFalse(result['person_analysis_calibrated'])
        with self.assertRaises(ValueError):
            validate_profile({'skin_tone_mst':11,'analysis_reviewed':True})
    def test_invalid_measurements_rejected(self):
        for value in (-1,0,True,float('nan'),float('inf'),'bad'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_profile({'measurements':{'height':value}})
    def test_profile_preferences_feed_ranking_and_explicit_choices_override(self):
        items=[dict(id='a',price_lkr=1000,colours=['blue']),dict(id='b',price_lkr=1000,colours=['red'])]
        profile=validate_profile({'colours':['red']})
        self.assertEqual(recommend(items,{'user_profile':profile})['recommendations'][0]['id'],'b')
        self.assertEqual(recommend(items,{'user_profile':profile,'colours':['blue']})['recommendations'][0]['id'],'a')
