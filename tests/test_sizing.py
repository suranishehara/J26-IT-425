import unittest
from sizing import assess_size,validate_sizes
from engine import recommend
from user_profile import validate_profile

class SizingTests(unittest.TestCase):
    def setUp(self):
        self.item={'garment_type':'dress','size_chart':[dict(label='M',unit='cm',basis='body_measurements',reference='Test chart',cut='relaxed',ranges={'chest':[88,96],'waist':[72,80],'hips':[96,104]})]}
        self.profile={'measurements_cm':{'chest':92,'waist':76,'hips':100},'fit_preference':'relaxed'}
    def test_full_chart_matches(self):
        result=assess_size(self.item,self.profile)
        self.assertEqual(result['matching_sizes'],['M'])
        self.assertTrue(result['preferred_cut_match'])
    def test_only_compatible_filter_and_size_bonus(self):
        item={**self.item,'id':'sized','price_lkr':2000}
        unknown={'id':'unknown','garment_type':'dress','price_lkr':1000}
        profile=validate_profile({'measurements':self.profile['measurements_cm'],'fit_preference':'relaxed'})
        result=recommend([unknown,item],{'budget':3000,'user_profile':profile,'only_size_compatible':True})
        self.assertEqual([row['id'] for row in result['recommendations']],['sized'])
        self.assertEqual(result['recommendations'][0]['score'],4)
        self.assertTrue(result['profile_context']['fit_assessed'])
    def test_missing_measurement_is_not_assumed_compatible(self):
        self.profile['measurements_cm'].pop('hips')
        self.assertEqual(assess_size(self.item,self.profile)['status'],'unknown')
    def test_outside_and_unknown_are_different(self):
        self.profile['measurements_cm']['waist']=90
        self.assertEqual(assess_size(self.item,self.profile)['status'],'incompatible')
        self.assertEqual(assess_size({'garment_type':'dress'},self.profile)['status'],'unknown')
    def test_rejects_finished_garment_dimensions_and_invalid_ranges(self):
        self.item['size_chart'][0]['basis']='garment_dimensions'
        with self.assertRaises(ValueError):validate_sizes(self.item['size_chart'])
        self.item['size_chart'][0]['basis']='body_measurements'
        self.item['size_chart'][0]['ranges']['waist']=[80,70]
        with self.assertRaises(ValueError):validate_sizes(self.item['size_chart'])
