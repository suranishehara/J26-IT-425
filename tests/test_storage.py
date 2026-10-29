import base64
import io
import tempfile
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from PIL import Image
from storage import save_reviewed, load_saved, image_bytes
from engine import recommend

class SavedGarmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / 'catalogue.sqlite3'
        buffer = io.BytesIO()
        Image.new('RGB', (20,30), 'blue').save(buffer, format='PNG')
        self.payload = dict(image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
            confirmed=True, reviewer='test-reviewer', record=dict(name='Floral blouse', family='global',
            price_lkr=2500, price_source='illustrative', garment_type='blouse', pattern_type='floral',
            drape_style='not_applicable', fabric_texture='unknown', textile_techniques=[],
            colours=['blue'], styles=['formal'], occasions=['office'], weather=['hot']))

    def test_saved_record_image_and_review_survive_reopening(self):
        saved = save_reviewed(self.payload, self.database)
        loaded = load_saved(self.database)
        self.assertEqual(loaded, [saved])
        self.assertEqual(loaded[0]['reviewer_id'], 'test-reviewer')
        self.assertTrue(image_bytes(saved['id'], self.database).startswith(b'\x89PNG'))

    def test_size_chart_persists_and_drives_compatibility(self):
        self.payload['record']['size_chart']=[dict(label='M',unit='cm',basis='body_measurements',reference='Fictional test chart',cut='regular',ranges={'chest':[88,96],'waist':[72,80]})]
        saved=save_reviewed(self.payload,self.database)
        profile={'measurements_cm':{'chest':92,'waist':76}}
        result=recommend(load_saved(self.database),{'budget':3000,'user_profile':profile,'only_size_compatible':True})
        self.assertEqual(result['recommendations'][0]['id'],saved['id'])
        self.assertEqual(result['recommendations'][0]['sizing']['matching_sizes'],['M'])

    def test_confirmed_attribute_correction_drives_ranking(self):
        blouse = save_reviewed(self.payload, self.database)
        self.payload['record'].update(name='Plain shirt', garment_type='shirt', pattern_type='solid', price_lkr=2000)
        save_reviewed(self.payload, self.database)
        results = recommend(load_saved(self.database), dict(budget=3000, garment_types=['blouse'], patterns=['floral']))
        self.assertEqual(results['recommendations'][0]['id'], blouse['id'])
        self.assertIn('Matches a preferred pattern: floral',results['recommendations'][0]['reasons'])

    def test_invalid_record_or_image_does_not_persist(self):
        for field,value in [('image','invalid'), ('confirmed',False)]:
            payload = {**self.payload, field:value}
            with self.assertRaises(ValueError):
                save_reviewed(payload,self.database)
        self.assertEqual(load_saved(self.database), [])

    def test_invalid_drape_correction_rejected(self):
        self.payload['record']['drape_style']='kandyan_osariya'
        with self.assertRaises(ValueError):
            save_reviewed(self.payload,self.database)
        self.assertEqual(load_saved(self.database), [])
