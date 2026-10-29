import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import app
from storage import save_reviewed, load_saved, image_bytes
import test_storage

class CatalogueAPITests(unittest.TestCase):
    def test_save_recommend_and_download_image_over_http(self):
        test_storage.SavedGarmentTests.setUp(self)
        with patch.object(app, 'save_reviewed', side_effect=lambda p: save_reviewed(p,self.database)), \
             patch.object(app, 'load_saved', side_effect=lambda: load_saved(self.database)), \
             patch.object(app, 'image_bytes', side_effect=lambda key: image_bytes(key,self.database)):
            server = app.ThreadingHTTPServer(('127.0.0.1',0), app.Handler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                base='http://127.0.0.1:'+str(server.server_port)
                with urlopen(base+'/api/health') as response:
                    health=json.load(response)
                    self.assertEqual(health['component'],2)
                    self.assertIn('person-analysis',health['features'])
                    self.assertEqual(response.headers['Cache-Control'],'no-store')
                def post(path,payload):
                    request=Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(request) as response:
                        return json.load(response)
                saved=post('/api/catalogue',self.payload)['garment']
                result=post('/api/recommendations',dict(budget=2800,garment_types=['blouse'],patterns=['floral']))
                self.assertEqual(result['recommendations'][0]['id'],saved['id'])
                profile=post('/api/profile',dict(unit='in',measurements={'waist':30},colours=['blue']))['profile']
                self.assertEqual(profile['measurements_cm']['waist'],76.2)
                personalized=post('/api/recommendations',dict(budget=2800,user_profile=profile))
                self.assertTrue(personalized['profile_context']['measurements_provided'])
                self.assertFalse(personalized['profile_context']['fit_assessed'])
                with urlopen(base+saved['image_url']) as response:
                    self.assertEqual(response.headers['Content-Type'],'image/png')
                    self.assertTrue(response.read().startswith(b'\x89PNG'))
                with urlopen(base+'/assets/demo/blouse.png') as response:
                    self.assertEqual(response.status,200)
                    self.assertTrue(response.read().startswith(b'\x89PNG'))
            finally:
                server.shutdown()
                server.server_close()
