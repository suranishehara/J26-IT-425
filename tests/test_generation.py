import base64
import io
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from services import generation

def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

class UploadValidationTests(unittest.TestCase):
    def payload(self):
        value = encode(Image.new('RGB', (384,512)))
        return dict(person=value, garment=value, mask=value, category='saree', seed=42)

    def test_empty_mask_cannot_launch_gpu_job(self):
        with self.assertRaisesRegex(ValueError, 'Paint'):
            generation.create(self.payload())

    def test_wrong_dimensions_rejected(self):
        payload = self.payload()
        payload['person'] = encode(Image.new('RGB',(12,12)))
        with self.assertRaisesRegex(ValueError, '384'):
            generation.create(payload)

    def test_invalid_seed_rejected_before_writing(self):
        payload = self.payload()
        payload['seed'] = -1
        with self.assertRaisesRegex(ValueError, 'Seed'):
            generation.create(payload)

    def test_native_upload_preserves_quality_and_input_resolution(self):
        payload=self.payload()
        image=encode(Image.new('RGB',(768,1024),'gray'))
        mask=Image.new('L',(768,1024))
        from PIL import ImageDraw
        ImageDraw.Draw(mask).rectangle((200,200,550,700),fill=255)
        payload.update(person=image,garment=image,mask=encode(mask),quality='high')
        with tempfile.TemporaryDirectory() as temporary, patch.object(generation,'JOBS',Path(temporary)), patch.object(generation.threading.Thread,'start'):
            job=generation.create(payload)
            import json
            folder=Path(temporary)/job['jobId']
            self.assertEqual(json.loads((folder/'request.json').read_text())['quality'],'high')
            with Image.open(folder/'person.png') as saved:
                self.assertEqual(saved.size,(768,1024))
