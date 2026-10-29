import base64
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from attributes import decode_image
from PIL import Image

class UploadTests(unittest.TestCase):
    def test_rejects_missing_or_unsupported_image(self):
        for payload in ({}, {"image":"file:///private/photo.png"}, []):
            with self.assertRaises(ValueError):
                decode_image(payload)
    def test_valid_image_is_rgb(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (20, 30)).save(buffer, format="PNG")
        image = decode_image({"image":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()})
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (20, 30))
