import base64
import io
import subprocess
import tempfile
import threading
from PIL import Image
from .generation import ROOT

LOCK = threading.Lock()

def detect(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a person image and category.')
    category = payload.get('category', 'upper-body')
    if category not in ('upper-body', 'dress', 'saree', 'kandyan-osariya'):
        raise ValueError('Unsupported category.')
    value = payload.get('person', '')
    if not isinstance(value, str) or not value.startswith('data:image/png;base64,'):
        raise ValueError('PNG person image required.')
    image = Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1], validate=True)))
    if image.size not in ((384,512), (768,1024)):
        raise ValueError('Person image must be 384x512 or 768x1024.')
    directory = ROOT / 'runtime' / 'uploads'
    directory.mkdir(parents=True, exist_ok=True)
    with LOCK, tempfile.TemporaryDirectory(dir=directory) as temporary:
        from pathlib import Path
        folder = Path(temporary)
        image.convert('RGB').save(folder / 'person.png')
        try:
            process = subprocess.run([str(ROOT / '.venv' / 'Scripts' / 'python.exe'),
                str(ROOT / 'scripts' / 'detect_clothing.py'), str(folder / 'person.png'),
                str(folder / 'mask.png'), category], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise ValueError('Clothing detection timed out. Check your connection during the first model download, or correct the mask manually.') from None
        if process.returncode:
            raise ValueError('Automatic detection failed. '+process.stderr[-700:])
        mask = base64.b64encode((folder / 'mask.png').read_bytes()).decode()
    return {'mask': 'data:image/png;base64,'+mask,
            'notice': 'Review the highlighted region. Saree and Osariya masks are approximate; correct missing draped areas if needed.'}
