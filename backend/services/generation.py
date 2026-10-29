"""Validated local uploads and serialized GPU jobs."""
import base64
import io
import json
import subprocess
import threading
import uuid
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / 'runtime' / 'outputs' / 'jobs'
LOCK = threading.Lock()
STATES = {}

def create(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected an image request object.')
    seed = int(payload.get('seed', 42))
    if not 0 <= seed < 2**32:
        raise ValueError('Seed must be between 0 and 4294967295.')
    category = payload.get('category')
    quality = payload.get('quality', 'balanced')
    if quality not in ('balanced', 'high'):
        raise ValueError('Choose balanced or high quality.')
    if category not in ('upper-body', 'dress', 'saree', 'kandyan-osariya'):
        raise ValueError('Choose a supported garment category.')
    images = {}
    for name in ('person', 'garment', 'mask'):
        value = payload.get(name, '')
        if not isinstance(value, str) or not value.startswith('data:image/png;base64,'):
            raise ValueError(f'{name}: PNG image required.')
        raw = base64.b64decode(value.split(',', 1)[1], validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.size not in ((384, 512), (768, 1024)):
            raise ValueError('Images must be 384x512 or 768x1024 pixels.')
        image.load()
        images[name] = image.convert('L' if name == 'mask' else 'RGB')
    if len({im.size for im in images.values()}) != 1:
        raise ValueError('Person, garment and mask dimensions must match.')
    mask = images['mask'].point(lambda p: 255 if p >= 128 else 0)
    coverage = sum(mask.histogram()[128:]) / (mask.width * mask.height)
    if not 0.01 < coverage < 0.9:
        raise ValueError('Paint the clothing replacement area before generating.')
    images['mask'] = mask
    job_id = uuid.uuid4().hex
    folder = JOBS / job_id
    folder.mkdir(parents=True)
    for name, image in images.items():
        image.save(folder / f'{name}.png')
    (folder / 'request.json').write_text(json.dumps({'seed': seed, 'category': category, 'quality': quality}))
    STATES[job_id] = {'jobId': job_id, 'status': 'queued'}
    threading.Thread(target=run, args=(job_id,), daemon=True).start()
    return dict(STATES[job_id])

def run(job_id):
    folder = JOBS / job_id
    with LOCK:
        STATES[job_id]['status'] = 'running'
        try:
            with (folder / 'inference.log').open('w', encoding='utf-8') as log:
                process = subprocess.run([str(ROOT / '.venv' / 'Scripts' / 'python.exe'),
                    str(ROOT / 'scripts' / 'catvton_worker.py'), str(folder)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if process.returncode:
                raise RuntimeError('Generation failed. Check inference.log in this job folder; missing weights, network errors or GPU memory may be responsible.')
            metrics = json.loads((folder / 'metrics.json').read_text())
            STATES[job_id].update(status='completed', imageUrl=f'/results/{job_id}/result.png', metrics=metrics)
        except Exception as error:
            STATES[job_id].update(status='failed', error=str(error))

def status(job_id):
    return dict(STATES.get(job_id, {'status': 'unknown', 'error': 'Job not found; the server may have restarted.'}))
