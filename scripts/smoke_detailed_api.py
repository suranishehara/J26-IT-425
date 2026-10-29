"""Validate native input detail and quality propagation through the HTTP API."""
import base64
import io
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'external' / 'VITON-HD' / 'datasets' / 'test'
person, garment = (data.parent / 'test_pairs_pilot.txt').read_text().splitlines()[0].split()
def encode(image):
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
def post(path,payload):
    try:
        with urlopen(Request('http://127.0.0.1:8000'+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
payload={'person':encode(Image.open(data/'image'/person).convert('RGB').resize((768,1024))),
         'garment':encode(Image.open(data/'cloth'/garment).convert('RGB').resize((768,1024))),
         'category':'upper-body','quality':'high','seed':42}
payload['mask']=post('/api/detect-mask',payload)['mask']
job=post('/api/try-on',payload)
print('Native-input detailed job:',job['jobId'],flush=True)
for _ in range(150):
    with urlopen('http://127.0.0.1:8000/api/jobs/'+job['jobId']) as response:
        state=json.load(response)
    if state['status']=='failed':
        raise RuntimeError(state)
    if state['status']=='completed':
        assert state['metrics']['resolution']==[768,1024]
        assert state['metrics']['steps']==50
        print(json.dumps(state),flush=True)
        break
    time.sleep(2)
else:
    raise TimeoutError('Detailed inference took over five minutes.')
