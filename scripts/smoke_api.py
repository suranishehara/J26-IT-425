"""Exercise the running local server with a public benchmark sample."""
import base64
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'runtime' / 'outputs' / 'catvton_smoke'
payload = {'category': 'upper-body', 'seed': 42}
for name in ('person','garment','mask'):
    payload[name] = 'data:image/png;base64,' + base64.b64encode((folder / f'{name}.png').read_bytes()).decode()
with urlopen('http://127.0.0.1:8000/') as response:
    assert b'local.js' in response.read(), 'Incorrect server on port 8000'
request = Request('http://127.0.0.1:8000/api/try-on', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
with urlopen(request) as response:
    job = json.load(response)
print('Job:', job['jobId'], flush=True)
for _ in range(180):
    with urlopen('http://127.0.0.1:8000/api/jobs/' + job['jobId']) as response:
        state = json.load(response)
    if state['status'] == 'failed':
        raise RuntimeError(state)
    if state['status'] == 'completed':
        with urlopen('http://127.0.0.1:8000' + state['imageUrl']) as response:
            assert response.read().startswith(b'\x89PNG'), 'Output is not a PNG'
        print('HTTP upload, generation, polling and result download passed.', flush=True)
        break
    time.sleep(2)
else:
    raise TimeoutError('Generation has not completed in six minutes.')
