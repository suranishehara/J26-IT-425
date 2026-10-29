"""Re-run an existing local job at higher synthesis resolution without overwriting it."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'runtime' / 'outputs' / 'jobs' / sys.argv[1]
target = ROOT / 'runtime' / 'outputs' / ('quality_' + sys.argv[1] + ('_refined' if len(sys.argv)>2 else ''))
target.mkdir(exist_ok=True)
for name in ('person.png','garment.png','mask.png'):
    shutil.copy2(source / name, target / name)
if len(sys.argv)>2:
    shutil.copy2(Path(sys.argv[2]),target / 'mask.png')
settings = json.loads((source / 'request.json').read_text())
settings['quality'] = 'high'
(target / 'request.json').write_text(json.dumps(settings))
subprocess.run([sys.executable,str(ROOT / 'scripts' / 'catvton_worker.py'),str(target)],check=True)
