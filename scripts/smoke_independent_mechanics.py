"""Artificial temporary calibration/assessment plumbing; no real evaluation evidence."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from backend.classifier import CATEGORIES
    with tempfile.TemporaryDirectory(prefix='fabric-independent-fixtures-') as temp:
        root = Path(temp); rows = []
        items = [('calibration', 'supported', category) for category in CATEGORIES]
        items += [('assessment', 'supported', category) for category in CATEGORIES]
        items += [('assessment', 'unsupported_fabric', 'linen'), ('assessment', 'non_fabric', 'object')]
        for i, (split, kind, category) in enumerate(items):
            path = root / f'fixture_{i}.png'
            Image.frombytes('RGB', (64, 64), random.Random(i + 100).randbytes(64 * 64 * 3)).save(path)
            rows.append({'id': str(i), 'path': path.name, 'specimen_id': str(i), 'source_group_id': str(i),
                'source_reference': 'ARTIFICIAL UNIT FIXTURE ONLY: NOT REAL RESEARCH DATA',
                'permission_reference': 'temporary generated test pixels', 'reviewer': 'automated-test',
                'image_kind': 'real', 'permission_confirmed': True, 'independence_reviewed': True,
                'region': 'unknown', 'kind': kind, 'category': category, 'split': split,
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'labels': {}})
        manifest = root / 'fixtures.jsonl'; manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        common = ['--manifest', str(manifest), '--image-root', str(root), '--model-id', 'convnext_v2']
        script = str(ROOT / 'scripts/evaluate_research_release.py')
        subprocess.run([sys.executable, script, '--stage', 'calibration', *common, '--output', str(root / 'calibration')], check=True)
        subprocess.run([sys.executable, script, '--stage', 'assessment', *common, '--calibration', str(root / 'calibration/calibration.json'), '--output', str(root / 'assessment')], check=True)
        result = json.loads((root / 'assessment/report.json').read_text())
        assert result['n'] == 9 and result['unknown']['unknown_n'] == 2
        assert result['supported']['classification']['n'] == 7 and result['final_receipt'] is None
        print('PASS: temporary artificial calibration freeze, disjoint assessment, known/unknown metrics and no final inference. All fixture images/results removed; no real calibration or accuracy claim.')


if __name__ == '__main__': main()
