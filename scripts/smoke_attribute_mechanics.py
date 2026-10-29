"""Temporary artificial fixtures only. No collected data or real accuracy claim."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='fabric-head-fixtures-') as temp:
        root = Path(temp); rows = []
        for i in range(12):
            generator = random.Random(i)
            image = root / f'fixture_{i}.png'
            Image.frombytes('RGB', (64, 64), generator.randbytes(64 * 64 * 3)).save(image)
            def annotation(value): return {'value': value, 'reference': 'ARTIFICIAL MECHANICS FIXTURE ONLY', 'reviewer': 'automated-test'}
            rows.append({'id': str(i), 'path': image.name, 'specimen_id': str(i), 'source_group_id': str(i),
                'image_kind': 'real', 'permission_confirmed': True, 'independence_reviewed': True,
                'source_reference': 'ARTIFICIAL UNIT FIXTURE: exercises real-record validator, never research data',
                'permission_reference': 'temporary generated test pixels', 'reviewer': 'automated-test',
                'file_sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'split': 'train' if i < 4 else 'validation' if i < 8 else 'assessment',
                'kind': 'supported', 'category': 'cotton', 'region': 'unknown',
                'labels': {'batik': annotation(bool(i % 2)), 'handloom': None,
                           'pattern_type': annotation('solid' if i % 2 else 'striped'),
                           'weave': annotation('plain' if i % 2 else 'twill'),
                           'fabric_texture': annotation('smooth_matte' if i % 2 else 'coarse_woven')}})
        manifest = root / 'fixtures.jsonl'
        manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        subprocess.run([sys.executable, str(ROOT / 'scripts/train_attribute_heads.py'), '--manifest', str(manifest),
                        '--image-root', str(root), '--output', str(root / 'run'), '--epochs', '2'], check=True)
        report = json.loads((root / 'run/report.json').read_text())
        assert report['encoder_frozen_verified']
        assert set(report['heads']) == {'batik', 'pattern_type', 'weave', 'fabric_texture'}
        assert 'handloom' in report['skipped_heads']
        assert not report['calibration_assessment_final_inference_performed']
        subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_attribute_heads.py'), '--manifest', str(manifest),
                        '--image-root', str(root), '--run', str(root / 'run'), '--output', str(root / 'assessment')], check=True)
        assessment = json.loads((root / 'assessment/report.json').read_text())
        assert assessment['n'] == 4 and assessment['split'] == 'assessment'
        print('PASS: temporary fixture extraction, masked head training, unknown-head skipping, frozen encoder, reload and disjoint assessment mechanics. No real attribute-accuracy claim; fixture images/weights removed with temporary directory.')


if __name__ == '__main__': main()
