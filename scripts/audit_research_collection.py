"""Validate collected data and label coverage; never invent or auto-label records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from backend.research_data import load_manifest, provider_hashes, vocabulary, value, sha, SPLITS
    output = Path(args.output)
    if output.exists(): raise ValueError('Choose a new audit output directory.')
    rows = load_manifest(args.manifest, args.image_root, provider_hashes())
    coverage = {split: {head: dict(Counter(str(value(row, head)) for row in rows if row['split'] == split and value(row, head) is not None)) for head in vocabulary()} for split in SPLITS}
    report = {'status': 'schema_audited_not_independently_certified', 'manifest_sha256': sha(args.manifest),
              'counts': dict(Counter(row['split'] for row in rows)), 'label_coverage': coverage,
              'specimens': len({row['specimen_id'] for row in rows}),
              'notice': 'Permission, label and independence assertions require human verification. Exact/decoded duplicates checked; near-duplicates and unrecorded shared sources can remain. No classifier inference performed.'}
    output.mkdir(parents=True); (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
