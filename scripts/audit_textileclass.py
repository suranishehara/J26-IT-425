"""Inspect raw images without changing them or running a classifier."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.dataset_audit import write_audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT / 'runtime/datasets/raw/Fabric Classification.v2'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/evaluations/textileclass_audit_01'))
    args = parser.parse_args()
    report = write_audit(args.root, args.output)
    print(json.dumps({k: report[k] for k in ('total_images', 'counts', 'structural_checks_pass')}, indent=2))
    for field, stats in report['groups'].items():
        print(field, {key: len(value) if isinstance(value, list) else value for key, value in stats.items()})
    print('Near-pair candidates:', report['near_duplicate_triage']['pair_counts'])
    print('Saved:', args.output)
