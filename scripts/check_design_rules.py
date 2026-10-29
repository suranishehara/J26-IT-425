"""Read-only rule validation; does not claim expert approval."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from backend.design_rules import load
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path)
    args = parser.parse_args()
    rules = load(args.path)
    print(json.dumps({'valid': True, 'version': rules['rule_version'], 'sha256': rules['sha256'],
                      'review': rules['review'], 'design_count': len(rules['designs'])}, indent=2))
