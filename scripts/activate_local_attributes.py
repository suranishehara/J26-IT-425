"""Explicitly activate a verified local attribute checkpoint, without replacing CLIP/IndoFashion."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from local_attributes import RELEASE,read_run
from attributes import MODEL_REVISION

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    directory=args.run.resolve()
    if not directory.is_relative_to(ROOT/'runtime/adaptation'):parser.error('Run must be inside runtime/adaptation.')
    if RELEASE.exists():parser.error('A local release is already active. Preserve/review it before replacing it.')
    report=read_run(directory,MODEL_REVISION)
    RELEASE.write_text(json.dumps({'run_directory':directory.relative_to(ROOT).as_posix(),
        'checkpoint_sha256':report['checkpoint_sha256'],'status':'trained_local_heads_requires_external_validation'},indent=2),encoding='utf-8')
    print('Activated:',directory)
    print('Restart the backend. This does not imply external research validation.')
