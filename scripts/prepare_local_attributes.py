"""Create annotation starter records, never generated ground-truth labels."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from taxonomy import ATTRIBUTES

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new annotation file.')
    rows=[]
    for path in sorted((ROOT/'frontend/assets/demo').glob('*.png')):
        rows.append(dict(sample_id='demo_'+path.stem,image_path=path.relative_to(ROOT).as_posix(),
            source_group='demo_'+path.stem,split='unassigned',**{attribute:'unknown' for attribute in ATTRIBUTES},
            source=dict(image_kind='synthetic',permission_confirmed=False,permission_reference=None),
            annotator=None,review_status='unreviewed',notes='Synthetic starter only. Review the image, labels and usage rights; not a real-world evaluation set.'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    print('Created',len(rows),'unreviewed starter records:',args.output)
    print('They are intentionally not training-ready. Use --image-root . from the project folder.')
