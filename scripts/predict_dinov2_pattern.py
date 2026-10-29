"""Smoke-test a saved experimental DINO pattern checkpoint without activating it."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dinov2_pattern import DinoEncoder,load_pattern_head,predict_pattern

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    args=parser.parse_args()
    from PIL import Image,ImageOps
    encoder=DinoEncoder()
    head,report=load_pattern_head(args.run,encoder.device)
    with Image.open(args.image) as source:image=ImageOps.exif_transpose(source).convert('RGB')
    result=predict_pattern(encoder,head,report,image)
    repeat=predict_pattern(encoder,head,report,image)
    assert result['suggested_label']==repeat['suggested_label']
    assert max(abs(a['relative_score']-b['relative_score']) for a,b in zip(result['alternatives'],repeat['alternatives']))<1e-6
    print(json.dumps(result,indent=2))
    print('Reload/inference/repeat check passed. No application model changed.')
