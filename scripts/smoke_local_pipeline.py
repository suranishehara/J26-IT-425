"""Exercise the pipeline with generated test pixels. NO research accuracy claim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output folder.')
    args.output.mkdir(parents=True)
    rows=[]
    for split,count in [('train',5),('validation',2),('test',2)]:
        for label in ('solid','striped'):
            for number in range(count):
                identifier=split+'_'+label+'_'+str(number)
                image=Image.new('RGB',(224,224),(90+len(rows)*3,160,190))
                draw=ImageDraw.Draw(image)
                if label=='striped':
                    for y in range(0,224,20):draw.rectangle((0,y,223,y+8),fill=(20,30,40))
                draw.rectangle((number+1,len(rows)+1,number+4,len(rows)+4),fill='white')
                image.save(args.output/(identifier+'.png'))
                rows.append(dict(sample_id=identifier,image_path=identifier+'.png',source_group=identifier,split=split,
                    garment_type='unknown',pattern_type=label,drape_style='unknown',fabric_texture='unknown',
                    source=dict(image_kind='synthetic',permission_confirmed=True,permission_reference='Program-generated unit/smoke-test pixels, not human/garment photographs'),
                    annotator='synthetic_fixture_generator_NOT_HUMAN',review_status='reviewed',notes='Software fixture only; never research ground truth.'))
    manifest=args.output/'fixture.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    def run(script,*arguments):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    run('train_local_attributes.py','--manifest',manifest.resolve(),'--image-root',args.output.resolve(),
        '--output',(args.output/'run').resolve(),'--attributes','pattern_type','--epochs','5')
    run('evaluate_local_attributes.py','--manifest',manifest.resolve(),'--image-root',args.output.resolve(),
        '--run',(args.output/'run').resolve(),'--output',(args.output/'test_evaluation').resolve(),'--split','test','--final-test')
    print('Synthetic training/evaluation mechanics passed. No production activation and no local-garment accuracy claim.')
