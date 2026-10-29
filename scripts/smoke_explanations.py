"""Real-model explanation smoke test; not an explanation-quality evaluation."""
import base64
import json
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from attributes import analyse,decode_image

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder.')
    args.output.mkdir(parents=True)
    image='data:image/png;base64,'+base64.b64encode((ROOT/'frontend/assets/demo/dress.png').read_bytes()).decode()
    checks=[]
    for mode in ('zero_shot','indofashion_probe'):
        result=analyse({'image':image,'classifier_mode':mode,'include_explanation':True})
        explanation=result['explanation']
        assert explanation['status']=='ok',explanation
        assert explanation['patch_grid']==[7,7]
        for field in ('model_crop','overlay'):
            decode_image({'image':explanation[field]}).save(args.output/(mode+'_'+field+'.png'))
        checks.append({key:value for key,value in explanation.items() if key not in ('model_crop','overlay')})
        repeat=analyse({'image':image,'classifier_mode':mode})
        assert repeat['attributes']==result['attributes'],'Explanation changed prediction state.'
        print(mode,explanation['label'],explanation['status'])
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Both model paths passed; predictions remain unchanged. No attribution-quality claim.')
