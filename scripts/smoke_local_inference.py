"""Test local-mode inference/Grad-CAM with a verified run; never activate it."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--image',type=Path,default=ROOT/'frontend/assets/demo/dress.png')
    args=parser.parse_args()
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from attributes import AttributeClassifier,MODEL_REVISION
    from local_attributes import read_run
    from explanations import explain
    report=read_run(args.run,MODEL_REVISION)
    classifier=AttributeClassifier()
    weights=load_file(str(args.run/'heads.safetensors'),device=classifier.device)
    classifier.local_heads={};classifier.local_report=report
    for attribute,details in report['heads'].items():
        head=torch.nn.Linear(report['embedding_dim'],len(details['labels'])).to(classifier.device)
        head.load_state_dict({'weight':weights[attribute+'.weight'],'bias':weights[attribute+'.bias']})
        classifier.local_heads[attribute]={'head':head.eval(),'labels':details['labels']}
    with Image.open(args.image) as source:image=source.convert('RGB')
    baseline=classifier.predict(image)
    local=classifier.predict(image,'local_attributes')
    for attribute in report['heads']:
        assert local['attributes'][attribute]['suggested_label'] in report['heads'][attribute]['labels']
        result=explain(classifier,image,local,attribute)
        assert result['target_score']=='Reviewed local attribute head pre-softmax logit'
        assert result['status'] in ('ok','uninformative')
    for attribute in set(local['attributes'])-set(report['heads']):
        assert local['attributes'][attribute]==baseline['attributes'][attribute]
    assert classifier.predict(image,'local_attributes')['attributes']==local['attributes']
    print('Local head inference, CLIP fallback, Grad-CAM and unchanged repeat predictions passed.')
    print('Software smoke test only. Production release was not created or modified; no accuracy claim.')
