"""Real HTTP hybrid-mode and DINO Grad-CAM checks, without changing user's server."""
import argparse
import base64
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app import Handler,ThreadingHTTPServer

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder.')
    args.output.mkdir(parents=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base='http://127.0.0.1:'+str(server.server_port)
    image='data:image/png;base64,'+base64.b64encode((ROOT/'frontend/assets/demo/dress.png').read_bytes()).decode()
    def request(payload):
        req=Request(base+'/api/analyse-garment',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=600) as response:return json.load(response)
    try:
        with urlopen(base+'/api/health') as response:assert json.load(response)['dinov2_pattern_available']
        baseline=request({'image':image,'classifier_mode':'zero_shot'})
        hybrid=request({'image':image,'classifier_mode':'dinov2_pattern','include_explanation':True,'explanation_attribute':'pattern_type'})
        assert hybrid['attributes']['pattern_type']['model']=='facebook/dinov2-small'
        for attribute in ('garment_type','drape_style','fabric_texture'):
            assert hybrid['attributes'][attribute]==baseline['attributes'][attribute]
        explanation=hybrid['explanation']
        assert explanation['model']=='facebook/dinov2-small'
        assert explanation['patch_grid']==[16,16]
        assert explanation['status'] in ('ok','uninformative')
        for field in ('model_crop','overlay'):
            (args.output/(field+'.png')).write_bytes(base64.b64decode(explanation[field].split(',',1)[1]))
        clip_explanation=request({'image':image,'classifier_mode':'dinov2_pattern','include_explanation':True,'explanation_attribute':'garment_type'})
        assert clip_explanation['explanation']['target_score']=='CLIP image/text pre-softmax similarity logit'
        repeat=request({'image':image,'classifier_mode':'dinov2_pattern'})
        assert repeat['attributes']==hybrid['attributes']
        import attributes
        core=attributes._classifier.dino_encoder.model.encoder
        assert not core.encoder.layer[-1].norm1._forward_hooks
        assert all(parameter.grad is None for parameter in core.parameters())
        metadata={key:value for key,value in explanation.items() if key not in ('overlay','model_crop')}
        (args.output/'checks.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('Hybrid HTTP prediction, genuine DINO Grad-CAM, CLIP fallback, repeat stability and hook/gradient cleanup passed.')
        print('Synthetic demo functional check only. No test-set evaluation or accuracy claim.')
    finally:server.shutdown();server.server_close()
