"""Exercise the optional drape mode and Grad-CAM via temporary local HTTP."""
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
    if args.output.exists():parser.error('Choose a new output directory.')
    rows=[json.loads(line) for line in (ROOT/'runtime/datasets/drape_reviewed_01/manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    source=next(row for row in rows if row['image_path']=='nivi/10.jpg')
    assert source['split']=='train','Smoke example must not be from reserved test images.'
    image='data:image/jpeg;base64,'+base64.b64encode((ROOT/'data/local_attributes/drape_candidates/real'/source['image_path']).read_bytes()).decode()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base='http://127.0.0.1:'+str(server.server_port)
    def request(image,mode,explain=False):
        payload={'image':image,'classifier_mode':mode,'include_explanation':explain,'explanation_attribute':'drape_style'}
        req=Request(base+'/api/analyse-garment',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=600) as response:return json.load(response)
    try:
        with urlopen(base+'/api/health') as response:health=json.load(response)
        assert health['local_attributes_configured']
        assert health['local_attribute_labels']['drape_style']==['nivi','kandyan_osariya']
        baseline=request(image,'zero_shot')
        result=request(image,'local_attributes',True)
        for attribute in ('garment_type','pattern_type','fabric_texture'):
            assert baseline['attributes'][attribute]==result['attributes'][attribute]
        assert result['attributes']['drape_style']['suggested_label'] in ('nivi','kandyan_osariya')
        explanation=result['explanation']
        assert explanation['target_score']=='Reviewed local attribute head pre-softmax logit'
        assert explanation['patch_grid']==[7,7]
        assert 'trained Sri Lankan drape head' in explanation['notice']
        assert explanation['status'] in ('ok','uninformative')
        dress='data:image/png;base64,'+base64.b64encode((ROOT/'frontend/assets/demo/dress.png').read_bytes()).decode()
        not_saree=request(dress,'local_attributes',True)
        assert not_saree['attributes']['drape_style']['suggested_label']=='not_applicable'
        assert not_saree['explanation']['status']=='not_applicable'
        repeat=request(image,'local_attributes')
        assert repeat['attributes']==result['attributes']
        args.output.mkdir(parents=True)
        for field in ('model_crop','overlay'):
            (args.output/(field+'.png')).write_bytes(base64.b64decode(explanation[field].split(',',1)[1]))
        metadata={key:value for key,value in explanation.items() if key not in ('overlay','model_crop')}
        metadata.update(smoke_sample=source['image_path'],sample_split='train',not_saree_gate='passed',test_features_used=0)
        (args.output/'checks.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('Drape HTTP prediction, trained-head Grad-CAM, unchanged CLIP fallback, non-saree gate and repeat check passed.')
        print('Training example used only for functional verification; no new accuracy or final-test claim.')
    finally:server.shutdown();server.server_close()
