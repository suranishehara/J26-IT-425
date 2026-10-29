"""Create a frozen catalogue/scenario packet and a blinded local rating page."""
import base64
import hashlib
import json
from pathlib import Path
import random
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from catalogue import validate_catalogue
from storage import load_saved,image_bytes
from engine import recommend
from cultural_guidance import RULES_PATH

SCENARIOS=[
 dict(id='wedding_traditional',description='Wedding in hot weather; budget LKR 10000; prefers ivory and traditional/formal styles.',profile=dict(occasion='wedding',weather='hot',budget=10000,colours=['ivory'],styles=['traditional','formal'])),
 dict(id='office_western',description='Office in hot weather; budget LKR 8000; prefers blue and Western/formal styles.',profile=dict(occasion='office',weather='hot',budget=8000,colours=['blue'],styles=['western','formal'])),
 dict(id='festival_mixed',description='Festival in mild weather; budget LKR 9000; open to traditional and Western options.',profile=dict(occasion='festival',weather='mild',budget=9000,colours=[],styles=[])),
 dict(id='casual_mixed',description='Casual outing in hot weather; budget LKR 6000; prefers blue but no fixed style.',profile=dict(occasion='casual',weather='hot',budget=6000,colours=['blue'],styles=[])),
]
def candidate_image(row):
    url=row.get('image_url','')
    raw=None
    if url.startswith('/assets/'):
        path=(ROOT/'frontend'/url.lstrip('/')).resolve()
        if path.is_relative_to((ROOT/'frontend').resolve()) and path.is_file():
            raw=path.read_bytes()
    elif url.startswith('/api/garment-images/'):
        raw=image_bytes(row['id'])
    return 'data:image/png;base64,'+base64.b64encode(raw).decode() if raw else None

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory.')
    catalogue=validate_catalogue(json.loads((ROOT/'data/garments.json').read_text(encoding='utf-8'))+load_saved())
    snapshots=[]
    for scenario in SCENARIOS:
        rankings={}
        for mode in ('baseline','sri_lanka_guidance'):
            result=recommend(catalogue,{**scenario['profile'],'recommendation_mode':mode,'limit':20})
            rankings[mode]=[row['id'] for row in result['recommendations']]
        pool=[row['id'] for row in catalogue if row['price_lkr']<=scenario['profile']['budget']]
        random.Random(42).shuffle(pool)
        snapshots.append({**scenario,'candidate_order':pool,'rankings':rankings})
    packet=dict(created_at=datetime.now(timezone.utc).isoformat(),scenarios=snapshots,catalogue=catalogue,
        cultural_rules=json.loads(RULES_PATH.read_text(encoding='utf-8')),candidate_images={row['id']:candidate_image(row) for row in catalogue},
        rating_scale={'0':'not relevant','1':'slightly relevant','2':'relevant','3':'highly relevant'},
        note='Constructed pilot scenarios and unvalidated catalogue metadata; ratings must be supplied by users or experts. Model identity/rank order are hidden on the rating page.')
    packet['packet_id']=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()
    template=(ROOT/'scripts/rating_page_template.html').read_text(encoding='utf-8')
    safe=json.dumps(packet).replace('<','\\u003c')
    args.output.mkdir(parents=True)
    (args.output/'packet.json').write_text(json.dumps(packet,indent=2),encoding='utf-8')
    (args.output/'rate_garments.html').write_text(template.replace('/*STUDY_PACKET*/',safe),encoding='utf-8')
    print('Open:',args.output/'rate_garments.html')
    print('Packet:',packet['packet_id'])
