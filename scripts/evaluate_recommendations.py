"""Score frozen baseline/guided rankings against a complete reviewer export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from ranking_metrics import ranking_metrics

def evaluate(packet,ratings,k):
    copy={key:value for key,value in packet.items() if key!='packet_id'}
    if hashlib.sha256(json.dumps(copy,sort_keys=True).encode()).hexdigest()!=packet['packet_id']:
        raise ValueError('Study packet has changed since it was created.')
    if ratings.get('packet_id')!=packet['packet_id'] or not ratings.get('reviewer_id'):
        raise ValueError('Ratings must identify the matching study packet and reviewer.')
    if not 1<=k<=20:
        raise ValueError('Use K from 1 to 20 (rankings store at most 20 positions).')
    rows=[]
    lookup={row['id']:row for row in packet['catalogue']}
    for scenario in packet['scenarios']:
        grades=ratings.get('judgments',{}).get(scenario['id'],{})
        if set(grades)!=set(scenario['candidate_order']):
            raise ValueError('Every eligible candidate needs a rating: '+scenario['id'])
        for mode,ranking in scenario['rankings'].items():
            metrics=ranking_metrics(ranking,grades,k)
            groups={}
            for family in ('global','traditional'):
                eligible=[identifier for identifier in grades if lookup[identifier]['family']==family]
                relevant=sum(grades[identifier]>=2 for identifier in eligible)
                retrieved=sum(grades[identifier]>=2 for identifier in ranking[:k] if identifier in eligible)
                groups[family]={'relevant_in_pool':relevant,'relevant_retrieved':retrieved,'recall_at_k':retrieved/relevant if relevant else None}
            rows.append(dict(scenario_id=scenario['id'],method=mode,metrics=metrics,family_recall_in_combined_ranking=groups))
    aggregate={}
    for mode in ('baseline','sri_lanka_guidance'):
        aggregate[mode]={}
        for metric in ('precision_at_k','recall_at_k','ndcg_at_k'):
            values=[row['metrics'][metric] for row in rows if row['method']==mode and row['metrics'][metric] is not None]
            aggregate[mode][metric]={'mean':sum(values)/len(values) if values else None,'defined_scenarios':len(values)}
    return dict(packet_id=packet['packet_id'],reviewer_id=ratings['reviewer_id'],k=k,scenarios=rows,aggregate=aggregate,
        note='Per-reviewer scenario averages, not statistical significance. Relevance >=2; graded NDCG uses 2^grade-1. Recall is relative to this complete eligible catalogue pool. No-relevant cases have undefined recall/NDCG.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--ratings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--k',type=int,default=5)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new report path.')
    result=evaluate(json.loads(args.packet.read_text(encoding='utf-8')),json.loads(args.ratings.read_text(encoding='utf-8')),args.k)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['aggregate'],indent=2))
