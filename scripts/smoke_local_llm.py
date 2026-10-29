"""Actual local model streaming smoke. Development prompts, not held-out quality scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def post(base,path,payload):
    return urlopen(Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=200)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='qwen3:4b')
    parser.add_argument('--retrieval', choices=['bm25', 'dense', 'hybrid'], default='bm25')
    parser.add_argument('--scenario', choices=['temple', 'black_tie'], default='temple')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new output directory.')
    base='http://127.0.0.1:8003'
    with urlopen(base+'/api/models') as response: inventory=json.load(response)
    identity=next((row for row in inventory['models'] if row['name']==args.model),None)
    if not identity: raise ValueError('Requested local model is not ready.')
    with post(base,'/api/session',{}) as response: session=json.load(response)
    results=[]
    first_prompt = 'What does black tie mean for a wedding invitation?' if args.scenario == 'black_tie' else 'What clothing guidance is available for visiting the Temple of the Tooth?'
    for prompt in (first_prompt, 'Can you explain that more simply and tell me what I still need to check?'):
        started=time.monotonic(); first=None; chunks=0; final=None
        with post(base,'/api/chat/stream',{'session_id':session['session_id'],'expected_revision':session['revision'],'message':prompt,'generation':'ollama','model':args.model,'retrieval':args.retrieval}) as response:
            for line in response:
                if not line.strip(): continue
                event=json.loads(line)
                if event['type']=='error': raise ValueError(event['error'])
                if event['type']=='delta':
                    chunks+=1
                    if first is None: first=time.monotonic()-started
                if event['type']=='done': final=event['data']
        elapsed=time.monotonic()-started
        if not final or final['methods']['generation']!='ollama_grounded_llm' or final['model']!=args.model or chunks<2: raise ValueError('Real streaming/model identity check failed.')
        if final['methods']['retrieval'] != args.retrieval: raise ValueError('Wrong retrieval method used.')
        if '<think>' in final['answer'] or '</think>' in final['answer']: raise ValueError('Reasoning-channel markers leaked into the answer.')
        session['revision']=final['revision']
        results.append({'prompt':prompt,'answer':final['answer'],'chunks':chunks,'first_delta_seconds':first,'elapsed_seconds':elapsed,
                        'citation_check':final['citation_check'],'knowledge_sha256':final['knowledge_sha256'],'methods':final['methods'],
                        'evidence_ids':[row['id'] for row in final['evidence']],
                        'conversation':final.get('conversation'),
                        'embedding_model':next((row['embedding_model'] for row in final['evidence'] if 'embedding_model' in row),None)})
    if args.scenario == 'black_tie' and not results[-1].get('conversation', {}).get('topic_reused'):
        raise ValueError('Follow-up did not retain the original topic.')
    report={'status':'passed','scenario':args.scenario,'recorded_utc':datetime.now(timezone.utc).isoformat(),'model':identity,'turns':results,
            'notice':'Actual local two-turn streaming smoke only. No claim of evaluated cultural accuracy, factual entailment or general latency.'}
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'passed','model':args.model,'turns':len(results),'chunks':[row['chunks'] for row in results],'output':str(args.output)}))


if __name__=='__main__': main()
