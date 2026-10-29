"""Explicit researcher-invoked setup, never imported/called by chat routes."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backend.llm import open_local, models


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen3:4b', 'all-minilm:22m'],default='qwen3:4b')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new setup output directory.')
    seen={}
    with open_local('/api/pull',{'model':args.model,'stream':True},timeout=120) as response:
        for line in response:
            if not line.strip(): continue
            event=json.loads(line)
            if event.get('error'): raise ValueError(event['error'])
            status=event.get('status',''); total=event.get('total',0); completed=event.get('completed',0)
            progress=int(100*completed/total)//10*10 if total else None
            key=event.get('digest',status)
            if key not in seen or seen[key]!=progress:
                seen[key]=progress
                print(json.dumps({'status':status,'percent':progress,'total_bytes':total,'completed_bytes':completed}),flush=True)
    available=models()
    identity=next((row for row in available['models'] if row['name']==args.model),None)
    if args.model == 'all-minilm:22m':
        from backend.semantic import identity as embedding_identity
        identity = embedding_identity()
    if not identity: raise ValueError('Pull ended but selected model is not in the local inventory.')
    with open_local('/api/version') as response: version=json.load(response)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps({'status':'ready','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'provider':'local_ollama','version':version,'model':identity,'notice':'Setup only, not response quality evaluation.'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ready','model':identity}),flush=True)


if __name__=='__main__': main()
