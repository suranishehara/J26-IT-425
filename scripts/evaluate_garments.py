"""Evaluate fixed CLIP garment-type predictions on a manifest split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from dataset_tools import classification_metrics

def select_records(rows,split,limit,seed):
    groups=defaultdict(list)
    seen=set()
    for row in rows:
        if row.get('split')!=split:
            continue
        # Avoid counting exact copies repeatedly within a split.
        key=row.get('source_group') or row['sample_id']
        if key in seen:
            continue
        seen.add(key)
        groups[row['garment_type']].append(row)
    chosen=[];rng=random.Random(seed)
    for label in sorted(groups):
        group=sorted(groups[label],key=lambda row:row['sample_id'])
        rng.shuffle(group)
        chosen.extend(group if limit==0 else group[:limit])
    return chosen

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test','train'],default='validation')
    parser.add_argument('--per-class',type=int,default=25,help='0 evaluates all records; default is a balanced pilot')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,required=True,help='New result directory')
    args=parser.parse_args()
    if args.per_class<0:
        parser.error('--per-class must be non-negative.')
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    chosen=select_records(rows,args.split,args.per_class,args.seed)
    if not chosen:
        parser.error('No examples in this split.')
    if args.output.exists():
        parser.error('Output already exists. Choose a new directory to preserve previous results.')
    from PIL import Image,ImageOps
    from attributes import AttributeClassifier,MODEL_ID,MODEL_REVISION
    from taxonomy import TAXONOMY_VERSION
    classifier=AttributeClassifier()
    args.output.mkdir(parents=True)
    pairs=[];errors=[]
    with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as file:
        for index,row in enumerate(chosen,1):
            try:
                with Image.open(row['image_path']) as source:
                    prediction=classifier.predict(ImageOps.exif_transpose(source).convert('RGB'))
                suggested=prediction['attributes']['garment_type']['suggested_label']
                pairs.append((row['garment_type'],suggested))
                result={'sample_id':row['sample_id'],'truth':row['garment_type'],'prediction':prediction}
            except Exception as error:
                result={'sample_id':row['sample_id'],'error':str(error)}
                errors.append(result)
            file.write(json.dumps(result)+'\n');file.flush()
            print(str(index)+'/'+str(len(chosen))+' '+row['sample_id'],flush=True)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'model':MODEL_ID,'revision':MODEL_REVISION,
        'taxonomy_version':TAXONOMY_VERSION,'manifest':str(args.manifest.resolve()),
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'split':args.split,'seed':args.seed,'per_class_limit':args.per_class,
        'selected':len(chosen),'successful':len(pairs),'errors':errors,
        'class_counts':dict(Counter(row['garment_type'] for row in chosen)),
        'metrics_successful_only':classification_metrics(pairs) if pairs else None,
        'accuracy_all_selected_counting_errors_wrong':sum(a==b for a,b in pairs)/len(chosen),
        'scope':'Garment type only. This does not evaluate pattern, fabric texture, Osariya drape, fit, skin tone or recommendation quality.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Report:',args.output/'report.json')
    if errors:
        print('Some images failed. Check coverage and errors before interpreting scores.')
