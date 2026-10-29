"""Compare frozen local heads and matched-label CLIP on a reviewed held-out split."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from local_attributes import audit_manifest,read_run
from taxonomy import ATTRIBUTES

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--final-test',action='store_true',help='Explicitly authorize final test feature extraction.')
    args=parser.parse_args()
    if args.split=='test' and not args.final_test:parser.error('Final test evaluation requires --final-test. Do not tune on these results.')
    if args.output.exists():parser.error('Choose a new output directory.')
    import torch
    from safetensors.torch import load_file
    from attributes import AttributeClassifier,MODEL_REVISION
    from dataset_tools import classification_metrics
    from train_garment_probe import extract
    report=read_run(args.run,MODEL_REVISION)
    rows,audit=audit_manifest(args.manifest,args.image_root)
    if report['audit']['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Use the frozen training manifest. New benchmark data requires a separately specified study.')
    frozen={row['sample_id']:row for row in report['source_records']}
    for row in rows:
        original=frozen.get(row['sample_id'],{})
        if any(original.get(key)!=row[key] for key in ('split','source_group','image_sha256')):
            raise ValueError('An image or split/group changed after training: '+row['sample_id'])
    selected=[r for r in rows if r['split']==args.split and any(r[a]!='unknown' for a in report['heads'])]
    if not selected:raise ValueError('No labelled images in the selected split.')
    args.output.mkdir(parents=True)
    classifier=AttributeClassifier()
    x,selected,failures=extract(selected,classifier,16,args.output,args.split)
    if failures:raise ValueError('Input processing failed; no successful evaluation report saved.')
    features=torch.from_numpy(x).float()
    weights=load_file(str(args.run/'heads.safetensors'))
    results={};predictions=[]
    for attribute,details in report['heads'].items():
        labels=details['labels']
        ids=[i for i,row in enumerate(selected) if row[attribute]!='unknown']
        if not ids:
            results[attribute]={'status':'no_known_labels'};continue
        with torch.no_grad():
            scores=torch.nn.functional.linear(features[ids],weights[attribute+'.weight'],weights[attribute+'.bias'])
            adapted=[labels[i] for i in scores.argmax(-1).tolist()]
            order=list(ATTRIBUTES[attribute])
            text=classifier.text_features[attribute].detach().cpu().clone()[[order.index(label) for label in labels]]
            baseline=[labels[i] for i in (features[ids]@text.T).argmax(-1).tolist()]
        truths=[selected[i][attribute] for i in ids]
        results[attribute]={'supported_labels':labels,'out_of_scope_truth_count':sum(t not in labels for t in truths),
            'local_head':classification_metrics(zip(truths,adapted)),
            'matched_label_zero_shot':classification_metrics(zip(truths,baseline))}
        predictions.extend({'sample_id':selected[i]['sample_id'],'attribute':attribute,'truth':truths[j],
            'adapted':adapted[j],'baseline':baseline[j]} for j,i in enumerate(ids))
    output={'split':args.split,'checkpoint_sha256':report['checkpoint_sha256'],'manifest_sha256':audit['manifest_sha256'],
        'metrics':results,'image_kinds':audit['counts'][args.split]['image_kinds'],
        'notice':'Matched candidate-label comparison; unsupported true labels count as errors. Real and synthetic image counts are reported. Do not tune on final test results.'}
    (args.output/'report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    (args.output/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
