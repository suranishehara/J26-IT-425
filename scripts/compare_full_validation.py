"""Compare frozen CLIP and the fixed probe on validation or frozen test; never train."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageOps,ImageDraw
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from attributes import AttributeClassifier,MODEL_ID,MODEL_REVISION
from dataset_tools import classification_metrics
from taxonomy import ATTRIBUTES,TAXONOMY_VERSION
from train_garment_probe import extract,LABELS

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_error_sheet(predictions,lookup,output):
    chosen=[]
    for label in LABELS:
        errors=[row for row in predictions if row['truth']==label and row['adapted']!=label]
        chosen.extend(sorted(errors,key=lambda row:-row['adapted_relative_score'])[:3])
    sheet=Image.new('RGB',(1000,330*((len(chosen)+3)//4)), 'white')
    draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(chosen):
        x=index%4*250;y=index//4*330
        with Image.open(lookup[row['sample_id']]['image_path']) as source:
            image=ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'),(240,265))
        sheet.paste(image,(x+(250-image.width)//2,y))
        draw.text((x+4,y+268),'Provider: '+row['truth']+' / predicted: '+row['adapted'],fill='black')
        draw.text((x+4,y+286),'Relative score: '+str(round(row['adapted_relative_score'],3)),fill='black')
        draw.text((x+4,y+304),row['sample_id'].replace('indofashion-validation-','ID ').replace('indofashion-test-','ID '),fill='black')
    if chosen:
        sheet.save(output/'adapted_error_contact_sheet.jpg')
    (output/'review_examples.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')

def make_html(report,output):
    summary=''.join('<tr><td>'+html.escape(name)+'</td><td>'+format(metric['accuracy']*100,'.2f')+'%</td><td>'+format(metric['macro_f1'],'.4f')+'</td></tr>' for name,metric in report['metrics'].items())
    tables=[]
    for name,metric in report['metrics'].items():
        cells=''.join('<tr><td>'+label+'</td><td>'+str(values['support'])+'</td><td>'+format(values['precision'],'.3f')+'</td><td>'+format(values['recall'],'.3f')+'</td><td>'+format(values['f1'],'.3f')+'</td></tr>' for label,values in metric['per_class'].items())
        labels=list(metric['confusion_matrix'])
        matrix='<tr><th>Truth / prediction</th>'+''.join('<th>'+label+'</th>' for label in labels)+'</tr>'
        matrix+=''.join('<tr><th>'+label+'</th>'+''.join('<td>'+str(metric['confusion_matrix'][label][p])+'</td>' for p in labels)+'</tr>' for label in labels)
        tables.append('<h2>'+name+'</h2><table><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'+cells+'</table><h3>Confusion counts</h3><table>'+matrix+'</table>')
    document='<!doctype html><meta charset="utf-8"><title>Full validation comparison</title><style>body{font:15px system-ui;max-width:1100px;margin:35px auto;padding:20px;color:#243e36}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #dce3d7;padding:9px;text-align:left}th{background:#edf2e8}img{max-width:100%}</style><h1>IndoFashion full validation comparison</h1><p>Fixed checkpoint. Provider labels unchanged. Test images read: 0. Garment type only.</p><p>'+str(report['selected'])+' selected; '+str(report['successful'])+' processed.</p><table><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th></tr>'+summary+'</table>'+''.join(tables)+'<h2>Examples requiring review</h2><p>High-score disagreements, up to three per class. These are candidates for manual review; labels have not been changed.</p><img src="adapted_error_contact_sheet.jpg" alt="Adapted classifier disagreements">'
    (output/'comparison.html').write_text(document,encoding='utf-8')
    if report['split']=='test':
        document=document.replace('Full validation comparison','Final frozen test comparison').replace('full validation comparison','final frozen test comparison').replace('Test images read: 0.','Test images evaluated: '+str(report['selected'])+'. No training or tuning performed.')
        (output/'comparison.html').write_text(document,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--probe',type=Path,default=ROOT/'runtime/adaptation/indofashion_probe_01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args()
    if args.output.exists() or args.batch_size<1:
        parser.error('Choose a new output directory and a positive batch size.')
    training_report=json.loads((args.probe/'report.json').read_text())
    if training_report['revision']!=MODEL_REVISION or training_report['labels']!=LABELS:
        raise ValueError('Checkpoint model revision or labels do not match.')
    if digest(args.manifest)!=training_report['manifest_sha256']:
        raise ValueError('Manifest changed since training.')
    all_rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows=[row for row in all_rows if row['split']==args.split]
    if not rows or any(row['garment_type'] not in LABELS for row in rows):
        raise ValueError('Evaluation split must have the supported four-class labels.')
    samples=json.loads((args.probe/'selected_samples.json').read_text())
    used=set(samples['training_sample_ids'])
    train_groups={row['source_group'] for row in all_rows if row['sample_id'] in used}
    if train_groups & {row['source_group'] for row in rows}:
        raise ValueError('Training/validation duplicate detected.')
    before=digest(args.probe/'probe.safetensors')
    if args.split=='test':
        release=json.loads((ROOT/'config/garment_probe_release.json').read_text())
        if before!=release['checkpoint_sha256'] or digest(ROOT/'backend/taxonomy.py')!=release['taxonomy_sha256']:
            raise ValueError('Checkpoint or taxonomy differs from the frozen release.')
        if MODEL_REVISION!=release['revision'] or LABELS!=release['labels']:
            raise ValueError('Frozen model revision or class order changed.')
        development_groups={row['source_group'] for row in all_rows if row['split']!='test'}
        if development_groups & {row['source_group'] for row in rows}:
            raise ValueError('Exact-image overlap between test and development splits.')
    args.output.mkdir(parents=True)
    start=time.monotonic();classifier=AttributeClassifier()
    features,successful,errors=extract(rows,classifier,args.batch_size,args.output,args.split)
    weights=load_file(str(args.probe/'probe.safetensors'),device=classifier.device)
    head=torch.nn.Linear(features.shape[1],len(LABELS)).to(classifier.device)
    head.load_state_dict(weights);head.eval()
    vectors=torch.from_numpy(features).to(classifier.device)
    order=list(ATTRIBUTES['garment_type'])
    with torch.inference_mode():
        logits=classifier.model.logit_scale.exp()*vectors@classifier.text_features['garment_type'].T
        original=[order[i] for i in logits.argmax(-1).cpu().tolist()]
        restricted=logits[:,[order.index(label) for label in LABELS]]
        matched=[LABELS[i] for i in restricted.argmax(-1).cpu().tolist()]
        adapted_scores=head(vectors).softmax(-1).cpu().numpy()
    adapted=[LABELS[i] for i in adapted_scores.argmax(-1)]
    truths=[row['garment_type'] for row in successful]
    predictions=[dict(sample_id=row['sample_id'],truth=row['garment_type'],original=original[i],matched_baseline=matched[i],adapted=adapted[i],adapted_relative_score=float(adapted_scores[i].max())) for i,row in enumerate(successful)]
    metrics={name:classification_metrics(zip(truths,values)) for name,values in [('original_eight_class_zero_shot',original),('matched_four_class_zero_shot',matched),('adapted_four_class_probe',adapted)]}
    pilot_ids=set(samples['validation_sample_ids'])
    cohorts={}
    for name,subset in [('previously_inspected_pilot',[row for row in predictions if row['sample_id'] in pilot_ids]),('remaining_validation',[row for row in predictions if row['sample_id'] not in pilot_ids])]:
        if subset and args.split=='validation':
            cohorts[name]={method:classification_metrics((row['truth'],row[field]) for row in subset) for method,field in [('matched_four_class_zero_shot','matched_baseline'),('adapted_four_class_probe','adapted')]}
    if digest(args.probe/'probe.safetensors')!=before:
        raise ValueError('Checkpoint changed during evaluation.')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),model=MODEL_ID,revision=MODEL_REVISION,taxonomy_version=TAXONOMY_VERSION,
        manifest_sha256=digest(args.manifest),checkpoint_sha256=before,checkpoint_unchanged=True,
        split=args.split,test_images_read=len(rows) if args.split=='test' else 0,training_performed=False,selected=len(rows),successful=len(successful),errors=errors,
        class_counts=dict(Counter(truths)),metrics=metrics,cohorts=cohorts,
        all_selected_accuracy_counting_failures_wrong={key:round(value['accuracy']*len(successful)/len(rows),8) for key,value in metrics.items()},
        paired_changes=dict(fixed=sum(row['matched_baseline']!=row['truth'] and row['adapted']==row['truth'] for row in predictions),regressed=sum(row['matched_baseline']==row['truth'] and row['adapted']!=row['truth'] for row in predictions)),
        elapsed_seconds=round(time.monotonic()-start,2),
        limitations=('Frozen final test comparison; no tuning permitted from these results. ' if args.split=='test' else 'Development validation. ')+'Original provider labels include apparent label noise. Four-class specialization; no fit/skin-tone/local-drape or recommendation claims. Exact duplicates were removed; near-duplicates may remain.')
    (args.output/'predictions.json').write_text(json.dumps(predictions,indent=2))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    make_error_sheet(predictions,{row['sample_id']:row for row in successful},args.output)
    make_html(report,args.output)
    print(json.dumps({name:{'accuracy':metric['accuracy'],'macro_f1':metric['macro_f1']} for name,metric in metrics.items()},indent=2),flush=True)

if __name__=='__main__':
    main()
