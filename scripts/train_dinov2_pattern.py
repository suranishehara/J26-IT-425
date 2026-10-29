"""Train DINOv2-Small pattern probe against the exact frozen CLIP development split."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dinov2_pattern import DinoEncoder,MODEL_ID,MODEL_REVISION,development_rows,aligned_validation
from local_attributes import audit_manifest,head_plan,taxonomy_hash,read_run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--clip-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder.')
    if not 1<=args.batch_size<=64:parser.error('Batch size must be 1–64.')
    from attributes import MODEL_REVISION as CLIP_REVISION
    baseline=read_run(args.clip_run,CLIP_REVISION)
    rows,audit=audit_manifest(args.manifest,args.image_root)
    if baseline['audit']['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Use exactly the CLIP training manifest.')
    frozen={r['sample_id']:r for r in baseline['source_records']}
    for row in rows:
        original=frozen.get(row['sample_id'],{})
        if any(original.get(key)!=row[key] for key in ('split','source_group','image_sha256')):
            raise ValueError('Image/group/split changed since CLIP training: '+row['sample_id'])
    plan=head_plan(rows,['pattern_type'])['pattern_type']
    labels=plan['labels']
    if labels!=baseline['heads']['pattern_type']['labels']:raise ValueError('Candidate labels differ from CLIP run.')
    baseline_predictions=json.loads((args.clip_run/'validation_predictions.json').read_text(encoding='utf-8'))
    aligned=aligned_validation(rows,baseline_predictions,labels)
    selected=development_rows(rows)
    # Require the exact feature sample IDs used by the CLIP experiment.
    import numpy as np
    with np.load(args.clip_run/'train_validation_features.npz',allow_pickle=False) as cache:
        if [r['sample_id'] for r in selected]!=cache['sample_ids'].tolist():
            raise ValueError('Training/validation extraction IDs/order differ from CLIP.')
    import torch
    from safetensors.torch import save_file
    from dataset_tools import classification_metrics
    from train_local_attributes import fit_head
    from train_garment_probe import extract
    args.output.mkdir(parents=True)
    encoder=DinoEncoder()
    if encoder.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    vectors,kept,failures=extract(selected,encoder,args.batch_size,args.output,'train_validation')
    if encoder.device=='cuda':torch.cuda.synchronize()
    extraction_seconds=time.perf_counter()-start
    if failures or [r['sample_id'] for r in kept]!=[r['sample_id'] for r in selected]:
        raise ValueError('Incomplete feature extraction; do not compare different sample sets.')
    x=torch.from_numpy(vectors).float().cpu()
    train_ids=[i for i,r in enumerate(kept) if r['split']=='train']
    validation_ids=[i for i,r in enumerate(kept) if r['split']=='validation']
    y=torch.tensor([labels.index(kept[i]['pattern_type']) for i in train_ids])
    vy=torch.tensor([labels.index(kept[i]['pattern_type']) for i in validation_ids])
    # Match CLIP's seed, epochs, head optimizer and validation-loss selection.
    head,history,best_epoch=fit_head(x[train_ids],y,x[validation_ids],vy,len(labels),baseline['epochs'],baseline['seed'])
    with torch.no_grad():predicted=[labels[i] for i in head(x[validation_ids]).argmax(-1).tolist()]
    truth=[kept[i]['pattern_type'] for i in validation_ids]
    comparisons={
        'clip_matched_zero_shot':classification_metrics((truth[j],aligned[kept[i]['sample_id']]['baseline']) for j,i in enumerate(validation_ids)),
        'clip_linear_head':classification_metrics((truth[j],aligned[kept[i]['sample_id']]['adapted']) for j,i in enumerate(validation_ids)),
        'dinov2_small_linear_head':classification_metrics(zip(truth,predicted))}
    predictions=[{'sample_id':kept[i]['sample_id'],'truth':truth[j],'dinov2':predicted[j],
        'clip_zero_shot':aligned[kept[i]['sample_id']]['baseline'],'clip_head':aligned[kept[i]['sample_id']]['adapted']} for j,i in enumerate(validation_ids)]
    checkpoint=args.output/'pattern_head.safetensors'
    save_file({key:value.detach().cpu().contiguous() for key,value in head.state_dict().items()},str(checkpoint))
    report={'created_at':datetime.now(timezone.utc).isoformat(),'model':MODEL_ID,'revision':MODEL_REVISION,
        'taxonomy_sha256':taxonomy_hash(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'embedding_dim':int(x.shape[1]),'labels':labels,'attribute':'pattern_type','training_plan':plan,'audit':audit,
        'seed':baseline['seed'],'epochs':baseline['epochs'],'selected_epoch':best_epoch,
        'method':'Frozen DINOv2-Small normalized 384-dimensional CLS features with trained linear pattern head.',
        'selection':'Lowest validation cross-entropy; identical AdamW lr=.02 weight_decay=.01 and epoch budget as CLIP.',
        'preprocessing':encoder.processor.to_dict(),'device':encoder.device,'batch_size':args.batch_size,
        'feature_extraction_seconds':extraction_seconds,'feature_images_per_second':len(kept)/extraction_seconds,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if encoder.device=='cuda' else None,
        'speed_comparison':'CLIP timing was not recorded in the original experiment; no direct speed superiority claim.',
        'comparison':comparisons,'clip_checkpoint_sha256':baseline['checkpoint_sha256'],
        'clip_report_sha256':hashlib.sha256((args.clip_run/'report.json').read_bytes()).hexdigest(),
        'source_records':baseline['source_records'],'test_features_used':0,
        'limitations':'Development validation used for checkpoint/model selection, not final evaluation. Small minority classes and provider-label noise. Native pretrained preprocessing differs by encoder. Not a fine-tuned backbone or a Sri Lankan drape/texture model. Not activated in application.'}
    for name,value in [('report',report),('training_history',history),('validation_predictions',predictions)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({name:{'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1']} for name,metrics in comparisons.items()},indent=2))
    print('Saved separate experiment:',args.output,'Test feature extraction: 0. Production unchanged.')
