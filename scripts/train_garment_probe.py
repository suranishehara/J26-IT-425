"""Train a four-class linear probe on frozen CLIP, without using test images."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from attributes import AttributeClassifier,MODEL_ID,MODEL_REVISION
from dataset_tools import classification_metrics
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_garments import select_records

LABELS=['blouse','dress','kurta','saree']

class Images(Dataset):
    def __init__(self,rows,processor):
        self.rows=rows;self.processor=processor
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        try:
            with Image.open(row['image_path']) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
                pixels=self.processor(images=image,return_tensors='pt')['pixel_values'][0]
            return index,pixels,None
        except Exception as error:
            return index,None,str(error)

def collate(batch):
    good=[(index,pixels) for index,pixels,error in batch if error is None]
    bad=[{'index':index,'error':error} for index,pixels,error in batch if error is not None]
    return ([index for index,_ in good],torch.stack([pixels for _,pixels in good]) if good else None,bad)

def extract(rows,classifier,batch_size,output,name):
    vectors=[];indices=[];failures=[]
    loader=DataLoader(Images(rows,classifier.processor),batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    with torch.inference_mode():
        for step,(ids,pixels,bad) in enumerate(loader):
            failures.extend({'sample_id':rows[x['index']]['sample_id'],'error':x['error']} for x in bad)
            if pixels is not None:
                features=classifier.model.get_image_features(pixel_values=pixels.to(classifier.device))
                features=features/features.norm(dim=-1,keepdim=True)
                vectors.append(features.cpu().numpy());indices.extend(ids)
            if step%10==0:
                print(name+': '+str(min((step+1)*batch_size,len(rows)))+'/'+str(len(rows)),flush=True)
    if not vectors:
        raise ValueError('No readable '+name+' images.')
    features=np.concatenate(vectors).astype('float32')
    kept=[rows[index] for index in indices]
    np.savez_compressed(output/(name+'_features.npz'),features=features,sample_ids=np.array([r['sample_id'] for r in kept]))
    return features,kept,failures

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True,help='Existing pilot predictions.jsonl; defines exact validation examples')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-class',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.per_class<10 or args.batch_size<1:
        parser.error('Use at least 10 training examples per class and a positive batch size.')
    if args.output.exists():
        parser.error('Choose a new output directory.')
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    manifest_bytes=args.manifest.read_bytes()
    rows=[json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    baseline=[json.loads(line) for line in args.baseline.read_text().splitlines() if line.strip()]
    lookup={r['sample_id']:r for r in rows}
    validation=[lookup[p['sample_id']] for p in baseline]
    if any(r['split']!='validation' for r in validation):
        raise ValueError('Comparison must contain validation examples only.')
    training=select_records(rows,'train',args.per_class,args.seed)
    if set(r['garment_type'] for r in training)!=set(LABELS):
        raise ValueError('Training requires exactly the four supported IndoFashion classes.')
    if set(r['source_group'] for r in training)&set(r['source_group'] for r in validation):
        raise ValueError('Training/validation duplicate leakage.')
    args.output.mkdir(parents=True)
    classifier=AttributeClassifier()
    start=time.monotonic()
    x,training,train_failures=extract(training,classifier,args.batch_size,args.output,'training')
    v,validation,val_failures=extract(validation,classifier,args.batch_size,args.output,'validation')
    # Fixed 10% internal holdout from the training split; external validation never selects the checkpoint.
    rng=np.random.default_rng(args.seed);fit_ids=[];holdout_ids=[]
    for label in LABELS:
        ids=np.array([i for i,r in enumerate(training) if r['garment_type']==label])
        rng.shuffle(ids);cut=max(1,len(ids)//10)
        holdout_ids.extend(ids[:cut]);fit_ids.extend(ids[cut:])
    device=classifier.device
    features=torch.from_numpy(x).to(device)
    targets=torch.tensor([LABELS.index(r['garment_type']) for r in training],device=device)
    fit_ids=torch.tensor(fit_ids,device=device);holdout_ids=torch.tensor(holdout_ids,device=device)
    head=torch.nn.Linear(x.shape[1],len(LABELS)).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01)
    best_loss=float('inf');best=None;history=[]
    for epoch in range(200):
        head.train();optimizer.zero_grad()
        loss=torch.nn.functional.cross_entropy(head(features[fit_ids]),targets[fit_ids])
        loss.backward();optimizer.step()
        with torch.no_grad():
            holdout_loss=torch.nn.functional.cross_entropy(head(features[holdout_ids]),targets[holdout_ids]).item()
        history.append({'epoch':epoch+1,'fit_loss':loss.item(),'internal_holdout_loss':holdout_loss})
        if holdout_loss<best_loss:
            best_loss=holdout_loss;best={key:value.detach().cpu().clone() for key,value in head.state_dict().items()};best_epoch=epoch+1
    head.load_state_dict(best);head.eval()
    val_features=torch.from_numpy(v).to(device)
    with torch.inference_mode():
        adapted_scores=head(val_features).softmax(-1).cpu().numpy()
        from taxonomy import ATTRIBUTES
        class_order=list(ATTRIBUTES['garment_type'])
        text=classifier.text_features['garment_type'][[class_order.index(label) for label in LABELS]]
        closed_scores=(classifier.model.logit_scale.exp()*val_features@text.T).softmax(-1).cpu().numpy()
    adapted=[LABELS[i] for i in adapted_scores.argmax(-1)]
    closed=[LABELS[i] for i in closed_scores.argmax(-1)]
    truths=[row['garment_type'] for row in validation]
    baseline_map={p['sample_id']:p for p in baseline}
    original=[baseline_map[row['sample_id']]['prediction']['attributes']['garment_type']['suggested_label'] for row in validation]
    metrics={'original_eight_class_zero_shot':classification_metrics(zip(truths,original)),
             'matched_four_class_zero_shot':classification_metrics(zip(truths,closed)),
             'adapted_four_class_probe':classification_metrics(zip(truths,adapted))}
    from safetensors.torch import save_file
    save_file(best,str(args.output/'probe.safetensors'))
    (args.output/'training_history.json').write_text(json.dumps(history,indent=2))
    selected={'training_sample_ids':[row['sample_id'] for row in training],
              'fit_sample_ids':[training[i]['sample_id'] for i in fit_ids.cpu().tolist()],
              'internal_holdout_sample_ids':[training[i]['sample_id'] for i in holdout_ids.cpu().tolist()],
              'validation_sample_ids':[row['sample_id'] for row in validation]}
    (args.output/'selected_samples.json').write_text(json.dumps(selected,indent=2))
    predictions=[dict(sample_id=row['sample_id'],truth=row['garment_type'],original=original[i],
                      matched_baseline=closed[i],adapted=adapted[i],adapted_relative_score=float(adapted_scores[i].max()))
                 for i,row in enumerate(validation)]
    (args.output/'comparison_predictions.json').write_text(json.dumps(predictions,indent=2))
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),model=MODEL_ID,revision=MODEL_REVISION,
        labels=LABELS,method='Frozen normalized CLIP image embeddings plus trained linear classification head',
        training_requested_per_class=args.per_class,training_counts=dict(Counter(r['garment_type'] for r in training)),
        fit_count=len(fit_ids),internal_training_holdout_count=len(holdout_ids),best_epoch=best_epoch,
        seed=args.seed,batch_size=args.batch_size,epochs=200,learning_rate=.02,weight_decay=.01,
        checkpoint_selection='Lowest loss on a stratified 10% holdout taken only from the training split.',
        validation_counts=dict(Counter(truths)),train_failures=train_failures,validation_failures=val_failures,
        test_images_read=0,metrics=metrics,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        elapsed_seconds=round(time.monotonic()-start,2),
        limitations='Four-class IndoFashion category adaptation only. Compare with matched four-class baseline; eight-class baseline is a different candidate-set task. No Osariya, skin-tone, fit or recommendation validation. Pilot validation was previously inspected; final generalization requires the untouched test set.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{'accuracy':value['accuracy'],'macro_f1':value['macro_f1']} for name,value in metrics.items()},indent=2),flush=True)

if __name__=='__main__':
    main()
