"""Train separate local attribute heads on frozen CLIP; never train on test features."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from local_attributes import audit_manifest,head_plan,taxonomy_hash
from taxonomy import ATTRIBUTES


def fit_head(x,y,v,vy,classes,epochs=100,seed=42):
    import torch
    torch.manual_seed(seed)
    head=torch.nn.Linear(x.shape[1],classes)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01)
    best_loss=float('inf');best=None;history=[];best_epoch=None
    for epoch in range(epochs):
        head.train();optimizer.zero_grad()
        loss=torch.nn.functional.cross_entropy(head(x),y)
        loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():validation_loss=torch.nn.functional.cross_entropy(head(v),vy).item()
        history.append({'epoch':epoch+1,'training_loss':loss.item(),'validation_loss':validation_loss})
        if validation_loss<best_loss:
            best_loss=validation_loss;best_epoch=epoch+1
            best={key:value.detach().clone().contiguous() for key,value in head.state_dict().items()}
    head.load_state_dict(best)
    return head.eval(),history,best_epoch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--attributes',nargs='+',choices=list(ATTRIBUTES),default=list(ATTRIBUTES))
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.epochs<=2000 or not 1<=args.batch_size<=128:parser.error('Use 1–2000 epochs and batch size 1–128.')
    rows,audit=audit_manifest(args.manifest,args.image_root)
    plan=head_plan(rows,args.attributes)
    if args.audit_only:
        print(json.dumps({'audit':audit,'training_plan':plan},indent=2));return
    if not args.output or args.output.exists():parser.error('Provide a new --output directory.')
    import torch
    from safetensors.torch import save_file
    from attributes import AttributeClassifier,MODEL_ID,MODEL_REVISION
    from dataset_tools import classification_metrics
    from train_garment_probe import extract
    args.output.mkdir(parents=True)
    classifier=AttributeClassifier()
    # No encoder gradients, fine-tuning, test feature extraction or model changes.
    selected=[row for row in rows if row['split'] in ('train','validation') and any(row[attr]!='unknown' for attr in plan)]
    x,selected,failures=extract(selected,classifier,args.batch_size,args.output,'train_validation')
    if failures:raise ValueError('Image processing failed; inspect input files. No checkpoint was saved.')
    features=torch.from_numpy(x).float().cpu()
    weights={};heads={};histories={};predictions=[]
    for attribute,details in plan.items():
        labels=details['labels']
        train_ids=[i for i,row in enumerate(selected) if row['split']=='train' and row[attribute]!='unknown']
        validation_ids=[i for i,row in enumerate(selected) if row['split']=='validation' and row[attribute]!='unknown']
        y=torch.tensor([labels.index(selected[i][attribute]) for i in train_ids])
        vy=torch.tensor([labels.index(selected[i][attribute]) for i in validation_ids])
        head,history,best_epoch=fit_head(features[train_ids],y,features[validation_ids],vy,len(labels),args.epochs,args.seed)
        with torch.no_grad():
            adapted=head(features[validation_ids]).argmax(-1).tolist()
            order=list(ATTRIBUTES[attribute])
            text=classifier.text_features[attribute].detach().cpu().clone()[[order.index(label) for label in labels]]
            baseline=(features[validation_ids]@text.T).argmax(-1).tolist()
        truth=[selected[i][attribute] for i in validation_ids]
        metrics={'matched_label_zero_shot':classification_metrics(zip(truth,[labels[i] for i in baseline])),
                 'local_head':classification_metrics(zip(truth,[labels[i] for i in adapted]))}
        heads[attribute]={**details,'selected_epoch':best_epoch,'validation_metrics':metrics}
        histories[attribute]=history
        for name,value in head.state_dict().items():weights[attribute+'.'+name]=value.contiguous()
        predictions.extend({'sample_id':selected[i]['sample_id'],'attribute':attribute,'truth':selected[i][attribute],
            'baseline':labels[baseline[j]],'adapted':labels[adapted[j]]} for j,i in enumerate(validation_ids))
        print(attribute,metrics,flush=True)
    save_file(weights,str(args.output/'heads.safetensors'))
    report={'created_at':datetime.now(timezone.utc).isoformat(),'model':MODEL_ID,'revision':MODEL_REVISION,
        'taxonomy_sha256':taxonomy_hash(),'checkpoint_sha256':hashlib.sha256((args.output/'heads.safetensors').read_bytes()).hexdigest(),
        'embedding_dim':int(features.shape[1]),'heads':heads,'audit':audit,'seed':args.seed,'epochs':args.epochs,
        'method':'Independent linear heads on shared frozen normalized CLIP image embeddings; unknown targets masked per head.',
        'selection':'Lowest per-head validation cross-entropy. Validation is development evidence, not independent final performance.',
        'test_features_used':0,'source_records':[{'sample_id':r['sample_id'],'split':r['split'],'source_group':r['source_group'],'image_sha256':r['image_sha256']} for r in rows],
        'limitations':'Only listed trained classes are supported by each head; other attributes fall back to zero-shot CLIP. Synthetic-only results do not establish real Sri Lankan garment performance. Labels, permission and source groups are reviewer assertions.'}
    for name,value in [('report',report),('training_history',histories),('validation_predictions',predictions)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Saved frozen heads:',args.output)

if __name__=='__main__':main()
