"""Propose deduplicated garment-group splits without moving/deleting originals."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from audit_drape_candidates import inventory,STYLES


def prepare(records,config,seed=42,labels_confirmed=False,groups_confirmed=False):
    by_path={row['image_path']:row for row in records}
    parents={path:path for path in by_path}
    def find(path):
        if parents[path]!=path:parents[path]=find(parents[path])
        return parents[path]
    def join(group):
        if len(group)<2:raise ValueError('Each grouping override needs at least two image paths.')
        if any(path not in by_path for path in group):raise ValueError('Grouping override refers to a missing image.')
        if len({by_path[path]['folder_label'] for path in group})>1:raise ValueError('Same-photo/garment group crosses folder labels; resolve the labels before splitting.')
        for path in group[1:]:parents[find(path)]=find(group[0])
    exact=defaultdict(list)
    for row in records:exact[row['decoded_sha256']].append(row['image_path'])
    photo_groups=[group for group in exact.values() if len(group)>1]+config.get('same_photo_groups',[])
    for group in photo_groups:join(group)
    photos=defaultdict(list)
    for path in by_path:photos[find(path)].append(path)
    excluded=[];canonical=[]
    for group in photos.values():
        keep=sorted(group,key=lambda path:(-by_path[path]['width']*by_path[path]['height'],path))[0]
        canonical.append(keep)
        for path in group:
            if path!=keep:excluded.append({'excluded':path,'retained':keep,'reason':'Exact/resized same photograph; original retained on disk.'})
    # Related alternate views remain as samples, but share one source group/split.
    for group in config.get('related_garment_groups',[]):join(group)
    grouped=defaultdict(list)
    for path in canonical:grouped[find(path)].append(path)
    split_for={};group_ids={}
    for style in STYLES:
        roots=[root for root,paths in grouped.items() if by_path[paths[0]]['folder_label']==style]
        roots.sort(key=lambda root:hashlib.sha256((str(seed)+'|'+min(grouped[root])).encode()).hexdigest())
        if len(roots)<3:raise ValueError(style+': at least three independent candidate groups are needed for three splits.')
        train_count=min(len(roots)-2,max(1,int(len(roots)*.7)))
        val_count=min(len(roots)-train_count-1,max(1,round(len(roots)*.15)))
        for index,root in enumerate(roots):
            split_for[root]='train' if index<train_count else 'validation' if index<train_count+val_count else 'test'
            group_ids[root]='drape_group_'+hashlib.sha256(min(grouped[root]).encode()).hexdigest()[:16]
    manifest=[]
    for path in sorted(canonical):
        row=by_path[path];root=find(path)
        manifest.append(dict(sample_id=row['candidate_id'],image_path=path,source_group=group_ids[root],
            grouping_status='user_confirmed' if groups_confirmed else 'provisional',split=split_for[root],
            garment_type='saree',drape_style=row['folder_label'],pattern_type='unknown',fabric_texture='unknown',
            annotator='user_supplied_folder_labels',review_status='reviewed' if labels_confirmed else 'unreviewed',
            source=dict(image_kind='real',url=row.get('source_url'),permission_confirmed=True,
                permission_reference='User stated on 2026-10-18 that the downloaded images are authorized; individual license/source references not provided.',
                individual_permission_reference=row.get('individual_permission_reference')),
            notes='Folder-proposed drape label. No independent expert review or legal verification is claimed. Confirm related garment views before training.'))
    counts={split:dict(Counter(row['drape_style'] for row in manifest if row['split']==split)) for split in ('train','validation','test')}
    return manifest,{'original_images':len(records),'retained_images':len(manifest),'excluded_copies':excluded,
        'split_counts':counts,'groups':len(grouped),'labels_confirmed':labels_confirmed,'source_groups_confirmed':groups_confirmed,
        'training_ready':labels_confirmed and groups_confirmed,
        'limitations':'Stratified candidate-group 70/15/15 proposal, not official splits. Default single-image groups may miss alternate views of the same garment. No originals moved/deleted. Not expert-verified labels.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--groups',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--confirm-folder-labels',action='store_true',help='Only after the researcher has checked each category label.')
    parser.add_argument('--confirm-source-groups',action='store_true',help='Only after reviewing same-garment/source grouping.')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory.')
    audit=inventory(args.input)
    if audit['failures']:raise ValueError('Unreadable candidate images; resolve before preparing splits.')
    config=json.loads(args.groups.read_text(encoding='utf-8'))
    rows,report=prepare(audit['records'],config,args.seed,args.confirm_folder_labels,args.confirm_source_groups)
    report.update(created_at=datetime.now(timezone.utc).isoformat(),image_root=str(args.input.resolve()),
        grouping_config_sha256=hashlib.sha256(args.groups.read_bytes()).hexdigest(),
        excluded_bytes_deleted=0,test_features_used=0)
    args.output.mkdir(parents=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    (args.output/'preparation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Use --image-root',args.input,'with this manifest. Training requires label/group confirmation first.')
