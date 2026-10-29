"""Inspect extracted official IndoFashion JSON and make a local manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from dataset_tools import map_category

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def records_from(path):
    text=path.read_text(encoding='utf-8-sig')
    try:
        data=json.loads(text)
    except json.JSONDecodeError:
        # Official *_data.json files contain one JSON object per line.
        data=[json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data,dict):
        if 'image_path' in data and 'class_label' in data:
            return [data]
        for key in ('data','images','samples'):
            if isinstance(data.get(key),list):
                return data[key]
        if all(isinstance(value,dict) for value in data.values()):
            return list(data.values())
    if isinstance(data,list):
        return data
    raise ValueError('Unsupported annotation layout: '+str(path))

def clean_duplicates(records):
    priority={'test':0,'validation':1,'train':2}
    labels={}
    for row in records:
        labels.setdefault(row['source_group'],set()).add(row['garment_type'])
    conflicts={key for key,value in labels.items() if len(value)>1}
    kept={};removed=[]
    for row in sorted(records,key=lambda row:priority[row['split']]):
        key=row['source_group']
        if key in conflicts:
            removed.append({'sample_id':row['sample_id'],'split':row['split'],'reason':'conflicting_labels_for_identical_image'})
        elif key in kept:
            removed.append({'sample_id':row['sample_id'],'split':row['split'],'reason':'exact_duplicate','retained_sample_id':kept[key]['sample_id']})
        else:
            kept[key]=row
    return list(kept.values()),removed

def prepare(dataset_root, images_root, output, deduplicate=False):
    if (output/'manifest.jsonl').exists() or (output/'import_report.json').exists():
        raise ValueError('Import output already exists. Use a new --output directory to preserve previous records.')
    output.mkdir(parents=True,exist_ok=True)
    manifest=[];missing=[];excluded=Counter();counts=Counter();hash_splits={}
    annotations={}
    for split,names in [('train',['train_data.json','train.json']),('validation',['val_data.json','val.json','valid.json','validation.json']),('test',['test_data.json','test.json'])]:
        choices=[dataset_root/name for name in names if (dataset_root/name).is_file()]
        if not choices:
            continue
        annotation=choices[0]
        annotations[split]={'path':str(annotation),'sha256':checksum(annotation)}
        for index,row in enumerate(records_from(annotation)):
            if not isinstance(row,dict):
                raise ValueError('Annotation record is not an object.')
            source_label=row.get('class_label')
            label=map_category(source_label)
            if label is None:
                excluded[str(source_label)]+=1
                continue
            raw_path=row.get('image_path')
            if not isinstance(raw_path,str) or not raw_path.strip():
                missing.append({'split':split,'index':index,'reason':'No image_path'})
                continue
            path=(images_root/Path(raw_path.replace('\\','/'))).resolve()
            if not path.is_relative_to(images_root):
                raise ValueError('Image path escapes the configured images root: '+raw_path)
            if not path.is_file():
                missing.append({'split':split,'image_path':raw_path})
                continue
            digest=checksum(path)
            hash_splits.setdefault(digest,set()).add(split)
            counts[split+':'+label]+=1
            manifest.append(dict(sample_id='indofashion-'+split+'-'+str(index),image_path=str(path),
                split=split,garment_type=label,source_label=source_label,source_group='sha256:'+digest,
                pattern_type='unknown',drape_style='unknown',fabric_texture='unknown',
                source_dataset='IndoFashion',label_source='original_category_annotation'))
    overlaps=[{'sha256':key,'splits':sorted(value)} for key,value in hash_splits.items() if len(value)>1]
    removed=[]
    original_counts=dict(counts)
    if deduplicate:
        manifest,removed=clean_duplicates(manifest)
        counts=Counter(row['split']+':'+row['garment_type'] for row in manifest)
    report=dict(annotations=annotations,counts=dict(counts),original_counts=original_counts,excluded_categories=dict(excluded),
                missing_files=missing,exact_duplicate_cross_split=overlaps,
                deduplicated=deduplicate,removed_records=removed,
                duplicate_policy='Keep test, then validation, then train; remove all identical images with conflicting mapped labels.' if deduplicate else 'Audit only; block cross-split duplicates.',
                note='Content hashes detect exact duplicates only, not near-duplicates or multiple photos of the same garment. Local drape/texture labels are not inferred.')
    (output/'import_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not annotations or not manifest:
        raise ValueError('No compatible extracted records found. Check annotation and image paths; see import_report.json.')
    if missing or (overlaps and not deduplicate):
        raise ValueError('Missing images or cross-split exact duplicates found. Resolve these before evaluation; see import_report.json.')
    target=output/'manifest.jsonl'
    if target.exists():
        raise ValueError('Manifest already exists. Use a new --output directory to preserve the previous import.')
    with target.open('w',encoding='utf-8') as file:
        for row in manifest:
            file.write(json.dumps(row)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Directory containing train_data.json, val_data.json and/or test_data.json (short filenames also supported)')
    parser.add_argument('--images-root',type=Path,help='Base directory for image_path values; defaults to --dataset')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'runtime'/'datasets'/'indofashion')
    parser.add_argument('--deduplicate',action='store_true',help='Write a derived manifest without exact duplicates; preserve test over validation over train. Raw files are unchanged.')
    args=parser.parse_args()
    try:
        report=prepare(args.dataset.resolve(),(args.images_root or args.dataset).resolve(),args.output.resolve(),args.deduplicate)
        print('Prepared manifest:',args.output/'manifest.jsonl')
        print(json.dumps(report['counts'],indent=2))
    except ValueError as error:
        parser.exit(1,str(error)+'\n')
