"""Reviewed local multi-attribute data contract and optional frozen CLIP heads."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from taxonomy import ATTRIBUTES

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'config/local_attribute_release.json'


def taxonomy_hash():
    return hashlib.sha256(json.dumps(ATTRIBUTES,sort_keys=True).encode()).hexdigest()


def audit_manifest(path,image_root):
    """Read all files for integrity/duplicate auditing; never extract test features."""
    from PIL import Image,ImageOps
    root=Path(image_root).resolve()
    raw=Path(path).read_bytes()
    rows=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not rows:raise ValueError('The annotation manifest is empty.')
    ids=set();groups={};hashes={};counts={};prepared=[]
    for row in rows:
        identifier=row.get('sample_id')
        if not isinstance(identifier,str) or not identifier.strip() or identifier in ids:
            raise ValueError('Every sample needs a unique non-empty sample_id.')
        ids.add(identifier)
        split=row.get('split')
        if split not in ('train','validation','test'):
            raise ValueError(identifier+': assign train, validation or test before training.')
        group=row.get('source_group')
        if row.get('grouping_status')=='provisional':
            raise ValueError(identifier+': confirm same-garment/source grouping before training. Proposed splits are not training-ready.')
        if not isinstance(group,str) or not group.strip():
            raise ValueError(identifier+': identify the physical garment/source group.')
        if group in groups and groups[group]!=split:
            raise ValueError('Source-group leakage across splits: '+group)
        groups[group]=split
        status=row.get('review_status')
        if status not in ('reviewed','provider_annotated') or not isinstance(row.get('annotator'),str) or not row['annotator'].strip():
            raise ValueError(identifier+': record reviewed labels or documented provider annotations and the annotator.')
        if status=='provider_annotated':
            provenance=row.get('label_provenance',{})
            if not isinstance(provenance,dict) or provenance.get('kind')!='provider_annotation' or not provenance.get('reference') or not provenance.get('mapping_version'):
                raise ValueError(identifier+': provider labels require reference and mapping-version provenance.')
        source=row.get('source',{})
        if not isinstance(source,dict) or source.get('permission_confirmed') is not True or not source.get('permission_reference'):
            raise ValueError(identifier+': record permission_confirmed and a permission/license reference.')
        if source.get('image_kind') not in ('real','synthetic'):
            raise ValueError(identifier+': source.image_kind must be real or synthetic.')
        labels={}
        for attribute,vocabulary in ATTRIBUTES.items():
            label=row.get(attribute,'unknown')
            if label not in (*vocabulary,'unknown'):
                raise ValueError(identifier+': unsupported '+attribute+' label.')
            labels[attribute]=label
        if labels['drape_style'] in ('nivi','kandyan_osariya') and labels['garment_type']!='saree':
            raise ValueError(identifier+': saree drape labels require garment_type=saree.')
        if labels['garment_type']=='saree' and labels['drape_style']=='not_applicable':
            raise ValueError(identifier+': a saree cannot have not_applicable drape.')
        if not isinstance(row.get('image_path'),str):raise ValueError(identifier+': image_path is required.')
        image_path=(root/row['image_path']).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(identifier+': image must exist inside --image-root.')
        with Image.open(image_path) as source_image:
            if source_image.width*source_image.height>20_000_000:raise ValueError(identifier+': image exceeds 20 megapixels.')
            decoded=ImageOps.exif_transpose(source_image).convert('RGB')
            digest=hashlib.sha256(str(decoded.size).encode()+decoded.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate decoded image: '+identifier+' and '+hashes[digest]+'. Remove duplicates before training.')
        hashes[digest]=identifier
        prepared.append({**row,**labels,'image_path':str(image_path),'image_sha256':digest})
    for split in ('train','validation','test'):
        subset=[row for row in prepared if row['split']==split]
        counts[split]={'images':len(subset),'image_kinds':dict(Counter(row['source']['image_kind'] for row in subset)),
            'annotation_status':dict(Counter(row['review_status'] for row in subset)),
            'labels':{attribute:dict(Counter(row[attribute] for row in subset)) for attribute in ATTRIBUTES}}
    return prepared,{'manifest_sha256':hashlib.sha256(raw).hexdigest(),'counts':counts,
        'duplicate_check':'Decoded RGB exact duplicates and source-group split overlap; near-duplicate detection is not implemented.',
        'test_feature_extraction':False,'notice':'All image bytes are read only for integrity/duplicate auditing. Training extracts features from train/validation only.'}


def head_plan(rows,attributes,min_train=5,min_validation=2):
    plan={}
    for attribute in attributes:
        if attribute not in ATTRIBUTES:raise ValueError('Unsupported attribute: '+attribute)
        train=Counter(row[attribute] for row in rows if row['split']=='train' and row[attribute]!='unknown')
        validation=Counter(row[attribute] for row in rows if row['split']=='validation' and row[attribute]!='unknown')
        if not train:
            continue
        labels=[label for label in ATTRIBUTES[attribute] if label in train]
        if len(labels)<2:raise ValueError(attribute+': needs at least two reviewed training classes.')
        if set(validation)-set(labels):raise ValueError(attribute+': validation has classes absent from training.')
        for label in labels:
            if train[label]<min_train or validation[label]<min_validation:
                raise ValueError(attribute+'/'+label+': needs at least '+str(min_train)+' training and '+str(min_validation)+' validation images.')
        plan[attribute]={'labels':labels,'train_counts':dict(train),'validation_counts':dict(validation)}
    if not plan:raise ValueError('No reviewed known labels for the requested attributes. Unknown labels are not training targets.')
    return plan


def read_run(directory,revision):
    directory=Path(directory).resolve()
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    if report.get('revision')!=revision or report.get('taxonomy_sha256')!=taxonomy_hash():
        raise ValueError('Local attribute heads do not match this model/taxonomy.')
    checkpoint=directory/'heads.safetensors'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report.get('checkpoint_sha256'):
        raise ValueError('Local attribute checkpoint checksum mismatch.')
    if not report.get('heads'):raise ValueError('No trained local heads in this run.')
    for attribute,details in report['heads'].items():
        if attribute not in ATTRIBUTES or len(details['labels'])<2 or not set(details['labels'])<=set(ATTRIBUTES[attribute]):
            raise ValueError('Invalid local-head label vocabulary.')
    return report


def configured_heads(revision):
    if not RELEASE.is_file():
        raise ValueError('Local attribute heads are not trained/activated yet. Use General CLIP or IndoFashion.')
    release=json.loads(RELEASE.read_text(encoding='utf-8'))
    directory=(ROOT/release['run_directory']).resolve()
    if not directory.is_relative_to(ROOT/'runtime/adaptation'):
        raise ValueError('Local model release must be inside runtime/adaptation.')
    report=read_run(directory,revision)
    if release.get('checkpoint_sha256')!=report['checkpoint_sha256']:
        raise ValueError('Active local-head release checksum mismatch.')
    return directory,report


def available_heads(revision):
    try:
        _,report=configured_heads(revision)
        return {attribute:details['labels'] for attribute,details in report['heads'].items()}
    except (OSError,ValueError,KeyError,TypeError):return {}


def load_heads(device,revision):
    from safetensors.torch import load_file
    import torch
    directory,report=configured_heads(revision)
    weights=load_file(str(directory/'heads.safetensors'),device=device)
    heads={}
    for attribute,details in report['heads'].items():
        head=torch.nn.Linear(report['embedding_dim'],len(details['labels'])).to(device)
        head.load_state_dict({'weight':weights[attribute+'.weight'],'bias':weights[attribute+'.bias']})
        heads[attribute]={'head':head.eval(),'labels':details['labels']}
    return heads,report
