"""Import a bounded DeepFashion pilot without changing raw images/annotations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from deepfashion_import import PROVIDER,MAPPING_VERSION,PATTERNS,TEXTURES,REGIONS,read_labels,source_group,group_split,region_crop


def safe_file(root,name,suffix=None):
    path=(root/Path(name).with_suffix(suffix) if suffix else root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('File escapes dataset directory.')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--parsing',type=Path,required=True)
    parser.add_argument('--mask-suffix',choices=['','_segm'],default='',help='Filename suffix before .png; downloaded segm files use _segm.')
    parser.add_argument('--pattern-labels',type=Path,required=True)
    parser.add_argument('--fabric-labels',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regions',nargs='+',choices=list(REGIONS),default=['upper'])
    parser.add_argument('--max-images',type=int,default=500,help='Pilot limit on raw images inspected, not all 44096 images.')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--assign-group-splits',action='store_true',help='Create reproducible 80/10/10 garment-group splits; not official benchmark splits.')
    parser.add_argument('--confirm-permission',action='store_true',help='Use only after YOU verify eligibility under the dataset research agreement.')
    parser.add_argument('--permission-reference',default='')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder; existing datasets are never overwritten.')
    if args.max_images<1:parser.error('--max-images must be positive.')
    if args.confirm_permission and not args.permission_reference.strip():parser.error('Record --permission-reference when confirming permission.')
    if not args.images.is_dir() or not args.parsing.is_dir():parser.error('Extract image and parsing folders first.')
    patterns=read_labels(args.pattern_labels)
    fabrics=read_labels(args.fabric_labels) if args.fabric_labels else {}
    # Stable seeded order avoids taking only one filename/category prefix for the pilot.
    names=sorted(patterns,key=lambda name:hashlib.sha256((str(args.seed)+'|'+name).encode()).hexdigest())[:args.max_images]
    from PIL import Image
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    records=[];skipped=Counter();failures=[];seen=set()
    for name in names:
        try:
            image_path=safe_file(args.images,name)
            mask_name=Path(name).with_name(Path(name).stem+args.mask_suffix+'.png').as_posix()
            mask_path=safe_file(args.parsing,mask_name)
            if not image_path.is_file():skipped['missing_image']+=1;continue
            if not mask_path.is_file():skipped['missing_parsing']+=1;continue
            group=source_group(name)
            with Image.open(image_path) as image,Image.open(mask_path) as mask:
                for region in dict.fromkeys(args.regions):
                    index=REGIONS[region][0]
                    pattern=PATTERNS.get(patterns[name][index],'unknown')
                    texture=TEXTURES.get(fabrics.get(name,[7,7,7])[index],'unknown')
                    if pattern==texture=='unknown':skipped['no_compatible_label']+=1;continue
                    crop=region_crop(image,mask,region)
                    if crop is None:skipped['small_missing_or_dress_region']+=1;continue
                    cropped,box,area=crop
                    digest=hashlib.sha256(str(cropped.size).encode()+cropped.tobytes()).hexdigest()
                    if digest in seen:skipped['duplicate_crop']+=1;continue
                    seen.add(digest)
                    identifier='dfmm_'+hashlib.sha256((name+'|'+region).encode()).hexdigest()[:24]
                    cropped.save(args.output/'images'/(identifier+'.png'))
                    records.append(dict(sample_id=identifier,image_path='images/'+identifier+'.png',source_group=group,
                        split=group_split(group,args.seed) if args.assign_group_splits else 'unassigned',
                        garment_type='unknown',pattern_type=pattern,drape_style='unknown',fabric_texture=texture,
                        annotator='DeepFashion-MultiModal original dataset annotators',review_status='provider_annotated',
                        source=dict(image_kind='real',dataset='DeepFashion-MultiModal',url=PROVIDER,
                            permission_confirmed=args.confirm_permission,permission_reference=args.permission_reference or None,
                            raw_image=name,region=region,mask_bbox=box,mask_pixels=area),
                        label_provenance=dict(kind='provider_annotation',reference=PROVIDER,mapping_version=MAPPING_VERSION,
                            original_pattern_code=patterns[name][index],original_fabric_code=fabrics.get(name,[7,7,7])[index]),
                        notes='Automatic conservative mapping of provider labels, NOT new independent human review; inspect crops/labels before research use.'))
        except (ValueError,OSError) as error:
            failures.append({'image':name,'error':str(error)})
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    report=dict(provider=PROVIDER,mapping_version=MAPPING_VERSION,mask_suffix=args.mask_suffix,raw_images_inspected=len(names),crops=len(records),skipped=dict(skipped),failures=failures,
        split_counts=dict(Counter(r['split'] for r in records)),pattern_counts=dict(Counter(r['pattern_type'] for r in records)),
        texture_counts=dict(Counter(r['fabric_texture'] for r in records)),
        annotation_sha256={'pattern':hashlib.sha256(args.pattern_labels.read_bytes()).hexdigest(),
            'fabric':hashlib.sha256(args.fabric_labels.read_bytes()).hexdigest() if args.fabric_labels else None},
        split_method='Seeded garment-group hash 80/10/10, not official benchmark split' if args.assign_group_splits else 'Unassigned; assign garment groups before training',
        limitations='Only images with available parsing are usable. No Sri Lankan drape labels. Knitted is the sole direct texture mapping and alone cannot train a multi-class texture head. Pilot selection is not class-balanced. Near duplicates and source identities require further review. Research-only dataset: do not publish images or derived crops.')
    (args.output/'import_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not records:raise SystemExit('No crops imported. Inspect paths, annotation names and import_report.json; do not train.')
    print('Raw files unchanged. Audit the manifest with --image-root',args.output)

if __name__=='__main__':main()
