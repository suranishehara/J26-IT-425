"""Non-destructive drape-image inventory; no split assignment or invented reviews."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

STYLES=('nivi','kandyan_osariya')
EXTENSIONS={'.jpg','.jpeg','.jfif','.png','.webp','.bmp'}


def perceptual_hash(image):
    import numpy as np
    from scipy.fft import dctn
    resized=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    block=dctn(resized,norm='ortho')[:8,:8].reshape(-1)
    bits=block[1:]>np.median(block[1:])
    return sum(int(value)<<i for i,value in enumerate(bits))


def inventory(root):
    root=Path(root).resolve();rows=[];failures=[];ignored=[];file_counts={}
    for style in STYLES:
        folder=root/style
        if not folder.is_dir():raise ValueError('Missing candidate folder: '+str(folder))
        files=sorted(folder.iterdir(),key=lambda path:path.name.lower())
        file_counts[style]=sum(path.is_file() for path in files)
        for path in files:
            if not path.is_file():continue
            relative=path.relative_to(root).as_posix()
            if path.suffix.lower() not in EXTENSIONS:ignored.append(relative);continue
            try:
                with Image.open(path) as source:
                    if source.width*source.height>20_000_000:raise ValueError('Image exceeds 20 megapixels.')
                    if getattr(source,'n_frames',1)>1:raise ValueError('Animated/multi-frame input; select one still frame.')
                    image=ImageOps.exif_transpose(source).convert('RGB')
                    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    phash=perceptual_hash(image)
                    rows.append(dict(candidate_id='drape_'+hashlib.sha256(relative.encode()).hexdigest()[:16],
                        image_path=relative,folder_label=style,width=image.width,height=image.height,
                        decoded_sha256=digest,perceptual_hash=hex(phash),
                        label_status='user_folder_label_not_independent_expert_review',source_url=None,
                        permission_status='user_asserted_authorized_2026-10-18',individual_permission_reference=None,
                        source_group=None,split='unassigned'))
            except (OSError,ValueError,Image.DecompressionBombError) as error:
                failures.append({'image_path':relative,'error':str(error)})
    hashes=defaultdict(list)
    for row in rows:hashes[row['decoded_sha256']].append(row)
    exact=[{'images':[row['image_path'] for row in group],
        'cross_label_conflict':len({row['folder_label'] for row in group})>1} for group in hashes.values() if len(group)>1]
    similar=[]
    for i,first in enumerate(rows):
        for second in rows[i+1:]:
            if first['decoded_sha256']==second['decoded_sha256']:continue
            distance=(int(first['perceptual_hash'],16)^int(second['perceptual_hash'],16)).bit_count()
            if distance<=6:similar.append({'first':first['image_path'],'second':second['image_path'],'phash_distance':distance})
    similar.sort(key=lambda row:row['phash_distance'])
    return {'file_counts':file_counts,'readable_counts':dict(Counter(row['folder_label'] for row in rows)),
        'unique_decoded_images':len(hashes),'exact_duplicate_groups':exact,'perceptual_similarity_candidates':similar,
        'failures':failures,'ignored_files':ignored,'records':rows,
        'notice':'No original files modified. Folder labels and authorization are user assertions, not expert/legal verification. Perceptual-hash similarity is only a review aid: it can miss alternate garment views and can flag different garments. No training/validation/test split has been assigned.'}


def contact_sheets(root,rows,output):
    paths=[]
    for style in STYLES:
        subset=[row for row in rows if row['folder_label']==style]
        for page in range((len(subset)+23)//24):
            sheet=Image.new('RGB',(6*190,4*245),'white');draw=ImageDraw.Draw(sheet)
            for index,row in enumerate(subset[page*24:(page+1)*24]):
                x=index%6*190;y=index//6*245
                with Image.open(root/row['image_path']) as source:
                    thumb=ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'),(180,205))
                sheet.paste(thumb,(x+(190-thumb.width)//2,y))
                draw.text((x+5,y+208),str(page*24+index+1)+': '+row['candidate_id'][-8:],fill='black')
                draw.text((x+5,y+225),Path(row['image_path']).name[:26],fill='black')
            path=output/(style+'_page_'+str(page+1)+'.jpg');sheet.save(path,quality=90);paths.append(path.name)
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory.')
    report=inventory(args.input)
    args.output.mkdir(parents=True)
    report['contact_sheets']=contact_sheets(args.input,report['records'],args.output)
    (args.output/'inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('records','perceptual_similarity_candidates')},indent=2))
    print('Perceptual similarity candidates:',len(report['perceptual_similarity_candidates']))
