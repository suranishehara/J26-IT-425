"""Conservative DeepFashion-MultiModal label and parsing conversion."""
import hashlib
from pathlib import Path
import re

PROVIDER='https://github.com/yumingj/DeepFashion-MultiModal'
MAPPING_VERSION='deepfashion_multimodal_1'
PATTERNS={0:'floral',2:'striped',3:'solid',4:'checked'}
# Cotton/chiffon/denim are not verified smoothness, shine or weave labels.
TEXTURES={4:'knitted'}
REGIONS={'upper':(0,(1,)),'lower':(1,(3,5,6)),'outer':(2,(2,))}


def read_labels(path):
    """Accept exact filename + three integer region labels; no guessed headers."""
    rows={}
    for number,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=4:raise ValueError(str(path)+': line '+str(number)+' must have filename and three labels.')
        name=fields[0].replace('\\','/')
        if name.startswith('/') or '..' in name.split('/') or ':' in name:
            raise ValueError('Unsafe annotation filename: '+name)
        try:values=[int(value) for value in fields[1:]]
        except ValueError:raise ValueError('Non-integer annotation at line '+str(number)) from None
        if any(value not in range(8) for value in values):raise ValueError('Annotation code must be from 0 to 7.')
        if name in rows:raise ValueError('Duplicate annotation filename: '+name)
        rows[name]=values
    if not rows:raise ValueError('No annotation rows found.')
    return rows


def source_group(name):
    # Keep category/gender prefix: id numbers alone need not be globally unique.
    normal=name.replace('\\','/')
    match=re.match(r'^(.*?id_\d+)(?:[-_/])',normal)
    if not match:raise ValueError('Cannot identify a garment group from filename: '+name)
    return 'deepfashion:'+match.group(1)


def group_split(group,seed=42):
    value=int(hashlib.sha256((str(seed)+'|'+group).encode()).hexdigest()[:8],16)%100
    return 'train' if value<80 else 'validation' if value<90 else 'test'


def region_crop(image,mask,region,min_pixels=4096):
    import numpy as np
    from PIL import Image
    if mask.mode not in ('L','P','I','I;16'):
        raise ValueError('Parsing mask must contain integer label IDs, not an RGB preview.')
    if image.size!=mask.size:raise ValueError('Image and parsing mask dimensions differ.')
    if image.getexif().get(274,1)!=1:raise ValueError('Image has an orientation transform; check mask alignment manually.')
    labels=np.asarray(mask)
    if labels.ndim!=2 or labels.min()<0 or labels.max()>23:raise ValueError('Invalid parsing-label values.')
    # Dress/romper images are omitted: their upper/lower annotation scope is ambiguous.
    if (labels==4).any() or (labels==21).any():return None
    selected=np.isin(labels,REGIONS[region][1])
    if int(selected.sum())<min_pixels:return None
    ys,xs=np.nonzero(selected)
    box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
    pixels=np.asarray(image.convert('RGB')).copy()
    pixels[~selected]=245
    cropped=Image.fromarray(pixels).crop(box)
    cropped.thumbnail((512,512),Image.Resampling.LANCZOS)
    return cropped,box,int(selected.sum())
