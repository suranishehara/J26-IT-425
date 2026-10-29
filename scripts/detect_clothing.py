"""CPU human parsing keeps VRAM free for CatVTON."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'runtime' / 'models'))

def build_mask(labels, category):
    import numpy as np
    from scipy import ndimage
    if category not in ('upper-body', 'dress', 'saree', 'kandyan-osariya'):
        raise ValueError('Unsupported category.')
    # ATR labels: upper garment=4, skirt=5, trousers=6, dress=7,
    # belt=8, scarf=17. All three full-outfit categories share one policy.
    classes = [4, 7, 17] if category == 'upper-body' else [4, 5, 6, 7, 8, 17]
    selected = np.isin(labels, classes)
    lower_dress = np.zeros_like(selected)
    if category == 'upper-body' and np.any(labels == 7):
        # A parser labels a one-piece dress as one region. For a top-only
        # replacement, retain its bodice and protect the skirt below the waist.
        # Without a measured waist this boundary is an approximation.
        dress_rows = np.where(labels == 7)[0]
        waist = int(dress_rows.min() + .45 * (dress_rows.max() - dress_rows.min() + 1))
        lower_dress = (labels == 7) & (np.arange(labels.shape[0])[:, None] >= waist)
        selected &= ~lower_dress
    # Close pinholes and allow a small margin without selecting unrelated background.
    scale = max(1, labels.shape[0] // 512)
    selected = ndimage.binary_fill_holes(ndimage.binary_closing(selected, iterations=2*scale))
    if category != 'upper-body':
        # Full-outfit references may have a continuous skirt although the source
        # person wears trousers. Include the background between the legs so it
        # can become fabric, rather than locking generation into two trouser legs.
        lower_rows = np.where(np.isin(labels, [5, 6, 7]))[0]
        if lower_rows.size:
            for row in range(int(lower_rows.min()), int(lower_rows.max()) + 1):
                columns = np.flatnonzero(selected[row])
                if columns.size:
                    selected[row, columns[0]:columns[-1] + 1] = True
    selected = ndimage.binary_dilation(selected, iterations=4*scale)
    protected = np.isin(labels, [1, 2, 3, 9, 10, 11, 16])
    if category == 'upper-body':
        protected |= np.isin(labels, [5, 6, 8, 12, 13])
        protected |= lower_dress
    return (selected & ~protected).astype('uint8') * 255

def detect(source, target, category):
    import torch
    import numpy as np
    from PIL import Image
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    torch.set_num_threads(4)
    processor = SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes').eval()
    image = Image.open(source).convert('RGB')
    with torch.inference_mode():
        logits = model(**processor(images=image, return_tensors='pt')).logits
        labels = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode='bilinear', align_corners=False).argmax(1)[0].numpy()
    mask = build_mask(labels, category)
    if np.count_nonzero(mask) < image.width * image.height * .01:
        raise ValueError('No clothing region detected. Use a clearer person photo or paint the mask manually.')
    Image.fromarray(mask).save(target)

if __name__ == '__main__':
    detect(sys.argv[1], sys.argv[2], sys.argv[3])
