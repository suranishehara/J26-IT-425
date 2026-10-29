"""Local image/mask inference. No uploaded image is sent to a remote API."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'runtime' / 'models'))
sys.path.insert(0, str(ROOT / 'external' / 'CatVTON'))

def main():
    import torch
    from PIL import Image, ImageFilter
    from model.pipeline import CatVTONPipeline
    job = Path(sys.argv[1]).resolve()
    settings = json.loads((job / 'request.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Start with the project .venv Python.')
    pipe = CatVTONPipeline(
        base_ckpt='booksforcharlie/stable-diffusion-inpainting',
        attn_ckpt='zhengchong/CatVTON',
        attn_ckpt_version='mix', weight_dtype=torch.float16, device='cuda',
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    torch.cuda.reset_peak_memory_stats()
    source = Image.open(job / 'person.png').convert('RGB')
    mask = Image.open(job / 'mask.png').convert('L')
    garment = Image.open(job / 'garment.png').convert('RGB')
    width, height = (768, 1024) if settings.get('quality', 'balanced') == 'high' else (384, 512)
    steps = 50 if settings.get('quality', 'balanced') == 'high' else 30
    source = source.resize((width,height), Image.Resampling.LANCZOS)
    garment = garment.resize((width,height), Image.Resampling.LANCZOS)
    mask = mask.resize((width,height), Image.Resampling.NEAREST)
    result = pipe(image=source, condition_image=garment, mask=mask,
                  width=width, height=height, num_inference_steps=steps,
                  generator=torch.Generator(device='cuda').manual_seed(settings['seed']))[0]
    result.save(job / 'generated.png')
    # Preserve pixels outside the user-selected replacement region exactly.
    # Feather inward only: do not alter pixels outside the selected region.
    import numpy as np
    soft = np.minimum(np.asarray(mask), np.asarray(mask.filter(ImageFilter.GaussianBlur(2))))
    result = Image.composite(result, source, Image.fromarray(soft))
    result.save(job / 'result.png')
    (job / 'metrics.json').write_text(json.dumps({
        'model': 'CatVTON mix', 'resolution': [width, height], 'steps': steps,
        'seed': settings['seed'], 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'category': settings['category'], 'adaptation': 'user-defined replacement mask; outside-mask compositing',
    }, indent=2))

if __name__ == '__main__':
    main()
