"""Immutable local checkpoint loading for offline research jobs, no downloads."""
import json
from .classifier import selected_release, ARCHITECTURE, CONVNEXT_ARCHITECTURE, CONVNEXTV2_ARCHITECTURE, sha


def load(model_id):
    import torch
    from safetensors.torch import load_file
    run, report, checksum = selected_release(model_id)
    if report['architecture'] == ARCHITECTURE:
        from .fabric_model import FabricNetwork, preprocessing
        model = FabricNetwork(len(report['labels'])); transform = preprocessing()
    elif report['architecture'] == CONVNEXT_ARCHITECTURE:
        from .convnext_model import ConvNextFabric, WEIGHTS
        model = ConvNextFabric(len(report['labels'])); transform = WEIGHTS.transforms()
    elif report['architecture'] == CONVNEXTV2_ARCHITECTURE:
        from .convnextv2_model import ConvNextV2Fabric
        model = ConvNextV2Fabric(len(report['labels'])); transform = model.transform
        if json.dumps(model.data_config, sort_keys=True) != json.dumps(report['training']['data_config'], sort_keys=True):
            raise ValueError('Preprocessing differs from frozen checkpoint.')
    else: raise ValueError('Unsupported architecture.')
    model.load_state_dict(load_file(str(run / 'model.safetensors'), device='cpu'), strict=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    identity = {'model_id': model_id, 'run': run.name, 'architecture': report['architecture'], 'labels': report['labels'],
                'checkpoint_sha256': checksum, 'report_sha256': sha(run / 'report.json'), 'training_manifest_sha256': report['manifest_sha256']}
    return model, transform, identity


def collect_logits(model, transform, rows, root, device='cpu'):
    import torch
    from PIL import Image, ImageOps
    from pathlib import Path
    from .research_data import sha
    if not rows: raise ValueError('No rows for this evaluation stage.')
    outputs = []; root = Path(root).resolve(); model.to(device).eval()
    with torch.inference_mode():
        for row in rows:
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root) or sha(path) != row['file_sha256']: raise ValueError('Evaluation input changed or escaped root.')
            with Image.open(path) as image: tensor = transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(device)
            result = model(tensor).cpu()
            if not bool(torch.isfinite(result).all()): raise ValueError('Non-finite logits.')
            outputs.append(result)
    return torch.cat(outputs)
