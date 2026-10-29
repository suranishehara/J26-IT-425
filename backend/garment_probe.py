"""Load the optional four-class IndoFashion adapter without changing CLIP."""
import json
import hashlib
from pathlib import Path

PROBE_DIRECTORY=Path(__file__).resolve().parents[1]/'runtime'/'adaptation'/'indofashion_probe_01'
PROBE_LABELS=['blouse','dress','kurta','saree']

def load_probe(device, model_revision):
    import torch
    from safetensors.torch import load_file
    if not (PROBE_DIRECTORY/'report.json').is_file():
        raise ValueError('The adapted classifier is not ready yet. Choose General CLIP.')
    report=json.loads((PROBE_DIRECTORY/'report.json').read_text())
    if report.get('revision')!=model_revision or report.get('labels')!=PROBE_LABELS:
        raise ValueError('The adapter is incompatible with this CLIP version.')
    release_path=Path(__file__).resolve().parents[1]/'config'/'garment_probe_release.json'
    if release_path.is_file():
        release=json.loads(release_path.read_text())
        checksum=hashlib.sha256((PROBE_DIRECTORY/'probe.safetensors').read_bytes()).hexdigest()
        if release.get('checkpoint_sha256')!=checksum:
            raise ValueError('The adapter does not match the frozen validated checkpoint.')
    weights=load_file(str(PROBE_DIRECTORY/'probe.safetensors'),device=device)
    head=torch.nn.Linear(weights['weight'].shape[1],len(PROBE_LABELS)).to(device)
    head.load_state_dict(weights)
    return head.eval()
