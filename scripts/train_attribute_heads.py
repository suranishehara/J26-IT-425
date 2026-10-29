"""Frozen V2 encoder plus masked attribute heads; real reviewed labels required."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000: raise ValueError('Invalid epoch count.')
    output = Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory.')
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file, load_file
    from backend.research_data import load_manifest, provider_hashes, sha
    from backend.research_inference import load
    from backend.attribute_heads import tasks_for, AttributeHeads, targets, masked_loss, evaluate_head
    torch.set_num_threads(4); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    rows = load_manifest(args.manifest, args.image_root, provider_hashes())
    tasks, skipped = tasks_for(rows)
    if not tasks: raise ValueError('No head has sufficient real reviewed train/validation labels. Collect data first.')
    model, transform, identity = load('convnext_v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval(); started = time.perf_counter()
    split_rows = {split: [row for row in rows if row['split'] == split and row['kind'] != 'non_fabric'] for split in ('train', 'validation')}
    if not all(split_rows.values()): raise ValueError('Need train and validation fabric images.')
    features = {}
    root = Path(args.image_root).resolve()
    with torch.inference_mode():
        for split, samples in split_rows.items():
            batch = []
            for row in samples:
                path = (root / row['path']).resolve()
                if not path.is_relative_to(root) or sha(path) != row['file_sha256']: raise ValueError('Training photo changed or escaped root.')
                with Image.open(path) as image: tensor = transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(device)
                batch.append(model.encode(tensor).cpu())
            features[split] = torch.cat(batch)
    features = {split: tensor.clone() for split, tensor in features.items()}
    # Never open calibration/assessment/final images for feature extraction.
    heads = AttributeHeads(tasks).to(device)
    train_x = features['train'].to(device); val_x = features['validation'].to(device)
    reports = {}
    for head, task in tasks.items():
        layer = heads.heads[head]
        target = targets(split_rows['train'], head, task['labels']).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.02, weight_decay=.01)
        best = -1.; best_state = None; best_epoch = None; history = []
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(layer(train_x), target, task['binary'])
            if loss is None or not bool(torch.isfinite(loss)): raise ValueError('Missing/non-finite masked training loss.')
            loss.backward(); optimizer.step()
            with torch.no_grad(): result = evaluate_head(layer(val_x).cpu(), split_rows['validation'], head, task)
            history.append({'epoch': epoch, 'loss': float(loss.detach()), 'validation_macro_f1': result['macro_f1']})
            if result['macro_f1'] > best:
                best = result['macro_f1']; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}
        layer.load_state_dict(best_state)
        with torch.no_grad(): final = evaluate_head(layer(val_x).cpu(), split_rows['validation'], head, task)
        reports[head] = {'task': task, 'validation': final, 'best_epoch': best_epoch, 'history': history}
        print(head, 'known validation', final['n'], 'macro-F1', final['macro_f1'], flush=True)
    # Exact frozen encoder state identity must still match its stored checkpoint.
    from backend.classifier import selected_release
    run, _, checksum = selected_release('convnext_v2')
    if checksum != identity['checkpoint_sha256']: raise ValueError('Encoder release changed during training.')
    source = load_file(str(run / 'model.safetensors'), device='cpu')
    if any(not torch.equal(value.detach().cpu(), source[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('Frozen encoder changed.')
    state = {k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()}
    output.mkdir(parents=True); save_file(state, str(output / 'attribute_heads.safetensors'))
    check = AttributeHeads(tasks); check.load_state_dict(load_file(str(output / 'attribute_heads.safetensors')), strict=True)
    with torch.no_grad():
        for name in tasks:
            actual = heads.heads[name](val_x).cpu(); repeated = check.heads[name](features['validation'])
            first = (actual.flatten() >= 0) if tasks[name]['binary'] else actual.argmax(-1)
            second = (repeated.flatten() >= 0) if tasks[name]['binary'] else repeated.argmax(-1)
            if not torch.equal(first, second):
                raise RuntimeError('Head reload prediction mismatch.')
            if not torch.allclose(actual, repeated, atol=1e-5, rtol=1e-4):
                raise RuntimeError('Head reload logit mismatch.')
    report = {'status': 'attribute_heads_development_not_activated', 'encoder': identity, 'manifest_sha256': sha(args.manifest),
        'heads': reports, 'skipped_heads': skipped, 'head_checkpoint_sha256': sha(output / 'attribute_heads.safetensors'),
        'seed': 42, 'epochs': args.epochs, 'device': device, 'elapsed_seconds': time.perf_counter() - started,
        'encoder_frozen_verified': True, 'calibration_assessment_final_inference_performed': False,
        'notice': 'Missing labels masked per head; batik and handloom are independent binary outputs. Labels/methods require expert evidence. Frozen encoder with newly trained heads, not an independently evaluated fabric-attribute system. No automatic activation.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
