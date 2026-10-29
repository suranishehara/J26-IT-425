"""Train/validate a frozen EfficientNet-B0 head; never opens test images."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'runtime/models/torch'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/training/efficientnet_b0_pilot_01'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000 or not 1 <= args.batch_size <= 128:
        raise ValueError('Invalid epochs or batch size.')
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; no trained run is overwritten.')
    import torch
    import torchvision
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    from torch.utils.data import Dataset, DataLoader
    from backend.fabric_model import FabricNetwork, ARCHITECTURE, WEIGHTS, preprocessing, classification_metrics

    preparation = json.loads((dataset / 'preparation.json').read_text())
    manifest_bytes = (dataset / 'manifest.jsonl').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != preparation['manifest_sha256']:
        raise ValueError('Manifest differs from the prepared checksum.')
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line]
    labels = preparation['labels']
    split_rows = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation', 'test')}
    for split in split_rows:
        groups = {r['group_id'] for r in split_rows[split]}
        for other in split_rows:
            if other != split and groups & {r['group_id'] for r in split_rows[other]}:
                raise ValueError('Group overlap across splits.')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = preprocessing()
    raw = Path(preparation['raw_root']).resolve()

    class Images(Dataset):
        def __init__(self, records): self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, index):
            row = self.records[index]
            if row['split'] == 'test': raise RuntimeError('Test inference is forbidden in this script.')
            path = (raw / row['path']).resolve()
            if not path.is_relative_to(raw): raise ValueError('Image escapes raw root.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']:
                raise ValueError('Raw image changed: ' + row['path'])
            with Image.open(path) as image:
                tensor = transform(ImageOps.exif_transpose(image).convert('RGB'))
            return tensor, labels.index(row['label'])

    output.mkdir(parents=True)
    model = FabricNetwork(len(labels), pretrained=True).to(device).eval()
    for parameter in model.features.parameters(): parameter.requires_grad_(False)
    started = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    features = {}
    for split in ('train', 'validation'):
        loader = DataLoader(Images(split_rows[split]), batch_size=args.batch_size, shuffle=False, num_workers=0)
        xx, yy = [], []
        with torch.inference_mode():
            for batch, (images, targets) in enumerate(loader):
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    encoded = model.encode(images.to(device))
                xx.append(encoded.float().cpu()); yy.append(targets)
                if (batch + 1) % 25 == 0: print(f'{split}: {min((batch + 1) * args.batch_size, len(loader.dataset))}/{len(loader.dataset)}', flush=True)
        features[split] = (torch.cat(xx), torch.cat(yy))
        print(split, len(features[split][1]), 'features extracted', flush=True)
    save_file({split + '_' + key: value for split, values in features.items() for key, value in zip(('features', 'labels'), values)}, str(output / 'features.safetensors'))
    x, y = (t.to(device) for t in features['train'])
    vx, vy = (t.to(device) for t in features['validation'])
    counts = torch.bincount(y, minlength=len(labels)).float()
    if bool((counts == 0).any()): raise ValueError('Missing training class.')
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (len(labels) * counts)))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.02, weight_decay=0.01)
    best, history, best_weights, best_epoch = -1, [], None, None
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model.head(x), y); loss.backward(); optimizer.step()
        with torch.no_grad():
            preds = model.head(vx).argmax(1).cpu().tolist()
        metric = classification_metrics(vy.cpu().tolist(), preds, labels)
        history.append({'epoch': epoch, 'training_loss': float(loss.detach()), 'validation_accuracy': metric['accuracy'], 'validation_macro_f1': metric['macro_f1']})
        if metric['macro_f1'] > best:
            best, best_epoch = metric['macro_f1'], epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        if epoch % 20 == 0: print('epoch', epoch, 'validation macro-F1', round(metric['macro_f1'], 4), flush=True)
    model.head.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model.head(vx).float().cpu()
    predictions = logits.argmax(1).tolist()
    final = classification_metrics(vy.cpu().tolist(), predictions, labels)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output / 'model.safetensors'))
    checkpoint_hash = hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest()
    prediction_rows = [{'path': r['path'], 'truth': r['label'], 'prediction': labels[p], 'relative_score': float(logits[i].softmax(0)[p])} for i, (r, p) in enumerate(zip(split_rows['validation'], predictions))]
    (output / 'validation_predictions.json').write_text(json.dumps(prediction_rows, indent=2), encoding='utf-8')
    pretrained_path = Path(torch.hub.get_dir()) / 'checkpoints' / WEIGHTS.url.rsplit('/', 1)[-1]
    report = {'architecture': ARCHITECTURE, 'labels': labels, 'seed': args.seed,
              'backbone': 'torchvision EfficientNet-B0 ImageNet1K V1; frozen',
              'pretrained_weights_url': WEIGHTS.url, 'pretrained_sha256': hashlib.sha256(pretrained_path.read_bytes()).hexdigest(),
              'checkpoint_sha256': checkpoint_hash, 'manifest_sha256': preparation['manifest_sha256'],
              'dataset_status': preparation['status'], 'counts': preparation['counts'],
              'training': {'method': 'L2-normalized frozen 1280-D features + weighted linear head', 'epochs': args.epochs, 'best_epoch': best_epoch, 'lr': 0.02, 'weight_decay': 0.01, 'batch_size_feature_extraction': args.batch_size, 'head_batch': 'full training feature matrix', 'preprocessing': 'official ImageNet1K V1 resize256 bicubic, centre224, ImageNet normalisation', 'mixed_precision_feature_extraction': device.type == 'cuda'},
              'validation': final, 'test_evaluated': False, 'test_images_opened': 0,
              'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
              'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
              'elapsed_seconds': time.perf_counter() - started,
              'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'history': history, 'notice': 'Development pilot only. Unknown specimen identities, noisy labels and augmented images limit generalisation claims. Not activated in application; test reserved.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'validation': final, 'test_evaluated': False}, indent=2), flush=True)


if __name__ == '__main__': main()
