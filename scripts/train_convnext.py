"""Frozen ConvNeXt development experiment, FP32 throughout; final test forbidden."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'runtime/models/torch'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/training/convnext_tiny_pilot_01'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000 or not 1 <= args.batch_size <= 32:
        raise ValueError('Invalid run size.')
    output = Path(args.output)
    if output.exists(): raise ValueError('Choose a new output; existing runs are never overwritten.')
    import torch
    import torchvision
    from safetensors.torch import save_file, load_file
    from torch.utils.data import DataLoader
    from backend.convnext_model import ConvNextFabric, WEIGHTS, ARCHITECTURE
    from backend.fabric_model import classification_metrics
    from backend.training_data import read_development, DevelopmentImages
    prep, rows = read_development(args.dataset)
    if not torch.cuda.is_available(): raise RuntimeError('This pilot requires CUDA.')
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    model = ConvNextFabric(len(prep['labels']), pretrained=True).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.head.parameters(): parameter.requires_grad_(True)
    output.mkdir(parents=True)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    features = {}
    for split in ('train', 'validation'):
        dataset = DevelopmentImages(prep['raw_root'], rows[split], prep['labels'])
        dataset.transform = WEIGHTS.transforms()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        xx, yy = [], []
        with torch.no_grad():
            for index, (images, labels) in enumerate(loader):
                xx.append(model.encode(images.cuda()).cpu()); yy.append(labels)
                if (index + 1) % 25 == 0:
                    print(split, min((index + 1) * args.batch_size, len(dataset)), '/', len(dataset), flush=True)
        features[split] = torch.cat(xx), torch.cat(yy)
    save_file({s + '_' + k: t for s, pair in features.items() for k, t in zip(('features', 'labels'), pair)}, str(output / 'features.safetensors'))
    x, y = [v.cuda() for v in features['train']]
    vx, vy = [v.cuda() for v in features['validation']]
    counts = torch.bincount(y, minlength=len(prep['labels'])).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.02, weight_decay=.01)
    history = []; best = -1; best_head = None; best_epoch = None
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model.head(x), y); loss.backward(); optimizer.step()
        with torch.no_grad(): logits = model.head(vx)
        metric = classification_metrics(vy.tolist(), logits.argmax(1).tolist(), prep['labels'])
        history.append({'epoch': epoch, 'loss': float(loss.detach()), 'validation_macro_f1': metric['macro_f1'], 'validation_accuracy': metric['accuracy']})
        if metric['macro_f1'] > best:
            best = metric['macro_f1']; best_epoch = epoch
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        if epoch % 20 == 0: print('epoch', epoch, 'macro-F1', metric['macro_f1'], flush=True)
    model.head.load_state_dict(best_head)
    with torch.no_grad(): logits = model.head(vx).cpu()
    metric = classification_metrics(vy.tolist(), logits.argmax(1).tolist(), prep['labels'])
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output / 'model.safetensors'))
    # Reload the actual saved weights and repeat full-network FP32 validation.
    model.load_state_dict(load_file(str(output / 'model.safetensors'), device='cuda'))
    dataset = DevelopmentImages(prep['raw_root'], rows['validation'], prep['labels']); dataset.transform = WEIGHTS.transforms()
    full = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False): full.append(model(images.cuda()).cpu())
    full = torch.cat(full)
    if not torch.equal(full.argmax(1), logits.argmax(1)): raise RuntimeError('Checkpoint reload predictions differ.')
    reference = json.loads((ROOT / 'runtime/training/efficientnet_b0_finetune_01/report.json').read_text())
    if reference['manifest_sha256'] != prep['manifest_sha256'] or reference['labels'] != prep['labels']:
        raise ValueError('EfficientNet comparison identity mismatch.')
    pretrained = Path(torch.hub.get_dir()) / 'checkpoints' / WEIGHTS.url.rsplit('/', 1)[-1]
    report = {'architecture': ARCHITECTURE, 'labels': prep['labels'], 'manifest_sha256': prep['manifest_sha256'],
        'counts': prep['counts'], 'seed': 42, 'validation': metric, 'best_epoch': best_epoch, 'history': history,
        'checkpoint_sha256': hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest(),
        'pretrained_weights_url': WEIGHTS.url, 'pretrained_sha256': hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        'training': {'method': 'Frozen ImageNet backbone and normalization; L2 768-D features plus weighted linear head',
                     'epochs': args.epochs, 'lr': .02, 'weight_decay': .01, 'feature_batch_size': args.batch_size,
                     'preprocessing': repr(WEIGHTS.transforms()), 'precision': 'float32', 'reload_predictions_verified': True},
        'comparison': {'efficientnet_frozen_fp32': reference['baseline_validation_recomputed'],
                       'efficientnet_partial_finetune_fp32': reference['validation'],
                       'notice': 'Same manifest/validation; official preprocessing differs by architecture. Frozen versus partially fine-tuned training budgets differ. No statistical superiority claim.'},
        'device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'elapsed_seconds': time.perf_counter() - started, 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(),
        'test_evaluated': False, 'test_images_opened': 0, 'activated_in_app': False,
        'notice': 'Validation-selected development pilot. No final test, calibration or OOD validation. Raw specimen independence remains unverified.'}
    save_file({'logits': full, 'labels': features['validation'][1]}, str(output / 'validation_logits.safetensors'))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1'], 'best_epoch': best_epoch}), flush=True)


if __name__ == '__main__': main()
