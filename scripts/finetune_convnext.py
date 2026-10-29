"""Five-epoch final-stage ConvNeXt pilot; no final-test inference or app mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure(model):
    model.eval()  # fixed stochastic depth; frozen earlier stages
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.features[-2:], model.norm, model.head):
        for parameter in module.parameters(): parameter.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=str(ROOT / 'runtime/training/convnext_tiny_pilot_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/training/convnext_tiny_finetune_01'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 1 <= args.batch_size <= 32: raise ValueError('Invalid run size.')
    output, baseline = Path(args.output), Path(args.baseline)
    if output.exists(): raise ValueError('Choose a new run directory.')
    import torch
    import torchvision
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    from backend.convnext_model import ConvNextFabric, WEIGHTS, ARCHITECTURE
    from backend.fabric_model import classification_metrics
    from backend.training_data import read_development, DevelopmentImages
    prep, rows = read_development(args.dataset)
    original = json.loads((baseline / 'report.json').read_text())
    if (original['architecture'], original['labels'], original['manifest_sha256']) != (ARCHITECTURE, prep['labels'], prep['manifest_sha256']):
        raise ValueError('Baseline/dataset mismatch.')
    if hashlib.sha256((baseline / 'model.safetensors').read_bytes()).hexdigest() != original['checkpoint_sha256']:
        raise ValueError('Baseline checksum mismatch.')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this pilot.')
    torch.set_num_threads(4); random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    model = ConvNextFabric(len(prep['labels'])).cuda()
    model.load_state_dict(load_file(str(baseline / 'model.safetensors'), device='cuda')); configure(model)
    protected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(tuple(f'features.{i}.' for i in range(len(model.features) - 2)))}
    datasets = {s: DevelopmentImages(prep['raw_root'], rows[s], prep['labels']) for s in ('train', 'validation')}
    for dataset in datasets.values(): dataset.transform = WEIGHTS.transforms()
    train = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42))
    validation = DataLoader(datasets['validation'], batch_size=args.batch_size, shuffle=False, num_workers=0)

    def evaluate():
        model.eval(); outputs, truth = [], []
        with torch.no_grad():
            for images, labels in validation:
                outputs.append(model(images.cuda()).cpu()); truth.extend(labels.tolist())
        logits = torch.cat(outputs)
        return classification_metrics(truth, logits.argmax(1).tolist(), prep['labels']), logits, torch.tensor(truth)

    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    best_metric, _, _ = evaluate(); best_epoch = 0
    if abs(best_metric['macro_f1'] - original['validation']['macro_f1']) > 1e-9:
        raise ValueError('Baseline validation did not reproduce.')
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    counts = torch.bincount(torch.tensor([prep['labels'].index(r['label']) for r in rows['train']]), minlength=len(prep['labels'])).float().cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts))
    optimizer = torch.optim.AdamW([{'params': model.features[-2:].parameters(), 'lr': .0001},
        {'params': model.norm.parameters(), 'lr': .0001}, {'params': model.head.parameters(), 'lr': .001}], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda'); history = []; output.mkdir(parents=True)
    for epoch in range(1, args.epochs + 1):
        model.eval(); losses = []
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): loss = loss_fn(model(images.cuda()), labels.cuda())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        metric, _, _ = evaluate()
        history.append({'epoch': epoch, 'mean_batch_loss': sum(losses) / len(losses), 'validation': metric})
        if metric['macro_f1'] > best_metric['macro_f1']:
            best_metric, best_epoch = metric, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        (output / 'progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('epoch', epoch, 'accuracy', metric['accuracy'], 'macro-F1', metric['macro_f1'], flush=True)
    for key, tensor in protected.items():
        if not torch.equal(tensor, model.state_dict()[key].cpu()): raise RuntimeError('Frozen stage changed: ' + key)
    model.load_state_dict(best_state)
    final, logits, truth = evaluate()
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(output / 'model.safetensors'))
    model.load_state_dict(load_file(str(output / 'model.safetensors'), device='cuda'))
    repeated, repeated_logits, _ = evaluate()
    if not torch.equal(logits.argmax(1), repeated_logits.argmax(1)): raise RuntimeError('Reload mismatch.')
    save_file({'logits': logits, 'labels': truth}, str(output / 'validation_logits.safetensors'))
    report = {'architecture': ARCHITECTURE, 'labels': prep['labels'], 'manifest_sha256': prep['manifest_sha256'],
        'counts': prep['counts'], 'seed': 42, 'baseline_checkpoint_sha256': original['checkpoint_sha256'],
        'checkpoint_sha256': hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest(),
        'validation': final, 'baseline_validation': original['validation'], 'best_epoch': best_epoch, 'history': history,
        'training': {'method': 'Final two feature modules, final normalization and head; remaining stages frozen; eval-mode stochastic depth',
            'epochs': args.epochs, 'batch_size': args.batch_size, 'backbone_lr': .0001, 'head_lr': .001, 'weight_decay': .01,
            'mixed_precision_training': True, 'evaluation_precision': 'float32', 'extra_augmentation': False,
            'frozen_state_verified': True, 'reload_predictions_verified': True, 'preprocessing': repr(WEIGHTS.transforms())},
        'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__, 'device': torch.cuda.get_device_name(0),
        'elapsed_seconds': time.perf_counter() - started, 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(),
        'test_evaluated': False, 'test_images_opened': 0, 'activated_in_app': False,
        'notice': 'Validation-selected development pilot, including epoch-zero fallback. Not independent test or specimen-level generalization evidence.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'best_epoch': best_epoch, 'accuracy': final['accuracy'], 'macro_f1': final['macro_f1']}), flush=True)


if __name__ == '__main__': main()
