"""Controlled last-stage EfficientNet fine-tuning. Never evaluates final test."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_finetuning(model):
    # Gradients update final two feature stages + head. Eval mode is intentional:
    # batch-normalisation statistics and stochastic-depth behaviour stay fixed.
    model.eval()
    for param in model.parameters(): param.requires_grad_(False)
    for param in model.features[-2:].parameters(): param.requires_grad_(True)
    for param in model.head.parameters(): param.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=str(ROOT / 'runtime/training/efficientnet_b0_pilot_01'))
    parser.add_argument('--dataset', default=str(ROOT / 'runtime/datasets/textileclass_pilot_02'))
    parser.add_argument('--output', default=str(ROOT / 'runtime/training/efficientnet_b0_finetune_01'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 1 <= args.batch_size <= 64: raise ValueError('Invalid run size.')
    baseline, output = Path(args.baseline), Path(args.output)
    if output.exists(): raise ValueError('Use a new run directory.')
    import torch
    import torchvision
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    from backend.fabric_model import FabricNetwork, ARCHITECTURE, classification_metrics
    from backend.training_data import read_development, DevelopmentImages
    prep, split_rows = read_development(args.dataset)
    original = json.loads((baseline / 'report.json').read_text())
    if original['architecture'] != ARCHITECTURE or original['manifest_sha256'] != prep['manifest_sha256'] or original['labels'] != prep['labels']:
        raise ValueError('Baseline and dataset identity mismatch.')
    if hashlib.sha256((baseline / 'model.safetensors').read_bytes()).hexdigest() != original['checkpoint_sha256']:
        raise ValueError('Baseline checksum mismatch.')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('This configured fine-tuning pilot requires CUDA.')
    torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    labels = prep['labels']
    model = FabricNetwork(len(labels)).to(device)
    model.load_state_dict(load_file(str(baseline / 'model.safetensors')))
    configure_finetuning(model)
    protected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if key.startswith(tuple(f'features.{i}.' for i in range(len(model.features) - 2))) or 'running_' in key or 'num_batches_tracked' in key}
    train = DataLoader(DevelopmentImages(prep['raw_root'], split_rows['train'], labels), batch_size=args.batch_size, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(args.seed))
    validation = DataLoader(DevelopmentImages(prep['raw_root'], split_rows['validation'], labels), batch_size=args.batch_size, shuffle=False, num_workers=0)

    def evaluate():
        model.eval()
        yy, scores = [], []
        with torch.inference_mode():
            for images, targets in validation:
                # Common FP32 evaluation for both baseline and candidate.
                scores.append(model(images.to(device)).cpu()); yy.extend(targets.tolist())
        logits = torch.cat(scores)
        return classification_metrics(yy, logits.argmax(1).tolist(), labels), logits

    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    measured_baseline, _ = evaluate()
    best_metric, best_epoch = measured_baseline, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    counts = torch.bincount(torch.tensor([labels.index(r['label']) for r in split_rows['train']]), minlength=len(labels)).float().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=counts.sum() / (len(labels) * counts))
    optimizer = torch.optim.AdamW([{'params': model.features[-2:].parameters(), 'lr': 0.0001},
                                   {'params': model.head.parameters(), 'lr': 0.001}], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    output.mkdir(parents=True)
    print('Baseline recomputed on validation:', measured_baseline['accuracy'], measured_baseline['macro_f1'], flush=True)
    for epoch in range(1, args.epochs + 1):
        model.eval()
        losses = []
        for images, targets in train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = loss_fn(model(images.to(device)), targets.to(device))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        metric, _ = evaluate()
        history.append({'epoch': epoch, 'mean_batch_train_loss': sum(losses) / len(losses), 'validation_accuracy': metric['accuracy'], 'validation_macro_f1': metric['macro_f1']})
        if metric['macro_f1'] > best_metric['macro_f1']:
            best_metric, best_epoch = metric, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        (output / 'progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('epoch', epoch, 'accuracy', round(metric['accuracy'], 4), 'macro-F1', round(metric['macro_f1'], 4), flush=True)
    for key, value in protected.items():
        if not torch.equal(value, model.state_dict()[key].detach().cpu()):
            raise RuntimeError('Frozen parameters/statistics changed: ' + key)
    model.load_state_dict(best_state)
    final, logits = evaluate()
    save_file({key: value.contiguous() for key, value in best_state.items()}, str(output / 'model.safetensors'))
    predictions = [{'path': row['path'], 'truth': row['label'], 'prediction': labels[int(logits[i].argmax())],
                    'relative_score': float(logits[i].softmax(0).max())} for i, row in enumerate(split_rows['validation'])]
    (output / 'validation_predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    report = {'architecture': ARCHITECTURE, 'labels': labels, 'seed': args.seed,
              'method': 'Final two feature stages and head fine-tuned from frozen-head baseline; BN/stochastic depth fixed in eval mode',
              'manifest_sha256': prep['manifest_sha256'], 'dataset_status': prep['status'], 'counts': prep['counts'],
              'baseline_checkpoint_sha256': original['checkpoint_sha256'],
              'checkpoint_sha256': hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest(),
              'baseline_validation_recomputed': measured_baseline, 'baseline_validation_reported': original['validation'],
              'validation': final, 'best_epoch': best_epoch, 'epochs': args.epochs, 'history': history,
              'training': {'backbone_lr': 0.0001, 'head_lr': 0.001, 'weight_decay': 0.01, 'batch_size': args.batch_size, 'clip_grad_norm': 1.0,
                           'extra_augmentation': False, 'mixed_precision': True, 'evaluation_precision': 'float32', 'frozen_state_verified': True},
              'device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
              'elapsed_seconds': time.perf_counter() - started, 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(),
              'test_evaluated': False, 'test_images_opened': 0, 'activated_in_app': False,
              'notice': 'Development pilot, selected on validation macro-F1, epoch 0 baseline retained if no improvement. Specimen independence is unverified; no real-world accuracy claim.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'best_epoch': best_epoch, 'validation_accuracy': final['accuracy'], 'validation_macro_f1': final['macro_f1'], 'test_evaluated': False}, indent=2), flush=True)


if __name__ == '__main__': main()
