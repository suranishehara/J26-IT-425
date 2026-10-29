"""Frozen-head then final-stage ConvNeXt V2 development pilot; test remains reserved."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--output')
    parser.add_argument('--nine-categories', action='store_true')
    parser.add_argument('--acknowledge-provisional-split', action='store_true')
    parser.add_argument('--head-epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.nine_categories and not args.acknowledge_provisional_split:
        raise ValueError('Review filename-group/similarity caveats and acknowledge this provisional development split.')
    args.dataset = args.dataset or str(ROOT / ('runtime/datasets/fabric_nine_categories_01' if args.nine_categories else 'runtime/datasets/textileclass_pilot_02'))
    args.output = args.output or str(ROOT / ('runtime/training/convnextv2_nine_categories_01' if args.nine_categories else 'runtime/training/convnextv2_tiny_pilot_01'))
    if not 1 <= args.head_epochs <= 1000 or not 1 <= args.finetune_epochs <= 30 or not 1 <= args.batch_size <= 32:
        raise ValueError('Invalid run size.')
    output = Path(args.output)
    if output.exists(): raise ValueError('Choose a new output directory; existing runs are never overwritten.')
    import torch
    import torchvision
    import timm
    from safetensors.torch import save_file, load_file
    from torch.utils.data import DataLoader
    from backend.convnextv2_model import ConvNextV2Fabric, download_pretrained, configure_finetuning, ARCHITECTURE, REPOSITORY, REVISION, LICENSE
    from backend.training_data import read_development, DevelopmentImages
    from backend.fabric_model import classification_metrics
    if args.nine_categories:
        from backend.category_dataset import read_prepared, development_images, digest
        prep, all_rows = read_prepared(args.dataset)
        if digest(ROOT / 'config/fabric_category_expansion.json') != prep['scope_sha256']:
            raise ValueError('Category scope changed since preparation; use a newly reviewed dataset version.')
        rows = {split: [r for r in all_rows if r['split'] == split] for split in ('train', 'validation')}
    else:
        prep, rows = read_development(args.dataset)
    if not torch.cuda.is_available(): raise RuntimeError('This development pilot requires CUDA.')
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    pretrained = download_pretrained(ROOT / 'runtime/models/huggingface')
    print('Pinned pretrained weights downloaded:', pretrained.name, flush=True)
    model = ConvNextV2Fabric(len(prep['labels']), pretrained_path=pretrained).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.head.parameters(): parameter.requires_grad_(True)
    datasets = {split: development_images(prep, rows[split], model.transform) if args.nine_categories else DevelopmentImages(prep['raw_root'], rows[split], prep['labels']) for split in ('train', 'validation')}
    for dataset in datasets.values(): dataset.transform = model.transform
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) for split, dataset in datasets.items()}
    output.mkdir(parents=True); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    print('Preprocessing:', model.data_config, flush=True)
    features = {}
    for split, loader in loaders.items():
        xx, yy = [], []
        with torch.no_grad():
            for index, (images, labels) in enumerate(loader):
                xx.append(model.encode(images.cuda()).cpu()); yy.append(labels)
                if (index + 1) % 25 == 0: print(split, min((index + 1) * args.batch_size, len(loader.dataset)), '/', len(loader.dataset), flush=True)
        features[split] = torch.cat(xx), torch.cat(yy)
    save_file({split + '_' + key: tensor for split, pair in features.items() for key, tensor in zip(('features', 'labels'), pair)}, str(output / 'features.safetensors'))
    x, y = [value.cuda() for value in features['train']]
    vx, vy = [value.cuda() for value in features['validation']]
    counts = torch.bincount(y, minlength=len(prep['labels'])).float()
    loss_weights = counts.sum() / (len(counts) * counts)
    if args.nine_categories:
        expected_weights = torch.tensor([prep['training_class_weights'][label] for label in prep['labels']], device='cuda')
        if not torch.allclose(loss_weights, expected_weights, atol=1e-6): raise ValueError('Loss weights differ from train-only prepared weights.')
        loss_weights = expected_weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.02, weight_decay=.01)
    best = -1.; best_head = None; best_head_epoch = 0; head_history = []
    for epoch in range(1, args.head_epochs + 1):
        optimizer.zero_grad(set_to_none=True); loss = loss_fn(model.head(x), y)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Non-finite loss.')
        loss.backward(); optimizer.step()
        with torch.no_grad(): cached_logits = model.head(vx).cpu()
        metric = classification_metrics(vy.tolist(), cached_logits.argmax(1).tolist(), prep['labels'])
        head_history.append({'epoch': epoch, 'training_loss': float(loss.detach()), 'validation': metric})
        if metric['macro_f1'] > best:
            best = metric['macro_f1']; best_head_epoch = epoch
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    model.head.load_state_dict(best_head)

    def evaluate():
        model.eval(); outputs = []; truth = []
        with torch.no_grad():
            for images, labels in loaders['validation']:
                outputs.append(model(images.cuda()).cpu()); truth.extend(labels.tolist())
        logits = torch.cat(outputs)
        return classification_metrics(truth, logits.argmax(1).tolist(), prep['labels']), logits, torch.tensor(truth)

    baseline_metric, _, _ = evaluate()
    with torch.no_grad(): expected = model.head(vx).argmax(1).cpu()
    if not torch.equal(evaluate()[1].argmax(1), expected): raise RuntimeError('Frozen-feature/full-network mismatch.')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output / 'frozen_model.safetensors'))
    print('Frozen head:', baseline_metric['accuracy'], baseline_metric['macro_f1'], 'epoch', best_head_epoch, flush=True)
    configure_finetuning(model)
    protected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(('backbone.stem.', 'backbone.stages.0.', 'backbone.stages.1.', 'backbone.stages.2.'))}
    optimizer = torch.optim.AdamW([{'params': model.backbone.stages[-1].parameters(), 'lr': .0001},
        {'params': model.backbone.head.parameters(), 'lr': .0001}, {'params': model.head.parameters(), 'lr': .001}], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    train = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42))
    best_metric, best_epoch = baseline_metric, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; history = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.eval(); losses = []
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): loss = loss_fn(model(images.cuda()), labels.cuda())
            if not bool(torch.isfinite(loss)): raise RuntimeError('Non-finite loss.')
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
        if not torch.equal(tensor, model.state_dict()[key].cpu()): raise RuntimeError('Protected stage changed: ' + key)
    model.load_state_dict(best_state); final, logits, truth = evaluate()
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(output / 'model.safetensors'))
    model.load_state_dict(load_file(str(output / 'model.safetensors'), device='cuda'))
    repeated = evaluate()[1]
    if not torch.equal(repeated.argmax(1), logits.argmax(1)): raise RuntimeError('Saved checkpoint reload mismatch.')
    save_file({'logits': logits, 'labels': truth}, str(output / 'validation_logits.safetensors'))
    comparisons = {}
    for name in (() if args.nine_categories else ('efficientnet_b0_finetune_01', 'convnext_tiny_finetune_01')):
        other = json.loads((ROOT / 'runtime/training' / name / 'report.json').read_text())
        if other['manifest_sha256'] != prep['manifest_sha256'] or other['labels'] != prep['labels']: raise ValueError('Comparison dataset mismatch.')
        comparisons[name] = {'validation': other['validation'], 'checkpoint_sha256': other['checkpoint_sha256']}
    report = {'architecture': ARCHITECTURE, 'labels': prep['labels'], 'manifest_sha256': prep['manifest_sha256'],
        'dataset_status': prep.get('status'), 'group_counts': prep.get('group_counts'),
        'scope_sha256': prep.get('scope_sha256'), 'similarity_triage': prep.get('similarity_triage'),
        'unsupported_prediction_heads': ['batik', 'handloom', 'pattern_type', 'weave', 'fabric_texture'],
        'comparison_scope': 'Frozen versus partially fine-tuned candidate on the same prepared validation split. Different datasets/class sets must not be compared as accuracy improvements.',
        'counts': prep['counts'], 'seed': 42, 'pretrained_repository': REPOSITORY, 'pretrained_revision': REVISION,
        'pretrained_sha256': hashlib.sha256(pretrained.read_bytes()).hexdigest(), 'license': LICENSE,
        'checkpoint_sha256': hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest(),
        'frozen_checkpoint_sha256': hashlib.sha256((output / 'frozen_model.safetensors').read_bytes()).hexdigest(),
        'frozen_validation': baseline_metric, 'frozen_best_epoch': best_head_epoch, 'head_history': head_history,
        'validation': final, 'best_epoch': best_epoch, 'history': history, 'comparison': comparisons,
        'training': {'method': 'L2-normalized frozen 768-D features/head followed by last-stage/head-normalization fine-tuning',
            'head_epochs': args.head_epochs, 'finetune_epochs': args.finetune_epochs, 'head_initial_lr': .02,
            'backbone_lr': .0001, 'head_lr': .001, 'weight_decay': .01, 'batch_size': args.batch_size,
            'data_config': model.data_config, 'preprocessing': repr(model.transform), 'extra_augmentation': False,
            'head_precision': 'float32', 'finetune_precision': 'AMP float16', 'evaluation_precision': 'float32',
            'class_weights': {label: float(loss_weights[i]) for i, label in enumerate(prep['labels'])},
            'frozen_state_verified': True, 'reload_predictions_verified': True},
        'device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'timm_version': timm.__version__, 'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(), 'test_evaluated': False, 'test_images_opened': 0,
        'activated_in_app': False, 'notice': 'Validation-selected research candidate. Pretraining datasets/transforms differ from earlier models; not a matched-pretraining architecture test. No independent generalization, calibration or OOD claim. Non-commercial checkpoint.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'accuracy': final['accuracy'], 'macro_f1': final['macro_f1'], 'best_epoch': best_epoch}), flush=True)


if __name__ == '__main__': main()
