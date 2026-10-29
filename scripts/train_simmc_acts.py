"""Train text-only native SIMMC dialogue-act baselines and DeBERTa; no app activation.

Fixed train/dev, dev-only model selection, deferred devtest access after selection.
Inputs contain only current user text + previous completed turn, never current reply
or annotation values. Training dependencies are lazy so app/tests remain lightweight.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'microsoft/deberta-v3-small'
REVISION = 'a36c739020e01763fe789b4b85e2df55d6180012'
WEIGHTS_SHA256 = 'd4b0ebcc7799a2c2dde1a39199cb0ed709f2c6b63a48b8c05133b77955f271a9'


def local_pretrained():
    """Pinned official weights, explicit progress and upstream LFS hash check."""
    from huggingface_hub import hf_hub_download
    directory = ROOT / 'runtime/models/deberta_v3_small_source'
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'spm.model', 'tokenizer_config.json'):
        target = directory / filename
        if not target.exists():
            content = Path(hf_hub_download(MODEL, filename, revision=REVISION)).read_bytes()
            with target.open('xb') as output:
                output.write(content)
    weights = directory / 'pytorch_model.bin'
    if not weights.exists():
        partial = directory / 'pytorch_model.bin.partial'
        offset = partial.stat().st_size if partial.exists() else 0
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/pytorch_model.bin?download=true'
        request = Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urlopen(request, timeout=60) as response:
            if offset and response.status != 206:
                raise ValueError('Server refused resumable download; partial file preserved.')
            with partial.open('ab' if offset else 'xb') as output:
                count = offset
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block); count += len(block)
                    if count // (16 * 1024 * 1024) != (count - len(block)) // (16 * 1024 * 1024):
                        print('Pretrained weights downloaded bytes', count, '/ 286059269', flush=True)
                    if count > 286059269:
                        raise ValueError('Oversized pretrained download.')
        if partial.stat().st_size != 286059269 or digest(partial) != WEIGHTS_SHA256:
            raise ValueError('Pretrained weight size/SHA-256 mismatch.')
        partial.rename(weights)
    if digest(weights) != WEIGHTS_SHA256:
        raise ValueError('Pinned pretrained weights failed integrity check.')
    return directory


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    # Generated experiment outputs, not source edits.
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def examples(data):
    rows = []
    for conversation in data['dialogue_data']:
        if conversation['domain'] != 'fashion':
            raise ValueError('Only the prepared fashion subset is supported.')
        previous = None
        for turn in conversation['dialogue']:
            # Fixed word budgets keep the current user at the front under truncation.
            current = ' '.join(turn['transcript'].split()[:64])
            text = 'Current user: ' + current
            if previous:
                text += '\nPrevious user: ' + ' '.join(previous['transcript'].split()[-24:])
                text += '\nPrevious assistant: ' + ' '.join(previous['system_transcript'].split()[-32:])
            rows.append({'dialogue_id': conversation['dialogue_idx'], 'turn_idx': turn['turn_idx'],
                         'text': text, 'label': turn['transcript_annotated']['act']})
            previous = turn
    return rows


def read_split(dataset, split, audit):
    filename = split + '.fashion.json'
    expected = next(row['sha256'] for row in audit['artifacts'] if row['path'] == filename)
    if digest(dataset / filename) != expected:
        raise ValueError('Prepared dataset hash mismatch.')
    return examples(json.loads((dataset / filename).read_text(encoding='utf-8')))


def metrics(truth, prediction, labels):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    return {'accuracy': float(accuracy_score(truth, prediction)),
            'macro_f1': float(f1_score(truth, prediction, labels=labels, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(truth, prediction, labels=labels, average='weighted', zero_division=0)),
            'per_class': classification_report(truth, prediction, labels=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, prediction, labels=labels).tolist(),
            'confusion_label_order': labels, 'n': len(truth)}


def select_best(items):
    # First candidate wins exact ties; no assessment information is accepted here.
    return max(range(len(items)), key=lambda index: items[index]['macro_f1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'runtime/datasets/simmc21_fashion_01')
    parser.add_argument('--output', type=Path, default=ROOT / 'runtime/training/simmc_acts_01')
    parser.add_argument('--torch-site', type=Path, help='Optional read-only dependency site; never modifies that environment')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--baseline-run', type=Path, help='Reuse this script\'s completed baseline after a neural setup failure; fitting data/policy must match')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new run directory; existing experiments are never overwritten.')
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 32:
        raise ValueError('Invalid training configuration.')
    if args.torch_site:
        sys.path.append(str(args.torch_site.resolve()))
    import os
    os.environ['HF_HOME'] = str(ROOT / 'runtime/models/huggingface')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import joblib
    import numpy as np
    import sklearn
    import torch
    import transformers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for this experiment; no silent CPU full fine-tuning.')
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    audit = json.loads((args.dataset / 'audit.json').read_text(encoding='utf-8'))
    train, dev = [read_split(args.dataset, split, audit) for split in ('train', 'dev')]
    if {row['dialogue_id'] for row in train} & {row['dialogue_id'] for row in dev}:
        raise ValueError('Train/dev conversation leakage.')
    labels = sorted({row['label'] for row in train})
    if set(row['label'] for row in dev) - set(labels):
        raise ValueError('Unknown validation label.')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'training',
              'seed': 42, 'labels': labels, 'task': 'SIMMC native user dialogue-act classification, text-only',
              'input_policy': 'Current user first 64 words, previous user last 24 and previous assistant last 32; no current reply, slots, objects or labels in features',
              'source_audit_sha256': digest(args.dataset / 'audit.json'),
              'splits': {key: {'turns': len(rows), 'dialogues': len({row['dialogue_id'] for row in rows})} for key, rows in [('train', train), ('dev', dev)]},
              'packages': {'torch': torch.__version__, 'transformers': transformers.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'hardware': torch.cuda.get_device_name(0), 'torch_source': str(Path(torch.__file__).resolve()),
              'pretrained_source': {'id': MODEL, 'revision': REVISION, 'licence': 'MIT'},
              'data_licence': 'CC-BY-NC-SA-4.0',
              'selection': 'Best validation macro-F1; no devtest predictions until all candidates are frozen',
              'limitations': ['One seed; no independent Sri Lankan evaluation.', 'Text-only partial task, not full official multimodal benchmark.',
                             'Native labels differ from app seven intents; no production integration.',
                             'Word-budget features identical but transformer additionally truncates at max_length tokens.',
                             'Public dataset may have been exposed during pretrained model training; source/speaker independence not established.',
                             'No token-level NER, dialogue-state or cultural explanation accuracy claimed.']}
    save(args.output / 'configuration.json', report)
    xtrain, xdev = [[row['text'] for row in rows] for rows in (train, dev)]
    ytrain, ydev = [[row['label'] for row in rows] for rows in (train, dev)]
    majority = Counter(ytrain).most_common(1)[0][0]
    report['majority'] = {'label': majority, 'dev': metrics(ydev, [majority] * len(dev), labels)}
    linear_results, linear_models = [], []
    if args.baseline_run:
        prior = json.loads((args.baseline_run / 'progress.json').read_text(encoding='utf-8'))
        for key in ('source_audit_sha256', 'input_policy', 'labels', 'seed'):
            if prior[key] != report[key]:
                raise ValueError('Prior baseline task/data mismatch.')
        if prior.get('status') != 'training' or 'assessment' in prior['tfidf_logistic']:
            raise ValueError('Reuse is restricted to a pre-assessment interrupted run.')
        baseline = joblib.load(args.baseline_run / 'tfidf_logistic.joblib')
        report['tfidf_logistic'] = prior['tfidf_logistic']
        report['baseline_reused_from'] = str(args.baseline_run.resolve())
        print('Reusing completed baseline from interrupted neural setup', flush=True)
    for c in (() if args.baseline_run else (.5, 2.0)):
        baseline = Pipeline([
            ('features', FeatureUnion([
                ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
                ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=50000, sublinear_tf=True))])),
            ('classifier', LogisticRegression(C=c, max_iter=500, random_state=42))])
        print('Fitting TF-IDF logistic regression C=', c, flush=True)
        baseline.fit(xtrain, ytrain)
        score = metrics(ydev, baseline.predict(xdev), labels)
        linear_results.append(score); linear_models.append(baseline)
        print('Validation', score['accuracy'], score['macro_f1'], flush=True)
    if not args.baseline_run:
        linear_index = select_best(linear_results)
        baseline = linear_models[linear_index]
        report['tfidf_logistic'] = {'C': (.5, 2.0)[linear_index], 'dev': linear_results[linear_index],
                                   'candidates': [{'C': c, 'dev': score} for c, score in zip((.5, 2.0), linear_results)]}
    joblib.dump(baseline, args.output / 'tfidf_logistic.joblib')
    save(args.output / 'progress.json', report)
    del linear_models
    print('Loading pinned pretrained DeBERTa', flush=True)
    # SentencePiece's Windows native file loader fails on this workspace's en dash.
    # Only the small vocabulary is copied to a managed ASCII temporary directory;
    # downloaded model/cache/checkpoints remain under Component 4.
    pretrained_directory = local_pretrained()
    vocabulary = pretrained_directory / 'spm.model'
    vocab_temp = tempfile.TemporaryDirectory(prefix='c4_spm_')
    ascii_vocab = Path(vocab_temp.name) / 'spm.model'
    if not str(ascii_vocab).isascii():
        raise ValueError('SentencePiece needs an ASCII temporary vocabulary path on this Windows installation.')
    ascii_vocab.write_bytes(Path(vocabulary).read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(pretrained_directory, local_files_only=True, use_fast=False, vocab_file=str(ascii_vocab))
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_directory, local_files_only=True, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)}, use_safetensors=False).cuda()
    mapping = {label: index for index, label in enumerate(labels)}
    encoded = {}
    for name, texts in [('train', xtrain), ('dev', xdev)]:
        print('Tokenizing', name, flush=True)
        encoded[name] = tokenizer(texts, max_length=args.max_length, truncation=True, padding='max_length', return_tensors='pt')
    def loader(split, rows, shuffle=False):
        tensors = encoded[split]
        values = [tensors[key] for key in tensors]
        target = torch.tensor([mapping[row['label']] for row in rows])
        dataset = torch.utils.data.TensorDataset(*values, target)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                           num_workers=0, generator=torch.Generator().manual_seed(42)), list(tensors)
    training_loader, keys = loader('train', train, True)
    validation_loader, _ = loader('dev', dev)
    def predict(data_loader):
        model.eval(); predictions = []
        with torch.inference_mode():
            for batch in data_loader:
                inputs = {key: value.cuda() for key, value in zip(keys, batch[:-1])}
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(**inputs).logits
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite prediction logits.')
                predictions.extend(labels[index] for index in logits.argmax(-1).cpu().tolist())
        return predictions
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    steps = len(training_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda')
    epoch_results, best = [], -1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train(); loss_total = 0; epoch_start = time.monotonic()
        for step, batch in enumerate(training_loader):
            inputs = {key: value.cuda() for key, value in zip(keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**inputs, labels=batch[-1].cuda()).loss
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_total += float(loss.detach())
            if step % 200 == 0:
                print(f'Epoch {epoch + 1}/{args.epochs} step {step}/{len(training_loader)} loss {float(loss.detach()):.4f}', flush=True)
        score = metrics(ydev, predict(validation_loader), labels)
        epoch_results.append(score)
        print('Epoch', epoch + 1, 'validation', score['accuracy'], score['macro_f1'], 'seconds', time.monotonic() - epoch_start, flush=True)
        if score['macro_f1'] > best:
            best = score['macro_f1']
            model.save_pretrained(args.output / 'deberta', safe_serialization=True)
            tokenizer.save_pretrained(args.output / 'deberta')
        report['deberta'] = {'epochs': epoch_results, 'selected_epoch': select_best(epoch_results) + 1,
                             'dev': epoch_results[select_best(epoch_results)], 'learning_rate': 2e-5,
                             'batch_size': args.batch_size, 'max_length': args.max_length, 'precision': 'CUDA AMP FP16'}
        save(args.output / 'progress.json', report)
    del optimizer, model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(args.output / 'deberta', local_files_only=True).cuda()
    reloaded_dev = predict(validation_loader)
    if metrics(ydev, reloaded_dev, labels)['macro_f1'] != report['deberta']['dev']['macro_f1']:
        raise ValueError('Saved checkpoint validation predictions did not reproduce.')
    # Selection is complete. Only now open the held-out assessment file.
    save(args.output / 'selection_frozen.json', {'labels': labels, 'linear_C': report['tfidf_logistic']['C'],
         'deberta_epoch': report['deberta']['selected_epoch'], 'checkpoint_sha256': digest(args.output / 'deberta/model.safetensors'),
         'heldout_not_yet_opened_by_training_script': True})
    assessment = read_split(args.dataset, 'devtest', audit)
    if ({row['dialogue_id'] for row in train} | {row['dialogue_id'] for row in dev}) & {row['dialogue_id'] for row in assessment}:
        raise ValueError('Assessment conversation leakage.')
    if set(row['label'] for row in assessment) - set(labels):
        raise ValueError('Unknown assessment label.')
    xassessment = [row['text'] for row in assessment]; truth = [row['label'] for row in assessment]
    encoded['devtest'] = tokenizer(xassessment, max_length=args.max_length, truncation=True, padding='max_length', return_tensors='pt')
    assessment_loader, _ = loader('devtest', assessment)
    predictions = {'majority': [majority] * len(truth), 'tfidf_logistic': baseline.predict(xassessment).tolist(),
                   'deberta': predict(assessment_loader)}
    for name, values in predictions.items():
        report[name]['assessment'] = metrics(truth, values, labels)
    with (args.output / 'assessment_predictions.jsonl').open('x', encoding='utf-8') as output:
        for index, row in enumerate(assessment):
            output.write(json.dumps({'dialogue_id': row['dialogue_id'], 'turn_idx': row['turn_idx'],
                                     'truth': row['label'], **{name: values[index] for name, values in predictions.items()}}) + '\n')
    report.update(status='completed', elapsed_seconds=time.monotonic() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint_sha256=digest(args.output / 'deberta/model.safetensors'),
                  baseline_sha256=digest(args.output / 'tfidf_logistic.joblib'),
                  checkpoint_reload_verified=True, production_activated=False)
    report['splits']['devtest'] = {'turns': len(assessment), 'dialogues': len({row['dialogue_id'] for row in assessment})}
    save(args.output / 'report.json', report)
    print(json.dumps({name: report[name]['assessment'] for name in predictions}, indent=2), flush=True)


if __name__ == '__main__':
    main()
