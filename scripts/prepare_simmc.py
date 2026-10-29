"""Explicit official SIMMC 2.1 download and text-only fashion audit; no training.

Pins upstream commit, resolves/verifies the metadata Git-LFS object, preserves
provider partitions and annotations. Does not access the official final test.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '6e3de4e3d11e2914edc3ab9123e6b0eb88933237'
REPOSITORY = 'https://github.com/facebookresearch/simmc2'
RAW = f'https://raw.githubusercontent.com/facebookresearch/simmc2/{COMMIT}/'
MEDIA = f'https://media.githubusercontent.com/media/facebookresearch/simmc2/{COMMIT}/'
SPLITS = {name: f'data/simmc2.1_dials_dstc11_{name}.json' for name in ('train', 'dev', 'devtest')}
FILES = [*SPLITS.values(), 'data/fashion_prefab_metadata_all.json',
         'LICENSE', 'README.md', 'data/README.md']


def fetch(url, limit=100_000_000):
    request = Request(url, headers={'User-Agent': 'Component4-SIMMC-research-intake/1.0'})
    with urlopen(request, timeout=60) as response:
        chunks, size = [], 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError('Download exceeds expected safety limit.')
            chunks.append(chunk)
    return b''.join(chunks)


def lfs_reference(raw):
    if not raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
        return None
    match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize (\d+)\n?', raw)
    if not match:
        raise ValueError('Invalid Git-LFS pointer.')
    return match[1].decode(), int(match[2])


def write_json(path, value):
    # Generated artifacts only; source files are edited with apply_patch.
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write('\n')


def download(destination):
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {'repository': REPOSITORY, 'commit': COMMIT,
                  'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                  'complete': False, 'licence': 'CC-BY-NC-SA-4.0', 'files': []}
    for relative in FILES:
        print('Downloading', relative, flush=True)
        url = RAW + relative
        raw = fetch(url)
        pointer = lfs_reference(raw)
        if pointer:
            digest, size = pointer
            url = MEDIA + relative
            raw = fetch(url)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('Git-LFS object hash/size mismatch.')
        if relative.endswith('.json'):
            json.loads(raw)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(raw)
        provenance['files'].append({'path': relative, 'url': url, 'bytes': len(raw),
                                    'sha256': hashlib.sha256(raw).hexdigest(),
                                    'lfs_object_verified': bool(pointer)})
    provenance['complete'] = True
    write_json(destination / 'download_manifest.json', provenance)
    return provenance


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def audit_split(data):
    if not isinstance(data.get('dialogue_data'), list):
        raise ValueError('Missing dialogue_data list.')
    fashion, identifiers, texts = [], set(), set()
    acts, slot_keys, requests, domains = Counter(), Counter(), Counter(), Counter()
    ambiguous = missing = turns = 0
    for row in data['dialogue_data']:
        domain = row.get('domain')
        if domain not in ('fashion', 'furniture'):
            raise ValueError('Unexpected dialogue domain; review upstream format.')
        domains[domain] += 1
        identifier = row['dialogue_idx']
        if identifier in identifiers:
            raise ValueError('Duplicate dialogue ID within provider split.')
        identifiers.add(identifier)
        if domain != 'fashion':
            continue
        fashion.append(row)
        positions = set()
        for turn in row['dialogue']:
            if turn['turn_idx'] in positions:
                raise ValueError('Repeated turn index.')
            positions.add(turn['turn_idx'])
            turns += 1
            for field in ('transcript', 'system_transcript'):
                if not isinstance(turn.get(field), str) or not turn[field].strip():
                    raise ValueError('Missing dialogue text.')
            texts.add(normalize(turn['transcript']))
            annotation = turn.get('transcript_annotated', {})
            if not isinstance(annotation.get('act'), str):
                missing += 1
                continue
            acts[annotation['act']] += 1
            attrs = annotation.get('act_attributes', {})
            slot_keys.update(attrs.get('slot_values', {}).keys())
            requests.update(attrs.get('request_slots', []))
            ambiguous += annotation.get('disambiguation_label') == 1
    summary = {'source_dialogues': len(data['dialogue_data']),
               'source_domain_counts': dict(domains), 'fashion_dialogues': len(fashion),
               'fashion_turn_pairs': turns, 'fashion_utterances': 2 * turns,
               'distinct_normalized_user_texts': len(texts),
               'ambiguous_user_turns': ambiguous, 'missing_user_act_labels': missing,
               'native_user_act_counts': dict(sorted(acts.items())),
               'user_slot_key_counts': dict(sorted(slot_keys.items())),
               'user_requested_slot_counts': dict(sorted(requests.items()))}
    return fashion, summary, identifiers, texts


def prepare(source, output):
    manifest = json.loads((source / 'download_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest.get('commit') != COMMIT:
        raise ValueError('Incomplete or unexpected source manifest.')
    for row in manifest['files']:
        raw = (source / row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Downloaded source hash mismatch: ' + row['path'])
    if {row['path'] for row in manifest['files']} != set(FILES):
        raise ValueError('Unexpected source inventory.')
    output.mkdir(parents=True, exist_ok=False)
    report = {'dataset': 'SIMMC 2.1 fashion text-only subset', 'commit': COMMIT,
              'source_manifest_sha256': hashlib.sha256((source / 'download_manifest.json').read_bytes()).hexdigest(),
              'created_utc': datetime.now(timezone.utc).isoformat(), 'licence': manifest['licence'],
              'split_policy': 'Provider partitions unchanged; train only for fitting, dev for selection; devtest held out from fitting/tuning',
              'final_test': 'Not downloaded, inspected or evaluated', 'splits': {}, 'artifacts': [],
              'limitations': [
                  'Fashion/furniture simulator-generated flows with manual paraphrasing, not independent Sri Lankan conversations.',
                  'Some catalogue attributes are synthetic; not verified cultural or material facts.',
                  'Text-only subset has no scene assets; visual reference tasks require additional context.',
                  'Native act/slot values retained, not automatically mapped to the app seven intents or token spans.',
                  'Cross-split repeated texts are diagnostics, not proof of duplicate conversations or independent speakers.',
                  'No model training, benchmark scores, production activation or Sri Lankan expert approval.']}
    ids, texts = {}, {}
    for split, relative in SPLITS.items():
        data = json.loads((source / relative).read_text(encoding='utf-8'))
        fashion, stats, ids[split], texts[split] = audit_split(data)
        stats['provider_header'] = {key: data.get(key) for key in ('split', 'version', 'year', 'domain')}
        report['splits'][split] = stats
        target = output / (split + '.fashion.json')
        subset = {**data, 'dialogue_data': fashion,
                  'subset_provenance': {'source': relative, 'commit': COMMIT,
                                        'filter': "dialogue.domain == 'fashion'", 'text_only': True}}
        write_json(target, subset)
        report['artifacts'].append({'path': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    report['cross_split_checks'] = {}
    for a, b in (('train', 'dev'), ('train', 'devtest'), ('dev', 'devtest')):
        overlap = ids[a] & ids[b]
        report['cross_split_checks'][a + '__' + b] = {
            'source_dialogue_id_overlap': len(overlap),
            'shared_normalized_fashion_user_texts': len(texts[a] & texts[b])}
        if overlap:
            raise ValueError('Provider dialogue ID leakage; review before use.')
    report['totals'] = {key: sum(row[key] for row in report['splits'].values())
                        for key in ('fashion_dialogues', 'fashion_turn_pairs', 'fashion_utterances')}
    write_json(output / 'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'runtime/datasets/raw/simmc21_official_01')
    parser.add_argument('--output', type=Path, default=ROOT / 'runtime/datasets/simmc21_fashion_01')
    parser.add_argument('--download', action='store_true', help='Explicitly download official files into a new raw directory')
    args = parser.parse_args()
    if args.output.exists() or (args.download and args.raw.exists()):
        raise ValueError('Choose fresh destination directories; existing artifacts are never overwritten.')
    if args.download:
        download(args.raw)
    report = prepare(args.raw, args.output)
    print(json.dumps({'totals': report['totals'], 'cross_split_checks': report['cross_split_checks'],
                      'output': str(args.output.resolve()), 'training': 'not_started'}, indent=2))


if __name__ == '__main__':
    main()
