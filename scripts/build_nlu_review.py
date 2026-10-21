"""Build an offline, human-operated review page; suggestions are never gold labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.nlp import parse, ALIASES
from scripts.prepare_nlu_data import INTENTS


def make_pack(queue, sources, source_sha256):
    by_id = {row['id']: row for row in sources}
    if len(by_id) != len(sources) or len({row['id'] for row in queue}) != len(queue):
        raise ValueError('Duplicate review/source IDs.')
    conversations = defaultdict(list)
    for row in sources:
        conversations[row['conversation_id']].append(row)
    items = []
    for item in queue:
        if not item['source_ids'] or any(identifier not in by_id for identifier in item['source_ids']):
            raise ValueError('Review item has missing source links.')
        suggestion = parse(item['text'])
        suggestion = {key: suggestion[key] for key in ('intent', 'entities', 'slots', 'remove_slots')}
        suggestion['slots'] = {key: value for key, value in suggestion['slots'].items()
                               if key not in ('terms', 'excluded_terms')}
        items.append({**item, 'suggestion': suggestion,
                      'review': {'status': 'pending', 'intent': None, 'entities': None, 'slots': None,
                                 'remove_slots': None, 'family_id': None, 'notes': '', 'reviewer_id': None,
                                 'assisted': False, 'reviewed_at': None}})
    payload = {'schema': 'nlu_review_v1', 'source_sha256': source_sha256,
               'purpose': 'Human annotation draft, no split or model training approval.',
               'intents': list(INTENTS), 'canonical_entities': list(ALIASES),
               'items': items, 'conversations': dict(conversations)}
    payload['pack_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def render(pack):
    template = (ROOT / 'review_tool/index.html').read_text(encoding='utf-8')
    script = (ROOT / 'review_tool/review.js').read_text(encoding='utf-8')
    # Untrusted workbook text is JSON data, never executable markup.
    data = json.dumps(pack, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return template.replace('/*PACK_DATA*/', data).replace('/*REVIEW_SCRIPT*/', script)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, default=ROOT / 'runtime/evaluations/volunteer_intake_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.intake / 'audit_report.json').read_text(encoding='utf-8'))
    queue = json.loads((args.intake / 'annotation_review_queue.json').read_text(encoding='utf-8'))
    sources = json.loads((args.intake / 'source_records.json').read_text(encoding='utf-8'))
    pack = make_pack(queue, sources, audit['source_sha256'])
    page = render(pack)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'review.html').write_text(page, encoding='utf-8')
    (args.output / 'review_pack.json').write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Offline review page:', (args.output / 'review.html').resolve())
    print('Pending items:', len(pack['items']), '; nothing labelled reviewed automatically.')


if __name__ == '__main__': main()
