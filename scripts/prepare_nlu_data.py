"""Validate explicit annotations, export normalized spans and evaluate unchanged rules.

Does not train a model or infer gold labels from the parser being evaluated.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.nlp import ALIASES, normalize, parse

INTENTS = ('recommend', 'explain', 'compare', 'update_preferences', 'knowledge_question', 'greeting', 'clarify')
SLOTS = {'occasion', 'weather', 'setting', 'style', 'colours', 'budget_lkr'}
VALUES = {'occasion': {'wedding', 'office', 'festival', 'casual', 'temple'},
          'weather': {'hot', 'cool', 'rainy'}, 'setting': {'outdoor', 'indoor'},
          'style': {'traditional', 'formal', 'casual', 'western'}}


def validate(data):
    if not isinstance(data.get('provenance'), str) or not data['provenance'].strip():
        raise ValueError('Describe annotation provenance.')
    if not isinstance(data.get('records'), list) or not data['records']:
        raise ValueError('Provide labelled records.')
    identifiers, texts, groups, output = set(), set(), {}, []
    for row in data['records']:
        for key in ('id', 'family_id', 'conversation_id', 'text'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Missing ' + key)
        if row.get('annotation_status') == 'pending' or row.get('permission_confirmed') is False:
            raise ValueError('Pending/unpermitted collection record: ' + row['id'])
        text = normalize(row['text'])
        if len(text) > 2000 or row['id'] in identifiers or text in texts:
            raise ValueError('Duplicate ID/text or oversized message: ' + row['id'])
        if row.get('split') not in ('train', 'dev'):
            raise ValueError('Only train/dev are accepted here; independently reserved final tests stay separate.')
        for field in ('family_id', 'conversation_id'):
            group = (field, row[field])
            if group in groups and groups[group] != row['split']:
                raise ValueError('Split leakage through ' + field)
            groups[group] = row['split']
        if row.get('intent') not in INTENTS:
            raise ValueError('Choose one defined primary intent.')
        if not isinstance(row.get('entities'), list) or not isinstance(row.get('slots'), dict) or not isinstance(row.get('remove_slots'), list):
            raise ValueError('Explicit entity, slot and removal annotations are required.')
        if set(row['slots']) - SLOTS or set(row['remove_slots']) - SLOTS:
            raise ValueError('Unknown scalar slot/removal.')
        for key, value in row['slots'].items():
            if key in VALUES and (not isinstance(value, str) or value not in VALUES[key]):
                raise ValueError('Invalid normalized slot value.')
            if key == 'budget_lkr' and (type(value) not in (int, float) or not 0 < value <= 10_000_000):
                raise ValueError('Invalid LKR budget.')
            if key == 'colours' and (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)):
                raise ValueError('Colours must be a list of normalized labels.')
        entities = []
        for entity in row['entities']:
            mention = entity.get('mention')
            if not isinstance(mention, str) or not mention or entity.get('value') not in ALIASES or type(entity.get('negated')) is not bool:
                raise ValueError('Invalid explicit fashion-term annotation.')
            mention = normalize(mention)
            # Repeated mentions require a supplied normalized start; never guess occurrence.
            start = entity.get('start')
            if start is None:
                if text.count(mention) != 1:
                    raise ValueError('Mention must be unique or supply its normalized start offset.')
                start = text.index(mention)
            if type(start) is not int or start < 0 or text[start:start + len(mention)] != mention:
                raise ValueError('Entity span does not match normalized text.')
            end = start + len(mention)
            if any(start < item['end'] and end > item['start'] for item in entities):
                raise ValueError('Overlapping entity spans.')
            entities.append({'type': 'fashion_term', 'start': start, 'end': end, 'mention': mention,
                             'value': entity['value'], 'negated': entity['negated']})
        identifiers.add(row['id']); texts.add(text)
        output.append({**row, 'normalized_text': text, 'entities': entities})
    return output


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}


def expected_slots(row):
    return {**row['slots'], 'terms': sorted({item['value'] for item in row['entities'] if not item['negated']}),
            'excluded_terms': sorted({item['value'] for item in row['entities'] if item['negated']})}


def evaluate(rows):
    reports = {}
    for split in ('train', 'dev'):
        subset = [row for row in rows if row['split'] == split]
        confusion = {label: {other: 0 for other in INTENTS} for label in INTENTS}
        tp = fp = fn = joint = 0
        details = []
        for row in subset:
            prediction = parse(row['text'])
            confusion[row['intent']][prediction['intent']] += 1
            def signatures(entities):
                return {(item['start'], item['end'], item['value'], item['negated']) for item in entities}
            gold, predicted = signatures(row['entities']), signatures(prediction['entities'])
            tp += len(gold & predicted); fp += len(predicted - gold); fn += len(gold - predicted)
            actual_slots = {key: sorted(value) if isinstance(value, list) else value for key, value in prediction['slots'].items()}
            slots_ok = actual_slots == expected_slots(row) and set(prediction['remove_slots']) == set(row['remove_slots'])
            joint += slots_ok
            details.append({'id': row['id'], 'gold_intent': row['intent'], 'predicted_intent': prediction['intent'],
                            'entities_exact': gold == predicted, 'slots_and_removals_exact': slots_ok,
                            'gold_slots': expected_slots(row), 'predicted_slots': prediction['slots']})
        per_class = {}
        for label in INTENTS:
            correct = confusion[label][label]
            per_class[label] = {**prf(correct, sum(confusion[other][label] for other in INTENTS if other != label),
                                    sum(confusion[label][other] for other in INTENTS if other != label)),
                                'support': sum(confusion[label].values())}
        reports[split] = {'n': len(subset), 'intent_counts': dict(Counter(row['intent'] for row in subset)),
                          'intent_accuracy': sum(confusion[label][label] for label in INTENTS) / len(subset) if subset else None,
                          'intent_macro_f1': sum(value['f1'] for value in per_class.values()) / len(INTENTS) if subset else None,
                          'macro_scope': 'All seven fixed intent labels; zero division gives zero.',
                          'per_class': per_class, 'confusion': confusion, 'entity_span_value_negation': prf(tp, fp, fn),
                          'slot_and_removal_exact_rate': joint / len(subset) if subset else None, 'details': details}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/nlu/starter.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes(); data = json.loads(raw); rows = validate(data)
    report = {'notice': data['provenance'], 'method': 'english_rules_v0.1', 'model_trained': False,
              'source_sha256': hashlib.sha256(raw).hexdigest(),
              'rules_sha256': hashlib.sha256((ROOT / 'backend/nlp.py').read_bytes()).hexdigest(),
              'recorded_utc': datetime.now(timezone.utc).isoformat(), 'splits': evaluate(rows)}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (('annotations.json', {'provenance': data['provenance'], 'records': rows}), ('baseline_report.json', report)):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({split: {key: item[key] for key in ('n', 'intent_accuracy', 'intent_macro_f1', 'slot_and_removal_exact_rate')}
                      for split, item in report['splits'].items()}, indent=2))
    print('Author-written development data only; no model trained. Output:', args.output.resolve())


if __name__ == '__main__': main()
