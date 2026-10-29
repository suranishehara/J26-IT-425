from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_nlu_data import validate, evaluate


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'data/nlu/starter.json').read_text(encoding='utf-8'))

    def test_starter_explicit_labels_and_offsets(self):
        rows = validate(self.data)
        self.assertEqual(len(rows), 28)
        for row in rows:
            for entity in row['entities']:
                self.assertEqual(row['normalized_text'][entity['start']:entity['end']], entity['mention'])
        report = evaluate(rows)
        self.assertEqual(report['dev']['n'], 14)
        self.assertLess(report['dev']['intent_accuracy'], 1)  # Retain hard examples, not only rule matches.

    def test_family_and_conversation_leakage(self):
        for field in ('family_id', 'conversation_id'):
            data = deepcopy(self.data)
            data['records'][2][field] = data['records'][0][field]
            with self.assertRaises(ValueError): validate(data)

    def test_duplicate_text_and_pending_annotations(self):
        data = deepcopy(self.data)
        data['records'][1]['text'] = data['records'][0]['text'].upper()
        with self.assertRaises(ValueError): validate(data)
        template = json.loads((ROOT / 'data/nlu/collection_template.json').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError): validate(template)

    def test_invalid_label_slot_and_span(self):
        for key, value in (('intent', 'invented'), ('slots', {'occasion': 'invented'}), ('split', 'test')):
            data = deepcopy(self.data)
            data['records'][0][key] = value
            with self.assertRaises(ValueError): validate(data)
        data = deepcopy(self.data)
        data['records'][0]['entities'][0]['mention'] = 'missing phrase'
        with self.assertRaises(ValueError): validate(data)

    def test_repeated_mentions_require_offsets(self):
        row = deepcopy(self.data['records'][0])
        row.update(text='cotton or cotton', entities=[{'mention': 'cotton', 'value': 'cotton', 'negated': False}], slots={})
        data = {'provenance': 'test', 'records': [row]}
        with self.assertRaises(ValueError): validate(data)
        row['entities'][0]['start'] = 10
        self.assertEqual(validate(data)[0]['entities'][0]['end'], 16)

    def test_gold_labels_are_not_parser_generated(self):
        from unittest.mock import patch
        with patch('scripts.prepare_nlu_data.parse', side_effect=AssertionError('Must not infer gold')):
            rows = validate(self.data)
        self.assertEqual(rows[2]['intent'], 'recommend')


if __name__ == '__main__': unittest.main()
