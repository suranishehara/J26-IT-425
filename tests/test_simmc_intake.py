import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_simmc import COMMIT, FILES, SPLITS, audit_split, lfs_reference, normalize, prepare


def fixture(identifier, domain='fashion'):
    return {'dialogue_idx': identifier, 'domain': domain, 'dialogue': [
        {'turn_idx': 0, 'transcript': 'Show me blue shirts.',
         'system_transcript': 'Which shirt?', 'transcript_annotated': {
             'act': 'REQUEST:GET', 'act_attributes': {
                 'slot_values': {'color': 'blue'}, 'request_slots': ['price']},
             'disambiguation_label': 1}}]}


class SimmcIntakeTests(unittest.TestCase):
    def test_fashion_filter_preserves_native_annotations(self):
        data = {'dialogue_data': [fixture(1), fixture(2, 'furniture')]}
        before = copy.deepcopy(data)
        rows, summary, ids, texts = audit_split(data)
        self.assertEqual(data, before)
        self.assertEqual(rows, [before['dialogue_data'][0]])
        self.assertEqual(summary['fashion_dialogues'], 1)
        self.assertEqual(summary['fashion_utterances'], 2)
        self.assertEqual(summary['ambiguous_user_turns'], 1)
        self.assertEqual(summary['native_user_act_counts'], {'REQUEST:GET': 1})
        self.assertEqual(summary['user_requested_slot_counts'], {'price': 1})
        self.assertEqual(ids, {1, 2})
        self.assertIn('show me blue shirts.', texts)

    def test_duplicate_ids_and_invalid_domain_rejected(self):
        with self.assertRaises(ValueError):
            audit_split({'dialogue_data': [fixture(1), fixture(1)]})
        with self.assertRaises(ValueError):
            audit_split({'dialogue_data': [fixture(1, 'unreviewed')]})
        bad = fixture(1)
        bad['dialogue'].append(copy.deepcopy(bad['dialogue'][0]))
        with self.assertRaises(ValueError):
            audit_split({'dialogue_data': [bad]})

    def test_lfs_pointer_and_unicode_normalization(self):
        pointer = ('version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 123\n').encode()
        self.assertEqual(lfs_reference(pointer), ('a' * 64, 123))
        self.assertIsNone(lfs_reference(b'{"data": 1}'))
        with self.assertRaises(ValueError):
            lfs_reference(b'version https://git-lfs.github.com/spec/v1\ninvalid')
        self.assertEqual(normalize(' BLUE   Shirt '), 'blue shirt')

    def test_preparation_hash_guard_and_reserved_split_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            raw.mkdir()
            inventory = []
            for relative in FILES:
                if relative in SPLITS.values():
                    split = next(key for key, value in SPLITS.items() if value == relative)
                    content = json.dumps({'split': split, 'dialogue_data': [fixture(list(SPLITS).index(split))]}).encode()
                else:
                    content = b'fixture source'
                target = raw / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                inventory.append({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            (raw / 'download_manifest.json').write_text(json.dumps({
                'complete': True, 'commit': COMMIT, 'files': inventory, 'licence': 'CC-BY-NC-SA-4.0'}), encoding='utf-8')
            result = prepare(raw, root / 'prepared')
            self.assertEqual(result['totals']['fashion_dialogues'], 3)
            self.assertEqual(json.loads((root / 'prepared/devtest.fashion.json').read_text())['split'], 'devtest')
            self.assertEqual(result['cross_split_checks']['train__dev']['shared_normalized_fashion_user_texts'], 1)
            self.assertIn('Not downloaded', result['final_test'])
            (raw / FILES[0]).write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                prepare(raw, root / 'other')


if __name__ == '__main__':
    unittest.main()
