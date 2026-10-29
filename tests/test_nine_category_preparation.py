from collections import Counter
import random
import unittest
from unittest.mock import patch
from backend.category_dataset import grouped_split, training_class_weights, development_images


class NineCategoryPreparationTests(unittest.TestCase):
    def test_training_requires_provisional_split_acknowledgement(self):
        from scripts.train_convnextv2 import main
        with patch('sys.argv', ['train', '--nine-categories']):
            with self.assertRaisesRegex(ValueError, 'provisional development split'):
                main()

    def rows(self):
        return [{'path': f'{label}_{group}_im_{view}.png', 'label': label, 'source_key': f'{label}_{group}',
                 'file_sha256': f'file-{label}-{group}-{view}', 'decoded_sha256': f'pixels-{label}-{group}-{view}'}
                for label in ('a', 'b') for group in range(5) for view in range(2)]

    def test_deterministic_grouped_membership(self):
        records = self.rows(); original, _ = grouped_split(records, ['a', 'b'])
        random.Random(1).shuffle(records)
        repeated, _ = grouped_split(records, ['a', 'b'])
        self.assertEqual(original, repeated)
        families = {}
        for row in original:
            self.assertEqual(families.setdefault(row['source_key'], row['split']), row['split'])
        for split in ('train', 'validation', 'test'):
            self.assertEqual({r['label'] for r in original if r['split'] == split}, {'a', 'b'})

    def test_duplicate_excluded_and_entire_families_linked(self):
        records = self.rows()
        records.append(dict(records[0], path='a_copy_im_1.png', source_key='a_copy'))
        records.append({'path': 'a_copy_im_2.png', 'source_key': 'a_copy', 'label': 'a', 'file_sha256': 'extra', 'decoded_sha256': 'extra-pixels'})
        accepted, excluded = grouped_split(records, ['a', 'b'])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]['reason'], 'exact_duplicate_copy')
        relatives = [r for r in accepted if r['source_key'] in ('a_0', 'a_copy')]
        self.assertEqual(len({r['group_id'] for r in relatives}), 1)
        self.assertEqual(len({r['split'] for r in relatives}), 1)
        self.assertEqual(len({r['decoded_sha256'] for r in accepted}), len(accepted))

    def test_conflicting_duplicate_labels_and_provider_relations_excluded(self):
        records = self.rows(); records.append(dict(records[0], path='b_bad.png', source_key='b_bad', label='b'))
        _, excluded = grouped_split(records, ['a', 'b'])
        self.assertEqual(len(excluded), 3)
        self.assertTrue(all(row['reason'].startswith('conflicting_labels') for row in excluded))
        _, excluded = grouped_split(self.rows(), ['a', 'b'], blocked_paths={'a_0_im_1.png'})
        self.assertEqual(len(excluded), 2)

    def test_weights_use_training_only(self):
        rows = [{'label': 'a', 'split': 'train'}] * 8 + [{'label': 'b', 'split': 'train'}] * 2
        rows += [{'label': 'b', 'split': 'test'}] * 100
        weights = training_class_weights(rows, ['a', 'b'])
        self.assertEqual(weights, {'a': .625, 'b': 2.5})
        self.assertAlmostEqual(8 * weights['a'], 2 * weights['b'])

    def test_test_rows_forbidden_for_development(self):
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            development_images({}, [{'split': 'test'}], None)
