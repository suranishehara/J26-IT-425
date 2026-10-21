from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_volunteer_workbook import audit, text_key


def record(identifier, participant, conversation, text, intent='greeting'):
    return {'id': identifier, 'participant_id': participant, 'conversation_id': conversation,
            'text': text, 'provided_intent': intent}


class IntakeTests(unittest.TestCase):
    def test_duplicates_link_participants_transitively(self):
        rows = [record('a', 'P1', 'C1', 'hello'), record('b', 'P1', 'C1', 'shirt', 'recommend'),
                record('c', 'P2', 'C2', 'HELLO'), record('d', 'P2', 'C2', 'goodbye'),
                record('e', 'P3', 'C3', 'goodbye')]
        report, queue, manifest, _ = audit(rows)
        self.assertEqual(report['indivisible_group_count'], 1)
        self.assertEqual(report['split_status'], 'blocked')
        self.assertEqual(report['unique_texts'], 3)
        self.assertEqual(sum(row['occurrence_count'] for row in manifest), 5)
        self.assertEqual(len(queue), 3)

    def test_whole_conversation_stays_grouped(self):
        rows = [record('a', 'P1', 'shared', 'one'), record('b', 'P2', 'shared', 'two')]
        self.assertEqual(audit(rows)[0]['indivisible_group_count'], 1)

    def test_conflicts_not_relabelled_and_no_fabricated_annotations(self):
        report, queue, _, _ = audit([record('a', 'P1', 'C1', 'hi'),
                                     record('b', 'P2', 'C2', 'hi', 'clarify')])
        self.assertEqual(len(report['conflicting_exact_text_labels']), 1)
        self.assertEqual(queue[0]['provided_intents'], ['clarify', 'greeting'])
        for name in ('reviewed_intent', 'entities', 'slots', 'remove_slots', 'split'):
            self.assertIsNone(queue[0][name])
        self.assertFalse(report['model_trained'])

    def test_disjoint_groups_are_not_automatically_approved(self):
        rows = [record('a', 'P1', 'C1', 'red shirt'), record('b', 'P2', 'C2', 'blue trousers'),
                record('c', 'P3', 'C3', 'green scarf')]
        report, queue, _, _ = audit(rows)
        self.assertEqual(report['indivisible_group_count'], 3)
        self.assertEqual(report['split_status'], 'requires_family_review_and_class_coverage_check')
        self.assertTrue(all(row['split'] is None for row in queue))

    def test_normalization_and_reproducibility(self):
        self.assertEqual(text_key('  HELLO\tthere '), 'hello there')
        rows = [record('a', 'P1', 'C1', 'hi')]
        self.assertEqual(audit(rows), audit(rows))

    def test_duplicate_ids_or_empty_input_rejected(self):
        with self.assertRaises(ValueError): audit([])
        with self.assertRaises(ValueError):
            audit([record('a', 'P1', 'C1', 'hi'), record('a', 'P2', 'C2', 'bye')])


if __name__ == '__main__': unittest.main()
