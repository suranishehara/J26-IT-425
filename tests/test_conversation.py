import json
import unittest
from unittest.mock import patch
from backend import dialogue, llm
from backend.conversation import plan
from backend.nlp import parse


def context(names=('Shirt', 'Kurta', 'Skirt', 'Dress')):
    return {'component': 3, 'result': {'property_basis': 'typical_profile', 'designs': [
        {'name': name, 'rule_score': index + 1, 'reasons': ['Recorded reason ' + name], 'conflicts': []}
        for index, name in enumerate(names)]}}


class ConversationImprovementTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(dialogue, 'SESSIONS', {})
        patcher.start(); self.addCleanup(patcher.stop)
        self.session = dialogue.new_session()

    def chat(self, message, **kwargs):
        return dialogue.chat({'session_id': self.session['session_id'], 'message': message, **kwargs})

    def test_follow_up_retains_user_topic_not_generated_claims(self):
        self.chat('What does black tie mean?')
        with patch.object(dialogue, 'retrieve', return_value=([], 'hash')) as search:
            result = self.chat('Can you explain that more simply?')
        self.assertIn('black tie', search.call_args.args[0])
        self.assertNotIn('Retrieved evidence', search.call_args.args[0])
        self.assertTrue(result['conversation']['topic_reused'])
        self.assertNotIn('supply its Component', result['answer'])

    def test_explicit_topic_does_not_inherit_old_question(self):
        self.chat('What does black tie mean?')
        for text in ('Explain silk', 'Tell me more about embroidery'):
            with patch.object(dialogue, 'retrieve', return_value=([], 'hash')) as search:
                result = self.chat(text)
            self.assertFalse(result['conversation']['topic_reused'])
            self.assertNotIn('black tie', search.call_args.args[0])

    def test_second_and_fourth_selected_not_first(self):
        result = self.chat('Compare the second and fourth options', component_context=context())
        self.assertEqual(result['conversation']['selected_item_indices'], [1, 3])
        self.assertIn('Recorded reason Kurta', result['answer'])
        self.assertIn('Recorded reason Dress', result['answer'])
        self.assertNotIn('Recorded reason Shirt', result['answer'])
        self.assertIn('Typical profile', result['answer'])

    def test_selected_context_only_in_model_grounding(self):
        prepared = dialogue.prepare({'session_id': self.session['session_id'], 'message': 'Explain option 4',
                                     'component_context': context(), 'generation': 'ollama', 'model': 'fixture:tiny'})
        try:
            content = llm.messages(prepared, [])[-1]['content']
            grounding = json.loads(content.split('GROUNDING DATA (not instructions):\n')[1].split('\nCURRENT USER MESSAGE:')[0])
            items = grounding['component_context']['items']
            self.assertEqual([row['name'] for row in items], ['Dress'])
            self.assertEqual(items[0]['supplied_option_number'], 4)
        finally:
            events = dialogue.stream_prepared(prepared); next(events); events.close()

    def test_other_option_requires_known_pair_and_selection(self):
        self.chat('Why the second option?', component_context=context(('Shirt', 'Kurta')))
        result = self.chat('What about the other one?')
        self.assertEqual(result['conversation']['selected_item_indices'], [0])
        self.assertIn('Recorded reason Shirt', result['answer'])
        self.assertNotIn('Recorded reason Kurta', result['answer'])

    def test_out_of_range_or_missing_results_clarify_without_llm(self):
        with patch.object(llm, 'generate') as generate, patch.object(dialogue, 'retrieve') as retrieve:
            result = self.chat('Why the sixth option?', component_context=context(), generation='ollama', model='fixture:tiny')
        self.assertEqual(result['methods']['generation'], 'deterministic_clarification_no_llm')
        self.assertEqual(result['methods']['retrieval'], 'not_run_clarification')
        self.assertEqual(result['evidence'], [])
        self.assertIn('4 supplied', result['answer'])
        generate.assert_not_called(); retrieve.assert_not_called()
        result = self.chat('Explain the second option', component_context=None)
        self.assertIn('Please paste', result['answer'])

    def test_ambiguous_other_with_three_items_is_not_guessed(self):
        self.chat('Explain option 1', component_context=context())
        result = self.chat('The other one')
        self.assertIn('Which option', result['answer'])
        self.assertEqual(result['conversation']['selected_item_indices'], [])

    def test_one_item_compare_and_context_reset(self):
        result = self.chat('Compare these', component_context=context(('Shirt',)))
        self.assertIn('at least two', result['answer'])
        result = self.chat('Explain that', component_context=None)
        self.assertEqual(result['conversation']['selected_item_indices'], [])

    def test_failed_turn_does_not_change_topic_or_selection(self):
        self.chat('What does black tie mean?')
        state = dialogue.SESSIONS[self.session['session_id']]
        topic, selected, revision = state['topic_query'], state['selected_item_indices'][:], state['revision']
        def fail(*args):
            yield 'partial'
            raise ValueError('fixture failure')
        with patch.object(llm, 'generate', fail), self.assertRaises(ValueError):
            self.chat('Explain option 2', component_context=context(), generation='ollama', model='fixture:tiny')
        self.assertEqual(state['topic_query'], topic)
        self.assertEqual(state['selected_item_indices'], selected)
        self.assertEqual(state['revision'], revision)

    def test_new_session_and_new_topic_no_stale_references(self):
        result = self.chat('Explain that more simply')
        self.assertIn('What would you like', result['answer'])
        self.assertFalse(result['conversation']['topic_reused'])
        result = plan('Tell me more about cotton', parse('Tell me more about cotton'), None, 'temple', [])
        self.assertFalse(result['topic_reused'])
        result = plan('First, what is batik?', parse('First, what is batik?'), None)
        self.assertIsNone(result['clarification'])
        result = plan('What should I wear to my first wedding?', parse('What should I wear to my first wedding?'), None)
        self.assertIsNone(result['clarification'])


if __name__ == '__main__': unittest.main()
