import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from backend.history import HistoryStore
from backend import dialogue
from backend.app import Handler
from test_streaming import server


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'history.sqlite3'
        self.store = HistoryStore(self.path)
        for name, value in (('STORE', self.store), ('SESSIONS', {})):
            patcher = patch.object(dialogue, name, value); patcher.start(); self.addCleanup(patcher.stop)

    def chat(self, identifier, text, **kwargs):
        return dialogue.chat({'session_id': identifier, 'message': text, **kwargs})

    def test_restart_restores_transcript_state_and_topic(self):
        identifier = dialogue.new_session()['session_id']
        first = self.chat(identifier, 'What does black tie mean for a wedding?')
        dialogue.SESSIONS.clear()
        reopened = dialogue.saved_chat(identifier)
        self.assertEqual(reopened['revision'], 1)
        self.assertEqual(reopened['turns'][0]['result']['answer'], first['answer'])
        self.assertEqual(reopened['state']['occasion'], 'wedding')
        second = self.chat(identifier, 'Explain that more simply')
        self.assertTrue(second['conversation']['topic_reused'])
        self.assertEqual(len(self.store.get(identifier)['turns']), 2)

    def test_manual_rename_survives_first_turn_and_restart(self):
        identifier = dialogue.new_session()['session_id']
        dialogue.history_action('rename', {'session_id': identifier, 'title': 'My wedding options'})
        self.chat(identifier, 'hello')
        self.assertEqual(HistoryStore(self.path).get(identifier)['title'], 'My wedding options')

    def test_auto_title_and_soft_delete_restore(self):
        identifier = dialogue.new_session()['session_id']
        self.chat(identifier, 'What is batik?')
        self.assertEqual(self.store.get(identifier)['title'], 'What is batik?')
        dialogue.history_action('delete', {'session_id': identifier})
        self.assertEqual(self.store.list()['chats'], [])
        self.assertEqual(self.store.list()['deleted'][0]['id'], identifier)
        with self.assertRaises(ValueError): dialogue.saved_chat(identifier)
        with self.assertRaises(ValueError): self.chat(identifier, 'why')
        dialogue.history_action('restore', {'session_id': identifier})
        self.assertEqual(dialogue.saved_chat(identifier)['revision'], 1)

    def test_filtered_component_context_can_resume(self):
        identifier = dialogue.new_session()['session_id']
        supplied = {'component': 3, 'result': {'property_basis': 'typical_profile', 'profile': {'secret': 'PRIVATE'}, 'designs': [
            {'name': 'Shirt', 'rule_score': 7, 'reasons': ['one'], 'image': 'PRIVATE'},
            {'name': 'Kurta', 'rule_score': 3, 'reasons': ['two']}]}}
        self.chat(identifier, 'Explain the second option', component_context=supplied)
        dialogue.SESSIONS.clear()
        reopened = dialogue.saved_chat(identifier)
        self.assertNotIn('PRIVATE', json.dumps(reopened))
        self.assertEqual(reopened['component_context']['result']['property_basis'], 'typical_profile')
        followup = self.chat(identifier, 'The other one', component_context=reopened['component_context'])
        self.assertEqual(followup['conversation']['selected_item_indices'], [0])

    def test_failure_and_db_failure_do_not_commit(self):
        identifier = dialogue.new_session()['session_id']
        def fail(*args):
            yield 'partial'; raise ValueError('fixture failure')
        with patch.object(dialogue.llm, 'generate', fail), self.assertRaises(ValueError):
            self.chat(identifier, 'What is cotton?', generation='ollama', model='fixture:tiny')
        self.assertEqual(self.store.get(identifier)['turns'], [])
        with patch.object(self.store, 'append', side_effect=sqlite3.OperationalError('disk full')), self.assertRaises(sqlite3.Error):
            self.chat(identifier, 'hello')
        self.assertEqual(dialogue.SESSIONS[identifier]['revision'], 0)
        self.assertIsNone(dialogue.SESSIONS[identifier]['busy'])
        self.assertEqual(self.store.get(identifier)['snapshot']['revision'], 0)

    def test_revision_conflict_and_busy_mutations_rejected(self):
        identifier = dialogue.new_session()['session_id']
        prepared = dialogue.prepare({'session_id': identifier, 'message': 'hello'})
        for action in ('rename', 'delete', 'restore'):
            with self.assertRaises(ValueError): dialogue.history_action(action, {'session_id': identifier, 'title': 'x'})
        events = dialogue.stream_prepared(prepared); next(events); events.close()
        self.chat(identifier, 'hello')
        with self.assertRaises(ValueError):
            self.store.append(identifier, 0, {'revision': 1}, {'user': 'stale', 'result': {'revision': 1}})
        self.assertEqual(len(self.store.get(identifier)['turns']), 1)

    def test_invalid_titles_and_ids(self):
        identifier = dialogue.new_session()['session_id']
        for title in ('', 'x'*81, 'line\nbreak', None):
            with self.assertRaises(ValueError): self.store.rename(identifier, title)
        with self.assertRaises(ValueError): self.store.get('../history.sqlite3')
        self.store.rename(identifier, '<script>alert(1)</script>')
        self.assertEqual(self.store.get(identifier)['title'], '<script>alert(1)</script>')  # UI uses textContent.

    def test_history_http_and_origin_guards(self):
        with server(Handler) as base:
            def post(route, value, origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin: headers['Origin'] = origin
                with urlopen(Request(base+route, data=json.dumps(value).encode(), headers=headers)) as response:
                    return json.load(response)
            identifier = post('/api/session', {})['session_id']
            post('/api/chat', {'session_id': identifier, 'message': 'hello'})
            with urlopen(base+'/api/chats') as response: self.assertEqual(len(json.load(response)['chats']), 1)
            with urlopen(base+'/api/chats/'+identifier) as response: self.assertEqual(json.load(response)['revision'], 1)
            post('/api/chats/rename', {'session_id': identifier, 'title': 'Test chat'})
            with self.assertRaises(HTTPError) as error:
                post('/api/chats/delete', {'session_id': identifier}, 'https://example.com')
            self.assertEqual(error.exception.code, 403)
            post('/api/chats/delete', {'session_id': identifier})
            post('/api/chats/restore', {'session_id': identifier})

    def test_history_frontend_actions(self):
        node = shutil.which('node')
        if node is None: self.skipTest('Node unavailable')
        result = subprocess.run([node, str(Path(__file__).with_name('history_ui.cjs'))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
