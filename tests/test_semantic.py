import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import semantic, dialogue, retrieval


class SemanticTests(unittest.TestCase):
    def setUp(self):
        semantic.CACHE.clear()
        self.addCleanup(semantic.CACHE.clear)

    def test_normalization_and_payload(self):
        with patch.object(semantic, 'open_local', return_value=io.BytesIO(b'{"embeddings":[[3,4]]}')) as call:
            self.assertEqual(semantic.embed(['test']), [[.6, .8]])
            self.assertFalse(call.call_args.args[1]['truncate'])
            self.assertEqual(call.call_args.args[0], '/api/embed')

    def test_reject_invalid_embeddings(self):
        for value in ([], [[0, 0]], [[True, 1]], [[1, float('nan')]], [[1], [2, 3]], 'bad'):
            with self.subTest(value=value), patch.object(semantic, 'open_local', return_value=io.BytesIO(json.dumps({'embeddings': value}).encode())):
                with self.assertRaises(ValueError): semantic.embed(['one'])

    def test_cache_identity_corpus_and_no_query_cache(self):
        rows = [{'id': 'a', 'title': 'A', 'text': 'a', 'scope': 'one'},
                {'id': 'b', 'title': 'B', 'text': 'b', 'scope': 'two'}]
        with patch.object(semantic, 'identity', return_value={'name': semantic.MODEL, 'digest': 'v1'}) as identity, \
             patch.object(semantic, 'embed', side_effect=lambda texts: [[1, 0], [0, 1]] if len(texts) == 2 else [[0, 1]]) as embed:
            self.assertEqual(semantic.rank('q', rows, 'corpus1')[0]['id'], 'b')
            semantic.rank('q', rows, 'corpus1')
            self.assertEqual(embed.call_count, 3)  # Public corpus once; every query freshly embedded.
            semantic.rank('q', rows, 'corpus2')
            self.assertEqual(embed.call_count, 5)
            identity.return_value = {'name': semantic.MODEL, 'digest': 'v2'}
            semantic.rank('q', rows, 'corpus2')
            self.assertEqual(embed.call_count, 7)
            self.assertEqual(len(semantic.CACHE), 1)

    def test_rrf_determinism_and_ranks(self):
        rows = [{'id': 'a', 'retrieval_score': 9}, {'id': 'b', 'retrieval_score': 2}]
        result = retrieval.fuse(rows, list(reversed(rows)))
        self.assertEqual([row['id'] for row in result], ['a', 'b'])
        self.assertAlmostEqual(result[0]['retrieval_score'], 1/61 + 1/62)
        self.assertEqual(result[0]['component_ranks'], {'bm25': 1, 'dense': 2})

    def test_missing_model_fails_without_fallback_or_commit(self):
        session = dialogue.new_session()
        self.addCleanup(lambda: dialogue.SESSIONS.pop(session['session_id'], None))
        with patch.object(semantic, 'identity', side_effect=ValueError('not installed')):
            with self.assertRaises(ValueError):
                dialogue.prepare({'session_id': session['session_id'], 'message': 'batik', 'retrieval': 'hybrid'})
        state = dialogue.SESSIONS[session['session_id']]
        self.assertEqual(state['revision'], 0)
        self.assertIsNone(state['busy'])
        self.assertEqual(state['state'], {})

    def test_retrieval_outside_session_lock(self):
        session = dialogue.new_session()
        self.addCleanup(lambda: dialogue.SESSIONS.pop(session['session_id'], None))
        def fake(*args, **kwargs):
            acquired = dialogue.LOCK.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired: dialogue.LOCK.release()
            return [], 'checksum'
        with patch.object(dialogue, 'retrieve', side_effect=fake):
            result = dialogue.chat({'session_id': session['session_id'], 'message': 'hello', 'retrieval': 'dense'})
        self.assertEqual(result['methods']['retrieval'], 'dense')

    def test_invalid_method_and_dimension_mismatch(self):
        with self.assertRaises(ValueError): retrieval.retrieve('hi', method='unimplemented')
        with patch.object(semantic, 'identity', return_value={'name': semantic.MODEL, 'digest': 'v'}), \
             patch.object(semantic, 'embed', side_effect=[[[1, 0]], [[1]]]):
            with self.assertRaises(ValueError):
                semantic.rank('q', [{'id': 'a', 'title': 'a', 'text': 'b', 'scope': 'c'}], 'hash')


if __name__ == '__main__': unittest.main()
