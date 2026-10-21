import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from backend.nlp import parse, update
from backend.retrieval import retrieve, load
from backend.context import adapt
from backend import dialogue
from backend.app import Handler


class NLPTests(unittest.TestCase):
    def test_budget_and_constraints(self):
        result = parse('Suggest something traditional for an outdoor wedding under LKR 15,000')
        self.assertEqual(result['intent'], 'recommend')
        for key,value in {'occasion':'wedding','setting':'outdoor','style':'traditional','budget_lkr':15000}.items(): self.assertEqual(result['slots'][key],value)
        self.assertIsNone(result['confidence'])

    def test_aliases_longest_and_negation(self):
        result = parse('Compare Kandyan saree and osari; not batik but handloom')
        self.assertEqual(result['slots']['terms'], ['osariya', 'handloom'])
        self.assertEqual(result['slots']['excluded_terms'], ['batik'])
        self.assertNotIn('saree', result['slots']['terms'])

    def test_state_retention_and_correction(self):
        state = update({}, parse('Wedding under 15k with batik'))
        state = update(state, parse('Actually office instead, not batik, handloom for hot weather'))
        self.assertEqual(state['occasion'],'office'); self.assertEqual(state['budget_lkr'],15000)
        self.assertEqual(state['weather'],'hot'); self.assertNotIn('batik',state['terms'])
        state = update(state, parse('Remove the budget'))
        self.assertNotIn('budget_lkr',state)

    def test_negated_occasion_removed(self):
        state=update({'occasion':'wedding'},parse('Not wedding'))
        self.assertNotIn('occasion',state)

    def test_invalid_message_and_budget(self):
        for value in ('',None,'x'*2001,'budget 0'):
            with self.assertRaises(ValueError): parse(value)


class RetrievalTests(unittest.TestCase):
    def test_known_query_and_scope(self):
        rows,checksum = retrieve('temple shoulders knees')
        self.assertEqual(rows[0]['id'],'sl_temple'); self.assertEqual(len(checksum),64)
        self.assertTrue(all(row.get('scope') and row.get('source') for row in rows))

    def test_unknown_and_determinism(self):
        self.assertEqual(retrieve('zxqv987')[0],[])
        self.assertEqual(retrieve('batik'),retrieve('batik'))

    def test_knowledge_integrity(self):
        rows,_=load(); self.assertEqual(len(rows),len({row['id'] for row in rows}))


class ConversationTests(unittest.TestCase):
    def setUp(self):
        patcher=patch.object(dialogue,'SESSIONS',{}); patcher.start(); self.addCleanup(patcher.stop)

    def test_isolated_sessions_and_revisions(self):
        a=dialogue.new_session(); b=dialogue.new_session()
        result=dialogue.chat({'session_id':a['session_id'],'message':'Suggest wedding outfit under 15000','expected_revision':0})
        self.assertEqual(result['revision'],1)
        self.assertEqual(dialogue.SESSIONS[b['session_id']]['state'],{})
        with self.assertRaises(ValueError): dialogue.chat({'session_id':a['session_id'],'message':'hot','expected_revision':0})

    def test_unknown_session_and_expiry(self):
        with self.assertRaises(ValueError): dialogue.chat({'session_id':'unknown','message':'hello'})
        session=dialogue.new_session(); dialogue.SESSIONS[session['session_id']]['updated']-=dialogue.TTL+1
        with self.assertRaises(ValueError): dialogue.chat({'session_id':session['session_id'],'message':'hello'})

    def test_context_caveats_and_no_invented_ranking(self):
        session=dialogue.new_session()
        result=dialogue.chat({'session_id':session['session_id'],'message':'Why this design?',
            'component_context':{'component':3,'result':{'property_basis':'typical_profile','designs':[
                {'name':'Shirt','rule_score':7,'reasons':['Matches recorded drapability'],'conflicts':[]}]}}})
        self.assertIn('Typical profile',result['answer']); self.assertIn('7 rule points',result['answer'])
        self.assertEqual(result['methods']['generation'],'extractive_template_no_llm')

    def test_failure_does_not_commit_turn(self):
        session=dialogue.new_session()
        with self.assertRaises(ValueError): dialogue.chat({'session_id':session['session_id'],'message':'why','component_context':{'component':9}})
        self.assertEqual(dialogue.SESSIONS[session['session_id']]['revision'],0)

    def test_no_evidence_abstains(self):
        session=dialogue.new_session()
        result=dialogue.chat({'session_id':session['session_id'],'message':'zxqv987'})
        self.assertEqual(result['evidence'],[]); self.assertIn('do not have matching evidence',result['answer'])


class ContextTests(unittest.TestCase):
    def test_reject_invalid_and_strip_private_fields(self):
        with self.assertRaises(ValueError): adapt({'component':3,'result':{'designs':[{'name':'x','rule_score':True}]}})
        result=adapt({'component':2,'result':{'profile':{'secret':'private'},'recommendations':[{'name':'Saree','score':2,'reasons':[],'image':'private'}]}})
        self.assertNotIn('private',json.dumps(result))


class HTTPTests(unittest.TestCase):
    def test_end_to_end_and_origin_guard(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            def post(path,payload,origin=None):
                headers={'Content-Type':'application/json'}
                if origin: headers['Origin']=origin
                with urlopen(Request(base+path,data=json.dumps(payload).encode(),headers=headers)) as response: return json.load(response)
            with urlopen(base+'/api/health') as response: self.assertFalse(json.load(response)['llm_enabled'])
            session=post('/api/session',{})
            result=post('/api/chat',{'session_id':session['session_id'],'message':'What should I wear to a temple?'})
            self.assertTrue(result['evidence'])
            with self.assertRaises(HTTPError) as error: post('/api/chat',{},'https://example.com')
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(HTTPError): post('/api/chat',{'message':'hello'})
        finally: server.shutdown(); server.server_close(); thread.join()


if __name__=='__main__': unittest.main()
