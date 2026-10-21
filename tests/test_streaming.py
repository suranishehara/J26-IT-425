"""Mock transport mechanics, not a model quality or real-generation benchmark."""
import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch
from backend import dialogue, llm
from backend.app import Handler


@contextmanager
def server(handler):
    service = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    try: yield f'http://127.0.0.1:{service.server_port}'
    finally: service.shutdown(); service.server_close(); thread.join()


class FakeOllama(BaseHTTPRequestHandler):
    next_chunk = threading.Event()
    last_payload = None
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'models':[{'name':'fixture:tiny','digest':'fixture-hash','size':1}, {'name':'fixture:cloud'}]}).encode())
    def do_POST(self):
        type(self).last_payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'message':{'thinking':'PRIVATE FIXTURE REASONING','content':''},'done':False}).encode()+b'\n'); self.wfile.flush()
        self.wfile.write(json.dumps({'message':{'content':'Hello '},'done':False}).encode()+b'\n'); self.wfile.flush()
        self.next_chunk.wait(4)
        self.wfile.write(json.dumps({'message':{'content':'there. [sl_temple]'},'done':True}).encode()+b'\n'); self.wfile.flush()


class StreamingTests(unittest.TestCase):
    def setUp(self):
        patcher=patch.object(dialogue,'SESSIONS',{}); patcher.start(); self.addCleanup(patcher.stop)
        FakeOllama.next_chunk.clear()

    def session_payload(self):
        session=dialogue.new_session()
        return {'session_id':session['session_id'], 'expected_revision':0, 'message':'What should I wear to a temple?', 'generation':'ollama','model':'fixture:tiny'}

    def test_actual_incremental_transport_and_commit(self):
        with server(FakeOllama) as provider, patch.object(llm,'BASE',provider), server(Handler) as app:
            payload=self.session_payload()
            request=Request(app+'/api/chat/stream',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=10) as response:
                start=json.loads(response.readline()); self.assertEqual(start['type'],'start')
                first=json.loads(response.readline()); self.assertEqual(first,{'type':'delta','text':'Hello '})
                self.assertEqual(dialogue.SESSIONS[payload['session_id']]['revision'],0)
                FakeOllama.next_chunk.set()
                rest=[json.loads(line) for line in response if line.strip()]
            self.assertEqual(rest[-1]['type'],'done')
            self.assertEqual(rest[-1]['data']['answer'],'Hello there. [sl_temple]')
            self.assertEqual(rest[-1]['data']['methods']['generation'],'ollama_grounded_llm')
            self.assertEqual(dialogue.SESSIONS[payload['session_id']]['revision'],1)
            self.assertEqual(FakeOllama.last_payload['think'],None)  # Native default for non-Qwen fixture model.
            self.assertNotIn('PRIVATE FIXTURE',rest[-1]['data']['answer'])
            self.assertIn('GROUNDING DATA',FakeOllama.last_payload['messages'][-1]['content'])

    def test_history_is_sent_with_grounding(self):
        payload=self.session_payload(); payload['generation']='baseline'
        dialogue.chat(payload); payload.update(expected_revision=1,message='Why?',generation='ollama')
        prepared=dialogue.prepare(payload)
        messages=llm.messages(prepared,prepared['history'])
        self.assertTrue(any(message['role']=='assistant' for message in messages))
        self.assertIn('current constraints override',messages[0]['content'])
        # Close the generator after start, releasing the reservation.
        events=dialogue.stream_prepared(prepared); next(events); events.close()

    def test_failure_does_not_commit_partial_answer(self):
        def fail(*args):
            yield 'partial'; raise ValueError('fixture failure')
        payload=self.session_payload()
        with patch.object(llm,'generate',fail):
            events=dialogue.stream_prepared(dialogue.prepare(payload))
            self.assertEqual(next(events)['type'],'start'); self.assertEqual(next(events)['type'],'delta')
            with self.assertRaises(ValueError): next(events)
        session=dialogue.SESSIONS[payload['session_id']]
        self.assertEqual(session['revision'],0); self.assertEqual(session['history'],[]); self.assertIsNone(session['busy'])

    def test_same_session_busy_other_session_unblocked(self):
        payload=self.session_payload(); events=dialogue.stream_prepared(dialogue.prepare(payload)); next(events)
        with self.assertRaises(ValueError): dialogue.prepare(payload)
        other=dialogue.new_session()
        self.assertEqual(dialogue.chat({'session_id':other['session_id'],'message':'hello'})['revision'],1)
        events.close(); self.assertIsNone(dialogue.SESSIONS[payload['session_id']]['busy'])

    def test_cloud_models_filtered_and_missing_not_downloaded(self):
        with server(FakeOllama) as provider, patch.object(llm,'BASE',provider):
            self.assertEqual([row['name'] for row in llm.models()['models']],['fixture:tiny'])
            with self.assertRaises(ValueError): list(llm.generate('missing:model',[]))

    def test_baseline_not_fake_tokenized(self):
        payload=self.session_payload(); payload['generation']='baseline'
        events=list(dialogue.stream_prepared(dialogue.prepare(payload)))
        self.assertEqual(sum(event['type']=='delta' for event in events),1)
        self.assertEqual(events[-1]['data']['methods']['generation'],'extractive_template_no_llm')

    def test_invalid_mode_and_model_rejected(self):
        for fields in ({'generation':'paid_api'}, {'generation':'ollama','model':'../../invalid space'}):
            with self.assertRaises(ValueError): dialogue.prepare({**self.session_payload(),**fields})

    def test_unknown_citation_is_flagged_not_called_verified(self):
        payload=self.session_payload()
        with patch.object(llm,'generate',return_value=iter(['Unsupported citation [invented_source].'])):
            result=dialogue.chat(payload)
        self.assertEqual(result['citation_check']['unknown_ids'],['invented_source'])
        self.assertEqual(result['citation_check']['status'],'id_check_only_not_factual_verification')

    def test_http_failure_emits_error_without_committing(self):
        def fail(*args):
            yield 'partial'; raise ValueError('mock failure')
        payload=self.session_payload()
        with patch.object(llm,'generate',fail), server(Handler) as app:
            request=Request(app+'/api/chat/stream',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request) as response: events=[json.loads(line) for line in response if line.strip()]
        self.assertEqual([event['type'] for event in events],['start','delta','error'])
        self.assertEqual(dialogue.SESSIONS[payload['session_id']]['revision'],0)
        self.assertIsNone(dialogue.SESSIONS[payload['session_id']]['busy'])
