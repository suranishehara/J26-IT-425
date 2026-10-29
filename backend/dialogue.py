"""Bounded transactional conversations with baseline or grounded local LLM replies."""
from copy import deepcopy
import threading
import time
import uuid
from .nlp import parse, update
from .retrieval import retrieve, load
from .context import adapt
from . import llm
from .conversation import plan, retrieval_query
import re

LOCK = threading.Lock()
SESSIONS = {}
TTL = 3600
MAX_SESSIONS = 200
STORE = None  # Enabled only by the app entry point; tests can supply an isolated store.


def snapshot(session):
    return deepcopy({key: session.get(key) for key in
                     ('revision', 'state', 'context', 'history', 'topic_query', 'selected_item_indices')})


def history_list():
    return STORE.list() if STORE is not None else {'enabled': False, 'chats': [], 'deleted': []}


def saved_chat(identifier):
    if STORE is None: raise ValueError('Persistent history is not enabled for this service.')
    with LOCK:
        if SESSIONS.get(identifier, {}).get('busy'): raise ValueError('A response is still in progress for this chat.')
        saved = STORE.get(identifier)
        SESSIONS[identifier] = {**saved['snapshot'], 'busy': None, 'updated': time.monotonic()}
    context = saved['snapshot']['context']
    supplied = None
    if context:
        key = 'recommendations' if context['component'] == 2 else 'designs'
        score = 'score' if context['component'] == 2 else 'rule_score'
        rows = [{**{key: value for key, value in row.items() if key != 'rule_score'}, score: row['rule_score']}
                for row in context['items']]
        supplied = {'component': context['component'], 'result': {key: rows}}
        if context['component'] == 3: supplied['result']['property_basis'] = context['property_basis']
    return {'id': identifier, 'title': saved['title'], 'revision': saved['snapshot']['revision'],
            'state': saved['snapshot']['state'], 'turns': saved['turns'], 'component_context': supplied}


def history_action(action, payload):
    if STORE is None: raise ValueError('Persistent history is not enabled for this service.')
    identifier = payload.get('session_id')
    STORE.identifier(identifier)
    with LOCK:
        if SESSIONS.get(identifier, {}).get('busy'): raise ValueError('Wait for the response to finish before changing this chat.')
        if action == 'rename': STORE.rename(identifier, payload.get('title'))
        elif action == 'delete':
            STORE.set_deleted(identifier, True)
            SESSIONS.pop(identifier, None)
        elif action == 'restore': STORE.set_deleted(identifier, False)
        else: raise ValueError('Unknown history action.')
    return history_list()


def new_session():
    with LOCK:
        now = time.monotonic()
        for key in list(SESSIONS):
            if now - SESSIONS[key]['updated'] > TTL: del SESSIONS[key]
        if len(SESSIONS) >= MAX_SESSIONS: raise ValueError('Session capacity reached. Try again later.')
        identifier = uuid.uuid4().hex
        created = {'revision': 0, 'state': {}, 'context': None, 'history': [], 'busy': None, 'updated': now,
                   'topic_query': None, 'selected_item_indices': []}
        if STORE is not None: STORE.create(identifier, snapshot(created))
        SESSIONS[identifier] = created
    return {'session_id': identifier, 'revision': 0, 'state': {}}


def answer(parsed, state, evidence, context, conversation=None):
    conversation = conversation or {}
    if conversation.get('clarification'): return conversation['clarification']
    parts = []
    if parsed['intent'] == 'greeting': return 'Hello! Ask about garments, fabrics, occasions or an explanation of a Component 2/3 result.'
    if context and (conversation.get('selected_item_indices') or parsed['intent'] in ('explain', 'compare')):
        selected = conversation.get('selected_item_indices') or list(range(min(2 if parsed['intent'] == 'compare' else 1, len(context['items']))))
        items = [context['items'][index] for index in selected]
        for index, row in zip(selected, items):
            parts.append(f"Option {index + 1} — {row['name']}:")
            parts.append(f"Supplied Component {context['component']} result: {row['name']} — {row['rule_score']:g} rule points.")
            parts.extend('Reported reason: ' + reason for reason in row['reasons'])
            parts.extend('Reported conflict: ' + reason for reason in row['conflicts'])
        if context['property_basis'] == 'typical_profile': parts.append('The physical properties come from a Typical profile, not measurements of your uploaded fabric.')
        parts.append('This is a read-only explanation of the JSON you supplied, not a freshly computed recommendation. Rule points are not probabilities or verified fit.')
    elif parsed['intent'] == 'explain' and not conversation.get('topic_reused'):
        parts.append('To explain a particular ranking, supply its Component 2 or 3 result in the context panel. I will not invent its reasons.')
    if evidence:
        parts.append('Retrieved evidence (scope applies):')
        for row in evidence:
            parts.append(f"[{row['id']}] {row['text']} Scope: {row['scope']}")
    elif not context:
        parts.append('I do not have matching evidence in the current small knowledge base. Please specify the garment, fabric or occasion; unsupported claims will remain unanswered.')
    if parsed['intent'] == 'compare' and not context:
        parts.append('There is not enough evidence here to declare one textile or tradition universally better. Which event, venue requirements and actual garment options should be compared?')
    if parsed['intent'] in ('recommend', 'update_preferences'):
        parts.append('I have updated the conversation constraints; no upstream recommendation was changed or executed.')
        if not state.get('occasion'): parts.append('What occasion should I use?')
        elif not state.get('budget_lkr'): parts.append('Do you have a budget in LKR, or should it remain unspecified?')
        else: parts.append('For actual ranked options, obtain a Component 2/3 response and supply it here. Live recommendation execution is not connected in this milestone.')
    return '\n\n'.join(parts)


def prepare(payload):
    if not isinstance(payload, dict): raise ValueError('Expected a chat object.')
    parsed = parse(payload.get('message'))
    identifier = payload.get('session_id')
    if not isinstance(identifier, str): raise ValueError('Start a session first.')
    mode = payload.get('generation', 'baseline')
    retrieval_method = payload.get('retrieval', 'bm25')
    if retrieval_method not in ('bm25', 'dense', 'hybrid'):
        raise ValueError('Choose bm25, dense or hybrid retrieval.')
    if mode not in ('baseline', 'ollama'): raise ValueError('Choose baseline or local Ollama generation.')
    model = payload.get('model')
    if mode == 'ollama' and (not isinstance(model, str) or not llm.MODEL_NAME.fullmatch(model)):
        raise ValueError('Choose an installed local model.')
    with LOCK:
        session = SESSIONS.get(identifier)
        if STORE is not None and (session is None or time.monotonic() - session['updated'] > TTL):
            stored = STORE.get(identifier)
            session = {**stored['snapshot'], 'busy': None, 'updated': time.monotonic()}
            SESSIONS[identifier] = session
        if session is None or time.monotonic() - session['updated'] > TTL: raise ValueError('Session missing/expired. Start a new conversation.')
        revision = payload.get('expected_revision')
        if revision is not None and (type(revision) is not int or revision != session['revision']): raise ValueError('Conversation revision changed; refresh or start a new conversation.')
        if session['revision'] >= 100: raise ValueError('Conversation turn limit reached; start a new conversation.')
        if session.get('busy'): raise ValueError('A response is already in progress for this session.')
        context = adapt(payload['component_context']) if 'component_context' in payload else session['context']
        state = update(session['state'], parsed)
        previous_selection = session.get('selected_item_indices') if context == session['context'] else []
        conversation = plan(payload['message'], parsed, context, session.get('topic_query'), previous_selection)
        query, topic_query = retrieval_query(payload['message'], state, conversation, session.get('topic_query'))
        token = uuid.uuid4().hex; session['busy'] = token
        snapshot = deepcopy({'session_id': identifier, 'revision': session['revision'], 'state': state,
                         'understanding': parsed, 'context': context,
                         'history': session.get('history', []), 'token': token, 'message': payload['message'],
                         'context_used': context is not None, 'retrieval_method': retrieval_method,
                         'mode': mode, 'model': model, 'conversation': conversation, 'topic_query': topic_query})
    try:
        # Embedding network calls must not hold the global session lock.
        if conversation['clarification']:
            _, checksum = load()
            evidence = []
        else:
            evidence, checksum = retrieve(query, method=retrieval_method)
        snapshot.update(evidence=evidence, knowledge_sha256=checksum)
        return snapshot
    except Exception:
        with LOCK:
            if session.get('busy') == token: session['busy'] = None
        raise


def stream_prepared(prepared):
    """Commit only completed responses. No global session lock during model I/O."""
    identifier, token = prepared['session_id'], prepared['token']
    try:
        metadata = {key: prepared[key] for key in ('session_id', 'state', 'understanding', 'evidence', 'knowledge_sha256', 'context_used')}
        metadata['conversation'] = prepared.get('conversation', {})
        clarification = metadata['conversation'].get('clarification')
        metadata['methods'] = {'nlu': 'english_rules_v0.1', 'retrieval': prepared.get('retrieval_method', 'bm25'),
                               'generation': 'ollama_grounded_llm' if prepared['mode'] == 'ollama' else 'extractive_template_no_llm'}
        if clarification:
            metadata['methods']['generation'] = 'deterministic_clarification_no_llm'
            metadata['methods']['retrieval'] = 'not_run_clarification'
            metadata['evidence'] = []
        yield {'type': 'start', 'data': metadata}
        parts = []
        if clarification:
            chunks = [clarification]
        elif prepared['mode'] == 'ollama':
            chunks = llm.generate(prepared['model'], llm.messages(prepared, prepared['history']))
        else:
            # One actual baseline answer, never fake typing or an artificial wait.
            chunks = [answer(prepared['understanding'], prepared['state'], prepared['evidence'], prepared['context'], metadata['conversation'])]
        for content in chunks:
            parts.append(content); yield {'type': 'delta', 'text': content}
        response = ''.join(parts)
        allowed = {row['id'] for row in metadata['evidence']}
        citations = set(re.findall(r'\[([a-z][a-z0-9_]*)\]', response))
        metadata['citation_check'] = {'cited_ids': sorted(citations), 'unknown_ids': sorted(citations - allowed),
                                      'missing_citation': bool(allowed and not citations and prepared['understanding']['intent'] != 'greeting'),
                                      'status': 'id_check_only_not_factual_verification'}
        metadata['model'] = prepared['model'] if prepared['mode'] == 'ollama' and not clarification else None
        with LOCK:
            session = SESSIONS.get(identifier)
            if session is None or session.get('busy') != token or session['revision'] != prepared['revision']:
                raise ValueError('Session changed while generating. Start a new conversation.')
            history = session.get('history', []) + [{'role': 'user', 'content': prepared['message']}, {'role': 'assistant', 'content': response[:4000]}]
            completed_session = {**session, 'revision': session['revision'] + 1, 'state': prepared['state'], 'context': prepared['context'],
                                 'history': history[-8:], 'busy': None, 'updated': time.monotonic(),
                                 'topic_query': session.get('topic_query') if clarification else prepared.get('topic_query'),
                                 'selected_item_indices': metadata['conversation'].get('selected_item_indices', [])}
            metadata.update(revision=completed_session['revision'], answer=response)
            if STORE is not None:
                STORE.append(identifier, session['revision'], snapshot(completed_session),
                             {'user': prepared['message'], 'result': metadata})
            session.update(completed_session)
        yield {'type': 'done', 'data': metadata}
    finally:
        with LOCK:
            session = SESSIONS.get(identifier)
            if session is not None and session.get('busy') == token: session['busy'] = None


def chat(payload):
    for event in stream_prepared(prepare(payload)):
        if event['type'] == 'done': return event['data']
