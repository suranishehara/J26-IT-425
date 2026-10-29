"""Conservative follow-up/reference rules. No learned NLU or factual verifier."""
import re
from .nlp import normalize

ORDINALS = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
            'sixth': 5, 'seventh': 6, 'eighth': 7, 'ninth': 8, 'tenth': 9}
FOLLOW_UP = re.compile(r'^(?:why\??|why (?:is |was )?that\??|(?:can you |could you |please )?explain (?:that|it)(?:\b.*)?|'
                       r'(?:can you |could you |please )?(?:make|say) (?:that|it) (?:simpler|shorter|clearer)(?:\b.*)?|'
                       r'(?:can you |could you |please )?tell me more(?:\b.*)?|what does that mean\??|'
                       r'(?:can you |could you )?(?:simplify|summarize|summarise) (?:that|it)(?:\b.*)?)$', re.I)


def plan(message, parsed, context, previous_query=None, previous_selection=None):
    text = normalize(message).strip()
    # Do not bring an earlier topic into an explicitly named new fabric/occasion.
    explicit = bool(parsed['entities'] or any(key not in ('terms', 'excluded_terms') for key in parsed['slots'])
                    or parsed['remove_slots'])
    if re.search(r'\b(?:about|regarding)\s+(?!(?:that|it|this)\b)\S+', text): explicit = True
    follow_up = bool(FOLLOW_UP.fullmatch(text)) and not explicit
    inherit = follow_up and bool(previous_query)
    references = []
    for match in re.finditer(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b'
                             r'(?=\s+(?:ones?|options?|designs?|recommendations?|results?|items?|choices?|and)\b|[?.!]*$)|'
                             r'\b(?:option|design|recommendation|result|item)\s*(?:number\s*)?(\d+)\b', text):
        value = ORDINALS[match[1]] if match[1] else int(match[2]) - 1
        if value not in references: references.append(value)
    other = bool(re.search(r'\b(?:the )?other (?:one|option|design|result)\b', text))
    compare = parsed['intent'] == 'compare' or bool(re.search(r'\bcompare\b', text))
    clarification = None
    selected = []
    count = len(context['items']) if context else 0
    if follow_up and not inherit and not context:
        clarification = 'What would you like me to explain? Please give the topic or supply the recommendation you mean.'
    if references or other:
        if not context:
            clarification = 'Please paste the Component 2 or 3 results you mean in the context panel, so I can identify that option.'
        elif any(index < 0 or index >= count for index in references):
            clarification = f'I have {count} supplied option(s). Which available option would you like me to explain or compare?'
        elif other and not references:
            if count == 2 and isinstance(previous_selection, list) and len(previous_selection) == 1:
                selected = [1 - previous_selection[0]]
            else:
                clarification = 'Which option do you mean? Please give its number from the supplied results.'
        else:
            selected = references
            if not compare and len(selected) > 1:
                clarification = 'Would you like a comparison of those options, or an explanation of just one?'
    elif context and parsed['intent'] in ('explain', 'compare'):
        if compare:
            selected = list(range(min(2, count)))
            if count < 2:
                clarification = 'Please supply at least two results to compare, or ask me to explain the one available option.'
        else:
            selected = previous_selection[:] if follow_up and previous_selection else [0]
    if compare and selected and len(selected) < 2 and not clarification:
        clarification = 'Which second option should I compare it with? Please give its number.'
    return {'follow_up': follow_up, 'topic_reused': inherit,
            'selected_item_indices': [] if clarification else selected,
            'clarification': clarification}


def retrieval_query(message, state, conversation, previous_query):
    topic = previous_query if conversation['topic_reused'] else message
    # Carry the prior user topic, never generated claims; avoid recursively growing queries.
    query = topic + ' ' + ' '.join(str(value) for key, value in state.items()
                                  if key not in ('excluded_terms', 'budget_lkr'))
    return query, topic[:2000]
