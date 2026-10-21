"""Inspectable English intent/slot baseline, not a trained language model."""
import re
import unicodedata

ALIASES = {'osariya': ['kandyan saree', 'kandyan sari', 'osariya', 'osari'],
           'saree': ['saree', 'sari'], 'batik': ['batik'], 'handloom': ['handloom', 'hand loom'],
           'cotton': ['cotton'], 'linen': ['linen'], 'silk': ['silk'], 'denim': ['denim'],
           'polyester': ['polyester'], 'nylon': ['nylon'], 'leather': ['leather'], 'viscose': ['viscose'], 'wool': ['wool'],
           'kurta': ['kurta'], 'blouse': ['blouse'], 'shirt': ['shirt'], 'dress': ['dress'], 'trousers': ['trousers', 'pants'], 'skirt': ['skirt']}


def normalize(text):
    return unicodedata.normalize('NFKC', text).casefold().replace('’', "'")


def terms(text):
    return re.findall(r'[a-z0-9]+', normalize(text))


def parse(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('Enter a message of 1–2000 characters.')
    lowered = normalize(text)
    if re.search(r'\b(compare|versus|vs|difference|better than)\b', lowered): intent = 'compare'
    elif re.search(r'\b(why|explain|reason)\b', lowered): intent = 'explain'
    elif re.search(r'\b(change|instead|make it|actually|switch)\b', lowered): intent = 'update_preferences'
    elif re.search(r'\b(recommend|suggest|wear|outfit|design)\b', lowered): intent = 'recommend'
    elif re.search(r'\b(what|define|meaning|how)\b', lowered): intent = 'knowledge_question'
    elif re.fullmatch(r'\s*(hi|hello|hey)[!. ]*', lowered): intent = 'greeting'
    else: intent = 'clarify'
    slots, entities, removals = {}, [], []
    occupied = []
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            for match in re.finditer(r'\b' + re.escape(alias) + r'\b', lowered):
                if any(match.start() >= a and match.end() <= b for a, b in occupied): continue
                occupied.append(match.span())
                negated = bool(re.search(r"\b(?:not|no|avoid|without|don't want)\s+(?:a\s+)?$", lowered[max(0, match.start()-25):match.start()]))
                entities.append({'type': 'fashion_term', 'value': canonical, 'mention': match.group(), 'start': match.start(), 'end': match.end(), 'negated': negated})
    slots['terms'] = list(dict.fromkeys(e['value'] for e in entities if not e['negated']))
    slots['excluded_terms'] = list(dict.fromkeys(e['value'] for e in entities if e['negated']))
    for name, mapping in {
        'occasion': {'wedding': ['wedding'], 'office': ['office', 'workplace'], 'festival': ['festival'], 'casual': ['casual outing'], 'temple': ['temple', 'religious visit']},
        'weather': {'hot': ['hot', 'warm weather'], 'cool': ['cool', 'cold'], 'rainy': ['rainy', 'rain']},
        'setting': {'outdoor': ['outdoor', 'outside'], 'indoor': ['indoor', 'inside']},
        'style': {'traditional': ['traditional'], 'formal': ['formal'], 'casual': ['casual'], 'western': ['western']},
    }.items():
        found = []
        for canonical, aliases in mapping.items():
            for alias in aliases:
                for match in re.finditer(r'\b' + re.escape(alias) + r'\b', lowered):
                    if re.search(r'\b(?:not|no|avoid)\s+$', lowered[max(0, match.start()-12):match.start()]):
                        removals.append(name); continue
                    found.append((match.start(), canonical))
        if found: slots[name] = max(found)[1]
    colours = re.findall(r'\b(blue|ivory|white|black|red|green|yellow|pink|purple|brown|beige|orange|grey|gray)\b', lowered)
    if colours: slots['colours'] = list(dict.fromkeys('grey' if value == 'gray' else value for value in colours))
    budget = re.search(r'\b(?:under|below|budget(?:\s+of)?|up to|max(?:imum)?)\s*(?:lkr|rs\.?)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k\b)?', lowered)
    if not budget: budget = re.search(r'\b(?:lkr|rs\.?)\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k\b)?', lowered)
    if budget:
        amount = float(budget[1].replace(',', '')) * (1000 if budget[2] else 1)
        if not 0 < amount <= 10_000_000: raise ValueError('Check the LKR budget.')
        slots['budget_lkr'] = amount
    if re.search(r'\b(no budget limit|remove (?:the )?budget|any budget)\b', lowered): removals.append('budget_lkr'); slots.pop('budget_lkr', None)
    return {'intent': intent, 'slots': slots, 'entities': entities, 'remove_slots': removals,
            'method': 'english_rules_v0.1', 'confidence': None}


def update(state, parsed):
    result = dict(state)
    for name in parsed['remove_slots']: result.pop(name, None)
    for name, value in parsed['slots'].items():
        if name not in ('terms', 'excluded_terms'): result[name] = value
    included = list(result.get('terms', [])); excluded = list(result.get('excluded_terms', []))
    for value in parsed['slots']['excluded_terms']:
        if value in included: included.remove(value)
        if value not in excluded: excluded.append(value)
    for value in parsed['slots']['terms']:
        if value in excluded: excluded.remove(value)
        if value not in included: included.append(value)
    result.update(terms=included, excluded_terms=excluded)
    return result
