"""Versioned, optional cultural hypotheses; never hard garment restrictions."""
import json
from pathlib import Path
RULES_PATH=Path(__file__).resolve().parents[1]/'data/cultural_guidance.json'

def guidance_score(graph,garment_id,occasion,mode):
    if mode not in ('baseline','sri_lanka_guidance'):
        raise ValueError('Unknown recommendation mode.')
    if mode=='baseline':
        return 0,[],[]
    config=json.loads(RULES_PATH.read_text(encoding='utf-8'))
    matches=[rule for rule in config['rules'] if rule['occasion']==occasion and graph.matches(garment_id,rule['relation'],rule['values'])]
    return min(config['max_bonus'],sum(rule['points'] for rule in matches)),[rule['explanation'] for rule in matches],[rule['id'] for rule in matches]
