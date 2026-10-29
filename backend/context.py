"""Read-only adapters for explicitly supplied upstream JSON, not live service calls."""
import math


def adapt(value):
    if value is None: return None
    if not isinstance(value, dict) or type(value.get('component')) is not int or value['component'] not in (2, 3):
        raise ValueError('Context needs component 2 or 3 and its result object.')
    result = value.get('result')
    if not isinstance(result, dict): raise ValueError('Component result must be an object.')
    component = value['component']; key = 'recommendations' if component == 2 else 'designs'
    rows = result.get(key)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20: raise ValueError('Supply 1–20 actual ' + key + ' from the component response.')
    output = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str) or not row['name'].strip() or len(row['name']) > 200: raise ValueError('Each result needs a short name.')
        score = row.get('score' if component == 2 else 'rule_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score): raise ValueError('Each result needs its numeric rule score.')
        reasons = row.get('reasons', [])
        conflicts = row.get('conflicts', [])
        for items in (reasons, conflicts):
            if not isinstance(items, list) or len(items) > 30 or any(not isinstance(item, str) or len(item) > 500 for item in items): raise ValueError('Invalid component explanation list.')
        output.append({'name': row['name'], 'rule_score': score, 'reasons': reasons[:5], 'conflicts': conflicts[:5]})
    basis = result.get('property_basis', 'unspecified') if component == 3 else 'recorded_garment_tags'
    if basis not in ('unspecified', 'typical_profile', 'user_record', 'recorded_garment_tags'): raise ValueError('Unknown upstream property basis.')
    return {'component': component, 'items': output[:5], 'property_basis': basis,
            'provenance': 'user_supplied_component_result_not_live_verified'}
