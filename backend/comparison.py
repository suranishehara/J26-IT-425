"""Controlled property ablation, not a suitability accuracy evaluation."""
from copy import deepcopy
from .engine import recommend
from .design_rules import load as load_rules


def compare(payload):
    rules = load_rules()
    aware = recommend(payload, rules=rules)
    baseline_payload = deepcopy(payload)
    baseline_payload.pop('typical_profile_id', None)
    baseline_payload['fabric'] = dict(aware['fabric'], properties={}, reviewed=True)
    baseline = recommend(baseline_payload, rules=rules)
    base = {design['id']: (rank, design) for rank, design in enumerate(baseline['designs'], 1)}
    rows = []
    for rank, design in enumerate(aware['designs'], 1):
        base_rank, base_design = base[design['id']]
        rows.append({'id': design['id'], 'name': design['name'],
                     'property_aware_rank': rank, 'baseline_rank': base_rank,
                     'rank_gain': base_rank - rank,
                     'property_aware_score': design['rule_score'],
                     'baseline_score': base_design['rule_score'],
                     'property_score_delta': design['rule_score'] - base_design['rule_score']})
    used = [key for key in ('drapability', 'weight_band') if key in aware['fabric']['properties']]
    aware['comparison'] = {
        'version': 'property-ablation-0.1', 'rows': rows,
        'properties_used_for_ranking': used,
        'context_only_properties': sorted(set(aware['fabric']['properties']) - set(used)),
        'ranking_changed': any(row['rank_gain'] for row in rows),
        'top_design_changed': aware['designs'][0]['id'] != baseline['designs'][0]['id'],
        'notice': 'Same measurements, fit, style, silhouette and rules in both runs. Baseline removes physical properties only. '
                  'Positive rank gain means moved up; score differences are rule points, not improvement in suitability. '
                  'Ease arithmetic is unchanged by properties in these starter rules. Ties use stable design IDs. '
                  'Independent blinded suitability ratings are still required.'}
    return aware
