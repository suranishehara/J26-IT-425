"""Transparent provisional design rules, not validated tailoring advice."""
import math
from .shared_contract import contract, measurements
from .classifier import REVIEW_CATEGORIES
from .design_rules import load as load_rules

MATERIALS = ['unknown', 'cotton', 'linen', 'nylon', 'leather', 'silk', 'polyester', 'viscose', 'wool', 'blend', 'other']
TECHNIQUES = ['unknown', 'batik', 'handloom', 'other']
SOURCES = ['unknown', 'user_measured', 'supplier_documented', 'expert_reviewed', 'typical_profile']
STYLES = ['any', 'casual', 'office', 'formal', 'traditional']
DESIGNS = load_rules()['designs']  # Compatibility for tools; requests load a validated snapshot.


def choice(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError('Unsupported ' + label + '.')
    return value


def fabric_record(value):
    if not isinstance(value, dict):
        raise ValueError('Fabric must be an object.')
    if value.get('reviewed') is not True:
        raise ValueError('Review the fabric labels and properties before continuing.')
    vocab = contract()
    result = {
        'fabric_category': choice(value.get('fabric_category', 'unknown'), ['unknown'] + REVIEW_CATEGORIES, 'fabric category'),
        'material': choice(value.get('material', 'unknown'), MATERIALS, 'material'),
        'technique': choice(value.get('technique', 'unknown'), TECHNIQUES, 'technique'),
        'pattern_type': choice(value.get('pattern_type', 'unknown'), ['unknown'] + vocab['pattern_type'], 'pattern'),
        'fabric_texture': choice(value.get('fabric_texture', 'unknown'), ['unknown'] + vocab['fabric_texture'], 'texture'),
        'label_source': 'user_reviewed_not_independently_verified',
        'properties': {},
    }
    props = value.get('properties', {})
    if not isinstance(props, dict):
        raise ValueError('Fabric properties must be an object.')
    options = {'drapability': ['unknown', 'fluid', 'moderate', 'structured'],
               'weight_band': ['unknown', 'light', 'medium', 'heavy']}
    for field in ('drapability', 'weight_band', 'gsm', 'stretch_percent'):
        record = props.get(field, {})
        if not isinstance(record, dict):
            raise ValueError(field + ' requires value, source and reference.')
        raw = record.get('value')
        if raw is None or raw == '' or raw == 'unknown':
            continue
        source = choice(record.get('source', 'unknown'), SOURCES, 'property source')
        ref = record.get('reference', '')
        if source == 'unknown' or not isinstance(ref, str) or not ref.strip() or len(ref) > 500:
            raise ValueError('Known ' + field + ' needs a source and a short measurement/document reference.')
        if field in options:
            parsed = choice(raw, options[field][1:], field)
        else:
            if isinstance(raw, bool):
                raise ValueError(field + ' must be numeric.')
            try:
                parsed = float(raw)
            except (ValueError, TypeError):
                raise ValueError(field + ' must be numeric.') from None
            valid_range = 0 < parsed <= 2000 if field == 'gsm' else 0 <= parsed <= 500
            if not math.isfinite(parsed) or not valid_range:
                raise ValueError('Check ' + field + ' range.')
        result['properties'][field] = {'value': parsed, 'source': source, 'reference': ref.strip()}
    if any(prop['source'] == 'typical_profile' for prop in result['properties'].values()):
        result['record_kind'] = 'typical_profile'
    return result


def inspect_record(payload):
    """Validate a draft without saving it or asserting human review."""
    from .typical_profiles import resolve
    payload = resolve(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get('fabric'), dict):
        raise ValueError('Fabric must be an object.')
    record = fabric_record({**payload['fabric'], 'reviewed': True})
    ranking = [name for name in ('drapability', 'weight_band') if name in record['properties']]
    return {'fabric': record, 'ranking_properties': ranking,
            'context_only_properties': [name for name in ('gsm', 'stretch_percent') if name in record['properties']],
            'missing_ranking_properties': [name for name in ('drapability', 'weight_band') if name not in record['properties']],
            'can_save_specimen': bool(record['properties']) and record.get('record_kind') != 'typical_profile',
            'notice': 'Record format checked; sources are not independently verified. Review confirmation is still required before saving or recommending.'}


def recommend(payload, rules=None):
    if not isinstance(payload, dict):
        raise ValueError('Expected a design request object.')
    from .typical_profiles import resolve
    payload = resolve(payload)
    rules = load_rules() if rules is None else rules
    designs, weights = rules['designs'], rules['weights']
    fabric = fabric_record(payload.get('fabric', {}))
    profile = measurements(payload.get('profile', {}))
    style = choice(payload.get('style', 'any'), STYLES, 'style')
    silhouette = choice(payload.get('silhouette', 'any'), ['any'] + [d['id'] for d in designs], 'silhouette')
    warnings = ['Prototype rules and ease ranges are illustrative and not validated by a pattern maker. Not a cutting pattern or fit guarantee.',
                'Material, print and technique do not determine physical drape, weight or stretch. No properties are inferred from the photograph.']
    if not fabric['properties']:
        warnings.append('No recorded physical properties: these are preference-based design ideas, not fabric-validated recommendations.')
    if profile['fit_preference'] == 'fitted':
        warnings.append('Fitted designs are not supported by these positive-ease starter rules. Consult a pattern maker; no negative ease is calculated.')
    results = []
    for design in designs:
        score, reasons, conflicts = 0, [], []
        if style != 'any' and style in design['styles']:
            score += weights['style']
            reasons.append('Matches selected style: ' + style)
        if silhouette == design['id']:
            score += weights['silhouette']
            reasons.append('Matches selected silhouette.')
        for field, acceptable, points in [('drapability', design['drapability'], weights['drapability']), ('weight_band', design['weight'], weights['weight_band'])]:
            prop = fabric['properties'].get(field)
            if not prop:
                continue
            if prop['value'] in acceptable:
                score += points
                reasons.append('Matches recorded ' + field + ': ' + prop['value'] + ' (' + prop['source'] + ').')
            else:
                score -= points
                conflicts.append('Recorded ' + field + ' is outside this illustrative rule preference.')
        ranges, missing = {}, []
        for field, bounds in design['ease'].items():
            body = profile['measurements_cm'].get(field)
            if body is None:
                missing.append(field)
                continue
            # Positive design-ease illustration; stretch and fabrication details
            # are not validated sufficiently to compute negative ease.
            extra = rules['relaxed_extra_cm'] if profile['fit_preference'] == 'relaxed' else 0
            low, high = bounds[0] + extra, bounds[1] + extra
            ranges[field] = {'body_cm': body, 'illustrative_ease_cm': [low, high],
                             'draft_finished_circumference_cm': [round(body + low, 2), round(body + high, 2)]}
        results.append({'id': design['id'], 'name': design['name'], 'rule_score': score,
                        'reasons': reasons or ['Starter design idea; no positive rule matches.'],
                        'conflicts': conflicts, 'measurements': ranges, 'missing_measurements': missing,
                        'status': 'provisional_not_fit_verified', 'rule_version': rules['rule_version']})
    results.sort(key=lambda item: (-item['rule_score'], item['id']))
    return {'fabric': fabric, 'profile': profile, 'designs': results, 'warnings': warnings,
            'property_basis': 'typical_profile' if fabric.get('record_kind') == 'typical_profile' else 'user_record',
            'rule_config': {'version': rules['rule_version'], 'sha256': rules['sha256'], 'review': rules['review']},
            'notice': 'Scores are transparent rule points, not probabilities or suitability accuracy. Seam allowance is not included.'}
