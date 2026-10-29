"""Validated, traceable configuration; review metadata is an assertion, not certification."""
import hashlib
import json
import math
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'config/design_rules.json'


def load(path=None):
    raw = Path(path or CONFIG).read_bytes()
    try:
        data = json.loads(raw)
        if data['schema_version'] != 1 or isinstance(data['schema_version'], bool): raise ValueError('Unsupported schema.')
        if not isinstance(data['rule_version'], str) or not data['rule_version'].strip(): raise ValueError('Missing rule version.')
        review = data['review']
        if review['status'] not in ('pending', 'reviewed'): raise ValueError('Invalid review status.')
        if any(not isinstance(review.get(key), str) for key in ('reviewer', 'reference')): raise ValueError('Invalid review metadata.')
        if review['status'] == 'reviewed' and not all(review[key].strip() for key in ('reviewer', 'reference')):
            raise ValueError('Reviewed rules require reviewer and reference.')
        def number(value, maximum):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError('Invalid rule number.')
        if set(data['weights']) != {'style', 'silhouette', 'drapability', 'weight_band'}: raise ValueError('Invalid scoring weights.')
        for value in data['weights'].values(): number(value, 100)
        number(data['relaxed_extra_cm'], 20)
        if not isinstance(data['designs'], list) or not 1 <= len(data['designs']) <= 100: raise ValueError('Invalid designs.')
        ids = set()
        for design in data['designs']:
            if not isinstance(design['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', design['id']) or design['id'] in ids: raise ValueError('Invalid or duplicate design ID.')
            ids.add(design['id'])
            if not isinstance(design['name'], str) or not design['name'].strip(): raise ValueError('Missing design name.')
            for key, allowed in [('styles', {'casual', 'office', 'formal', 'traditional'}), ('drapability', {'fluid', 'moderate', 'structured'}), ('weight', {'light', 'medium', 'heavy'})]:
                values = design[key]
                if not isinstance(values, list) or not values or any(not isinstance(value, str) or value not in allowed for value in values) or len(values) != len(set(values)):
                    raise ValueError('Invalid design ' + key)
            if not isinstance(design['ease'], dict) or not design['ease'] or not set(design['ease']) <= {'chest', 'waist', 'hips'}: raise ValueError('Invalid ease fields.')
            for bounds in design['ease'].values():
                if not isinstance(bounds, list) or len(bounds) != 2: raise ValueError('Invalid ease range.')
                for value in bounds: number(value, 100)
                if bounds[0] > bounds[1]: raise ValueError('Reversed ease range.')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed design-rule configuration.') from error
    return {**data, 'sha256': hashlib.sha256(raw).hexdigest()}
