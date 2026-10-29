"""Illustrative variants, never measurements of an uploaded photo."""
PROFILES = [
    {'id': 'cotton_oxford', 'name': 'Cotton Oxford shirting', 'category': 'cotton',
     'gsm': 194, 'stretch_percent': 0, 'drapability': 'structured', 'weight_band': 'medium',
     'url': 'https://www.moodfabrics.com/products/cotton-oxford-shirting-denim-blue-deadstock-447586'},
    {'id': 'polyester_chiffon', 'name': 'Polyester chiffon', 'category': 'polyester',
     'gsm': 70, 'stretch_percent': 0, 'drapability': 'fluid', 'weight_band': 'light',
     'url': 'https://www.moodfabrics.com/products/netta-black-polyester-high-multi-chiffon-mp5000-196'},
    {'id': 'rigid_cotton_denim', 'name': 'Rigid cotton denim', 'category': 'denim',
     'gsm': 432, 'stretch_percent': 0, 'drapability': 'structured', 'weight_band': 'heavy',
     'url': 'https://www.moodfabrics.com/products/structured-cotton-denim-dark-blue-deadstock-331499'},
]


def catalogue():
    return [{'id': row['id'], 'name': row['name'], 'category': row['category'], 'source_url': row['url'],
             'fabric': fabric(row['id'])} for row in PROFILES]


def fabric(profile_id):
    if not isinstance(profile_id, str): raise ValueError('Invalid typical profile.')
    row = next((row for row in PROFILES if row['id'] == profile_id), None)
    if row is None: raise ValueError('Unknown typical profile.')
    reference = 'Typical profile; supplier example, not uploaded sample: ' + row['url']
    return {'reviewed': True, 'fabric_category': row['category'], 'material': 'unknown',
            'properties': {key: {'value': row[key], 'source': 'typical_profile', 'reference': reference}
                           for key in ('drapability', 'weight_band', 'gsm', 'stretch_percent')},
            'typical_profile_id': profile_id}


def resolve(payload):
    """Ignore client property overrides for an explicitly selected preset."""
    if not isinstance(payload, dict): raise ValueError('Expected an object.')
    if 'typical_profile_id' not in payload: return payload
    if payload.get('specimen_id'): raise ValueError('Choose a saved sample or typical profile, not both.')
    if payload.get('reviewed') is not True: raise ValueError('Confirm the typical profile before continuing.')
    return {**payload, 'fabric': fabric(payload['typical_profile_id'])}
