"""Validate optional user-entered measurements; no photographic inference."""
import math

MEASUREMENTS = ('height', 'chest', 'waist', 'hips', 'shoulder_width', 'inseam')

def validate_profile(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a user profile object.')
    unit = payload.get('unit', 'cm')
    if unit not in ('cm', 'in'):
        raise ValueError('Measurement unit must be cm or in.')
    measurements = payload.get('measurements', {})
    if not isinstance(measurements, dict):
        raise ValueError('Measurements must be an object.')
    values = {}
    for name in MEASUREMENTS:
        raw = measurements.get(name)
        if raw is None or raw == '':
            continue
        try:
            if isinstance(raw, bool):
                raise ValueError()
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError(name.replace('_', ' ') + ' must be a number.') from None
        centimetres = value * (2.54 if unit == 'in' else 1)
        if not math.isfinite(centimetres) or not 0 < centimetres <= 400:
            raise ValueError('Check ' + name.replace('_', ' ') + ': enter a positive measurement no greater than 400 cm.')
        values[name] = round(centimetres, 2)
    fit = payload.get('fit_preference', 'no_preference')
    if fit not in ('no_preference', 'fitted', 'regular', 'relaxed'):
        raise ValueError('Unsupported fit preference.')
    preferences = {}
    for field in ('colours', 'styles'):
        items = payload.get(field, [])
        if not isinstance(items, list) or len(items) > 30 or any(not isinstance(item, str) or len(item) > 60 for item in items):
            raise ValueError(field + ' must be a list of short text values.')
        preferences[field] = list(dict.fromkeys(item.strip().lower() for item in items if item.strip()))
    mst=payload.get('skin_tone_mst')
    if mst is not None and (isinstance(mst,bool) or not isinstance(mst,int) or not 1<=mst<=10):
        raise ValueError('Monk reference must be an integer from 1 to 10 or left unset.')
    ratio=payload.get('projected_shoulder_hip_ratio')
    if ratio is not None and (isinstance(ratio,bool) or not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or not 0<ratio<=10):
        raise ValueError('Enter a positive image-space ratio no greater than 10, or leave it blank.')
    if (mst is not None or ratio is not None) and payload.get('analysis_reviewed') is not True:
        raise ValueError('Review and confirm the optional person-analysis fields before applying.')
    return {'measurements_cm':values, 'measurement_source':'user_entered',
            'fit_preference':fit, **preferences,
            'skin_tone_mst':mst,'projected_shoulder_hip_ratio':ratio,
            'analysis_reviewed':payload.get('analysis_reviewed') is True,
            'person_analysis_calibrated':False,
            'notice':'Measurements are user-entered. Fit is not assessed until garment sizing data is available.'}

def profile_from_normalized(payload):
    """Revalidate normalized browser data before using it in ranking."""
    if not isinstance(payload, dict):
        raise ValueError('Expected a user profile object.')
    return validate_profile({**payload, 'unit':'cm', 'measurements':payload.get('measurements_cm', {})})
