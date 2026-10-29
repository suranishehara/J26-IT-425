"""Compare self-reported body measurements with supplied BODY-size ranges."""
import math

def validate_sizes(charts):
    if charts is None:
        return []
    if not isinstance(charts,list) or len(charts)>30:
        raise ValueError('Size chart must be a list of up to 30 sizes.')
    labels=set()
    for chart in charts:
        if not isinstance(chart,dict) or chart.get('unit')!='cm' or chart.get('basis')!='body_measurements':
            raise ValueError('Sizing requires body-measurement ranges in cm, not finished garment dimensions.')
        label=chart.get('label')
        if not isinstance(label,str) or not label.strip() or label in labels:
            raise ValueError('Each size needs a unique label.')
        labels.add(label)
        if not isinstance(chart.get('reference'),str) or not chart['reference'].strip():
            raise ValueError('Provide the size-chart source or reference.')
        ranges=chart.get('ranges',{})
        if not isinstance(ranges,dict) or not ranges or set(ranges)-{'chest','waist','hips'}:
            raise ValueError('Supported size ranges are chest, waist and hips.')
        for bounds in ranges.values():
            if not isinstance(bounds,list) or len(bounds)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=400 for v in bounds) or bounds[0]>bounds[1]:
                raise ValueError('Size ranges need positive minimum/maximum cm values in ascending order.')
        if chart.get('cut','regular') not in ('fitted','regular','relaxed'):
            raise ValueError('Unsupported garment cut.')
    return charts

def assess_size(garment,profile):
    scope='Saree blouse only; saree drape/length not assessed.' if garment.get('garment_type')=='saree' else 'Body-size chart compatibility only; not a fit guarantee.'
    if not profile or not profile.get('measurements_cm'):
        return dict(status='not_assessed',reason='Enter body measurements to assess size.',scope=scope,matching_sizes=[])
    charts=validate_sizes(garment.get('size_chart'))
    if not charts:
        return dict(status='unknown',reason='No supplied body-size chart.',scope=scope,matching_sizes=[])
    required={'chest','waist','hips'} if garment.get('garment_type')=='dress' else {'waist','hips'} if garment.get('garment_type') in ('skirt','trousers') else {'chest','waist'}
    measured=profile['measurements_cm']
    complete=[c for c in charts if required<=set(c['ranges']) and set(c['ranges'])<=set(measured)]
    if not complete:
        return dict(status='unknown',reason='Missing required body measurements or incomplete size ranges.',scope=scope,matching_sizes=[])
    matches=[c for c in complete if all(lo<=measured[key]<=hi for key,(lo,hi) in c['ranges'].items())]
    if not matches and len(complete)<len(charts):
        return dict(status='unknown',reason='Available measurements cannot rule out the incomplete size variants.',scope=scope,matching_sizes=[])
    return dict(status='compatible' if matches else 'incompatible',matching_sizes=[c['label'] for c in matches],
                preferred_cut_match=any(c.get('cut','regular')==profile.get('fit_preference') for c in matches),
                reason='Within the supplied body-size chart: '+', '.join(c['label'] for c in matches) if matches else 'Outside the supplied complete body-size ranges.',scope=scope)
