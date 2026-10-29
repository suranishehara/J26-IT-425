"""Conservative UI review heuristics, not calibrated confidence or OOD detection."""
from PIL import ImageOps, ImageStat


def quality(image):
    gray = ImageOps.grayscale(image)
    gray.thumbnail((256, 256))
    histogram = gray.histogram()
    pixels = sum(histogram)
    dark = sum(histogram[:16]) / pixels
    bright = sum(histogram[240:]) / pixels
    contrast = ImageStat.Stat(gray).stddev[0]
    warnings = []
    if min(image.size) < 224:
        warnings.append('Low image resolution: capture a closer, higher-resolution fabric photograph.')
    if dark > .35:
        warnings.append('Large very dark areas: check shadows and lighting.')
    if bright > .35:
        warnings.append('Large very bright areas: check glare, exposure and background.')
    if contrast < 10:
        warnings.append('Little visible tonal detail: check focus and exposure; uniform fabric can also cause this.')
    return {'status': 'capture_review_recommended' if warnings else 'no_heuristic_warning',
            'warnings': warnings, 'dark_pixel_fraction': round(dark, 4),
            'bright_pixel_fraction': round(bright, 4), 'grayscale_stddev': round(contrast, 4),
            'notice': 'Capture heuristics only. Not a validated blur, fabric-presence or out-of-distribution detector. Passing does not prove image suitability.'}


def decision(suggestions, capture):
    if not suggestions:
        return {'status': 'no_prediction', 'suggested_category': 'unknown', 'allow_quick_apply': False,
                'reasons': ['No classifier prediction is available.']}
    top = suggestions[0]['relative_score']
    margin = top - suggestions[1]['relative_score'] if len(suggestions) > 1 else 0
    reasons = []
    if top < .60:
        reasons.append('Top relative score is below the provisional 60% review threshold.')
    if margin < .15:
        reasons.append('The two leading categories are close (less than 15 percentage points apart).')
    if capture['warnings']:
        reasons.append('Capture-quality warnings need review before using the category.')
    uncertain = bool(reasons)
    return {'status': 'uncertain_review_required' if uncertain else 'candidate_review_required',
            'suggested_category': 'unknown' if uncertain else suggestions[0]['category'],
            'allow_quick_apply': not uncertain, 'top_two_margin': round(margin, 6),
            'reasons': reasons or ['Candidate clears provisional review heuristics; human review is still required.'],
            'policy': {'version': 'review-heuristic-0.1', 'minimum_top_score': .60, 'minimum_margin': .15},
            'notice': 'Thresholds are unvalidated review heuristics, not calibrated confidence or OOD rejection. A confident prediction may still be wrong.'}
