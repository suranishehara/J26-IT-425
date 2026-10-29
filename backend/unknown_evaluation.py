"""Metrics for an explicitly labelled, external unknown-input development set."""
import math


def metrics(rows):
    known = [r for r in rows if r['kind'] == 'supported']
    unknown = [r for r in rows if r['kind'] != 'supported']
    if not known or not unknown:
        raise ValueError('Need supported and unknown examples; neither may be fabricated.')
    if any(not math.isfinite(r['max_score']) or not 0 <= r['max_score'] <= 1 or type(r['accepted']) is not bool for r in rows):
        raise ValueError('Invalid evaluated score or decision.')
    accepted_known = [r for r in known if r['accepted']]
    # Rank AUROC: unknown-positive score = 1 - maximum supported-class score.
    # Pairwise form includes half credit for ties, and avoids threshold tuning.
    auc = sum((a['max_score'] > b['max_score']) + .5 * (a['max_score'] == b['max_score'])
              for a in known for b in unknown) / (len(known) * len(unknown))
    return {'supported_n': len(known), 'unknown_n': len(unknown),
            'unknown_detection_auroc_using_one_minus_max_score': auc,
            'supported_quick_apply_coverage': len(accepted_known) / len(known),
            'supported_top1_accuracy': sum(r['prediction'] == r['truth'] for r in known) / len(known),
            'supported_accuracy_when_quick_apply_allowed': sum(r['prediction'] == r['truth'] for r in accepted_known) / len(accepted_known) if accepted_known else None,
            'unknown_false_quick_apply_rate': sum(r['accepted'] for r in unknown) / len(unknown),
            'unknown_rejection_rate': sum(not r['accepted'] for r in unknown) / len(unknown),
            'notice': 'Image-weighted metrics; repeated specimen views are not independent. Rejection follows the decision policy stated by the enclosing report. '
                      'AUROC uses raw model scores only; neither measures verified fibre composition.'}
