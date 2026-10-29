"""Transparent demonstration knowledge graph and ranking engine."""

import math
from user_profile import profile_from_normalized
from sizing import assess_size
from cultural_guidance import guidance_score

RELATIONS = {"occasions": "suitable_for", "weather": "comfortable_in",
             "colours": "has_colour", "styles": "has_style", "fabric": "made_from",
             "garment_type": "has_type", "pattern_type": "has_pattern",
             "drape_style": "has_drape", "fabric_texture": "has_visible_texture",
             "textile_techniques": "uses_textile_technique"}

class KnowledgeGraph:
    def __init__(self, garments):
        self.adjacency = {}
        for garment in garments:
            edges = {}
            for field, relation in RELATIONS.items():
                values = garment.get(field, [])
                if isinstance(values, str):
                    values = [values]
                edges[relation] = set(values)
            self.adjacency[garment["id"]] = edges

    def matches(self, garment_id, relation, values):
        return self.adjacency[garment_id].get(relation, set()) & set(values)

def recommend(garments, profile):
    if not isinstance(profile, dict):
        raise ValueError("Expected a preference object.")
    user_profile = profile_from_normalized(profile['user_profile']) if 'user_profile' in profile else None
    profile = dict(profile)
    if user_profile:
        for field in ('colours', 'styles'):
            profile.setdefault(field, user_profile[field])
    try:
        budget = float(profile.get("budget", 10000))
        limit = int(profile.get("limit", 5))
    except (TypeError, ValueError):
        raise ValueError("Budget and limit must be numbers.") from None
    if not math.isfinite(budget) or budget <= 0 or not 1 <= limit <= 20:
        raise ValueError("Enter a positive budget and a limit from 1 to 20.")
    for field in ("colours", "styles", "garment_types", "patterns"):
        values = profile.get(field, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(field + " must be a list of text values.")
    for field in ("occasion", "weather"):
        if not isinstance(profile.get(field, ""), str):
            raise ValueError(field + " must be text.")
    graph = KnowledgeGraph(garments)
    mode=profile.get('recommendation_mode','baseline')
    if mode not in ('baseline','sri_lanka_guidance'):
        raise ValueError('Unknown recommendation mode.')
    results = []
    fit_counts={'compatible':0,'incompatible':0,'unknown':0,'not_assessed':0}
    strict=profile.get('only_size_compatible',False)
    if not isinstance(strict,bool):
        raise ValueError('Size-compatible filter must be true or false.')
    for garment in garments:
        if garment["price_lkr"] > budget:
            continue
        sizing=assess_size(garment,user_profile)
        fit_counts[sizing['status']]+=1
        if strict and sizing['status']!='compatible':
            continue
        price_note = 'illustrative price' if garment.get('price_source', 'illustrative') == 'illustrative' else 'user-entered price'
        reasons = ["Within your LKR budget (" + price_note + ")"]
        score = 1
        checks = [
            ("suitable_for", [profile.get("occasion", "")], 2, "Tagged for your occasion"),
            ("comfortable_in", [profile.get("weather", "")], 2, "Tagged for your weather"),
            ("has_colour", profile.get("colours", []), 3, "Matches a preferred colour"),
            ("has_style", profile.get("styles", []), 3, "Matches a preferred style"),
            ("has_type", profile.get("garment_types", []), 3, "Matches a preferred garment type"),
            ("has_pattern", profile.get("patterns", []), 2, "Matches a preferred pattern"),
        ]
        for relation, values, weight, reason in checks:
            matches = graph.matches(garment["id"], relation, values)
            if matches:
                score += weight
                reasons.append(reason + ": " + ", ".join(sorted(matches)))
        if sizing['status']=='compatible':
            score+=2
            reasons.append(sizing['reason']+' (supplied chart, not a fit guarantee)')
            if sizing['preferred_cut_match']:
                score+=1
                reasons.append('Matches your preferred garment cut.')
        bonus,cultural_reasons,rule_ids=guidance_score(graph,garment['id'],profile.get('occasion',''),mode)
        score+=bonus
        reasons.extend(cultural_reasons)
        results.append({**garment, "score": score, "reasons": reasons,'sizing':sizing,
                        'cultural_guidance':{'bonus':bonus,'rule_ids':rule_ids,'status':'proposed_requires_expert_review'}})
    results.sort(key=lambda row: (-row["score"], row["price_lkr"], row["id"]))
    return {"recommendations": results[:limit], "eligible_count": len(results),
            'recommendation_mode':mode,'cultural_rules_version':'1.0',
            "profile_context": {'measurements_provided':bool(user_profile and user_profile['measurements_cm']),
                                'person_analysis_used_for_ranking':False,
                                'fit_assessed':bool(fit_counts['compatible']+fit_counts['incompatible']),
                                'sizing_counts_within_budget':fit_counts,
                                'notice':user_profile['notice'] if user_profile else None},
            "notice": "Includes demonstration and saved garments. Prices and suitability tags need review. Scores are rule scores."}
