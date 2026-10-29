"""Metrics on complete scenario-candidate relevance judgments, grades 0..3."""
import math

def ranking_metrics(ranked_ids,judgments,k=5):
    if not isinstance(k,int) or isinstance(k,bool) or k<1:
        raise ValueError('K must be a positive integer.')
    if len(set(ranked_ids))!=len(ranked_ids):
        raise ValueError('Ranking contains duplicate IDs.')
    if any(identifier not in judgments for identifier in ranked_ids):
        raise ValueError('All ranked candidates must have judgments.')
    if any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<=3 for v in judgments.values()):
        raise ValueError('Relevance grades must be integers 0..3.')
    top=ranked_ids[:k]
    relevant=sum(judgments[identifier]>=2 for identifier in top)
    total=sum(value>=2 for value in judgments.values())
    def dcg(values):
        return sum((2**value-1)/math.log2(index+2) for index,value in enumerate(values))
    ideal=dcg(sorted(judgments.values(),reverse=True)[:k])
    return {'precision_at_k':relevant/k,'recall_at_k':relevant/total if total else None,
            'ndcg_at_k':dcg([judgments[x] for x in top])/ideal if ideal else None,
            'relevant_in_pool':total,'k':k,'judged_candidates':len(judgments),
            'precision_denominator':'K, even if fewer than K recommendations exist',
            'relevance_threshold':2}
