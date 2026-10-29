"""Reproducible development retrieval report and unrated human-review packet.

No LLM calls, invented human ratings, training, or writes to upstream components.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.retrieval import load, retrieve
from backend import dialogue


def ranking_metrics(ranked, relevant, k):
    if not relevant:
        return {'recall_at_k': None, 'mrr_at_k': None, 'ndcg_at_k': None,
                'no_hit': not ranked[:k]}
    relevant = set(relevant)
    ranked = ranked[:k]
    hits = [i for i, value in enumerate(ranked, 1) if value in relevant]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {'recall_at_k': len(hits) / len(relevant),
            'mrr_at_k': 1 / hits[0] if hits else 0,
            'ndcg_at_k': sum(1 / math.log2(i + 1) for i in hits) / ideal,
            'no_hit': None}


def evaluate(suite, k=3, method='bm25'):
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError('K must be between 1 and 20.')
    entries, checksum = load()
    allowed = {row['id'] for row in entries if row['status'] != 'draft'}
    identifiers = set()
    results, reviews = [], []
    for case in suite['cases']:
        if case['id'] in identifiers:
            raise ValueError('Duplicate case ID.')
        identifiers.add(case['id'])
        if len(case['relevant_ids']) != len(set(case['relevant_ids'])) or set(case['relevant_ids']) - allowed:
            raise ValueError('Invalid relevant evidence IDs.')
        # Use the real dialogue preparation/state query, not a separate retrieval approximation.
        session = dialogue.new_session()
        try:
            response = dialogue.chat({'session_id': session['session_id'], 'message': case['question'], 'retrieval': method})
            # Recreate the actual query for arbitrary K, matching prepare().
            query = case['question'] + ' ' + ' '.join(str(value) for key, value in response['state'].items()
                                                     if key not in ('excluded_terms', 'budget_lkr'))
            evidence, _ = retrieve(query, k=k, method=method)
            ranked = [row['id'] for row in evidence]
            results.append({'id': case['id'], 'group': case['group'], 'ranked_ids': ranked,
                            'relevant_ids': case['relevant_ids'],
                            'embedding_model': next((row['embedding_model'] for row in evidence if 'embedding_model' in row), None),
                            **ranking_metrics(ranked, case['relevant_ids'], k)})
            reviews.append({'case_id': case['id'], 'group': case['group'], 'question': case['question'],
                            'system': 'baseline_' + method, 'answer': response['answer'],
                            'evidence': response['evidence'], 'review_focus': case['review_focus'],
                            'reviewer_id': None, 'ratings': {name: None for name in
                                ('relevance', 'factual_consistency', 'explanation_quality', 'cultural_scope')},
                            'supported_claims': None, 'total_checkable_claims': None,
                            'unsupported_claims': [], 'notes': None})
        finally:
            with dialogue.LOCK:
                dialogue.SESSIONS.pop(session['session_id'], None)
    buckets = defaultdict(list)
    for row in results:
        buckets['overall'].append(row)
        buckets[row['group']].append(row)
    summary = {}
    for group, rows in buckets.items():
        summary[group] = {'n': len(rows)}
        for metric in ('recall_at_k', 'mrr_at_k', 'ndcg_at_k', 'no_hit'):
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[group][metric] = sum(values) / len(values) if values else None
            summary[group][metric + '_n'] = len(values)
    return {'notice': suite['purpose'], 'method': method + '_state_augmented_development_only',
            'knowledge_sha256': checksum, 'k': k, 'summary': summary, 'cases': results,
            'human_ratings_status': 'not_collected'}, reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT / 'data/evaluation/development.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a report')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--method', choices=['bm25', 'dense', 'hybrid', 'all'], default='bm25')
    args = parser.parse_args()
    raw = args.suite.read_bytes()
    suite = json.loads(raw)
    if args.method == 'all':
        reports, reviews = {}, []
        for method in ('bm25', 'dense', 'hybrid'):
            item, packet = evaluate(suite, args.k, method)
            reports[method] = item
            reviews.extend(packet)
        report = {'comparison': reports, 'notice': suite['purpose'], 'human_ratings_status': 'not_collected'}
    else:
        report, reviews = evaluate(suite, args.k, args.method)
    report.update(suite_sha256=hashlib.sha256(raw).hexdigest(),
                  created_utc=datetime.now(timezone.utc).isoformat())
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, value in (('report.json', report), ('human_review.json', reviews)):
        (args.output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({method: item['summary'] for method, item in report['comparison'].items()}
                     if 'comparison' in report else report['summary'], indent=2))
    print('Development fixtures only. Human ratings not collected. Output:', args.output.resolve())


if __name__ == '__main__':
    main()
