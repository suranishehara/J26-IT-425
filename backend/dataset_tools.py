"""IndoFashion category mapping and transparent classification metrics."""
import re

# Only map semantically compatible source labels. Others stay excluded.
INDOFASHION_LABELS = {
    'saree':'saree', 'sarees':'saree', 'blouse':'blouse', 'blouses':'blouse',
    'women kurta':'kurta', 'women kurtas':'kurta', 'kurta women':'kurta',
    'men kurta':'kurta', 'men kurtas':'kurta', 'kurta men':'kurta',
    'gown':'dress', 'gowns':'dress',
}
def map_category(label):
    if not isinstance(label,str):
        return None
    return INDOFASHION_LABELS.get(re.sub(r'[_\s-]+',' ',label.strip().lower()))

def classification_metrics(pairs):
    pairs=list(pairs)
    if not pairs:
        raise ValueError('No successful labelled predictions to evaluate.')
    labels=sorted({label for pair in pairs for label in pair})
    matrix={truth:{prediction:0 for prediction in labels} for truth in labels}
    for truth,prediction in pairs:
        matrix[truth][prediction]+=1
    per_class={}
    for label in labels:
        tp=matrix[label][label]
        support=sum(matrix[label].values())
        predicted=sum(row[label] for row in matrix.values())
        precision=tp/predicted if predicted else 0
        recall=tp/support if support else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        per_class[label]=dict(support=support,precision=precision,recall=recall,f1=f1)
    supported=[row for row in per_class.values() if row['support']]
    return dict(n=len(pairs), accuracy=sum(a==b for a,b in pairs)/len(pairs),
        macro_f1=sum(row['f1'] for row in supported)/len(supported),
        macro_f1_scope='Classes present in ground-truth labels; out-of-scope predictions still count as errors.',
        per_class=per_class,confusion_matrix=matrix)
