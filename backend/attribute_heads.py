"""Independent masked heads; unknown technique labels never mean false."""
import torch
from torch import nn
from .research_data import BINARY_HEADS, value, vocabulary
from .fabric_model import classification_metrics


def tasks_for(rows):
    tasks, skipped = {}, {}
    for head, choices in vocabulary().items():
        train = [value(row, head) for row in rows if row['split'] == 'train' and value(row, head) is not None]
        validation = [value(row, head) for row in rows if row['split'] == 'validation' and value(row, head) is not None]
        supported = [choice for choice in choices if choice in train]
        if len(supported) < 2 or not validation:
            skipped[head] = 'Need at least two distinct labelled training values and labelled validation samples.'
            continue
        tasks[head] = {'labels': supported, 'binary': head in BINARY_HEADS,
                       'train_known_n': len(train), 'validation_known_n': len(validation)}
    return tasks, skipped


class AttributeHeads(nn.Module):
    def __init__(self, tasks, dimension=768):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(dimension, 1 if task['binary'] else len(task['labels'])) for name, task in tasks.items()})


def targets(rows, head, labels):
    # -1 = unknown/masked; -2 = known but unsupported by this head.
    return torch.tensor([-1 if value(row, head) is None else labels.index(value(row, head)) if value(row, head) in labels else -2 for row in rows], dtype=torch.long)


def masked_loss(logits, target, binary=False):
    mask = target >= 0
    if not bool(mask.any()): return None
    truth = target[mask]
    if binary:
        positives = (truth == 1).sum().float()
        negatives = (truth == 0).sum().float()
        if not bool(positives > 0 and negatives > 0): raise ValueError('Binary training needs explicit positive and negative labels.')
        return nn.functional.binary_cross_entropy_with_logits(logits[mask].flatten(), truth.float(), pos_weight=negatives / positives)
    counts = torch.bincount(truth, minlength=logits.shape[1]).float()
    if bool((counts == 0).any()): raise ValueError('Every trained categorical output needs labelled support.')
    return nn.functional.cross_entropy(logits[mask], truth, weight=counts.sum() / (len(counts) * counts))


def evaluate_head(logits, rows, head, task):
    indices = [i for i, row in enumerate(rows) if value(row, head) is not None]
    if not indices: raise ValueError('No known validation labels.')
    known = [value(rows[i], head) for i in indices]
    labels = task['labels'] + [choice for choice in vocabulary()[head] if choice in known and choice not in task['labels']]
    prediction = (logits[indices].flatten() >= 0).long().tolist() if task['binary'] else logits[indices].argmax(1).tolist()
    # Supported labels occupy the same first positions in evaluation vocabulary.
    result = classification_metrics([labels.index(v) for v in known], prediction, [str(v) for v in labels])
    result.update(supported_labels=task['labels'], unknown_masked_n=len(rows) - len(indices),
                  out_of_scope_truth_n=sum(v not in task['labels'] for v in known),
                  macro_f1_scope='Supported training labels plus known validation-only values; unknown labels excluded, unsupported truth counted as errors.')
    return result
