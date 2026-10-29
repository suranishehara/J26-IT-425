"""Development-only scalar temperature fitting and explicit probability metrics."""
import math
import torch


def validate(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError('Expected nonempty N by C logits.')
    if labels.ndim != 1 or len(labels) != len(logits) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected aligned integer labels.')
    if not bool(torch.isfinite(logits).all()) or bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError('Non-finite logits or invalid class index.')
    return logits, labels.long()


def fit_temperature(logits, labels):
    logits, labels = validate(logits, labels)
    def loss(log_temperature):
        return float(torch.nn.functional.cross_entropy(logits / math.exp(log_temperature), labels))
    left, right = math.log(.05), math.log(20.)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    for _ in range(80):
        if loss(a) < loss(b):
            right = b; b = a; a = right - ratio * (right - left)
        else:
            left = a; a = b; b = left + ratio * (right - left)
    candidates = [0., math.log(.05), math.log(20.), (left + right) / 2]
    return math.exp(min(candidates, key=loss))


def probability_metrics(logits, labels, temperature=1., bins=10):
    logits, labels = validate(logits, labels)
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive.')
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError('Invalid bin count.')
    scaled = logits / temperature
    probabilities = scaled.softmax(1)
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(labels).double()
    target = torch.nn.functional.one_hot(labels, logits.shape[1]).double()
    rows = []; ece = 0.
    indices = torch.clamp((confidence * bins).long(), max=bins - 1)
    for index in range(bins):
        mask = indices == index; n = int(mask.sum())
        accuracy = float(correct[mask].mean()) if n else None
        mean_confidence = float(confidence[mask].mean()) if n else None
        if n: ece += n / len(labels) * abs(accuracy - mean_confidence)
        rows.append({'lower': index / bins, 'upper': (index + 1) / bins, 'n': n,
                     'accuracy': accuracy, 'mean_confidence': mean_confidence})
    return {'n': len(labels), 'accuracy': float(correct.mean()),
            'negative_log_likelihood': float(torch.nn.functional.cross_entropy(scaled, labels)),
            'multiclass_brier_sum': float(((probabilities - target) ** 2).sum(1).mean()),
            'ece_equal_width': ece, 'bins': rows, 'temperature': temperature}
