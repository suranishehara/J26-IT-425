"""EfficientNet-B0 frozen-feature baseline; no other component's model reused."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
ARCHITECTURE = 'efficientnet_b0_imagenet1k_v1_frozen_l2_linear'


class FabricNetwork(nn.Module):
    def __init__(self, classes, pretrained=False):
        super().__init__()
        base = efficientnet_b0(weights=WEIGHTS if pretrained else None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1280, classes)

    def encode(self, images):
        return F.normalize(self.pool(self.features(images)).flatten(1), dim=1)

    def forward(self, images):
        return self.head(self.encode(images))


def preprocessing():
    return WEIGHTS.transforms()


def classification_metrics(truth, predicted, labels):
    matrix = [[0] * len(labels) for _ in labels]
    for a, b in zip(truth, predicted):
        matrix[a][b] += 1
    per_class = {}
    for i, name in enumerate(labels):
        tp, support = matrix[i][i], sum(matrix[i])
        called = sum(row[i] for row in matrix)
        precision, recall = tp / called if called else 0, tp / support if support else 0
        per_class[name] = {'support': support, 'precision': precision, 'recall': recall,
                           'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}
    return {'n': len(truth), 'accuracy': sum(matrix[i][i] for i in range(len(labels))) / len(truth) if truth else None,
            'macro_f1': sum(p['f1'] for p in per_class.values()) / len(labels),
            'balanced_accuracy': sum(p['recall'] for p in per_class.values()) / len(labels),
            'per_class': per_class, 'confusion_matrix': matrix, 'label_order': labels}
