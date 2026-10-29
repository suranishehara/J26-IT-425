"""Separate ConvNeXt-Tiny experiment; does not alter the active classifier."""
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

WEIGHTS = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
ARCHITECTURE = 'convnext_tiny_imagenet1k_v1_frozen_l2_linear'


class ConvNextFabric(nn.Module):
    def __init__(self, classes, pretrained=False):
        super().__init__()
        base = convnext_tiny(weights=WEIGHTS if pretrained else None)
        self.features = base.features
        self.pool = base.avgpool
        self.norm = base.classifier[0]
        self.head = nn.Linear(768, classes)

    def encode(self, images):
        return F.normalize(self.norm(self.pool(self.features(images))).flatten(1), dim=1)

    def forward(self, images):
        return self.head(self.encode(images))
