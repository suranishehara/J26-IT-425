"""Pinned non-commercial research candidate. Inference never downloads weights."""
from pathlib import Path
from torch import nn
from torch.nn import functional as F

MODEL = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
REPOSITORY = 'timm/' + MODEL
REVISION = 'b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817'
PRETRAINED_SHA256 = '6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9'
ARCHITECTURE = 'timm_convnextv2_tiny_fcmae_in22k_in1k_l2_linear'
LICENSE = 'CC-BY-NC-4.0'


class ConvNextV2Fabric(nn.Module):
    def __init__(self, classes, pretrained=False, pretrained_path=None):
        super().__init__()
        if pretrained:
            raise ValueError('Pass an explicit local pretrained_path; downloads are training-only.')
        import timm
        from timm.data import resolve_model_data_config, create_transform
        self.backbone = timm.create_model(MODEL, pretrained=False, num_classes=1000 if pretrained_path else 0)
        if pretrained_path:
            from safetensors.torch import load_file
            self.backbone.load_state_dict(load_file(str(pretrained_path), device='cpu'), strict=True)
            self.backbone.reset_classifier(0)
        self.data_config = resolve_model_data_config(self.backbone)
        self.transform = create_transform(**self.data_config, is_training=False)
        self.head = nn.Linear(self.backbone.num_features, classes)

    def encode(self, images):
        return F.normalize(self.backbone(images), dim=1)

    def forward(self, images):
        return self.head(self.encode(images))


def download_pretrained(cache):
    import hashlib
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=REPOSITORY, filename='model.safetensors', revision=REVISION,
                               cache_dir=str(cache), token=False))
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRETRAINED_SHA256:
        raise ValueError('Pinned pretrained checkpoint checksum mismatch.')
    return path


def configure_finetuning(model):
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.backbone.stages[-1], model.backbone.head, model.head):
        for parameter in module.parameters(): parameter.requires_grad_(True)
