"""Check the independent environment and a small real CUDA operation."""
import sys
from importlib.metadata import version

def main():
    print('Python:', sys.executable)
    for package in ('torch', 'torchvision', 'transformers', 'numpy', 'Pillow',
                    'scipy', 'networkx', 'huggingface-hub', 'safetensors', 'accelerate', 'mediapipe'):
        print(package + ':', version(package))
    import torchvision
    import transformers
    import torch
    from transformers import CLIPModel, SegformerForSemanticSegmentation
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        value = (torch.ones(4, device='cuda') * 2).sum().item()
        assert value == 8
        print('CUDA computation: passed')
    else:
        print('CPU only: recommendation engine works; future image analysis will be slower.')
    print('Environment verification passed. No model weights downloaded.')

if __name__ == '__main__':
    main()
