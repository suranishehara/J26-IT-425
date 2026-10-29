"""Opt-in, checksum-pinned CPU inference. No physical-property inference."""
import hashlib
import importlib.util
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/fabric_classifier_release.json'
CATEGORIES = ['cotton', 'cotton_mixed', 'denim', 'polyester', 'silk', 'viscose', 'wool']
NINE_CATEGORIES = ['cotton', 'denim', 'leather', 'linen', 'nylon', 'polyester', 'silk', 'viscose', 'wool']
REVIEW_CATEGORIES = list(dict.fromkeys(NINE_CATEGORIES + CATEGORIES))
ARCHITECTURE = 'efficientnet_b0_imagenet1k_v1_frozen_l2_linear'
CONVNEXT_ARCHITECTURE = 'convnext_tiny_imagenet1k_v1_frozen_l2_linear'
CONVNEXTV2_ARCHITECTURE = 'timm_convnextv2_tiny_fcmae_in22k_in1k_l2_linear'
MODELS = {
    'convnext_v2': ('Experimental ConvNeXt V2-Tiny', CONVNEXTV2_ARCHITECTURE, None),
    'convnext_v1': ('Experimental ConvNeXt-Tiny', CONVNEXT_ARCHITECTURE, 'fabric_classifier_release.convnext_v1.json'),
    'efficientnet_b0': ('Experimental EfficientNet-B0', ARCHITECTURE, 'fabric_classifier_release.efficientnet_rollback.json'),
}


def selected_config(model_id=None):
    if model_id is None: return CONFIG
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise ValueError('Unsupported fabric classifier selection.')
    filename = MODELS[model_id][2]
    return ROOT / 'config' / filename if filename else CONFIG


def selected_release(model_id=None):
    run, report, checksum = verify_release(config_path=selected_config(model_id))
    if model_id is not None and report['architecture'] != MODELS[model_id][1]:
        raise ValueError('Selected model does not match its pinned architecture.')
    expected = NINE_CATEGORIES if model_id in (None, 'convnext_v2') else CATEGORIES
    if report['labels'] != expected:
        raise ValueError('Selected model does not match its pinned category order.')
    return run, report, checksum


def catalogue():
    return [{'id': model_id, 'label': spec[0], **availability(model_id)} for model_id, spec in MODELS.items()]


def model_name(architecture):
    if architecture == CONVNEXTV2_ARCHITECTURE: return 'ConvNeXt V2-Tiny'
    return 'ConvNeXt-Tiny' if architecture == CONVNEXT_ARCHITECTURE else 'EfficientNet-B0'


def notice_for(architecture, labels=CATEGORIES):
    return NOTICE.replace('seven-category', str(len(labels)) + '-category') + (' This checkpoint is CC-BY-NC-4.0; non-commercial research use only.' if architecture == CONVNEXTV2_ARCHITECTURE else '')
LOCK = threading.Lock()
_CACHE = None
NOTICE = ('Experimental seven-category suggestion, not verified fibre composition. '
          'Scores are relative softmax values, not calibrated accuracy. Unsupported fabrics '
          'may receive an incorrect supported label; there is no validated out-of-distribution detector. '
          'No batik/handloom, pattern, weave or physical properties are inferred. Review or select unknown.')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release(root=ROOT, config_path=None):
    root = Path(root).resolve()
    config_path = Path(config_path) if config_path else root / 'config/fabric_classifier_release.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('enabled') is not True or config.get('status') != 'experimental_validation_only':
        raise ValueError('Experimental classifier is disabled.')
    relative = config.get('run_directory')
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Release directory must be project-relative.')
    run = (root / relative).resolve()
    if not run.is_relative_to((root / 'runtime/training').resolve()):
        raise ValueError('Release directory must stay inside runtime/training.')
    report_path, checkpoint = run / 'report.json', run / 'model.safetensors'
    if not report_path.resolve().is_relative_to(run) or not checkpoint.resolve().is_relative_to(run):
        raise ValueError('Release files escape their run directory.')
    if sha(report_path) != config.get('report_sha256'):
        raise ValueError('Release report checksum mismatch.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    allowed = [CATEGORIES, NINE_CATEGORIES] if report.get('architecture') == CONVNEXTV2_ARCHITECTURE else [CATEGORIES]
    if report.get('architecture') not in (ARCHITECTURE, CONVNEXT_ARCHITECTURE, CONVNEXTV2_ARCHITECTURE) or report.get('labels') not in allowed:
        raise ValueError('Unsupported classifier architecture or category order.')
    if 'labels' in config and config['labels'] != report['labels']:
        raise ValueError('Pinned release category order mismatch.')
    if report.get('manifest_sha256') != config.get('manifest_sha256'):
        raise ValueError('Release manifest identity mismatch.')
    checksum = sha(checkpoint)
    if checksum != config.get('checkpoint_sha256') or checksum != report.get('checkpoint_sha256'):
        raise ValueError('Release checkpoint checksum mismatch.')
    return run, report, checksum


def availability(model_id=None):
    try:
        run, report, checksum = selected_release(model_id)
        packages = ['torch', 'torchvision', 'safetensors']
        if report['architecture'] == CONVNEXTV2_ARCHITECTURE: packages.append('timm')
        for package in packages:
            if importlib.util.find_spec(package) is None:
                raise ValueError('Install Component 3 ML dependencies to use experimental inference.')
        return {'status': 'available_experimental', 'available': True, 'trained': True,
                'model': model_name(report['architecture']) + ' · experimental', 'run': run.name,
                'checkpoint_sha256': checksum, 'categories': list(report['labels']),
                'license': report.get('license', 'See source checkpoint terms'),
                'validation_accuracy': report['validation']['accuracy'],
                'validation_macro_f1': report['validation']['macro_f1'],
                'notice': notice_for(report['architecture'], report['labels'])}
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        return {'status': 'unavailable', 'available': False, 'trained': False,
                'notice': 'Classifier unavailable: ' + str(error) + ' Manual review remains available.'}


def predict(image, model_id=None):
    """Validate files before using even the cached model; never download weights."""
    global _CACHE
    try:
        with LOCK:
            run, report, checksum = selected_release(model_id)
            labels = report['labels']
            import torch
            from safetensors.torch import load_file
            if _CACHE is None or _CACHE['checksum'] != checksum or _CACHE.get('architecture') != report['architecture'] or _CACHE.get('labels') != labels:
                torch.set_num_threads(4)
                if report['architecture'] == CONVNEXTV2_ARCHITECTURE:
                    from .convnextv2_model import ConvNextV2Fabric
                    network = ConvNextV2Fabric(len(labels)).eval()
                    transform = network.transform
                    if json.dumps(network.data_config, sort_keys=True) != json.dumps(report['training']['data_config'], sort_keys=True):
                        raise ValueError('Candidate preprocessing configuration differs from training.')
                elif report['architecture'] == CONVNEXT_ARCHITECTURE:
                    from .convnext_model import ConvNextFabric, WEIGHTS
                    network = ConvNextFabric(len(labels), pretrained=False).eval()
                    transform = WEIGHTS.transforms()
                else:
                    from .fabric_model import FabricNetwork, preprocessing
                    network = FabricNetwork(len(labels), pretrained=False).eval()
                    transform = preprocessing()
                network.load_state_dict(load_file(str(run / 'model.safetensors'), device='cpu'), strict=True)
                for parameter in network.parameters(): parameter.requires_grad_(False)
                _CACHE = {'checksum': checksum, 'architecture': report['architecture'], 'labels': list(labels), 'network': network, 'transform': transform}
            if report['architecture'] == CONVNEXTV2_ARCHITECTURE and json.dumps(_CACHE['network'].data_config, sort_keys=True) != json.dumps(report['training']['data_config'], sort_keys=True):
                raise ValueError('Candidate preprocessing configuration differs from training.')
            tensor = _CACHE['transform'](image).unsqueeze(0)
            with torch.inference_mode():
                logits = _CACHE['network'](tensor)
            if logits.shape != (1, len(labels)) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Invalid model output.')
            probabilities = logits.softmax(1)[0]
            top = probabilities.argsort(descending=True)[:3].tolist()
            suggestions = [{'category': labels[i], 'relative_score': float(probabilities[i])} for i in top]
            return {'classifier': {'status': 'experimental_review_required', 'available': True,
                                   'trained': True, 'model': model_name(report['architecture']), 'model_id': model_id, 'run': run.name,
                                   'checkpoint_sha256': checksum, 'device': 'cpu', 'categories': list(labels),
                                   'score_kind': 'uncalibrated_softmax', 'license': report.get('license', 'See source checkpoint terms'),
                                   'notice': notice_for(report['architecture'], labels)},
                    'predictions': {'fabric_category': suggestions}, 'notice': notice_for(report['architecture'], labels)}
    except Exception as error:
        # Optional inference must fail closed even for backend/DLL/safetensor
        # exceptions; the validated upload and manual review remain usable.
        return {'classifier': {'status': 'unavailable', 'available': False, 'trained': False},
                'predictions': {}, 'notice': 'Experimental inference unavailable: ' + str(error) + ' Enter reviewed labels manually.'}
