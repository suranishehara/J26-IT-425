"""Load Component 3's bundled measurement validator and vocabulary."""
import hashlib
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / 'shared'


def load(name):
    path = SOURCE / (name + '.py')
    if not path.is_file():
        raise RuntimeError('Component 2 shared module missing: ' + str(path))
    spec = importlib.util.spec_from_file_location('component2_shared_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile_module = load('user_profile')
taxonomy_module = load('taxonomy')


def measurements(payload):
    # Only measurements and fit are consumed. Skin appearance and body-shape
    # estimates are neither requested nor used by Component 3.
    if not isinstance(payload, dict):
        raise ValueError('Profile must be an object.')
    return profile_module.validate_profile({
        'unit': payload.get('unit', 'cm'),
        'measurements': payload.get('measurements', {}),
        'fit_preference': payload.get('fit_preference', 'no_preference'),
    })


def contract():
    return {
        'taxonomy_version': taxonomy_module.TAXONOMY_VERSION,
        'pattern_type': list(taxonomy_module.ATTRIBUTES['pattern_type']),
        'fabric_texture': list(taxonomy_module.ATTRIBUTES['fabric_texture']),
        'drape_style': list(taxonomy_module.ATTRIBUTES['drape_style']),
        'measurement_fields': list(profile_module.MEASUREMENTS),
        'measurement_source_sha256': hashlib.sha256((SOURCE / 'user_profile.py').read_bytes()).hexdigest(),
        'notice': 'Drape style describes garment arrangement; drapability describes a material property. They are not interchangeable.',
    }
