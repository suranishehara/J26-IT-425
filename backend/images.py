"""Decode/inspect an upload in memory; no untrained material predictions."""
import base64
import binascii
import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from .photo_review import quality, decision


def inspect(payload):
    if not isinstance(payload, dict) or payload.get('permission') is not True:
        raise ValueError('Confirm permission to process this photograph locally.')
    encoded = payload.get('image', '')
    mode = payload.get('mode', 'review_only')
    if mode not in ('review_only', 'experimental'):
        raise ValueError('Choose review_only or experimental mode.')
    model_id = payload.get('model_id')
    if mode == 'experimental':
        from .classifier import selected_config
        selected_config(model_id)
    if not isinstance(encoded, str) or not encoded.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/webp;base64,')):
        raise ValueError('Upload a PNG, JPEG or WebP photograph.')
    try:
        raw = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        if len(raw) > 6_000_000:
            raise ValueError('Image must be below 6 MB.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as original:
                if original.width * original.height > 20_000_000:
                    raise ValueError('Image exceeds 20 megapixels.')
                if min(original.size) < 64:
                    raise ValueError('Use a photograph at least 64 pixels in each dimension.')
                if original.format not in ('PNG', 'JPEG', 'WEBP'):
                    raise ValueError('Unsupported image format.')
                image = ImageOps.exif_transpose(original).convert('RGB')
                preview = image.copy()
                preview.thumbnail((1000, 1000))
                buffer = io.BytesIO()
                preview.save(buffer, format='PNG')
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError('Invalid or oversized photograph.') from None
    result = {'status': 'review_required', 'classifier': {'status': 'not_requested', 'trained': False},
            'preview': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            'image_size': list(image.size), 'predictions': {},
            'notice': 'Photograph validated locally in review-only mode. No model was requested. Enter reviewed labels; physical properties are not estimated from this photo.'}
    if mode == 'experimental':
        from .classifier import predict
        result.update(predict(image, model_id=model_id))
    result['capture_quality'] = quality(image)
    result['review_decision'] = decision(result['predictions'].get('fabric_category', []), result['capture_quality'])
    return result
