"""Local CLIP zero-shot garment attribute baseline, loaded once per process."""
import base64
import io
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "runtime" / "models"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from taxonomy import ATTRIBUTES, PROMPT_TEMPLATES, TAXONOMY_VERSION

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
LOCK = threading.Lock()
_classifier = None

class AttributeClassifier:
    def __init__(self, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        torch.set_num_threads(4)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = CLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval().to(self.device)
        self.text_features = {}
        self.probe = None
        self.local_heads = None
        self.local_report = None
        self.dino_encoder = None
        self.dino_head = None
        self.dino_report = None
        with torch.inference_mode():
            for attribute, labels in ATTRIBUTES.items():
                descriptions = [template.format(description=description)
                                for description in labels.values() for template in PROMPT_TEMPLATES]
                inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(self.device)
                vectors = self.model.get_text_features(**inputs)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
                vectors = vectors.reshape(len(labels), len(PROMPT_TEMPLATES), -1).mean(dim=1)
                self.text_features[attribute] = vectors / vectors.norm(dim=-1, keepdim=True)

    def predict(self, image, mode='zero_shot'):
        if mode not in ('zero_shot','indofashion_probe','local_attributes','dinov2_pattern'):
            raise ValueError('Unsupported classifier mode.')
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            vector = self.model.get_image_features(**inputs)
            vector = vector / vector.norm(dim=-1, keepdim=True)
            results = {}
            for attribute, features in self.text_features.items():
                scores = (self.model.logit_scale.exp() * vector @ features.T).softmax(dim=-1)[0].cpu()
                ranked = sorted(zip(ATTRIBUTES[attribute], scores.tolist()), key=lambda item: -item[1])
                margin = ranked[0][1] - ranked[1][1]
                results[attribute] = {
                    "suggested_label": ranked[0][0],
                    "relative_score": round(ranked[0][1], 4),
                    "margin": round(margin, 4),
                    "needs_review": True,
                    "allowed_labels": list(ATTRIBUTES[attribute]),
                    "ambiguous": margin < .15,
                    "alternatives": [{"label": label, "relative_score": round(score, 4)}
                                     for label, score in ranked[:3]],
                }
            if mode=='indofashion_probe':
                from garment_probe import load_probe, PROBE_LABELS
                if self.probe is None:
                    self.probe=load_probe(self.device,MODEL_REVISION)
                scores=self.probe(vector).softmax(-1)[0].cpu().tolist()
                ranked=sorted(zip(PROBE_LABELS,scores),key=lambda item:-item[1])
                margin=ranked[0][1]-ranked[1][1]
                results['garment_type']={
                    'suggested_label':ranked[0][0], 'relative_score':round(ranked[0][1],4),
                    'margin':round(margin,4),'needs_review':True,'ambiguous':margin<.15,
                    'allowed_labels':list(ATTRIBUTES['garment_type']),
                    'alternatives':[{'label':label,'relative_score':round(score,4)} for label,score in ranked[:3]],
                    'method':'IndoFashion four-class linear probe on frozen CLIP',
                    'scope':'Predicts blouse, dress, kurta or saree only; use General CLIP for other categories.'}
        if mode=='local_attributes':
            if self.local_heads is None:
                from local_attributes import load_heads
                self.local_heads,self.local_report=load_heads(self.device,MODEL_REVISION)
            with torch.inference_mode():
                for attribute,details in self.local_heads.items():
                    scores=details['head'](vector).softmax(-1)[0].cpu().tolist()
                    ranked=sorted(zip(details['labels'],scores),key=lambda item:-item[1])
                    margin=ranked[0][1]-ranked[1][1]
                    results[attribute]={'suggested_label':ranked[0][0],'relative_score':round(ranked[0][1],4),
                        'margin':round(margin,4),'needs_review':True,'ambiguous':margin<.15,
                        'allowed_labels':list(ATTRIBUTES[attribute]),'trained_labels':details['labels'],
                        'method':('Sri Lankan two-class drape head' if attribute=='drape_style' else 'Reviewed local linear head')+' on frozen CLIP; uncalibrated score',
                        'alternatives':[{'label':label,'relative_score':round(score,4)} for label,score in ranked[:3]]}
        if mode=='dinov2_pattern':
            from dinov2_pattern import DinoEncoder,configured_run,load_pattern_head,predict_pattern
            if self.dino_encoder is None:
                directory=configured_run()
                encoder=DinoEncoder(self.device)
                head,report=load_pattern_head(directory,self.device)
                self.dino_encoder,self.dino_head,self.dino_report=encoder,head,report
            pattern=predict_pattern(self.dino_encoder,self.dino_head,self.dino_report,image)
            scores=pattern['alternatives']
            margin=scores[0]['relative_score']-scores[1]['relative_score']
            results['pattern_type']={**pattern,'margin':margin,'ambiguous':margin<.15,'needs_review':True,
                'allowed_labels':list(ATTRIBUTES['pattern_type']),
                'method':'DINOv2-Small trained pattern head; development validation only'}
        if results["garment_type"]["suggested_label"] != "saree":
            results["drape_style"]["suggested_label"] = "not_applicable"
            results["drape_style"]["relative_score"] = None
            results["drape_style"]["margin"] = None
            results["drape_style"]["rule_applied"] = "Drape style applies to saree predictions only."
        local_notice=''
        if mode=='local_attributes':
            local_notice='Local trained heads: '+', '.join(self.local_heads)+'. Missing heads use General CLIP. Each local head supports only its recorded trained labels. '
            if 'drape_style' in self.local_heads:
                local_notice+='Drape suggestions distinguish Nivi vs Kandyan/Osariya only, when General CLIP predicts saree. This is a small researcher-labelled development pilot, not final-test validation. Other drapes may be forced into a known class; choose unknown/other manually when needed. '
        if mode=='dinov2_pattern':
            local_notice='Hybrid mode: DINOv2 for solid/striped/checked/floral pattern only; General CLIP for garment type, drape and texture. Pattern head has development validation only. '
        return {"model": MODEL_ID, "revision": MODEL_REVISION,
                'classifier_mode':mode,
                "taxonomy_version": TAXONOMY_VERSION, "device": self.device,
                "attributes": results,
                "notice": local_notice+"Review all suggestions. Scores compare these candidate labels and are not calibrated accuracy. Fabric composition, batik technique and handloom production are not verified from the photograph."}

def decode_image(payload):
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
        raise ValueError("Upload a garment image.")
    value = payload["image"]
    if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
        raise ValueError("Use a PNG, JPEG or WebP image.")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Image is too large. Use at most 20 megapixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1600, 1600))
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Could not read this garment image.") from None

def analyse(payload):
    global _classifier
    image = decode_image(payload)
    mode=payload.get('classifier_mode','zero_shot')
    if mode not in ('zero_shot','indofashion_probe','local_attributes','dinov2_pattern'):
        raise ValueError('Unsupported classifier mode.')
    include_explanation=payload.get('include_explanation',False)
    if not isinstance(include_explanation,bool):
        raise ValueError('include_explanation must be true or false.')
    attribute=payload.get('explanation_attribute','garment_type')
    if include_explanation and attribute not in ATTRIBUTES:
        raise ValueError('Unsupported explanation attribute.')
    with LOCK:
        if _classifier is None:
            _classifier = AttributeClassifier()
        result=_classifier.predict(image,mode)
        if include_explanation:
            from explanations import explain
            result['explanation']=explain(_classifier,image,result,attribute)
        return result
